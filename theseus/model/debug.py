"""Explore a captured Flax invocation without continuing its execution."""

from __future__ import annotations

import copy
import inspect
from dataclasses import is_dataclass
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from functools import wraps
from contextvars import ContextVar
from typing import Any

import jax
from flax import linen as nn
from flax.core import meta
from flax.core.scope import Scope, subtract_filters
from flax.linen import module as linen_module
from loguru import logger

from theseus.config import configuration

_active: ContextVar[bool] = ContextVar("theseus_debug_active", default=False)


class _Captured(BaseException):
    """Unwind only the forward owned by this debugger."""


class DebugInputs(Mapping[str, Any]):
    """Signature-bound arguments with mapping access and notebook completion."""

    def __init__(self, values: Mapping[str, Any]) -> None:
        self._values = dict(values)

    def __getitem__(self, name: str) -> Any:
        return self._values[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._values[name]
        except KeyError:
            raise AttributeError(name) from None

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | self._values.keys())


class Debugger:
    """An open invocation: unpack as ``layer, inputs`` or use with a context.

    Captures the first call at an exact path. Parameters must already exist;
    constructors follow normal Flax naming and shape checks. Only originally
    compact methods permit inline construction. Closing discards exploration,
    but does not undo side effects of the supplied trace (including batching).
    Sessions must be closed in the Python context in which they were opened.
    """

    def __init__(
        self, trace: Callable[[], Any], config: Any, path: str | tuple[str, ...]
    ) -> None:
        if isinstance(path, str):
            path = tuple(path.split("/")) if path else ()
        if not isinstance(path, tuple) or any(
            not isinstance(part, str) or not part or "/" in part for part in path
        ):
            raise ValueError("Expected an exact slash-separated path or tuple")
        self._path = path
        self._obj: nn.Module | None = None
        self._closed = False
        self._grouped = False
        self._contexts = self._context(config)
        try:
            with nn.intercept_methods(self._intercept):
                try:
                    trace()
                except _Captured:
                    pass
            if self._obj is None:
                raise LookupError(
                    f"Trace did not call module at path {'/'.join(path)!r}"
                )
            # Flax normally owns this stack for one method call. Here its
            # lifetime deliberately spans notebook statements instead.
            linen_module._context.module_stack.append(self._obj)
            self._contexts.callback(linen_module._context.module_stack.pop)
            self._contexts.enter_context(nn.intercept_methods(self._intercept))
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _context(config: Any) -> ExitStack:
        if _active.get():
            raise RuntimeError("Close the active debugger before tracing another job")
        contexts = ExitStack()
        token = _active.set(True)
        contexts.callback(_active.reset, token)
        try:
            contexts.enter_context(configuration(copy.deepcopy(config)))
            contexts.enter_context(jax.disable_jit())
        except BaseException:
            contexts.close()
            raise
        return contexts

    @classmethod
    @contextmanager
    def multiple(
        cls, trace: Callable[[], Any], config: Any, paths: list[str]
    ) -> Iterator[tuple[list[Debugger], list[DebugInputs]]]:
        """Capture ordered paths in separate traces and explore them together.

        The default trainer trace reuses its node's cached batch; custom traces
        own their input semantics. Calls through a debugger use that layer's
        Flax scope. For inline module construction or direct access
        to nested modules, use ``with layer.activate():`` to select its scope.
        All sessions close together, including on capture or analysis failure.
        """
        if not paths:
            raise ValueError("Select at least one debug path")
        layers: list[Debugger] = []
        try:
            for path in paths:
                layer = cls(trace, config, path)
                layers.append(layer)
                # Preserve the captured invocation, but release its ambient
                # Flax/config/JIT contexts before tracing the next selection.
                layer._contexts.close()
                layer._grouped = True
            with cls._context(config):
                yield layers, [layer._inputs for layer in layers]
        finally:
            for layer in reversed(layers):
                layer.close()

    @contextmanager
    def activate(self) -> Iterator[None]:
        """Select this invocation's scope for exploration within a group."""
        self._check_open()
        if not self._grouped:
            yield
            return
        with ExitStack() as contexts:
            linen_module._context.module_stack.append(self._obj)
            contexts.callback(linen_module._context.module_stack.pop)
            contexts.enter_context(nn.intercept_methods(self._intercept))
            yield

    @classmethod
    def find(
        cls, trace: Callable[[], Any], config: Any, module_type: type[nn.Module]
    ) -> list[tuple[str, ...]]:
        """Discover distinct paths executed by a trace, in encounter order."""
        if not isinstance(module_type, type) or not issubclass(module_type, nn.Module):
            raise TypeError("find() expects a Flax module type")
        paths: dict[tuple[str, ...], None] = {}

        def discover(
            next_call: Callable[..., Any], args: Any, kwargs: Any, context: Any
        ) -> Any:
            if context.method_name == "__call__" and isinstance(
                context.module, module_type
            ):
                paths[context.module.path] = None
            return next_call(*args, **kwargs)

        with cls._context(config), nn.intercept_methods(discover):
            trace()
        return list(paths)

    def _intercept(
        self, next_call: Callable[..., Any], args: Any, kwargs: Any, context: Any
    ) -> Any:
        module = context.module
        if self._obj is None:
            if context.method_name == "__call__" and module.path == self._path:
                bound = inspect.signature(context.orig_method).bind(*args, **kwargs)
                bound.apply_defaults()
                # Traverse Python bookkeeping before deepcopy: device arrays,
                # including setup attributes and RNG keys, always retain identity.
                memo: dict[int, Any] = {}
                visited: set[int] = set()
                pending = [module, bound.arguments]
                while pending:
                    value = pending.pop()
                    if id(value) in visited:
                        continue
                    visited.add(id(value))
                    if isinstance(value, jax.core.Tracer):
                        raise RuntimeError(
                            "Debug trace must run an eager forward, outside autodiff or other JAX transforms"
                        )
                    if isinstance(value, jax.Array):
                        memo[id(value)] = value
                    elif isinstance(value, Mapping):
                        pending.extend(value.values())
                    elif isinstance(value, (tuple, list)):
                        pending.extend(value)
                    elif isinstance(value, (nn.Module, Scope)) or (
                        is_dataclass(value) and not isinstance(value, type)
                    ):
                        pending.extend(vars(value).values())
                self._obj = copy.deepcopy(module, memo)
                self._inputs = DebugInputs(copy.deepcopy(bound.arguments, memo))
                # Even a custom trace using init/mutable params cannot grant
                # the exploratory execution permission to create parameters.
                for value in memo.values():
                    if isinstance(value, Scope):
                        value.mutable = subtract_filters(value.mutable, "params")
                raise _Captured
            return next_call(*args, **kwargs)

        result = next_call(*args, **kwargs)
        assert self._obj.scope is not None
        if module.scope is not None and module.scope.root is self._obj.scope.root:
            if context.method_name == "__call__":
                for name, value in module.scope.variables().get("params", {}).items():
                    if not isinstance(value, Mapping):
                        logger.debug(
                            "DEBUG | {}/{} | reused parameter shape={}",
                            "/".join(module.path),
                            name,
                            getattr(meta.unbox(value), "shape", None),
                        )
        return result

    def __iter__(self) -> Iterator[Any]:
        self._check_open()
        yield self
        yield self._inputs

    def __enter__(self) -> tuple[Debugger, DebugInputs]:
        self._check_open()
        return self, self._inputs

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("This debugger is closed")

    def __getattr__(self, name: str) -> Any:
        self._check_open()
        if name.startswith("__") and not name.endswith("__"):
            name = f"_{type(self._obj).__name__.lstrip('_')}{name}"
        value = getattr(self._obj, name)
        if self._grouped and callable(value):

            @wraps(value)
            def call(*args: Any, **kwargs: Any) -> Any:
                with self.activate():
                    return value(*args, **kwargs)

            return call
        return value

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._check_open()
        assert self._obj is not None
        with self.activate():
            return self._obj(*args, **kwargs)

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(dir(self._obj)))

    def close(self) -> None:
        """Discard the exploratory scope and restore the surrounding context."""
        if self._closed:
            return
        self._closed = True
        try:
            self._contexts.close()
        finally:
            self._obj = None
