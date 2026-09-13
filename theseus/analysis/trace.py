"""Cached numerical analysis inside a pure trainer trace, without eager debugging."""

import inspect
from typing import Any, Callable

import flax.linen as nn
import jax

from theseus.model.debug import DebugInputs


class AnalysisTrace:
    """Discover abstractly once; execute capture and analysis in the same JIT."""

    def __init__(
        self,
        trace: Callable[..., Any],
        target: type[nn.Module],
        select: Callable[[list[str]], str | list[str]],
        analyze: Callable[..., Any],
    ) -> None:
        self.trace, self.target = trace, target
        self.select, self.analyze = select, analyze
        self.selected: str | list[str] | None = None
        self.compute = jax.jit(self._compute)

    def __call__(self, *args: Any) -> Any:
        if self.selected is None:
            paths: dict[str, None] = {}

            def discover(next_call: Any, args: Any, kwargs: Any, context: Any) -> Any:
                if context.method_name == "__call__" and isinstance(
                    context.module, self.target
                ):
                    paths["/".join(context.module.path)] = None
                return next_call(*args, **kwargs)

            with nn.intercept_methods(discover):
                jax.eval_shape(self.trace, *args)
            if not paths:
                raise LookupError(f"No modules matching {self.target.__name__}")
            selected = self.select(list(paths))
            selections = [selected] if isinstance(selected, str) else selected
            if not isinstance(selections, list) or not selections:
                raise ValueError("Select a path or a nonempty list of paths")
            if any(
                not isinstance(path, str) or path not in paths for path in selections
            ):
                raise ValueError("Selected an undiscovered module")
            self.selected = selected
        return self.compute(*args)

    def _compute(self, *args: Any) -> Any:
        selected = self.selected
        assert selected is not None
        paths = [selected] if isinstance(selected, str) else selected
        captures: dict[str, tuple[nn.Module, DebugInputs]] = {}
        result: list[Any] = []
        analyzing = False

        def capture(next_call: Any, args: Any, kwargs: Any, context: Any) -> Any:
            nonlocal analyzing
            path = "/".join(context.module.path)
            if (
                not analyzing
                and context.method_name == "__call__"
                and path in paths
                and path not in captures
            ):
                bound = inspect.signature(context.orig_method).bind(*args, **kwargs)
                bound.apply_defaults()
                captures[path] = context.module, DebugInputs(bound.arguments)
                if len(captures) == len(set(paths)):
                    layers, inputs = zip(
                        *(captures[path] for path in paths), strict=True
                    )
                    analyzing = True
                    try:
                        result.append(
                            self.analyze(
                                layers[0]
                                if isinstance(selected, str)
                                else list(layers),
                                inputs[0]
                                if isinstance(selected, str)
                                else list(inputs),
                            )
                        )
                    finally:
                        analyzing = False
            return next_call(*args, **kwargs)

        with nn.intercept_methods(capture):
            self.trace(*args)
        if not result:
            raise LookupError("The current trace no longer calls every selected module")
        return result[0]
