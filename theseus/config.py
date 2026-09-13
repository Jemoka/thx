import dataclasses
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import field as f
from dataclasses import fields, is_dataclass
from typing import Any, Union, Dict, Tuple, List, TypeVar, Generator, cast
from collections import defaultdict

from flax.linen import Module as nn

from omegaconf import OmegaConf, DictConfig, ListConfig

T = TypeVar("T")
_current_config: ContextVar[DictConfig | ListConfig | None] = ContextVar(
    "_current_config", default=None
)

MISSING_VALUE = "???"


def field(
    key: str, default: Any = None, default_factory: Any = None, **kwargs: Any
) -> Any:
    if default is not None and default_factory is not None:
        raise ValueError("Cannot specify both default and default_factory")
    if default is not None:
        return f(metadata={"th_config_field": key}, default=default, **kwargs)
    elif default_factory is not None:
        return f(
            metadata={"th_config_field": key}, default_factory=default_factory, **kwargs
        )
    else:
        return f(metadata={"th_config_field": key}, **kwargs)


def dedupe(seq: Any) -> Any:
    out: List[Any] = []
    for x in seq:
        if not any(x == y for y in out):
            out.append(x)
    return out


def generate_canonical_config(
    *classes: Any, _non_recurse_cls: List[Any] = []
) -> Tuple[Dict[Any, Any], Dict[Any, Any]]:
    """generate a full configuration from components

    Args:
        *classes (List[Type]): list of dataclass types to extract fields from

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: canonical field types and default values
    """

    # if the type of any field is itself a dataclass, recurse into it
    expanded_classes = []
    for cls in classes:
        for fld in fields(cls):
            if is_dataclass(fld.type):
                expanded_classes.append(fld.type)
    if len(expanded_classes) > 0:
        return generate_canonical_config(
            *expanded_classes, _non_recurse_cls=list(classes)
        )

    classes = tuple(list(classes) + list(_non_recurse_cls))

    all_fields = [
        (j.metadata.get("th_config_field"), j.type)
        for i in classes
        for j in fields(i)
        if j.metadata.get("th_config_field") is not None
        if is_dataclass(j.type) is False
    ]

    all_fields_lub = defaultdict(list)
    for i, j in all_fields:
        all_fields_lub[i].append(j)
    canonical_fields = {
        key: value[0] if len(value) == 1 else Union[tuple(value)]
        for key, value in all_fields_lub.items()
    }

    all_field_defaults = [
        (
            j.metadata.get("th_config_field"),
            j.default
            if not isinstance(j.default, dataclasses._MISSING_TYPE)
            else j.default_factory(),  # type: ignore
        )
        for i in classes
        for j in fields(i)
        if j.metadata.get("th_config_field") is not None
        and j.default is not j.default_factory
        and is_dataclass(j.type) is False
    ]
    all_field_defaults_lub = defaultdict(list)
    for i, j in all_field_defaults:
        if j is not None:
            all_field_defaults_lub[i].append(j)
    canonical_field_defaults = {
        key: value[0] if len(dedupe(value)) == 1 else None
        for key, value in all_field_defaults_lub.items()
    }

    for i in all_fields_lub.keys():
        if i not in canonical_field_defaults:
            canonical_field_defaults[i] = None

    return canonical_fields, canonical_field_defaults


def nest_slash_keys(flat: Dict[str, Any]) -> Dict[str, Any]:
    """Nests a flat dictionary with slash-separated keys into a nested dictionary.

    Args:
        flat (Dict[str, Any]): Flat dictionary with keys containing slashes.
    Returns:
        Dict[str, Any]: Nested dictionary.
    """

    root: Dict[str, Any] = {}
    for k, v in flat.items():
        parts = k.split("/")
        cur = root
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
            if not isinstance(cur, dict):
                raise ValueError(
                    f"Key collision: {k} tries to nest under a non-dict at {p}"
                )
        leaf = parts[-1]
        if leaf in cur and isinstance(cur[leaf], dict):
            raise ValueError(f"Key collision: {k} wants a value but existing is a dict")
        cur[leaf] = v
    return root


def build_default_config(types: Dict[str, Any], defaults: Dict[str, Any]) -> DictConfig:
    """Builds a default OmegaConf configuration from types and defaults.
    Args:
        types (Dict[str, Any]): Dictionary of configuration types.
        defaults (Dict[str, Any]): Dictionary of default values.

    Returns:
        OmegaConf: OmegaConf configuration with defaults filled in.
    """

    flat_out: Dict[str, Any] = {}
    for k in types.keys():
        if k in defaults and defaults[k] is not None:
            flat_out[k] = defaults[k]
        else:
            flat_out[k] = MISSING_VALUE
    cfg = OmegaConf.create(nest_slash_keys(flat_out))
    OmegaConf.set_struct(cfg, True)
    return cfg


def build(*classes: Any) -> DictConfig:
    """Builds a default OmegaConf configuration from dataclass types.

    Args:
            *classes (DataclassInstance | type[DataclassInstance]): List of dataclass types.

    Returns:
            OmegaConf: OmegaConf configuration with defaults filled in.
    """

    types, defaults = generate_canonical_config(*classes)
    return build_default_config(types, defaults)


def hydrate(cls: Any, config: DictConfig | ListConfig, **overrides: Any) -> Any:
    """Hydrates a dataclass instance from an OmegaConf configuration.

    Args:
        cls: Dataclass type to instantiate.
        config: OmegaConf configuration.
        **overrides: Per-instance overrides applied after config-derived kwargs.
            Used to differentiate sibling instances (e.g. ``centered=True`` on a
            specific RMSNorm) without polluting the shared config tree.

    Returns:
        DataclassInstance: Instantiated dataclass with values from config.
    """

    flat_config: Dict[str, Any] = {}
    config_obj = OmegaConf.to_container(config, resolve=True)

    if isinstance(config_obj, dict):
        stack: List[Tuple[List[str], Any]] = [([], config_obj)]
        while stack:
            prefix, value = stack.pop()
            # A field may own a mapping as a value, not just scalar leaves.
            if prefix:
                flat_config["/".join(prefix)] = value
            if isinstance(value, dict):
                for key, nested in value.items():
                    stack.append((prefix + [str(key)], nested))

    init_kwargs: Dict[str, Any] = {}
    for fld in fields(cls):
        key = fld.metadata.get("th_config_field")
        if is_dataclass(fld.type):
            init_kwargs[fld.name] = hydrate(fld.type, config)
        if key is not None and key in flat_config:
            init_kwargs[fld.name] = flat_config[key]

    init_kwargs.update(overrides)

    if not issubclass(cls, nn):
        init_config = OmegaConf.create(init_kwargs)
        init_kwargs_typed = OmegaConf.merge(OmegaConf.structured(cls), init_config)

        return OmegaConf.to_object(init_kwargs_typed)
    else:
        return cls(**init_kwargs)


@contextmanager
def configuration(config: DictConfig | ListConfig) -> Generator[None, None, None]:
    """Context manager that sets the current config for configure() calls.

    Usage:
        with Configurate(config):
            model = configure(ModelConfig)
    """
    token = _current_config.set(config)
    try:
        yield
    finally:
        _current_config.reset(token)


def configure(cls: Any, **overrides: Any) -> Any:
    """Hydrates a dataclass from the current Configurate context.

    Must be called within a Configurate context manager.

    Args:
        cls: Dataclass type to instantiate.
        **overrides: Per-instance overrides applied after config-derived values.

    Returns:
        Instantiated dataclass with values from the current config.
    """
    config = _current_config.get()
    if config is None:
        raise RuntimeError("configure() must be called within a Configurate context")
    return hydrate(cls, config, **overrides)


def current_config() -> DictConfig | ListConfig | None:
    """Returns the current configuration in the Configurate context.

    Returns:
        OmegaConf | None: Current configuration or None if not in context.
    """
    return _current_config.get()


@contextmanager
def patch() -> Generator[DictConfig, None, None]:
    """Context manager that opens the current config for free-form mutation.

    If a config context is already active, temporarily disables struct mode so
    arbitrary keys can be added. Mutations persist into the outer context after
    the block — this is intentional so that callers like from_pretrained() can
    enrich a sparse backbone config with architecture fields that remain
    available for the rest of the job lifecycle.

    If no context is active, creates a fresh empty config that lives only for
    the duration of the block.

    Usage::

        with patch() as cfg:
            cfg.architecture.n_layers = 32
            cfg.architecture.attention_bias = False
            model.init(
                jax.random.PRNGKey(
                    0
                ),
                dummy,
            )
    """
    current = _current_config.get()
    if current is not None:
        OmegaConf.set_struct(current, False)
        try:
            yield cast(DictConfig, current)
        finally:
            OmegaConf.set_struct(current, True)
    else:
        cfg = OmegaConf.create({})
        token = _current_config.set(cfg)
        try:
            yield cfg
        finally:
            OmegaConf.set_struct(cfg, True)
            _current_config.reset(token)
