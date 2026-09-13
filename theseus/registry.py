"""
registry.py
Decorator-based registry for jobs, analyses, datasets, and evaluations.

Decorators (@job, @analysis, @dataset, @evaluation) can be imported cheaply and used
to register classes at definition time — no heavy submodule imports happen
until the registry dicts are actually read.

Usage:
    from theseus.registry import job, analysis, dataset, evaluation

    @job("gpt/train/pretrain")
    class PretrainGPT(BaseTrainer[...]): ...

    @analysis("gpt/analyze/attention")
    class GPTAttention(AttentionHeatmapAnalysis, GPTPretrain): ...

    @dataset("alpaca")
    class Alpaca(ChatTemplateDataset): ...

    @evaluation("bbq")
    class BBQEval(RolloutEvaluation): ...

User-defined jobs in scripts are recognized automatically — decorate with
@job/@analysis/@dataset/@evaluation and the class will coexist with built-in entries
the moment any code reads from the registry.
Analyses also appear in JOBS; the analysis templates themselves are unregistered.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Iterator, TypeVar

T = TypeVar("T")

_registered = False


def ensure_registered() -> None:
    """Trigger registration of all built-in decorated classes.

    Safe to call multiple times — only the first call imports the submodules.
    Any classes already registered via decorators (e.g. user-defined jobs)
    are preserved.
    """
    global _registered
    if _registered:
        return
    _registered = True

    import theseus.data.datasets  # noqa: F401 — dataset decorators
    import theseus.experiments  # noqa: F401 — experiment job decorators
    import theseus.evaluation.datasets  # noqa: F401 — evaluation decorators
    import theseus.analysis  # noqa: F401 — analysis job decorators


class _LazyRegistry(dict[str, Any]):
    """A dict that calls ensure_registered() on any read access."""

    # --- writes are always direct (decorators must not trigger registration) ---

    # --- reads trigger registration first ---

    def __getitem__(self, key: str) -> Any:
        ensure_registered()
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        ensure_registered()
        return super().get(key, default)

    def __contains__(self, key: object) -> bool:
        ensure_registered()
        return super().__contains__(key)

    def __iter__(self) -> Iterator[str]:
        ensure_registered()
        return super().__iter__()

    def __len__(self) -> int:
        ensure_registered()
        return super().__len__()

    def keys(self) -> Any:
        ensure_registered()
        return super().keys()

    def values(self) -> Any:
        ensure_registered()
        return super().values()

    def items(self) -> Any:
        ensure_registered()
        return super().items()

    def __repr__(self) -> str:
        ensure_registered()
        return super().__repr__()


JOBS: dict[str, type] = _LazyRegistry()
ANALYSES: dict[str, type] = _LazyRegistry()
DATASETS: dict[str, type] = _LazyRegistry()
EVALUATIONS: dict[str, Callable[[], Any]] = _LazyRegistry()


def is_builtin_job(cls: Any) -> bool:
    """Whether a class is defined in the installed Theseus package."""
    if cls.__module__ == "__main__":
        return False
    module = sys.modules.get(cls.__module__)
    source = getattr(module, "__file__", None)
    return bool(
        source
        and Path(source).resolve().is_relative_to(Path(__file__).parent.resolve())
    )


def job(key: str) -> Callable[[T], T]:
    """Register a job class under the given key."""

    def decorator(cls: T) -> T:
        setattr(cls, "JOB_NAME", key)
        setattr(cls, "JOB_BUILTIN", is_builtin_job(cls))
        JOBS[key] = cls  # type: ignore[assignment]
        return cls

    return decorator


def analysis(key: str) -> Callable[[T], T]:
    """Register an analysis as both a job and a discoverable analysis."""

    def decorator(cls: T) -> T:
        registered = job(key)(cls)
        ANALYSES[key] = registered  # type: ignore[assignment]
        return registered

    return decorator


def dataset(key: str) -> Callable[[T], T]:
    """Register a dataset class under the given key."""

    def decorator(cls: T) -> T:
        setattr(cls, "DATASET_KEY", key)
        DATASETS[key] = cls  # type: ignore[assignment]
        return cls

    return decorator


def evaluation(key: str) -> Callable[[T], T]:
    """Register an evaluation callable under the given key.

    ``PerplexityEvaluation`` subclasses must use a key ending in ``_ppl``
    so downstream consumers (e.g. the boundary-eval bar plots) can group
    unbounded perplexity metrics away from 0-1 scored ones. Bits-per-byte
    evaluations use ``_bpb`` for the same reason.
    ``PerplexityComparisonEvaluation`` returns an accuracy, not a
    perplexity, and is intentionally exempt.
    """

    def decorator(cls: T) -> T:
        setattr(cls, "EVALUATION_KEY", key)
        # Local import to avoid circular imports at module load.
        from theseus.evaluation.base import (
            BitsPerByteEvaluation,
            PerplexityComparisonEvaluation,
            PerplexityEvaluation,
        )

        if (
            isinstance(cls, type)
            and issubclass(cls, BitsPerByteEvaluation)
            and not key.endswith("_bpb")
        ):
            raise AssertionError(
                f"BitsPerByteEvaluation registered as '{key}' must use a "
                f"'_bpb' suffix (e.g. '{key}_bpb') so BPB metrics "
                f"can be grouped away from 0-1 scored evals in plots."
            )

        if (
            isinstance(cls, type)
            and issubclass(cls, PerplexityEvaluation)
            and not issubclass(cls, BitsPerByteEvaluation)
            and not issubclass(cls, PerplexityComparisonEvaluation)
            and not key.endswith("_ppl")
        ):
            raise AssertionError(
                f"PerplexityEvaluation registered as '{key}' must use a "
                f"'_ppl' suffix (e.g. '{key}_ppl') so perplexity metrics "
                f"can be grouped away from 0-1 scored evals in plots. "
                f"Note: PerplexityComparisonEvaluation is not perplexity."
            )

        EVALUATIONS[key] = cls  # type: ignore[assignment]
        return cls

    return decorator


__all__ = [
    "JOBS",
    "ANALYSES",
    "DATASETS",
    "EVALUATIONS",
    "ensure_registered",
    "is_builtin_job",
    "job",
    "analysis",
    "dataset",
    "evaluation",
]
