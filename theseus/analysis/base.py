"""Analysis jobs using the trainer's existing configuration and state lifecycle.

Use ``class MyAnalysis(AnalysisBase[C, M], MyTrainer)``. BaseTrainer owns config,
construction, restoration/surgery, and cleanup; analysis replaces run().
Pass a checkpoint as ``base=`` to restore, or omit it to analyze initialized state.
Execute through the normal job entry point in either case.
Constructing an analysis does not call setup or initialize training state.

Implement select(paths), analyze(layer, inputs), and plot(results). Analysis
returns a JAX-compatible PyTree; the framework gathers it before host zero calls
plot(), which returns a Figure, JSON value, or None. Crop/reduce large arrays in
analyze() before returning them. Analysis does not tick the trainer's node.
select() may return a string or a nonempty list of strings. A list supplies
parallel lists of bound modules and DebugInputs to analyze(), in selection order;
even a one-item list stays a list. The first invocation of each path is captured
inside one compiled trace. Notebook Debugger remains a separate eager interface.
Each JSON/PDF is saved in results and as an artifact on the current node.
Query with store.query().node(node).artifact().select(), or use get(node).
Each analysis payload contains filename and content (the exact file bytes).
Tandem analyses attach to the trainer's current node; NAME identifies the file
and artifact, so distinct analyses on the same node must use distinct names.

For tandem use, declare ANALYSIS on the trainer. from_trainer() creates a borrowed
view without construction or state initialization. Its node/state follow the
trainer. Each run reads fresh state and prepares inputs outside the cached pure
trace. The trainer calls run() at validation cadence; it alone
owns setup, ticking, and cleanup. training/analyze disables this integration.
Plotting helpers
accept CPU data and never decide what to gather. CONFIG is the combined trainer
and analysis dataclass schema; the trainer constructor hydrates it into self.args.
Compose specialized schemas explicitly through dataclass inheritance.
"""

from abc import abstractmethod
from functools import cached_property, partial
import json
from typing import Any, Generic, Self, TypeAlias

import jax
import flax.linen as nn
from jax.experimental import multihost_utils
from flax.training import train_state
from matplotlib.figure import Figure
from loguru import logger

from theseus.config import configuration, configure
from theseus.job import CometLoggingJob
from theseus.model.debug import DebugInputs
from theseus.model.module import Module
from theseus.training.base import BaseTrainer, C, M

from .plots import bar, heatmap, line, scatter
from .trace import AnalysisTrace

JSONValue: TypeAlias = (
    str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
)


class AnalysisBase(BaseTrainer[C, M], Generic[C, M]):
    """Trainer-backed analysis; subclasses implement selection and computation."""

    TARGET: type[Module]
    NAME: str = "analysis"
    _trainer: BaseTrainer[Any, M] | None = None
    _state: train_state.TrainState

    line = staticmethod(line)
    scatter = staticmethod(scatter)
    heatmap = staticmethod(heatmap)
    bar = staticmethod(bar)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        owner = next(base for base in cls.__mro__ if "run" in vars(base))
        if not issubclass(owner, AnalysisBase):
            raise TypeError(
                f"{cls.__name__} resolves run() to {owner.__name__}. "
                "Put the analysis before the trainer: "
                "class MyAnalysis(AttentionHeatmapAnalysis, MyTrainer). "
                "Alternatively, explicitly override run()."
            )

    @classmethod
    def from_trainer(cls, trainer: BaseTrainer[Any, M]) -> Self:
        """Borrow the live trainer; never initialize state or open owned resources."""
        job = object.__new__(cls)
        job._trainer = trainer
        job.node = trainer.node
        job.spec = trainer.spec
        job.store = trainer.store
        job.base = trainer.base
        job.model = trainer.model
        job.mesh = trainer.mesh
        job.sharding_context = trainer.sharding_context
        job._debug_config = trainer._debug_config
        with configuration(trainer._debug_config):
            job.args = configure(cls.CONFIG)
        return job

    @property
    def state(self) -> train_state.TrainState:
        """Observe replacement training states, including after donated JIT calls."""
        return self._trainer.state if self._trainer is not None else self._state

    @state.setter
    def state(self, state: train_state.TrainState) -> None:
        if self._trainer is not None:
            raise RuntimeError("A borrowed analysis cannot replace the trainer's state")
        self._state = state

    def batch(self, slice: str = "train") -> Any:
        if self._trainer is not None:
            return self._trainer.batch(slice)
        return super().batch(slice)

    @cached_property
    def _analysis_trace(self) -> AnalysisTrace:
        owner = self._trainer if self._trainer is not None else self
        return AnalysisTrace(
            partial(owner.trace, sharding=self.sharding_context),
            self.TARGET,
            self.select,
            self.analyze,
        )

    def setup(self, resume: bool = False) -> None:
        if self._trainer is not None:
            raise RuntimeError("Call run() directly on a borrowed analysis")
        super().setup(resume=resume)

    def tick(self) -> None:
        if self._trainer is not None:
            raise RuntimeError("A borrowed analysis uses the trainer's clock")
        super().tick()

    def finish(self) -> None:
        if self._trainer is None:
            super().finish()

    @abstractmethod
    def select(self, paths: list[str]) -> str | list[str]:
        """Choose one path or a nonempty ordered list (the root path is '')."""
        ...

    @abstractmethod
    def analyze(
        self,
        layer: nn.Module | list[nn.Module],
        inputs: DebugInputs | list[DebugInputs],
    ) -> Any:
        """Pure numerical computation; return a PyTree of arrays/scalars/None.

        A string selection supplies a bound module and DebugInputs; a list
        supplies matching lists. Do not fetch batches, read self.state, render,
        convert tracers to NumPy, or mutate Python state here. Configuration may
        be read at trace time; parameters and inputs come from the supplied layer.
        """
        ...

    @abstractmethod
    def plot(self, results: Any) -> Figure | JSONValue:
        """Render gathered CPU results on host zero; None suppresses the artifact."""
        ...

    def run(self) -> None:
        if self._trainer is None and self._comet is None:
            with configuration(self._debug_config):
                CometLoggingJob.run(self)

        owner = self._trainer if self._trainer is not None else self
        batch = owner._to_global(owner._reshape_batch(self.batch()))
        batch = jax.tree.map(lambda value: value[0], batch)
        with configuration(self._debug_config):
            result = self._analysis_trace(self.state, batch, jax.random.PRNGKey(0))
        result = multihost_utils.process_allgather(result, tiled=True)
        if jax.process_index() != 0:
            return
        result = self.plot(jax.device_get(result))
        if result is None:
            return
        comet = self._trainer._comet if self._trainer is not None else self._comet
        filename = f"{self.NAME}.{'pdf' if isinstance(result, Figure) else 'json'}"
        if isinstance(result, Figure):
            with self.spec.result(filename, mode="wb", encoding=None) as f:
                assert f is not None
                result.savefig(f, format="pdf", bbox_inches="tight")
        else:
            with self.spec.result(filename) as f:
                assert f is not None
                json.dump(result, f, indent=2, allow_nan=False)
        self.artifact(
            self.NAME,
            {"analysis": self.NAME},
            {
                "filename": filename,
                "content": self.spec.result_path(filename).read_bytes(),
            },
        )
        if comet is not None:
            if isinstance(result, Figure):
                comet.log_figure(
                    figure_name=self.NAME, figure=result, step=self.node.seq
                )
            else:
                comet.log_asset(
                    file_data=str(self.spec.result_path(filename)),
                    file_name=filename,
                    step=self.node.seq,
                )
        logger.info("ANALYSIS | {} | saved result", self.NAME)
