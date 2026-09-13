import json
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import matplotlib as mpl
from matplotlib.figure import Figure
import numpy as np
from scipy.special import softmax
import pytest

from theseus.analysis import AnalysisBase, AttentionHeatmapAnalysis, AttentionAnalysisConfig
from theseus.analysis import plots
from theseus.analysis.trace import AnalysisTrace
from theseus.base import ExecutionSpec
from theseus.config import build, configuration, configure, field
from theseus.model.attention.base import SelfAttention
from theseus.model.attention.grouped import GroupedSelfAttention
from theseus.model.debug import Debugger
from theseus.registry import ANALYSES, JOBS, analysis
from theseus.training.base import BaseTrainer, BaseTrainerConfig


class Trainer(BaseTrainer):
    CONFIG = BaseTrainerConfig
    MODEL = SelfAttention
    DATASET = []

    def run(self):
        raise AssertionError("Must not train during analysis")


class AttentionJob(AttentionHeatmapAnalysis, Trainer):
    pass


def test_composition_and_order_guard():
    assert AttentionJob.run is AnalysisBase.run
    assert AttentionJob.MODEL is Trainer.MODEL
    assert issubclass(AttentionJob.CONFIG, Trainer.CONFIG)
    assert issubclass(AttentionJob.CONFIG, AttentionAnalysisConfig)
    schemas = AttentionJob.config()
    assert AttentionJob.CONFIG in schemas
    assert AttentionJob.state_restore is Trainer.state_restore

    with pytest.raises(TypeError, match="Put the analysis before the trainer"):
        class WrongOrder(Trainer, AttentionHeatmapAnalysis):
            pass

    class ExplicitOrder(Trainer, AttentionHeatmapAnalysis):
        def run(self):
            AnalysisBase.run(self)

    assert ExplicitOrder.run is not Trainer.run


def test_analysis_registration_reuses_job_decorator(monkeypatch):
    import theseus.registry as registry

    monkeypatch.setattr(registry, "_registered", True)
    register_job = Mock(wraps=registry.job)
    monkeypatch.setattr(registry, "job", register_job)
    key = "test/analyze/attention"
    try:
        assert analysis(key)(AttentionJob) is AttentionJob
        assert ANALYSES[key] is JOBS[key] is AttentionJob
        assert AttentionJob.JOB_NAME == key
        register_job.assert_called_once_with(key)
        assert AttentionHeatmapAnalysis not in ANALYSES.values()
    finally:
        ANALYSES.pop(key, None)
        JOBS.pop(key, None)
        del AttentionJob.JOB_NAME


@pytest.fixture
def captured():
    cfg = build(AttentionAnalysisConfig)
    cfg.analysis.max_tokens = 3
    model = SelfAttention(n_embd=8, n_head=2, activation_dtype="float32")
    x = jax.random.normal(jax.random.key(1), (2, 5, 8))
    variables = {"params": model.init(jax.random.key(0), x)["params"]}
    return SimpleNamespace(
        cfg=cfg, model=model, x=x, variables=variables,
        trace=lambda: model.apply(variables, x),
    )


def prepare_probe(probe, fixture):
    probe._debug_config = fixture.cfg
    probe._state = fixture.variables
    probe.batch = Mock(return_value=fixture.x)
    probe._reshape_batch = lambda batch: batch[None]
    probe._to_global = lambda batch: batch
    probe._analysis_trace = AnalysisTrace(
        lambda state, batch, key: fixture.model.apply(state, batch),
        probe.TARGET, probe.select, probe.analyze,
    )
    return probe


@pytest.mark.parametrize("kind", ["json", "figure", "none", "error"])
@pytest.mark.parametrize("remote", [False, True])
def test_run_persistence_and_capture_cleanup(captured, tmp_path, kind, remote):
    e = captured

    class Probe(AnalysisBase):
        TARGET = SelfAttention
        NAME = "probe"
        spec = ExecutionSpec.local(str(tmp_path), name="analysis")

        def find(self, target):
            return Debugger.find(e.trace, e.cfg, target)

        def debug(self, path):
            self.session = Debugger(e.trace, e.cfg, path)
            return self.session

        def select(self, paths):
            return paths[0]

        def analyze(self, layer, inputs):
            assert not jax.config.jax_disable_jit
            assert isinstance(inputs.x, jax.core.Tracer)
            if kind == "error":
                raise ValueError("analysis failed")
            return jnp.asarray(inputs.x.shape)

        def plot(self, results):
            if kind == "figure":
                return self.line(x=[0, 1], y=[1, 2])
            if kind == "json":
                return {"shape": results.tolist()}
            return None

    probe = prepare_probe(object.__new__(Probe), e)
    probe._comet = Mock() if remote else None
    probe._debug_config = e.cfg
    from theseus.base import Node
    from theseus.store import RecordStore
    probe.node = Node(name="probe", seq=7)
    probe.store = RecordStore(probe.spec.hardware)
    with jax.disable_jit(False):
        if kind == "error":
            with pytest.raises(ValueError, match="analysis failed"):
                probe.run()
        else:
            probe.run()
        assert not jax.config.jax_disable_jit
    probe.store.close()
    artifacts = probe.store.query().node(probe.node).artifact().select()
    if kind in ("json", "figure"):
        filename = "probe.json" if kind == "json" else "probe.pdf"
        payload = artifacts[0]["payload"][0]
        assert payload == {
            "filename": filename,
            "content": probe.spec.result_path(filename).read_bytes(),
        }
        assert probe.get(probe.node)["payload"] == [payload]
    else:
        assert artifacts == []
    if kind == "json":
        assert json.loads(probe.spec.result_path("probe.json").read_text()) == {"shape": [2, 5, 8]}
    elif kind == "figure":
        assert probe.spec.result_path("probe.pdf").read_bytes().startswith(b"%PDF")
    else:
        assert not probe.spec.result_path("probe.json").exists()
        assert not probe.spec.result_path("probe.pdf").exists()
    if remote:
        if kind == "figure":
            probe._comet.log_figure.assert_called_once()
            logged = probe._comet.log_figure.call_args.kwargs
            assert logged["figure_name"] == "probe"
            assert isinstance(logged["figure"], Figure)
            assert logged["step"] == 7
            probe._comet.log_asset.assert_not_called()
        elif kind == "json":
            probe._comet.log_asset.assert_called_once_with(
                file_data=str(probe.spec.result_path("probe.json")),
                file_name="probe.json",
                step=7,
            )
            probe._comet.log_figure.assert_not_called()
        else:
            assert not probe._comet.mock_calls


def test_nonzero_host_executes_but_does_not_save(captured, monkeypatch):
    e = captured

    class Probe(AnalysisBase):
        TARGET = SelfAttention
        spec = Mock()
        find = Mock(return_value=[()])
        select = Mock(return_value="")
        analyze = Mock(return_value={"value": 1})
        plot = Mock()
        artifact = Mock()

        def debug(self, path):
            return Debugger(e.trace, e.cfg, path)

    monkeypatch.setattr(jax, "process_index", lambda: 1)
    monkeypatch.setattr("theseus.analysis.base.multihost_utils.process_allgather", lambda value, **kw: value)
    probe = prepare_probe(object.__new__(Probe), e)
    probe._comet = Mock()
    probe.run()
    probe.analyze.assert_called_once()
    probe.spec.result.assert_not_called()
    probe.artifact.assert_not_called()
    assert not probe._comet.mock_calls


@pytest.mark.parametrize("missing, chosen, error", [(True, "", LookupError), (False, "missing", ValueError)])
def test_selection_fails_before_capture(captured, missing, chosen, error):
    class Probe(AnalysisBase):
        TARGET = GroupedSelfAttention if missing else SelfAttention
        select = Mock(return_value=chosen)
        debug = Mock()
        analyze = Mock()
        plot = Mock()

    probe = prepare_probe(object.__new__(Probe), captured)
    probe._comet = Mock()
    with pytest.raises(error):
        probe.run()
    probe.debug.assert_not_called()


@pytest.mark.parametrize("grouped, gated", [(False, False), (True, False), (True, True)])
def test_attention_qk_matches_projection_and_preserves_sharding(grouped, gated):
    cfg = build(AttentionAnalysisConfig)
    cfg.analysis.head = 3
    cfg.analysis.sample = 1
    cfg.analysis.max_tokens = 3
    model_type = GroupedSelfAttention if grouped else SelfAttention
    kwargs = {"n_kv_head": 2, "q_output_gate": gated} if grouped else {}
    model = model_type(n_embd=16, n_head=4, activation_dtype="float32", **kwargs)
    x = jax.random.normal(jax.random.key(1), (jax.device_count() * 2, 5, 16))
    variables = {"params": model.init(jax.random.key(0), x)["params"]}
    mesh = Mesh(np.array(jax.devices()), ("batch",))
    x = jax.device_put(x, NamedSharding(mesh, P("batch")))
    variables = jax.device_put(variables, NamedSharding(mesh, P()))
    job = object.__new__(AttentionJob)
    with configuration(cfg):
        job.args = configure(AttentionJob.CONFIG)
    before = jax.tree.map(lambda x: np.asarray(x).copy(), variables)
    with Debugger(lambda: model.apply(variables, x), cfg, "") as (layer, inputs):
        figure = job.plot(np.asarray(job.analyze(layer, inputs)))
        assert inputs.x is x
        assert inputs.x.sharding == x.sharding
        method = model._project_with_gate if gated else model.project
        projected = model.apply(variables, x, method=method)
        q, k, v = projected[:3]
        q, k, _ = model.apply(variables, q, k, v, method=model.preprocess_qkv)
        scores = np.asarray(q[1, :3, 3]) @ np.asarray(k[1, :, 1 if grouped else 3]).T / 2
        mask = np.arange(5)[None, :] <= np.arange(3)[:, None]
        expected = softmax(np.where(mask, scores, -np.inf), axis=-1)[:, :3]
        actual = figure.axes[0].collections[0].get_array().reshape(3, 3)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    for old, new in zip(jax.tree.leaves(before), jax.tree.leaves(variables)):
        np.testing.assert_array_equal(old, new)


@pytest.mark.parametrize("field, value", [("head", -1), ("sample", 20), ("max_tokens", 0)])
def test_attention_rejects_bad_selection(captured, field, value):
    e = captured
    e.cfg.analysis[field] = value
    job = object.__new__(AttentionJob)
    with configuration(e.cfg):
        job.args = configure(AttentionJob.CONFIG)
    with Debugger(e.trace, e.cfg, "") as (layer, inputs):
        with pytest.raises(ValueError):
            job.analyze(layer, inputs)
    with configuration(e.cfg):
        job.args.layer = 5
        with pytest.raises(ValueError, match="analysis/layer"):
            job.select([""])


@pytest.mark.parametrize("draw", [plots.line, plots.scatter, plots.bar, plots.heatmap])
def test_plots_have_scoped_style_and_render(draw, tmp_path):
    rc = dict(mpl.rcParams)
    data = [[1, 2], [3, 4]] if draw is plots.heatmap else None
    kwargs = {} if draw is plots.heatmap else {"x": [0, 1], "y": [1, 3]}
    figure = draw(data, title="Readable", xlabel="X", ylabel="Y", **kwargs)
    assert isinstance(figure, Figure)
    ax = figure.axes[0]
    assert ax.xaxis.label.get_fontsize() == 14.5
    assert ax.xaxis.label.get_fontfamily() == ["sans-serif"]
    assert not ax.spines["top"].get_visible()
    assert not any(line.get_visible() for line in ax.get_xgridlines())
    assert dict(mpl.rcParams) == rc
    figure.savefig(tmp_path / "plot.pdf")


def test_composed_args_use_trainer_constructor_without_state_initialization(monkeypatch):
    cfg = build(*AttentionJob.config())
    cfg.training.batch_size = 12
    cfg.analysis.head = 1
    cfg.analysis.layer = 1

    def construct(self, spec, base=None):
        self.args = configure(self.CONFIG)
        self.spec = spec

    monkeypatch.setattr(BaseTrainer, "__init__", construct)
    with configuration(cfg):
        job = AttentionJob(object())
    assert isinstance(job.args, BaseTrainerConfig)
    assert isinstance(job.args, AttentionAnalysisConfig)
    assert job.args.batch_size == 12
    assert job.args.head == 1
    assert job.select(["first", "second"]) == "second"
    assert not hasattr(job, "state")
    assert Trainer.CONFIG is BaseTrainerConfig


def test_explicit_config_composition():
    @dataclass
    class CustomTrainerConfig(BaseTrainerConfig):
        custom: int = field("training/custom", default=7)

    @dataclass
    class CombinedConfig(CustomTrainerConfig, AttentionAnalysisConfig):
        pass

    class CustomTrainer(Trainer):
        CONFIG = CustomTrainerConfig

    class CustomAnalysis(AttentionHeatmapAnalysis, CustomTrainer):
        CONFIG = CombinedConfig

    with configuration(build(*CustomAnalysis.config())):
        args = configure(CustomAnalysis.CONFIG)
    assert args.custom == 7
    assert args.head == 0
    assert args.batch_size == 512


@pytest.mark.parametrize("empty", [False, True])
def test_attention_masks_and_normalizes_before_cropping(empty):
    class FullAttention(SelfAttention):
        def build_mask(self, t, padding_mask, **kwargs):
            return jnp.ones((1, 1, t, t), dtype=bool) & (not empty)

    cfg = build(AttentionAnalysisConfig)
    cfg.analysis.max_tokens = 2
    model = FullAttention(n_embd=4, n_head=1, activation_dtype="float32")
    x = jnp.zeros((1, 4, 4))
    variables = {"params": model.init(jax.random.key(0), x)["params"]}
    job = object.__new__(AttentionJob)
    with configuration(cfg):
        job.args = configure(AttentionJob.CONFIG)
    with Debugger(lambda: model.apply(variables, x), cfg, "") as (layer, inputs):
        figure = job.plot(np.asarray(job.analyze(layer, inputs)))
    values = figure.axes[0].collections[0].get_array().reshape(2, 2)
    np.testing.assert_allclose(values, 0 if empty else .25)


@pytest.mark.parametrize("restore", [False, True])
def test_attention_job_runs_on_initialized_or_checkpointed_gpt(tmp_path, monkeypatch, restore):
    from theseus.base import Topology
    from theseus.base.chip import SUPPORTED_CHIPS
    from theseus.model.models.base import GPT

    class GPTAttention(AttentionHeatmapAnalysis):
        MODEL = GPT
        DATASET = []

    cfg = build(*GPTAttention.config())
    cfg.architecture.n_layers = 1
    cfg.architecture.n_embd = 8
    cfg.architecture.n_head = 2
    cfg.architecture.vocab_size = 16
    cfg.architecture.block_size = 4
    cfg.architecture.rope = False
    cfg.architecture.dtype.activation = "float32"
    cfg.training.batch_size = jax.device_count()
    cfg.training.per_device_batch_size = 1
    cfg.analysis.max_tokens = 4
    batch = np.tile(np.arange(4, dtype=np.int32), (jax.local_device_count(), 1))
    monkeypatch.setattr(GPTAttention, "_init_data", Mock())
    monkeypatch.setattr(GPTAttention, "evaluator", Mock(return_value=None))
    monkeypatch.setattr(GPTAttention, "batch", Mock(return_value={
        "x": batch, "y": batch, "padding_mask": np.ones_like(batch, dtype=bool),
    }))
    spec = ExecutionSpec.local(str(tmp_path), name="attention")
    spec.topology = Topology.new(SUPPORTED_CHIPS["cpu"])
    with configuration(cfg):
        base = None
        expected = None
        if restore:
            source = GPTAttention(spec)
            source.setup()
            source.tick()
            source.tick()
            source.state = source.state.replace(params=jax.tree.map(lambda x: x + .5, source.state.params))
            expected = source.state.params
            base = source.node
            source.save(source.state, {"steps": 0, "accumulate_steps": source.accumulate_steps})
            source.finish()
        job = GPTAttention(spec, base=base)
        assert not hasattr(job, "state")
        initialize = Mock(wraps=job.initialize)
        monkeypatch.setattr(job, "initialize", initialize)
        job(resume=restore)
        assert initialize.call_count == (0 if restore else 1)
        assert int(job.state.step) == 0
        if restore:
            assert job.node.serialize() == base.serialize()
            assert job.node.seq == 2
            for saved, loaded in zip(jax.tree.leaves(expected), jax.tree.leaves(job.state.params)):
                np.testing.assert_array_equal(saved, loaded)
        assert spec.result_path("attention.pdf").read_bytes().startswith(b"%PDF")


def test_trainer_collects_analysis_config_without_recursing():
    from omegaconf import OmegaConf

    @dataclass
    class HostConfig(BaseTrainerConfig):
        batch_size: int = field("training/batch_size", default=8)

    class Host(Trainer):
        CONFIG = HostConfig

    class ChildAnalysis(AttentionHeatmapAnalysis, Host):
        pass

    Host.ANALYSIS = [ChildAnalysis]
    schemas = Host.config()
    assert AttentionAnalysisConfig in schemas
    cfg = build(*schemas)
    assert cfg.analysis.head == 0
    assert cfg.training.analyze is True
    assert OmegaConf.is_missing(cfg.training, "batch_size")
    cfg.training.batch_size = 16
    with configuration(cfg):
        assert configure(Host.CONFIG).batch_size == 16
        assert configure(ChildAnalysis.CONFIG).batch_size == 16


@pytest.mark.parametrize("enabled", [False, True])
def test_init_analysis_borrows_only_when_enabled(enabled):
    child = Mock()
    trainer = SimpleNamespace(
        args=SimpleNamespace(analyze=enabled), ANALYSIS=[child],
        state=SimpleNamespace(params={}), main_process=Mock(return_value=False),
        evaluator=Mock(return_value=None),
    )
    BaseTrainer._init_counters_and_eval(trainer)
    if enabled:
        child.from_trainer.assert_called_once_with(trainer)
        assert trainer.analyses == [child.from_trainer.return_value]
    else:
        child.from_trainer.assert_not_called()
        assert trainer.analyses == []


def test_borrowed_analysis_follows_live_state_node_and_comet(captured, tmp_path, monkeypatch):
    from theseus.base import Node
    from flax.training.train_state import TrainState
    import optax

    e = captured
    trainer = SimpleNamespace(
        spec=ExecutionSpec.local(str(tmp_path), name="host"),
        store=Mock(), base=None, node=Node(name="host", nonce="abcdef"), model=e.model, mesh=None,
        state=TrainState.create(apply_fn=e.model.apply, params=e.variables["params"], tx=optax.identity()),
        sharding_context=None,
        batch=Mock(return_value=e.x),
        _reshape_batch=lambda batch: batch[None], _to_global=lambda batch: batch,
        _debug_config=e.cfg, _comet=None,
    )
    trainer.trace = Mock(side_effect=lambda state, batch, key, **kw: trainer.model.apply({"params": state.params}, batch))
    constructor = Mock(side_effect=AssertionError("Must not construct another trainer"))
    monkeypatch.setattr(AttentionJob, "__init__", constructor)
    view = AttentionJob.from_trainer(trainer)
    constructor.assert_not_called()
    view.run()
    assert view.node is trainer.node
    assert view.state is trainer.state
    old_state = trainer.state
    trainer.state = trainer.state.replace(params=jax.tree.map(lambda x: x + .1, trainer.state.params))
    trainer.batch.return_value = e.x + .2
    trainer.node.update(trainer.node.next())
    trainer._comet = Mock()
    view.run()
    assert view.state is trainer.state and view.state is not old_state
    assert view.node is trainer.node
    trainer._comet.log_figure.assert_called_once()
    assert trainer._comet.log_figure.call_args.kwargs["step"] == trainer.node.seq
    assert trainer.trace.call_count == 2  # one abstract discovery and one JIT trace
    assert trainer.batch.call_count == 2  # refreshed outside JIT every invocation
    assert trainer.spec.result_path("attention.pdf").is_file()
    trainer.store.artifact.assert_called_with(
        trainer.node, "attention", {"analysis": "attention"},
        {"filename": "attention.pdf", "content": trainer.spec.result_path("attention.pdf").read_bytes()},
    )
    view.finish()
    trainer.store.close.assert_not_called()
    trainer._comet.end.assert_not_called()
    with pytest.raises(RuntimeError, match="run"):
        view.setup()
    with pytest.raises(RuntimeError, match="clock"):
        view.tick()
    with pytest.raises(RuntimeError, match="state"):
        view.state = old_state
    assert view.node is trainer.node
    assert view.state is trainer.state


@pytest.mark.parametrize("selection", ["", [""], ["", ""], [], ["", "missing"]])
def test_selection_shape_and_validation(captured, tmp_path, selection):
    e = captured

    class Probe(AnalysisBase):
        TARGET = SelfAttention
        NAME = "multi"

        def select(self, paths):
            return selection

        def analyze(self, layers, inputs):
            if isinstance(selection, list):
                assert isinstance(layers, list) and isinstance(inputs, list)
                assert len(layers) == len(inputs) == len(selection)
                assert all(isinstance(item.x, jax.core.Tracer) for item in inputs)
                for layer, item in zip(layers, inputs):
                    layer.project(item.x)
            else:
                assert isinstance(layers, SelfAttention)
                assert isinstance(inputs.x, jax.core.Tracer)
            return {"count": len(selection) if isinstance(selection, list) else 1}

        def plot(self, results):
            return {"count": int(results["count"])}

    probe = prepare_probe(object.__new__(Probe), e)
    probe._comet = Mock()
    probe._debug_config = e.cfg
    probe.trace = Mock(side_effect=e.trace)
    probe.spec = ExecutionSpec.local(str(tmp_path), name="multi")
    probe.node = SimpleNamespace(seq=0)
    probe.artifact = Mock()
    invalid = selection == [] or "missing" in selection
    if invalid:
        with pytest.raises(ValueError):
            probe.run()
        probe.artifact.assert_not_called()
        return
    probe.run()
    count = len(selection) if isinstance(selection, list) else 1
    assert json.loads(probe.spec.result_path("multi.json").read_text()) == {"count": count}


def test_compiled_analysis_refreshes_parameters_and_inputs(captured):
    e = captured
    def analyze(layer, inputs):
        assert "trace_batch" not in inputs
        assert isinstance(inputs.x, jax.core.Tracer)
        return layer.project(inputs.x)[0]
    measured = Mock(side_effect=analyze)
    compiled = AnalysisTrace(lambda params, x, key: e.model.apply(params, x), SelfAttention, lambda paths: "", measured)
    for offset in (0., .5, 1.):
        params = jax.tree.map(lambda value: value + offset, e.variables)
        x = e.x + offset
        actual = compiled(params, x, jax.random.key(0))
        expected = e.model.apply(params, x, method=e.model.project)[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    assert measured.call_count == 1
    assert compiled.compute._cache_size() == 1
