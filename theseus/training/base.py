"""
a very basic trainer
"""

import copy
import time
from functools import cached_property, partial
from dataclasses import asdict, dataclass, is_dataclass
from pprint import pformat
from typing import (
    Generic,
    TypeVar,
    Type,
    Dict,
    Any,
    Optional,
    List,
    Tuple,
    Callable,
    TYPE_CHECKING,
    cast,
)


import numpy as np

import jax
import jax.numpy as jnp
from jax import random as jax_random
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding, PartitionSpec as P

import flax
from flax.training import train_state

import optax

from loguru import logger

from theseus.base import ExecutionSpec, Node
from theseus.base.axis import ParameterShardingContext
from theseus.job import BasicJob, CometLoggingJob, RestoreableJob
from theseus.config import field, configure, configuration, current_config
from theseus.data.datasets import DatasetComponent

from theseus.model.module import Module
from theseus.model.debug import Debugger

from theseus.base import PyTree, Axis, Topology
from theseus.training.profiler import Profiler
from theseus.training.optimizers import AdamW, Optimizer
from theseus.training.schedules import Schedule
from theseus.training.utils import (
    find_accumulation_steps,
    scalar_metadata,
)
from theseus.training.flywheel.strategy import Strategy, Sampling
from theseus.evaluation.base import Evaluation, Evaluator

if TYPE_CHECKING:
    from theseus.analysis.base import AnalysisBase

M = TypeVar("M", bound=Module)


@dataclass
class BaseTrainerConfig:
    # Training hyperparameters
    batch_size: int = field("training/batch_size", default=512)
    per_device_batch_size: int = field(
        "training/per_device_batch_size", default=-1
    )  # -1 = auto-estimate based on VRAM
    total_tokens: int = field("training/tokens", default=1_000_000_000)

    # Learning rate schedule (WSD: Warmup-Stable-Decay)
    lr: float = field("optimization/lr", default=3e-4)
    warmup_pct: float = field("training/warmup_pct", default=0.01)
    decay_pct: float = field("training/decay_pct", default=0.1)

    # run validation or not?
    validate: bool = field("training/validation", default=True)
    evaluate: bool = field("training/evaluate", default=True)
    analyze: bool = field("training/analyze", default=True)

    # Some Architecture
    # We need to know block size for data handling; the model will
    # ask for the rest of the architecture parameters itself
    block_size: int = field("architecture/block_size", default=512)
    param_dtype: str = field("architecture/dtype/param", default="float32")
    activation_dtype: str = field("architecture/dtype/activation", default="bfloat16")

    # Logging/checkpointing
    report_interval: int = field("logging/report_interval", default=32)
    checkpoint_interval: int = field("logging/checkpoint_interval", default=1024)
    validation_interval: int = field("logging/validation_interval", default=512)
    validation_steps: int = field("training/validation_steps", default=2048)


C = TypeVar("C", bound=BaseTrainerConfig)


class BaseTrainer(RestoreableJob[C], CometLoggingJob[C], Generic[C, M]):
    """
    Generic pretrainer for GPT-style models.

    DATASET must be declared by concrete trainers (use [] for custom batching).
    ANALYSIS declares analysis classes run sequentially before validation and
    checkpointing, including the terminal step. Their CONFIG schemas join the
    trainer's config; conflicting defaults require explicit configuration.
    training/analyze disables borrowed analysis creation and execution.
    """

    MODEL: Type[M]
    CONFIG: Type[C]
    DATASET: Sampling | Type[DatasetComponent] | List[Sampling | Type[DatasetComponent]]
    EVALUATION: List[Type[Evaluation]] = []
    ANALYSIS: list[type["AnalysisBase[Any, Any]"]] = []
    SCHEDULE: Schedule[Any] | None = None
    OPTIMIZER: Optimizer[Any] = AdamW

    #### configuration ####

    @classmethod
    def config(cls) -> List[Type[Any]]:
        return (
            cls._config()
            + [cls.CONFIG, *(analysis.CONFIG for analysis in cls.ANALYSIS)]
            + super().config()
        )

    @staticmethod
    def _normalize_ds(
        datasets: Sampling
        | Type[DatasetComponent]
        | List[Sampling | Type[DatasetComponent]],
    ) -> List[Sampling]:
        """Preserve explicit rates; bare datasets equally share the remainder."""
        declared = datasets if isinstance(datasets, list) else [datasets]

        accounted_fraction = sum(
            dataset.rate for dataset in declared if isinstance(dataset, Sampling)
        )
        # Match Strategy's tolerance when explicit rates round above one.
        if accounted_fraction - 1.0 >= 1e-6:
            raise ValueError(
                f"Explicit sampling rates exceed 1, got {accounted_fraction}"
            )
        accounted_count = sum(
            1 for dataset in declared if isinstance(dataset, Sampling)
        )
        additional_fraction = max(0.0, 1.0 - accounted_fraction) / max(
            1, len(declared) - accounted_count
        )

        return [
            dataset
            if isinstance(dataset, Sampling)
            else Sampling(dataset, rate=additional_fraction)
            for dataset in declared
        ]

    @classmethod
    def _config(cls) -> List[Type[Any]]:
        cfg: List[Type[Any]] = [
            *cls.MODEL.gather(),
            *Evaluator.config(cls.EVALUATION),
            *Strategy.config(cls._normalize_ds(cls.DATASET)),
        ]

        cfg.append(cls.OPTIMIZER.config)

        if cls.SCHEDULE is not None:
            cfg.append(cls.SCHEDULE.config)

        return cfg

    @cached_property
    def schedule(self) -> optax.Schedule:
        """Build and cache the declared schedule, or a constant learning rate."""
        if self.SCHEDULE is None:
            return optax.constant_schedule(self.args.lr)
        return self.SCHEDULE.schedule(self.total_steps, configure(self.SCHEDULE.config))

    @cached_property
    def optimizer(self) -> optax.GradientTransformation:
        """Build and cache the optimizer using the trainer's cached schedule."""
        return self.OPTIMIZER.optimizer(self.schedule, configure(self.OPTIMIZER.config))

    @cached_property
    def model(self) -> M:
        return cast(M, configure(self.MODEL))

    #### construction ####

    def __init__(self, spec: ExecutionSpec, base: Optional[Node] = None) -> None:
        """Build a basic trainer

        Args:
            spec (ExecutionSpec): execution specification

        Raises:
            AssertionError: if topology is not provided in spec
        """

        self._profiler: Profiler | None = None
        super().__init__(spec, base=base)
        self._debug_config = copy.deepcopy(current_config())
        preview = spec.model_copy(update={"hardware": spec.hardware.redacted()})
        logger.info(f"TOPOLOGY | \n{preview.model_dump_json(indent=2)}\n")

        self.args = configure(self.CONFIG)
        config_dump = asdict(self.args) if is_dataclass(self.args) else self.args
        logger.info(f"CONFIG | \n{pformat(config_dump, sort_dicts=False)}\n")
        logger.debug("TRAINER | Initializing Topology")
        topology = self._init_topology(spec)
        self.model  # Hydrate while the configuration context is active.
        self.data_seed = int(jax_random.key_data(self.key)[-1])
        self.key, self.init_key, self.dropout_key = jax_random.split(self.key, num=3)
        # Populate both caches while configuration is active, before JAX tracing.
        self.optimizer
        logger.debug("TRAINER | Initializing Batch Config")
        self._init_batch_config(topology)
        logger.debug("TRAINER | Hydrating Data Loaders")
        self._init_data(spec)

    #### initialization ####

    def _init_topology(self, spec: ExecutionSpec) -> Topology:
        """Initialize topology, mesh, and compute total steps."""
        # first get the requested topology from spec
        assert spec.topology is not None, (
            "Topology must be provided to perform training"
        )

        topology = spec.topology
        self.mesh = spec.topology.mesh
        self.replicas = spec.topology.replicas
        self.local_replicas = spec.topology.local_replicas
        self.total_steps = int(
            self.args.total_tokens / self.args.batch_size / self.args.block_size
        )
        return topology

    def _init_batch_config(self, topology: Topology) -> None:
        """Compute batch size, accumulation steps, and log configuration."""
        # Compute batch size (auto-estimate or manual override)
        if self.args.per_device_batch_size < 0:
            raise ValueError(
                "You specified -1 for the per-device batch size, but auto-estimation requires running with dispatch harness which you didn't appear to use; please either use `theseus bootstrap` or `theseus submit` endpoints OR specify a positive integer for `per_device_batch_size` in your config."
            )
        else:
            fitted_bs = self.args.per_device_batch_size

        self.per_device_batch_size, self.accumulate_steps = find_accumulation_steps(
            self.args.batch_size, fitted_bs, topology
        )

        # Total micro-batches to process per node
        self.total_batches = self.total_steps * self.accumulate_steps

        # Log batch configuration
        if self.main_process():
            logger.info(
                "BATCHING | {} batchsize/node * ({} local * {} prox = {} dp) * {} accumulation = {} batchsize",
                self.per_device_batch_size,
                self.local_replicas,
                jax.process_count(),
                self.replicas,
                self.accumulate_steps,
                self.args.batch_size,
            )
            logger.info(
                "STEPS | {} micro batches // {} accumulation = {} steps",
                self.total_batches,
                self.accumulate_steps,
                self.total_batches // self.accumulate_steps,
            )
            logger.info(
                "TOKENS | {} steps * {} batchsize * {} blocksize = {} tokens",
                self.total_batches // self.accumulate_steps,
                self.args.batch_size,
                self.args.block_size,
                (self.total_batches // self.accumulate_steps)
                * self.args.batch_size
                * self.args.block_size,
            )

    def _init_data(self, spec: ExecutionSpec) -> None:
        """Initialize dataset strategy and data loaders."""
        # make dataset strategy
        self.strategy = Strategy(
            spec,
            self.args.block_size,
            self._normalize_ds(self.DATASET),
        )
        self.train_dl = self.strategy.get_async_batches(
            self.per_device_batch_size * self.local_replicas * self.accumulate_steps,
            split="train",
            node=self.node,
            seed=self.data_seed,
        )
        val_batch_size = max(
            self.per_device_batch_size * self.local_replicas,
            (
                self.args.validation_steps
                // (self.per_device_batch_size * self.local_replicas)
            )
            * (self.per_device_batch_size * self.local_replicas),
        )
        self.val_dl = self.strategy.get_async_batches(
            val_batch_size,
            split="val",
            node=Node(name="validation"),
            seed=self.data_seed,
        )

    def _init_counters_and_eval(self) -> None:
        """Initialize step counters and the evaluator."""
        self.total_params = (
            sum(x.size for x in jax.tree_util.tree_leaves(self.state.params)) / 1e6
        )
        if self.main_process():
            logger.info(f"MODEL | Total Parameters: {self.total_params:.2f}m")

        # bake evaluator
        self.inference: Evaluator[M] = self.evaluator()  # type: ignore

        # analysis (plots and stuff)
        self.analyses = (
            [analysis.from_trainer(self) for analysis in self.ANALYSIS]
            if self.args.analyze
            else []
        )

        # weeeeeeeeeeee
        # print the model
        if self.main_process():
            logger.info(self.model)

    #### state lifecycle ####

    def _init_sharding(self) -> None:
        """Resolve template layouts with Flax and validate every partition."""

        #### first we have to figure out what the state even looks ilke ####
        shapes = jax.eval_shape(self._new_state, self.init_key)
        state_pspec = flax.linen.get_partition_spec(
            shapes
        )  # <- these are *logical* names like n_embd

        #### logical -> physical mapping rules ####

        assert self.spec.topology is not None
        policy = (
            self.spec.topology.shard
        )  # <- this is the ShardingPolicy, which says what the user asked
        plan = (
            self.model.sharding
        )  # <- this is how the user tells us which logical axes maps to what hardware

        ## optimizer state mapping ##
        # If we are using ZERO-1, then we store the optimizer state into splits logically mapped by
        # the model's declared ZERO sharding policy. This is true for both storage and updates.
        # Indeed parameters maybe gathered into all ranks, so we need the gradients
        # back into a zero-shapde thing.
        optim_state_rules = plan._tp_x_zero if policy.zero else plan._tp

        ## training state *STORAGE* mapping ##
        # If we are using FSDP, then we store parameters according to mesh=(tp, zero) such that
        # things are underlying data worker sharded and on top tensor parallel sharded so that
        # we can gather into tp shards doring forward; if we are not, then we will just store it
        # using the TP order since during forward pass we save ourselves one reorganization into
        # the TP shape
        parameter_storage_rules = optim_state_rules if policy.fsdp else plan._tp
        ## training state *FORWARD/BACKWARD* mapping ##
        # during forward/backward, we shard parameters logically into TP shards. OFC if TP is not
        # on, then this gathers into all ranks fully which is indeed what we want for
        # both FSDP and Zero.
        parameter_fwdbwd_rules = plan._tp

        #### cached layouts ####

        self.state_sharding = flax.linen.logical_to_mesh_sharding(  # type: ignore[attr-defined]
            state_pspec, self.mesh, rules=parameter_storage_rules
        ).replace(
            opt_state=flax.linen.logical_to_mesh_sharding(  # type: ignore[attr-defined]
                state_pspec.opt_state, self.mesh, rules=optim_state_rules
            )
        )
        self.parameter_update_sharding = flax.linen.logical_to_mesh_sharding(  # type: ignore[attr-defined]
            state_pspec.params, self.mesh, rules=optim_state_rules
        )

        #### partition validation ####
        # we just check if people have a shape in their model that doesn't divide evenly
        # by their topology
        unboxed_shapes = flax.core.meta.map_axis_meta(
            lambda box: (
                box.unbox(apply_constraint=False)
                if isinstance(box, flax.core.meta.Partitioned)
                else box.unbox()
            ),
            shapes,
        )

        for name, layouts, values in (
            ("state", self.state_sharding, unboxed_shapes),
            ("update", self.parameter_update_sharding, unboxed_shapes.params),
        ):
            layouts = jax.tree.broadcast(layouts, values)
            for (path, value), layout in zip(
                jax.tree_util.tree_flatten_with_path(values)[0],
                jax.tree.leaves(layouts),
                strict=True,
            ):
                try:
                    layout.devices_indices_map(value.shape)
                except ValueError as error:
                    raise ValueError(
                        f"{name}{jax.tree_util.keystr(path)}: shape {value.shape}, "
                        f"partition {layout.spec}: {error}"
                    ) from error

        #### sharding for compiled training and validation ####
        # make one of the sharding contexts and then also anontate our template
        # using the sharding rules we figured out after the shenanigans
        parameter, structure = jax.tree_util.tree_flatten(self.state_sharding.params)
        update = structure.flatten_up_to(self.parameter_update_sharding)  # type: ignore[attr-defined]
        self.sharding_context = ParameterShardingContext(
            mesh=self.mesh,
            parameters=structure,
            parameter_storage_sharding=tuple(parameter),
            parameter_update_sharding=tuple(update),
            parameter_fwdbwd_mapping=parameter_fwdbwd_rules,
            parameter_update_mapping=optim_state_rules,
        )
        self.state_template: train_state.TrainState = jax.eval_shape(
            jax.jit(self._new_state, out_shardings=self.state_sharding),
            self.init_key,
        )

    def _cast_params(self, params: PyTree[jax.Array]) -> PyTree[jax.Array]:
        """Cast all params to the model's configured param dtype.

        For already-sharded JAX arrays, preserve their current sharding through
        the cast.
        """
        target = np.dtype(self.model.param_dtype)

        def cast_leaf(x: Any) -> Any:
            if isinstance(x, np.ndarray):
                return x.astype(target, copy=False)
            if isinstance(x, jax.Array):
                y = x.astype(jnp.dtype(target))
                if isinstance(x, jax.core.Tracer):
                    return y
                return jax.device_put(y, x.sharding)
            return x

        return jax.tree_util.tree_map(cast_leaf, params)  # type: ignore[no-any-return]

    def _new_state(self, key: jax.Array) -> train_state.TrainState:
        dummy_input = jnp.ones((1, self.args.block_size), dtype=jnp.int32)
        variables = self.model.init(key, dummy_input)
        params = self._cast_params(variables["params"])
        return self._make_state(params)

    def _make_state(self, params: PyTree[jax.Array]) -> train_state.TrainState:
        return train_state.TrainState.create(  # type: ignore
            apply_fn=self.model.apply, params=params, tx=self.optimizer
        )

    def _set_state(self, state: train_state.TrainState) -> None:
        self.state = state
        self.state_sharding = jax.tree.map(lambda leaf: leaf.sharding, state)
        self._init_counters_and_eval()

    @property
    def template(self) -> train_state.TrainState:
        self._init_sharding()
        return self.state_template

    def initialize(self) -> None:
        """Initialize a new training state from scratch."""
        logger.debug("TRAINER | Initializing Model Parameters, State, and Optimizers")
        sharding = jax.tree.map(lambda leaf: leaf.sharding, self.template)
        self._set_state(jax.jit(self._new_state, out_shardings=sharding)(self.init_key))

    def surgery(self, partial: PyTree[bool]) -> PyTree[Any]:
        """Initialize only state leaves missing from a restored checkpoint."""
        sharding = jax.tree.map(
            lambda leaf, needed: leaf.sharding if needed else None,
            self.template,
            partial,
        )
        return jax.jit(
            lambda key: jax.tree.map(
                lambda value, needed: value if needed else None,
                self._new_state(key),
                partial,
            ),
            out_shardings=sharding,
        )(self.init_key)

    def apply(self, state: PyTree[Any], metadata: Dict[str, Any]) -> None:
        """Install the restored state, including its completed optimizer step."""
        self._set_state(cast(train_state.TrainState, state))

    def evaluator(self) -> Optional[Evaluator[M]]:
        """define what evaluator to use"""
        return Evaluator.from_trainer(self)

    #### debugging ####

    @classmethod
    def trace(
        cls,
        state: train_state.TrainState,
        batch: PyTree[jax.Array],
        key: jax.Array,
        *,
        sharding: ParameterShardingContext,
    ) -> Any:
        """Pure inspection execution; state, batch and RNG are explicit inputs."""
        with (
            jax.sharding.use_abstract_mesh(sharding.mesh.abstract_mesh),
            flax.linen.logical_axis_rules(sharding.parameter_fwdbwd_mapping),
        ):
            return cls.forward(
                state, cast(PyTree[jax.Array], state.params), batch, key=key
            )

    def find(self, module_type: Type[Module]) -> List[Tuple[str, ...]]:
        """Discover paths using trace(); the default reuses the node's cached batch."""
        batch = self._to_global(self._reshape_batch(self.batch()))
        batch = jax.tree.map(lambda value: value[0], batch)
        return Debugger.find(
            partial(
                self.trace,
                self.state,
                batch,
                jax_random.PRNGKey(0),
                sharding=self.sharding_context,
            ),
            self._debug_config,
            module_type,
        )

    def debug(self, path: str | Tuple[str, ...]) -> Debugger:
        """Open the first invocation at an exact path; close after exploration."""
        batch = self._to_global(self._reshape_batch(self.batch()))
        batch = jax.tree.map(lambda value: value[0], batch)
        return Debugger(
            partial(
                self.trace,
                self.state,
                batch,
                jax_random.PRNGKey(0),
                sharding=self.sharding_context,
            ),
            self._debug_config,
            path,
        )

    #### batches ####

    def batch(self, slice: str = "train") -> PyTree[np.ndarray]:
        """get the next batch from the dataset strategy"""
        from typing import cast as type_cast

        if slice == "train":
            return type_cast(PyTree[np.ndarray], self.train_dl.get_batch())
        else:
            return type_cast(PyTree[np.ndarray], self.val_dl.get_batch())

    def _reshape_batch(self, batch: PyTree[np.ndarray]) -> PyTree[np.ndarray]:
        """Reshape batch for sharding; assumes dict batches."""
        from typing import cast as type_cast

        per = self.per_device_batch_size * self.local_replicas

        def _reshape(arr: np.ndarray) -> np.ndarray:
            usable = (arr.shape[0] // per) * per
            return arr[:usable].reshape(-1, per, arr.shape[-1])

        return type_cast(PyTree[np.ndarray], jax.tree_util.tree_map(_reshape, batch))

    def _to_global(self, batch: PyTree[np.ndarray]) -> PyTree[jax.Array]:
        """Move host-local numpy batch to global arrays with standard sharding."""
        from typing import cast as type_cast

        pspec = P(None, Axis.BATCH, None)  # type: ignore

        def convert(arr: np.ndarray) -> jax.Array:
            result: jax.Array = multihost_utils.host_local_array_to_global_array(
                arr,
                self.mesh,
                pspec,
            )
            return result

        return type_cast(PyTree[jax.Array], jax.tree_util.tree_map(convert, batch))

    #### steps ####

    @staticmethod
    def forward(
        state: train_state.TrainState,
        params: PyTree[jax.Array],
        batch: PyTree[jax.Array],
        key: Optional[jax.Array] = None,
        deterministic: bool = False,
        intermediates: bool = False,
    ) -> Any:
        from typing import cast as type_cast

        batch_dict = type_cast(Dict[str, jax.Array], batch)
        x = batch_dict["x"]
        y = batch_dict["y"]
        padding_mask = batch_dict["padding_mask"]

        if hasattr(state, "merge_params"):
            params = state.merge_params(params)

        dropout_key = None
        if not deterministic and key is not None:
            _, dropout_key = jax_random.split(key)

        rngs = {"dropout": dropout_key} if dropout_key is not None else {}
        mutable = ["scalars"]
        if intermediates:
            mutable.extend(("intermediates", "plots"))

        (logits, loss), mutated = state.apply_fn(
            {"params": params},
            x,
            y,
            padding_mask=padding_mask,
            deterministic=deterministic,
            rngs=rngs,
            mutable=mutable,
        )
        meta: Dict[str, Any] = scalar_metadata(mutated.get("scalars", {}))
        if intermediates:
            meta.update(
                {
                    "intermediates": mutated.get("intermediates", {}),
                    "plots": mutated.get("plots", {}),
                }
            )
        return logits, loss, meta

    @classmethod
    @jax.named_call
    def train_step(
        cls,
        state: train_state.TrainState,
        batch: PyTree[jax.Array],  # (S, B, T) each
        key: jax.Array,
        accumulate_steps: int,
        *,
        sharding: ParameterShardingContext,
        dtype: str = "float32",
        fsdp: bool = False,
        activation_checkpointing: bool = False,
    ) -> Tuple[train_state.TrainState, jax.Array, Any, jax.Array]:
        """Compute gradients over S micro-batches and apply one optimizer step.

        Args:
            state: Current training state
            batch: (x, y, padding_mask) each with shape (S, B, T)
                   S = accumulation steps, B = batch size, T = sequence length
            key: PRNG key for dropout
            accumulate_steps: Number of micro-batches (S)

        Returns:
            (updated_state, loss, meta, grad_norm); meta is from the last micro-batch
        """

        parameter_storage_sharding = sharding.parameters.unflatten(
            sharding.parameter_storage_sharding
        )
        parameter_update_sharding = sharding.parameters.unflatten(
            sharding.parameter_update_sharding
        )
        compute_params = jax.tree.map(lambda p: p.astype(dtype), state.params)
        compute_params = jax.lax.with_sharding_constraint(  # type: ignore[no-untyped-call]
            compute_params, parameter_storage_sharding
        )

        def train_eval(
            state: train_state.TrainState,
            params: PyTree[jax.Array],
            batch: PyTree[jax.Array],  # (B, T) each
            key: jax.Array,
            accumulate_steps: int,
        ) -> Tuple[jax.Array, PyTree[jax.Array], Any]:
            payload = batch

            @partial(jax.checkpoint, policy=None if activation_checkpointing else lambda op, *_, **__: op.name not in {"sharding_constraint", "convert_element_type", "reshape", "transpose"})  # fmt: skip
            def loss_fn(params: PyTree[jax.Array]) -> Tuple[jax.Array, Any]:
                with (
                    jax.named_scope("forward"),
                    jax.sharding.use_abstract_mesh(sharding.mesh.abstract_mesh),
                    flax.linen.logical_axis_rules(sharding.parameter_fwdbwd_mapping),
                ):
                    logits, loss, meta = cls.forward(
                        state,
                        params,
                        payload,
                        key=key,
                        deterministic=False,
                    )
                return loss / accumulate_steps, meta

            with jax.named_scope("forward_backward"):
                (loss, meta), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                    params
                )  # loss: scalar, grads: PyTree

            return loss, grads, meta

        def reduce(
            carry: Tuple[PyTree[jax.Array], jax.Array, jax.Array],
            batch_item: Any,  # PyTree with single batch (B, T)
        ) -> Tuple[Tuple[PyTree[jax.Array], jax.Array, jax.Array], Any]:
            grad, loss, key = carry
            params = compute_params
            if fsdp:
                # Bound FSDP memory across microbatches; allow prefetch between layers.
                grad, loss, key, params, batch_item = jax.lax.optimization_barrier(  # type: ignore[no-untyped-call]
                    (grad, loss, key, params, batch_item)
                )
            key, subkey = jax_random.split(key)
            loss_single, grad_single, meta = train_eval(
                state, params, batch_item, subkey, accumulate_steps
            )
            grad_single = jax.lax.with_sharding_constraint(  # type: ignore[no-untyped-call]
                grad_single, parameter_storage_sharding
            )

            grad_acc = jax.tree_util.tree_map(lambda a, g: a + g, grad, grad_single)
            grad_acc = jax.lax.with_sharding_constraint(  # type: ignore[no-untyped-call]
                grad_acc, parameter_storage_sharding
            )
            loss_acc = loss + loss_single

            return (grad_acc, loss_acc, key), meta

        grad_zero = jax.tree_util.tree_map(jnp.zeros_like, state.params)
        grad_zero = jax.lax.with_sharding_constraint(  # type: ignore[no-untyped-call]
            grad_zero, parameter_storage_sharding
        )

        # scan over S micro-batches, accumulating gradients; collect per-step meta
        loss_sum: jax.Array
        with jax.named_scope("gradient_accumulation"):
            (grad_sum, loss_sum, _), metas = jax.lax.scan(
                reduce, (grad_zero, jnp.array(0.0), key), batch
            )

        # take the last micro-batch's metadata
        last_meta: Any = jax.tree_util.tree_map(lambda x: x[-1], metas)

        grad_sum = jax.lax.with_sharding_constraint(  # type: ignore[no-untyped-call]
            grad_sum, parameter_update_sharding
        )
        state = state.replace(
            params=jax.lax.with_sharding_constraint(  # type: ignore[no-untyped-call]
                state.params, parameter_update_sharding
            )
        )
        grad_norm: jax.Array = optax.global_norm(grad_sum)
        # Optimizers may request their own intermediate layouts (e.g. Muon).
        with (
            jax.named_scope("optimizer_update"),
            jax.sharding.use_abstract_mesh(sharding.mesh.abstract_mesh),
            flax.linen.logical_axis_rules(sharding.parameter_update_mapping),
        ):
            state = state.apply_gradients(grads=grad_sum)  # type: ignore[no-untyped-call]
        state = state.replace(
            params=jax.lax.with_sharding_constraint(  # type: ignore[no-untyped-call]
                state.params, parameter_storage_sharding
            )
        )

        return state, loss_sum, last_meta, grad_norm

    @classmethod
    def val_step(
        cls,
        state: train_state.TrainState,
        batch: PyTree[jax.Array],  # (S, B, T) each
        *,
        sharding: ParameterShardingContext,
    ) -> Tuple[jax.Array, jax.Array, Any]:
        """Compute validation loss over S micro-batches.

        Args:
            state: Current training state
            batch: (x, y, padding_mask) each with shape (S, B, T)
                   S = accumulation size, B = batch size, T = sequence length

        Returns:
            (loss_sum, token_count, last_meta)
        """

        def reduce(
            carry: Tuple[jax.Array, jax.Array],
            xb_item: Any,  # PyTree with single batch item (B, T)
        ) -> Tuple[Tuple[jax.Array, jax.Array], Any]:
            from typing import cast as type_cast

            loss_sum, count = carry

            # Cast to PyTree for forward call
            xb_pytree: PyTree[jax.Array] = type_cast(PyTree[jax.Array], xb_item)
            params_pytree: PyTree[jax.Array] = type_cast(
                PyTree[jax.Array], state.params
            )
            with (
                jax.sharding.use_abstract_mesh(sharding.mesh.abstract_mesh),
                flax.linen.logical_axis_rules(sharding.parameter_fwdbwd_mapping),
            ):
                _, loss_i, meta = cls.forward(
                    state,
                    params_pytree,
                    xb_pytree,
                    deterministic=True,
                    intermediates=True,
                )  # loss_i: scalar

            # Extract mask from batch (expected to be dict)
            xb_dict: Dict[str, jax.Array] = type_cast(Dict[str, jax.Array], xb_item)
            mask = xb_dict.get("padding_mask")
            if mask is None and "padding_mask_pos" in xb_dict:
                mask = jnp.stack(
                    [xb_dict["padding_mask_pos"], xb_dict["padding_mask_neg"]], axis=1
                )
            assert mask is not None, "No padding mask found in batch"

            n = mask.sum()  # count real tokens: scalar
            return (loss_sum + loss_i * n, count + n), meta

        # scan over S micro-batches, accumulating weighted loss
        (loss_sum, count), metas = jax.lax.scan(
            reduce, (jnp.array(0.0), jnp.array(0)), batch
        )

        last_meta: Any = jax.tree_util.tree_map(lambda x: x[-1], metas)

        return loss_sum, count, last_meta

    def _make_valid_step(
        self,
    ) -> Callable[..., Tuple[float, Dict[str, float]]]:
        batch = self._to_global(self._reshape_batch(self.batch("val")))
        data_shard = NamedSharding(self.mesh, P(None, Axis.BATCH, None))  # type: ignore

        valid_step_inner_jit = jax.jit(
            partial(self.val_step, sharding=self.sharding_context),
            in_shardings=(self.state_sharding, data_shard),
            out_shardings=(None, None, None),
        )

        def valid_step_wrapper(
            state: train_state.TrainState,
            step: int = 0,
        ) -> Tuple[float, Dict[str, float]]:
            loss_sum, count, meta = valid_step_inner_jit(state, batch)

            if self.main_process():
                local_meta = jax.tree.map(
                    lambda value: value.addressable_shards[0].data, meta
                )
                scalar_meta = {
                    name: value
                    for name, value in local_meta.items()
                    if name not in ("intermediates", "plots")
                }
                if scalar_meta:
                    self.log(scalar_meta)
                self.artifact(
                    "validation",
                    {"step": step},
                    {
                        "intermediates": local_meta.get("intermediates", {}),
                        "plots": local_meta.get("plots", {}),
                    },
                )

            loss_sum = jax.device_get(loss_sum)
            count = jax.device_get(count)

            # if these come back as per-device arrays, reduce them here
            loss_sum_local = float(jnp.sum(loss_sum))
            count_local = float(jnp.sum(count))

            loss_sum_g = multihost_utils.process_allgather(jnp.asarray(loss_sum_local))
            count_g = multihost_utils.process_allgather(jnp.asarray(count_local))

            loss = float(jnp.sum(loss_sum_g) / jnp.sum(count_g))

            score = 1 / loss
            metrics = {"val/loss": loss, "val/score": score}

            return score, metrics

        return valid_step_wrapper

    def _make_train_step(
        self,
    ) -> Callable[
        [
            train_state.TrainState,
            PyTree[jax.Array],
            jax.Array,
            int,
        ],
        Tuple[train_state.TrainState, jax.Array, Any, jax.Array],
    ]:
        data_shard = NamedSharding(self.mesh, P(None, Axis.BATCH, None))  # type: ignore
        assert self.spec.topology is not None
        step = partial(
            self.train_step,
            dtype=self.args.activation_dtype,
            fsdp=self.spec.topology.shard.fsdp,
            activation_checkpointing=self.spec.topology.shard.activation_checkpointing,
        )
        train_step = jax.jit(
            partial(step, sharding=self.sharding_context),
            in_shardings=(self.state_sharding, data_shard, None, None),
            out_shardings=(self.state_sharding, None, None, None),
            donate_argnums=(0,),
        )
        return train_step

    def finish(self) -> None:
        profiler = getattr(self, "_profiler", None)
        if profiler is not None:
            profiler.close()
            self._profiler = None
        for name in ("train_dl", "val_dl"):
            loader = getattr(self, name, None)
            if loader is not None:
                loader.close()
        super().finish()

    #### training ####

    def mfu(self) -> None:
        """Prepare theoretical compute seconds per update; None means unavailable."""
        self._mfu_seconds: float | None = None
        assert self.spec.topology is not None
        topology = self.spec.topology
        peak = getattr(topology.chip.flops, self.model.activation_dtype, None)
        overhead = self.OPTIMIZER.flops_per_param
        if peak is None or overhead is None:
            return

        # LoRA stores the model's original variables separately from trainable params.
        params = getattr(self.state, "base_params", self.state.params)
        with configuration(self._debug_config):
            model_flops = self.model.bind({"params": params}).flops(
                self.args.block_size
            )
        nparams = sum(leaf.size for leaf in jax.tree.leaves(self.state.params))
        flops = model_flops * self.args.batch_size + overhead * nparams
        # TPU v2/v3 expose two JAX devices per physical chip.
        chips = topology.device_count / (
            2 if topology.chip.name in {"tpu-v2", "tpu-v3"} else 1
        )
        self._mfu_seconds = flops / (peak * 1e12 * chips)

    def train(self) -> None:
        if getattr(self, "_profiler", None) is None:
            self._profiler = Profiler(node=self.node)
            self._profiler.start()

        assert self._profiler is not None

        # Restore selects the saved batch; only training advances past it.
        if self.base is not None and self.node.serialize() == self.base.serialize():
            BasicJob.tick(self)

        if self.main_process():
            logger.info("BEGIN TRAINING")

        train_step = self._make_train_step()
        state_type = type(self.state)
        report_time = self.__dict__.get("_mfu_started")
        report_tokens = (
            int(self.state.step) * self.args.batch_size * self.args.block_size
        )
        mfu_work: float | None = 0.0
        if self.args.validate:
            valid_step = self._make_valid_step()

        # Resume after the last completed optimizer step.
        for step in range(int(self.state.step) + 1, self.total_steps + 1):
            logger.debug("DATA | {} | START", step)
            batch = self._to_global(self._reshape_batch(self.batch()))
            logger.debug("DATA | {} | PLACED", step)

            if type(self.state) is not state_type:
                train_step = self._make_train_step()
                if self.args.validate:
                    valid_step = self._make_valid_step()
                state_type = type(self.state)
                if report_time is not None:
                    self.mfu()

            self.dropout_key, subkey = jax_random.split(self.dropout_key)
            with (
                self._profiler.measure_step(),
                jax.profiler.StepTraceAnnotation("train", step_num=step),
            ):
                self.state, loss, train_meta, grad_norm = train_step(
                    self.state,
                    batch,
                    subkey,
                    self.accumulate_steps,
                )
            if report_time is not None:
                mfu_work = (
                    mfu_work + self._mfu_seconds
                    if mfu_work is not None and self._mfu_seconds is not None
                    else None
                )
            logger.debug("COMPUTATION | {} | FINISHED", step)
            train_metrics = {}

            if step % self.args.report_interval == 0:
                multihost_utils.sync_global_devices("report:pre")
                train_metrics["train/lr"] = float(self.schedule(self.state.step))
                loss_val = float(loss)
                if report_time is not None:
                    now = time.perf_counter()
                    tokens = step * self.args.batch_size * self.args.block_size
                    if now > report_time:
                        assert self.spec.topology is not None
                        train_metrics["train/tokens_per_second_per_device"] = (
                            (tokens - report_tokens)
                            / (now - report_time)
                            / self.spec.topology.device_count
                        )
                    if mfu_work is not None and now > report_time:
                        train_metrics["train/mfu"] = mfu_work / (now - report_time)
                    report_tokens = tokens
                    report_time, mfu_work = now, 0.0

                if self.main_process():
                    train_metrics["train/tokens"] = (
                        step * self.args.batch_size * self.args.block_size
                    )
                    train_metrics["train/loss"] = loss_val
                    train_metrics["train/grad_norm"] = float(jax.device_get(grad_norm))
                    train_meta_host = jax.device_get(train_meta)
                    train_metrics.update(
                        {
                            name: np.asarray(value).item()
                            for name, value in train_meta_host.items()
                        }
                    )

                    self.log(train_metrics)
                    logger.info(
                        "TRAIN | {}/{} | loss {}",
                        step,
                        self.total_steps,
                        loss_val,
                    )
                multihost_utils.sync_global_devices("report:post")

            if self.main_process():
                logger.debug("STEP | {} | {}", step, train_metrics)

            # Perform periodic validation at the configured offset and always
            # evaluate the exact terminal optimizer step. Evaluation owns its
            # node through checkpointing; tick() happens only after both finish.
            if (
                step == self.total_steps
                or step % self.args.validation_interval
                == self.args.validation_interval // 3
            ):
                val_metrics = {}

                if self.args.analyze:
                    for analysis in self.analyses:
                        analysis.run()

                if self.args.validate:
                    _, metrics = valid_step(self.state, step=step)
                    val_metrics.update(metrics)

                if self.args.evaluate and self.inference is not None:
                    eval_metrics = self.inference.evaluate()
                    val_metrics.update(eval_metrics)

                val_metrics["train/tokens"] = (
                    step * self.args.batch_size * self.args.block_size
                )
                if self.main_process():
                    self.log(val_metrics)
                    logger.info("VAL | {}", step)

                self.checkpoint()
            elif (
                step % self.args.checkpoint_interval
                == self.args.checkpoint_interval // 2
            ):
                # The evaluation residue is handled above, so coincident modular
                # schedules checkpoint this node exactly once.
                self.checkpoint()

            self.tick()

    #### logging and checkpoints ####

    def checkpoint(self) -> None:
        """Save the current training state, including its optimizer step."""

        super().save(self.state, {})

        if self.main_process():
            self.log({"checkpoint": 1})
            logger.info(
                "CHECKPOINT | saved checkpoint at {} at step {}",
                self.node.serialize(),
                int(self.state.step),
            )

    #### job lifecycle ####

    def run(self) -> None:
        """main entry point to run training, called on all nodes"""
        super().run()
        self._mfu_started = time.perf_counter()
        self.mfu()
        self.train()
