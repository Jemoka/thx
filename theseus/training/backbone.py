"""
BackbonedTrainer: finetune from a pretrained HuggingFace backbone.

Instead of configuring model architecture from scratch, reads two config keys:
  - architecture/backbone/implementation: "llama", "qwen", or "gpt_neox"
  - architecture/backbone/weights: HuggingFace model ID (e.g. "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")

The model class and initial weights are loaded via from_pretrained,
bypassing the normal configure() path for architecture parameters.
"""

from dataclasses import dataclass
from functools import cached_property
from typing import Any, List, Type

import jax

from theseus.config import field, configure
from theseus.model.module import Module
from theseus.model.models.contrib.llama import Llama
from theseus.model.models.contrib.qwen import Qwen
from theseus.model.models.contrib.qwen_3_5 import Qwen3_5
from theseus.model.models.contrib.qwen_3_5_moe import Qwen3_5MoE
from theseus.model.models.contrib.gpt_neox import GPTNeoX
from theseus.training.base import BaseTrainer, BaseTrainerConfig

BACKBONES: dict[str, Any] = {
    "llama": Llama,
    "qwen": Qwen,
    "qwen_3_5": Qwen3_5,
    "qwen_3_5_moe": Qwen3_5MoE,
    "gpt_neox": GPTNeoX,
}


@dataclass
class BackboneConfig:
    implementation: str = field("architecture/backbone/implementation")
    weights: str = field("architecture/backbone/weights")


@dataclass
class ModelDtypeConfig:
    param_dtype: str = field("architecture/dtype/param", default="float32")
    activation_dtype: str = field("architecture/dtype/activation", default="bfloat16")


def load_backbone() -> tuple[Any, Any]:
    """Read the declared HuggingFace architecture and its host parameter tree."""
    backbone = configure(BackboneConfig)
    dtype = configure(ModelDtypeConfig)
    model, params = BACKBONES[backbone.implementation].from_pretrained(
        backbone.weights,
        param_dtype=dtype.param_dtype,
        activation_dtype=dtype.activation_dtype,
    )

    return model, params


class BackbonedTrainer(BaseTrainer[BaseTrainerConfig, Module]):
    """Trainer that initializes from a pretrained HuggingFace backbone."""

    MODEL: Type[Any] = Module
    CONFIG = BaseTrainerConfig

    @classmethod
    def _config(cls) -> List[Type[Any]]:
        from theseus.evaluation.base import Evaluator
        from theseus.training.flywheel.strategy import Strategy

        return [
            BackboneConfig,
            ModelDtypeConfig,
            *Evaluator.config(cls.EVALUATION),
            *Strategy.config(cls._normalize_ds(cls.DATASET)),
            cls.OPTIMIZER.config,
            *([cls.SCHEDULE.config] if cls.SCHEDULE is not None else []),
        ]

    @cached_property
    def model(self) -> Any:
        model, self._pretrained_params = load_backbone()
        return model

    def initialize(self) -> None:
        # Copy each host leaf directly to its final mesh placement before
        # building optimizer state, avoiding a full-model transient on one GPU.
        from flax.core import unfreeze

        template = self.template
        sharding = jax.tree.map(lambda leaf: leaf.sharding, template)
        params = jax.tree.map(
            lambda value, target: jax.device_put(value, target),
            unfreeze(self._pretrained_params),
            sharding.params,
        )
        self._set_state(
            jax.jit(self._make_state, out_shardings=sharding)(self._cast_params(params))
        )
