"""Standalone evaluation jobs using native checkpoint restoration."""

from functools import cached_property
from typing import Any

import jax
from flax.core import unfreeze

from theseus.evaluation.base import Evaluator, Evaluation
from theseus.inference.base import InferenceConfig
from theseus.model.models import GPT
from theseus.model.module import Module
from theseus.registry import job
from theseus.training.backbone import BackboneConfig, ModelDtypeConfig, load_backbone


@job("gpt/evaluate")
class Evaluate(Evaluator[GPT]):
    """Evaluate initialized or restored GPT weights with a declared EVALUATION."""

    MODEL = GPT


@job("backbone/evaluate")
class BackboneEvaluate(Evaluator[Module]):
    """Evaluate HuggingFace weights or a native checkpoint based on that model."""

    MODEL = Module

    @classmethod
    def config(
        cls, evaluations: list[type[Evaluation]] | None = None
    ) -> list[type[Any]]:
        return [
            BackboneConfig,
            ModelDtypeConfig,
            InferenceConfig,
            *Evaluator.config(cls.EVALUATION if evaluations is None else evaluations),
        ]

    @cached_property
    def model(self) -> Any:
        model, self._pretrained_params = load_backbone()
        return model

    def initialize(self) -> None:
        template = self.template
        assert isinstance(template, dict)
        params = jax.tree.map(
            lambda value, target: jax.device_put(value, target.sharding),
            unfreeze(self._pretrained_params),
            template["params"],
        )
        self.apply({"params": params}, {})
