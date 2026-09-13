from dataclasses import dataclass
from typing import Any, List, Union

import flax
import jax
import numpy as np
import optax
from flax.training import train_state
from jax import random as jax_random
from loguru import logger

from theseus.base import ExecutionSpec, PyTree
from theseus.config import configure, field
from theseus.data.datasets import ChatTemplate, ChatTurn
from theseus.data.tokenizer import get_tokenizer, TokenizerConfig
from theseus.inference.base import InferenceJob
from theseus.model.models.base import GPT
from theseus.model.module import Module
from theseus.registry import job
from theseus.training.backbone import BACKBONES, BackboneConfig, ModelDtypeConfig


@dataclass
class ChatConfig:
    block_size: int = field("architecture/block_size", default=512)
    per_device_batch_size: int = field("training/per_device_batch_size", default=-1)
    max_new_tokens: int = field("inference/max_new_tokens", default=256)
    temperature: float = field("inference/temperature", default=0.8)
    top_p: float = field("inference/top_p", default=0.9)
    system_prompt: str = field("inference/system_prompt", default="")


@job("gpt/debug/chat/continuation")
class Chat(InferenceJob[ChatConfig, GPT]):
    MODEL = GPT

    @classmethod
    def config(cls) -> List[Any]:
        return [ChatConfig, TokenizerConfig]

    def _generate(
        self, tok: Any, inp: Union[str, ChatTemplate]
    ) -> tuple[List[int], str]:
        [gen_ids] = self.rollout(
            [inp],
            tok,
            max_new_tokens=self.args.max_new_tokens,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            return_type="output_indices",
        )
        eot = tok.eot_token
        if eot in gen_ids:
            gen_ids = gen_ids[: gen_ids.index(eot)]
        return gen_ids, tok.decode(gen_ids)

    def _banner(self, mode: str) -> None:
        logger.info(
            "CHAT | ready  mode={}  max_new_tokens={}  temperature={}  top_p={}",
            mode,
            self.args.max_new_tokens,
            self.args.temperature,
            self.args.top_p,
        )
        print(f"\n--- chat:{mode} (ctrl-c to quit) ---\n")

    def run(self) -> None:
        tok = get_tokenizer(configure(TokenizerConfig))
        self._banner("continuation")

        while True:
            try:
                prompt = input("> ")
            except (EOFError, KeyboardInterrupt):
                print("\nbye")
                break

            if not prompt.strip():
                continue

            _, text = self._generate(tok, prompt)
            print(text)
            print()


class TurnsRunMixin:
    """Provides a turns-mode ``run`` that drives a chat-encoded conversation.

    Routes ChatTemplate inputs through the active tokenizer's chat-template
    API (HuggingFace ``apply_chat_template``, or ChatML for tiktoken) via
    ``rollout``. The mixin assumes ``self`` provides ``_generate``,
    ``_banner``, and an ``args.system_prompt`` config field — i.e. it is
    mixed onto a ``Chat`` (or ``Chat`` subclass) instance.
    """

    def run(self) -> None:
        tok = get_tokenizer()
        self._banner("turns")  # type: ignore[attr-defined]

        system_prompt: str = self.args.system_prompt  # type: ignore[attr-defined]
        history: ChatTemplate = []
        if system_prompt:
            history.append(ChatTurn(role="system", message=system_prompt))

        while True:
            try:
                prompt = input("> ")
            except (EOFError, KeyboardInterrupt):
                print("\nbye")
                break

            if not prompt.strip():
                continue

            history.append(ChatTurn(role="user", message=prompt))
            _, text = self._generate(tok, history)  # type: ignore[attr-defined]
            print(text)
            print()
            history.append(ChatTurn(role="assistant", message=text.strip()))


@job("gpt/debug/chat/turns")
class ChatTurns(TurnsRunMixin, Chat):
    pass


@job("backbone/debug/chat/continuation")
class ChatPretrained(Chat):
    """Chat with a pretrained HuggingFace backbone loaded via from_pretrained."""

    MODEL = Module  # actual model class chosen at runtime via BACKBONES

    @classmethod
    def config(cls) -> List[Any]:
        return [ChatConfig, BackboneConfig, ModelDtypeConfig, TokenizerConfig]

    def __init__(self, spec: ExecutionSpec) -> None:
        super().__init__(spec)

        assert spec.topology is not None, "Topology required for chat"
        self.mesh = spec.topology.mesh
        self.replicas = spec.topology.replicas
        self.local_replicas = spec.topology.local_replicas

        backbone_cfg = configure(BackboneConfig)
        dtype_cfg = configure(ModelDtypeConfig)

        model_cls = BACKBONES[backbone_cfg.implementation]
        self.model, params = model_cls.from_pretrained(
            backbone_cfg.weights,
            param_dtype=dtype_cfg.param_dtype,
            activation_dtype=dtype_cfg.activation_dtype,
        )

        self.key, self.dropout_key = jax_random.split(self.key)

        params = self._cast_params(params)

        def make_state(p: PyTree[Any]) -> train_state.TrainState:
            return train_state.TrainState.create(  # type: ignore[no-untyped-call]
                apply_fn=self.model.apply, params=p, tx=optax.identity()
            )

        state_shapes = jax.eval_shape(make_state, params)
        self.state_sharding = flax.linen.logical_to_mesh_sharding(  # type: ignore[attr-defined]
            flax.linen.get_partition_spec(state_shapes),
            self.mesh,
            rules=self.model.sharding._tp,
        )
        self.state = jax.jit(make_state, out_shardings=self.state_sharding)(params)

        self.per_device_batch_size = self.args.per_device_batch_size
        self.block_size = self.args.block_size

    def _cast_params(self, params: PyTree[Any]) -> PyTree[Any]:
        target = np.dtype(self.model.param_dtype)

        def cast_leaf(x: Any) -> Any:
            if isinstance(x, np.ndarray):
                return x.astype(target, copy=False)
            if isinstance(x, jax.Array):
                return x.astype(target)
            return x

        return jax.tree_util.tree_map(cast_leaf, params)


@job("backbone/debug/chat/turns")
class ChatTurnsPretrained(TurnsRunMixin, ChatPretrained):
    pass
