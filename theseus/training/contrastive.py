from dataclasses import dataclass
from typing import cast as type_cast
from typing import Any, Dict, Optional, List, Type, Generic

import jax
import jax.numpy as jnp
import jax.random as jax_random
from flax.training import train_state


from theseus.base import PyTree
from theseus.model.module import Module
from theseus.config import field, configure
from theseus.training.base import BaseTrainer, BaseTrainerConfig, M
from theseus.training.backbone import BackbonedTrainer


@dataclass
class DPOConfig:
    beta: float = field("optimization/dpo/beta", default=0.1)
    label_smoothing: float = field("optimization/dpo/label_smoothing", default=0.0)


class ContrastiveTrainState(train_state.TrainState):  # type: ignore[no-untyped-call]
    base: PyTree[Any]
    beta: float
    label_smooth: float


class ContrastiveTrainer(BaseTrainer[BaseTrainerConfig, M], Generic[M]):
    """Contrastive trainer scaffold."""

    CONFIG = BaseTrainerConfig

    @classmethod
    def _config(cls) -> List[Type[Any]]:
        return super()._config() + [DPOConfig]

    def _make_state(self, params: PyTree[jax.Array]) -> ContrastiveTrainState:
        cfg = configure(DPOConfig)
        return ContrastiveTrainState.create(  # type: ignore[no-any-return, no-untyped-call]
            apply_fn=self.model.apply,
            params=params,
            tx=self.optimizer,
            base=jax.tree.map(lambda x: x.astype(jnp.bfloat16), params),
            label_smooth=cfg.label_smoothing,
            beta=cfg.beta,
        )

    @staticmethod
    def forward(
        state: train_state.TrainState,
        params: PyTree[jax.Array],
        batch: PyTree[jax.Array],
        key: Optional[jax.Array] = None,
        deterministic: bool = False,
        intermediates: bool = False,
    ) -> Any:
        cstate = type_cast(ContrastiveTrainState, state)
        batch_dict: Dict[str, jax.Array] = type_cast(Dict[str, jax.Array], batch)

        # evaluation path: skip contrastive loss, just return logits
        if batch_dict.get("x") is not None:
            return BaseTrainer.forward(
                state,
                params,
                batch,
                key,
                deterministic,
                intermediates=intermediates,
            )

        # unpack dataset
        pos = batch_dict["pos"]
        neg = batch_dict["neg"]
        padding_mask_pos = batch_dict["padding_mask_pos"]
        padding_mask_neg = batch_dict["padding_mask_neg"]

        # build dropout details / extra variables
        dropout_key = None
        if not deterministic and key is not None:
            _, dropout_key = jax_random.split(key)
        kwargs: Dict[str, Any] = {
            "deterministic": deterministic,
        }
        if dropout_key is not None:
            kwargs["rngs"] = {"dropout": dropout_key}

        # compute logprobs for pos and neg using the policy parameters
        if intermediates:
            kwargs["mutable"] = ["intermediates"]

        result_pos = cstate.apply_fn(
            {"params": params},
            pos[:, :-1],
            pos[:, 1:],
            padding_mask=padding_mask_pos[:, :-1],
            **kwargs,
        )
        result_neg = cstate.apply_fn(
            {"params": params},
            neg[:, :-1],
            neg[:, 1:],
            padding_mask=padding_mask_neg[:, :-1],
            **kwargs,
        )

        intermediates_meta: Dict[str, Any] = {}
        if intermediates:
            (logits, loss_pos), mutated_pos = result_pos
            (_, loss_neg), mutated_neg = result_neg
            intermediates_meta = {
                "pos": dict(mutated_pos.get("intermediates", {})),
                "neg": dict(mutated_neg.get("intermediates", {})),
            }
        else:
            (logits, loss_pos) = result_pos
            (_, loss_neg) = result_neg

        # compute baseline loss
        (_, loss_base_pos) = cstate.apply_fn(
            {"params": cstate.base},
            pos[:, :-1],
            pos[:, 1:],
            padding_mask=padding_mask_pos[:, :-1],
            **kwargs,
        )
        (_, loss_base_neg) = cstate.apply_fn(
            {"params": cstate.base},
            neg[:, :-1],
            neg[:, 1:],
            padding_mask=padding_mask_neg[:, :-1],
            **kwargs,
        )
        # detach
        loss_base_pos = jax.lax.stop_gradient(loss_base_pos)
        loss_base_neg = jax.lax.stop_gradient(loss_base_neg)

        # compute DPO loss and rewards
        # model.loss() returns per-token-averaged NLL; standard DPO needs total
        # (summed) log-probs.  Recover total NLL by multiplying back by the
        # number of real target tokens (positions where target != -1).
        n_pos = padding_mask_pos[:, 1:].sum()
        n_neg = padding_mask_neg[:, 1:].sum()

        logits = -(
            (loss_pos * n_pos - loss_neg * n_neg)
            - (loss_base_pos * n_pos - loss_base_neg * n_neg)
        )

        beta = cstate.beta
        label_smooth = cstate.label_smooth

        loss = (
            -jax.nn.log_sigmoid(beta * logits) * (1 - label_smooth)
            - jax.nn.log_sigmoid(-beta * logits) * label_smooth
        )

        # reward sign fix
        chosen_rewards = jax.lax.stop_gradient(
            -beta * (loss_pos - loss_base_pos) * n_pos
        )
        rejected_rewards = jax.lax.stop_gradient(
            -beta * (loss_neg - loss_base_neg) * n_neg
        )

        metrics: Dict[str, Any] = {
            "rewards/chosen": chosen_rewards,
            "rewards/rejected": rejected_rewards,
            "rewards/reward_accuracy": jnp.mean(chosen_rewards > rejected_rewards),
            "rewards/reward_margin": chosen_rewards - rejected_rewards,
            "policy/nll_chosen": loss_pos,
            "policy/nll_rejected": loss_neg,
            "ref/nll_chosen": loss_base_pos,
            "ref/nll_rejected": loss_base_neg,
            **intermediates_meta,
        }

        return logits, loss, metrics


class BackbonedContrastiveTrainer(BackbonedTrainer, ContrastiveTrainer[Module]):
    """Contrastive trainer scaffold for backboned training."""

    @classmethod
    def _config(cls) -> List[Type[Any]]:
        return super()._config() + [DPOConfig]
