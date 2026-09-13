#!/usr/bin/env python3
"""Smoke test for GRPOTrainer.

Defines a trivial `high_number` evaluation: the reward is the integer parsed
out of the model's decoded response (higher is better). Registers a tiny
`grpo/smoke/high_number` job. Runs a handful of training steps so the
rollout → reward → GRPO-loss pipeline gets exercised end to end.
"""

from typing import Any, List, Tuple

from theseus.quick import quick
from theseus.registry import evaluation, job
from theseus.evaluation.base import RolloutEvaluation
from theseus.data.tokenizer import get_tokenizer
from theseus.model.models import Llama
from theseus.training.grpo import GRPOTrainer


@evaluation("high_number")
class HighNumberEval(RolloutEvaluation):
    """Reward = integer parsed out of the rollout's decoded text.

    No ground truth is needed: whatever number the model emits *is* the reward.
    """

    _PROMPTS: List[str] = [
        "Number: ",
        "Output a large integer: ",
        "Reply with digits: ",
        "N = ",
    ]

    def __init__(self) -> None:
        self.encoder = get_tokenizer()

    @property
    def name(self) -> str:
        return "high_number"

    def max_new_tokens(self, inference: Any) -> int:
        return 8

    def __len__(self) -> int:
        # Match the GRPO batch size so one eval call = one training batch.
        return 16

    def get(self, indx: int) -> Tuple[str, str]:
        return self._PROMPTS[indx % len(self._PROMPTS)], ""

    def clean(self, y_hat: str) -> str:
        return y_hat.strip()

    def score(self, ys: List[str], y_hats: List[str]) -> List[float]:
        """Per-sample reward = number of digit chars in the decoded rollout.

        Dense, monotone signal — every extra digit raises the reward by 1, so
        within-group variance shows up after just a few different rollouts.
        """
        return [float(sum(c.isdigit() for c in (y or ""))) for y in y_hats]


@job("grpo/smoke/high_number")
class HighNumberGRPO(GRPOTrainer[Llama]):
    """Smoke-test GRPO trainer: reward is the eval's parsed value.

    With a single RL component, the default ``reward_postprocess`` (identity)
    already gives the per-rollout score from `high_number`'s scoring. No
    override needed.
    """

    MODEL = Llama

    RL_EVALUATION = [HighNumberEval]


if __name__ == "__main__":
    with quick() as j:
        j.build("grpo/smoke/high_number", "smoke_test_grpo")
        j.config.architecture.n_layers = 2
        j.config.architecture.n_embd = 64
        j.config.architecture.n_head = 2
        j.config.architecture.n_kv_head = 2
        j.config.architecture.block_size = 32
        j.config.architecture.intermediate_size = 128
        j.config.architecture.rope_theta = 10000.0
        j.config.architecture.rms_norm_eps = 1e-6
        j.config.architecture.attention_bias = False
        j.config.architecture.bias = False
        # cl100k_base vocab — include the standard EOT but stay under the
        # invalid-token range so random samples can always decode.
        j.config.architecture.vocab_size = 100257

        # GRPO settings.
        j.config.optimization.ppo.beta = (
            0.0  # KL relaxed for smoke (verify reward signal)
        )
        j.config.optimization.ppo.clip_eps = 0.2
        j.config.optimization.ppo.discount = 1.0
        j.config.optimization.ppo.sample_temperature = 1.0
        j.config.optimization.ppo.sample_top_p = 1.0
        j.config.optimization.grpo.group_size = 8

        # Trainer batching.
        j.config.training.batch_size = 16
        j.config.training.per_device_batch_size = 16
        j.config.training.tokens = 102400  # 200 optimizer steps at 16*32
        j.config.optimization.lr = 3e-3  # higher LR — small model, smoke run
        j.config.training.validation = False
        j.config.training.evaluate = False

        # RL components — drives the rollout source.

        # Logging.
        j.config.logging.report_interval = 1
        j.config.logging.checkpoint_interval = 100000
        j.config.logging.validation_interval = 100000
        j.config.logging.remote = False

        print("Starting GRPO smoke test:")
        print(f"  group_size = {j.config.optimization.grpo.group_size}")
        print(f"  batch_size = {j.config.training.batch_size}")
        print(f"  tokens     = {j.config.training.tokens}")
        print()

        j()

        print("\nGRPO smoke test completed successfully!")
