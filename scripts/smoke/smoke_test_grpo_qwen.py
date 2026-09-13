#!/usr/bin/env python3
"""Smoke test for BackbonedGRPOTrainer with Qwen 2.5 0.5B.

Same toy "emit digits" reward as scripts/smoke_test_grpo.py, but the policy
starts from the Qwen2.5-0.5B-Instruct HF checkpoint rather than a random
init. Verifies the from-pretrained → rollout → reward → GRPO-loss pipeline
end to end.

Heads-up: this *will* be slow on CPU (≈500M params, fp32, autoregressive
decode). Keep step counts tiny.
"""

from typing import Any, List, Tuple

from theseus.quick import quick
from theseus.registry import evaluation, job
from theseus.evaluation.base import RolloutEvaluation
from theseus.data.tokenizer import get_tokenizer
from theseus.training.grpo import BackbonedGRPOTrainer


@evaluation("high_number_qwen")
class HighNumberQwenEval(RolloutEvaluation):
    """Reward = number of digit chars emitted.

    Same shape as scripts/smoke_test_grpo.py:HighNumberEval, just renamed
    so the registry doesn't collide if both scripts get imported.
    """

    _PROMPTS: List[str] = [
        "Please reply with one large integer and nothing else: ",
        "Output a number, just digits: ",
        "Give me a number with many digits: ",
        "Emit only digits, e.g. N = ",
    ]

    def __init__(self) -> None:
        self.encoder = get_tokenizer()

    @property
    def name(self) -> str:
        return "high_number_qwen"

    def max_new_tokens(self, inference: Any) -> int:
        return 12

    def __len__(self) -> int:
        # Match the GRPO batch size so one eval call = one training batch.
        return 4

    def get(self, indx: int) -> Tuple[str, str]:
        return self._PROMPTS[indx % len(self._PROMPTS)], ""

    def clean(self, y_hat: str) -> str:
        return y_hat.strip()

    def score(self, ys: List[str], y_hats: List[str]) -> List[float]:
        """Per-sample reward = number of digit chars in the decoded rollout."""
        return [float(sum(c.isdigit() for c in (y or ""))) for y in y_hats]


@job("grpo/smoke/high_number_qwen")
class HighNumberGRPOQwen(BackbonedGRPOTrainer):
    """Backboned GRPO trainer: Qwen 2.5 0.5B + the digits-reward smoke task.

    With a single RL component, the default ``reward_postprocess`` (identity)
    already gives the per-rollout score from `high_number_qwen`'s scoring.
    """

    RL_EVALUATION = [HighNumberQwenEval]


if __name__ == "__main__":
    with quick() as j:
        j.build("grpo/smoke/high_number_qwen", "smoke_test_grpo_qwen")
        j.config.architecture.backbone.implementation = "qwen"
        j.config.architecture.backbone.weights = "Qwen/Qwen2.5-0.5B-Instruct"
        j.config.architecture.block_size = 64
        j.config.architecture.dtype.param = "float32"
        j.config.architecture.dtype.activation = "bfloat16"  # mixed precision

        # Use the matching Qwen tokenizer rather than the default cl100k.
        j.config.tokenizer.backend = "huggingface"
        j.config.tokenizer.name = "Qwen/Qwen2.5-0.5B-Instruct"

        # GRPO settings — KL relaxed so we can see raw learning signal.
        j.config.optimization.ppo.beta = 0.0
        j.config.optimization.ppo.clip_eps = 0.2
        j.config.optimization.ppo.discount = 1.0
        j.config.optimization.ppo.sample_temperature = 1.0
        j.config.optimization.ppo.sample_top_p = 1.0
        j.config.optimization.grpo.group_size = 4

        # Tiny token budget — 500M params on CPU is slow per step.
        # Just verify load + 1 step end-to-end.
        j.config.training.batch_size = 4
        j.config.training.per_device_batch_size = 4
        j.config.training.tokens = 256  # ~1 optimizer step at 4*64
        j.config.optimization.lr = 1e-5  # small LR for a real-policy fine-tune

        j.config.training.validation = False
        j.config.training.evaluate = False

        j.config.logging.report_interval = 1
        j.config.logging.checkpoint_interval = 100000
        j.config.logging.validation_interval = 100000
        j.config.logging.remote = False

        print("Starting Qwen-backboned GRPO smoke test:")
        print(f"  backbone   = {j.config.architecture.backbone.weights}")
        print(f"  group_size = {j.config.optimization.grpo.group_size}")
        print(f"  batch_size = {j.config.training.batch_size}")
        print(f"  tokens     = {j.config.training.tokens}")
        print()

        j()

        print("\nQwen-backboned GRPO smoke test completed successfully!")
