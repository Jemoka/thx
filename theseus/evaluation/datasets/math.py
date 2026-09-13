"""MATH benchmark rollout evaluation (Hendrycks et al., 2021).

Competition-math problems with LaTeX answers. The dataset's ``solution`` field
ends with a ``\\boxed{...}`` containing the final answer; we extract that as
ground truth, and grade rollouts by extracting their final ``\\boxed{...}``.
"""

import re
from typing import Any, Tuple

from datasets import load_dataset

from theseus.data.datasets import ChatTemplate, ChatTurn
from theseus.evaluation.base import RolloutEvaluation
from theseus.registry import evaluation
from theseus.data.tokenizer import (
    decode_chat_template,
    encode_chat_template,
    get_tokenizer,
)


_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def _last_boxed(text: str) -> str:
    matches = _BOXED_RE.findall(text)
    return matches[-1].strip() if matches else ""


def _normalize_latex(ans: str) -> str:
    s = ans.strip()
    s = s.replace(" ", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", "").replace("\\;", "").replace("\\!", "")
    s = s.rstrip(".")
    return s


def template(problem: str) -> ChatTemplate:
    return [
        ChatTurn(
            role="user",
            message=(
                "Solve the following math problem. Reason step by step, then "
                "put the final answer inside \\boxed{}.\n\n"
                f"{problem}"
            ),
        ),
    ]


@evaluation("math")
class MathEval(RolloutEvaluation):
    """MATH benchmark rollout evaluation (test split)."""

    def __init__(self) -> None:
        self.ds = load_dataset("hendrycks/competition_math", split="test")
        self.encoder = get_tokenizer()

    @property
    def name(self) -> str:
        return "math"

    def max_new_tokens(self, inference: Any) -> int:
        return 512

    def get(self, indx: int) -> Tuple[str, str]:
        item = self.ds[indx]
        gt = _last_boxed(item["solution"]) or item["solution"].strip()
        prompt = encode_chat_template(
            template(item["problem"]),
            self.encoder,
            prompt=True,
            tokenize=False,
        )
        return prompt, gt

    def __len__(self) -> int:
        return len(self.ds)

    def clean(self, y_hat: str) -> str:
        chats: ChatTemplate = decode_chat_template(y_hat)
        for turn in chats:
            if turn.role == "assistant":
                boxed = _last_boxed(turn.message)
                return boxed if boxed else turn.message.strip()
        return ""

    def check(self, y: str, y_hat: str) -> bool:
        return _normalize_latex(y) == _normalize_latex(y_hat)
