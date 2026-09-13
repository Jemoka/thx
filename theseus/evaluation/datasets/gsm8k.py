"""GSM8K rollout evaluation (Cobbe et al., 2021).

Grade-school math word problems. Ground-truth answer follows a ``#### <int>``
sentinel in the dataset's ``answer`` field. Models are graded by extracting
either the final ``\\boxed{...}``, a final ``#### N`` line, or the last integer
in the rollout.
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


_GT_RE = re.compile(r"####\s*(-?[0-9.,]+)")
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_HASH_RE = re.compile(r"####\s*(-?[0-9.,]+)")
_LAST_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _normalize(num_str: str) -> str:
    return num_str.replace(",", "").strip().rstrip(".")


def template(question: str) -> ChatTemplate:
    return [
        ChatTurn(
            role="user",
            message=(
                "Solve the following math word problem. "
                "Show your reasoning, then give the final answer after '#### '.\n\n"
                f"{question}"
            ),
        ),
    ]


@evaluation("gsm8k")
class GSM8KEval(RolloutEvaluation):
    """GSM8K rollout evaluation (test split)."""

    def __init__(self) -> None:
        self.ds = load_dataset("gsm8k", "main", split="test")
        self.encoder = get_tokenizer()

    @property
    def name(self) -> str:
        return "gsm8k"

    def max_new_tokens(self, inference: Any) -> int:
        return 256

    def get(self, indx: int) -> Tuple[str, str]:
        item = self.ds[indx]
        m = _GT_RE.search(item["answer"])
        gt = _normalize(m.group(1)) if m else item["answer"].strip()
        prompt = encode_chat_template(
            template(item["question"]),
            self.encoder,
            prompt=True,
            tokenize=False,
        )
        return prompt, gt

    def __len__(self) -> int:
        return len(self.ds)

    def clean(self, y_hat: str) -> str:
        chats: ChatTemplate = decode_chat_template(y_hat)
        text = ""
        for turn in chats:
            if turn.role == "assistant":
                text = turn.message
                break
        if not text:
            return ""

        m = _BOXED_RE.search(text)
        if m:
            return _normalize(m.group(1))
        m = _HASH_RE.search(text)
        if m:
            return _normalize(m.group(1))
        matches = _LAST_NUM_RE.findall(text)
        if matches:
            return _normalize(matches[-1])
        return text.strip()

    def check(self, y: str, y_hat: str) -> bool:
        try:
            return float(_normalize(y)) == float(_normalize(y_hat))
        except (ValueError, TypeError):
            return _normalize(y) == _normalize(y_hat)
