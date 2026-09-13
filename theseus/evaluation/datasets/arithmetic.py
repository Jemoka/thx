"""
EleutherAI/arithmetic rollout evaluation.

The dataset is stored as one JSONL file per arithmetic task on the Hub. We load
all files into one evaluation, strip the ``Q:/A:`` framing, wrap the math
question in a chat template, and grade the assistant's first numeric token
against the ground-truth integer answer.
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


ARITHMETIC_DATA_FILES = [
    "two_digit_addition.jsonl",
    "two_digit_subtraction.jsonl",
    "three_digit_addition.jsonl",
    "three_digit_subtraction.jsonl",
    "four_digit_addition.jsonl",
    "four_digit_subtraction.jsonl",
    "five_digit_addition.jsonl",
    "five_digit_subtraction.jsonl",
    "two_digit_multiplication.jsonl",
    "single_digit_three_ops.jsonl",
]


_QA_RE = re.compile(r"Q:\s*(.*?)\s*A:\s*$", re.DOTALL)
_FIRST_INT_RE = re.compile(r"-?\d+")
_DATA_DIR = "hf://datasets/EleutherAI/arithmetic/data"


def load_arithmetic_dataset() -> Any:
    """Load all arithmetic JSONL files from the Hub as one dataset.

    The packaged ``EleutherAI/arithmetic`` loading script is not usable on
    current ``datasets`` releases, so this uses the JSON builder directly.
    """

    return load_dataset(
        "json",
        data_files=[f"{_DATA_DIR}/{file}" for file in ARITHMETIC_DATA_FILES],
        split="train",
    )


def _extract_question(context: str) -> str:
    m = _QA_RE.search(context)
    return m.group(1).strip() if m else context.strip()


def template(question: str) -> ChatTemplate:
    return [
        ChatTurn(
            role="user",
            message=(
                "Solve the following arithmetic problem. "
                "Respond with only the integer answer.\n\n"
                f"{question}"
            ),
        ),
    ]


@evaluation("arithmetic")
class ArithmeticEval(RolloutEvaluation):
    """EleutherAI/arithmetic rollout evaluation."""

    def __init__(self) -> None:
        self.ds = load_arithmetic_dataset()
        self.encoder = get_tokenizer()

    @property
    def name(self) -> str:
        return "arithmetic"

    def max_new_tokens(self, inference: Any) -> int:
        return 20

    def get(self, indx: int) -> Tuple[str, str]:
        item = self.ds[indx]
        question = _extract_question(item["context"])
        answer = item["completion"].strip()
        prompt = encode_chat_template(
            template(question),
            self.encoder,
            prompt=True,
            tokenize=False,
        )
        return prompt, answer

    def __len__(self) -> int:
        return len(self.ds)

    def clean(self, y_hat: str) -> str:
        chats: ChatTemplate = decode_chat_template(y_hat)
        for turn in chats:
            if turn.role == "assistant":
                m = _FIRST_INT_RE.search(turn.message)
                if m:
                    return m.group(0)
                return turn.message.strip()
        return ""

    def check(self, y: str, y_hat: str) -> bool:
        try:
            return int(y) == int(y_hat)
        except (ValueError, TypeError):
            return y.strip() == y_hat.strip()
