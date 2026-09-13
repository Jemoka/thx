"""HellaSwag rollout evaluation (Zellers et al., 2019).

4-way multiple choice over sentence-ending continuations.
"""

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


def template(ctx: str, endings: list[str]) -> ChatTemplate:
    choices_text = "\n".join(f"{chr(65 + i)}: {c}" for i, c in enumerate(endings))
    return [
        ChatTurn(
            role="user",
            message=(
                "Pick the most likely continuation of the passage.\n\n"
                f"Passage: {ctx}\n\n"
                f"{choices_text}\n\n"
                "Answer with only the letter (A, B, C, or D):"
            ),
        ),
    ]


@evaluation("hellaswag")
class HellaSwagEval(RolloutEvaluation):
    """HellaSwag evaluation (validation split)."""

    def __init__(self) -> None:
        self.ds = load_dataset("Rowan/hellaswag", split="validation")
        self.encoder = get_tokenizer()

    @property
    def name(self) -> str:
        return "hellaswag"

    def max_new_tokens(self, inference: Any) -> int:
        return 10

    def get(self, indx: int) -> Tuple[str, str]:
        item = self.ds[indx]
        ctx = item["ctx"]
        endings: list[str] = item["endings"]
        answer = chr(65 + int(item["label"]))
        prompt = encode_chat_template(
            template(ctx, endings),
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
                return turn.message.strip().upper()[:1]
        return ""

    def check(self, y: str, y_hat: str) -> bool:
        return y.strip().upper() == y_hat.strip().upper()
