from dataclasses import dataclass

from datasets import load_dataset
from typing import Any, Tuple

from theseus.config import configure, field
from theseus.data.datasets import ChatTemplate, ChatTurn
from theseus.evaluation.base import RolloutEvaluation
from theseus.registry import evaluation
from theseus.data.tokenizer import (
    decode_chat_template,
    encode_chat_template,
    get_tokenizer,
)


@dataclass
class SIQAEvalConfig:
    split: str = field("eval/siqa/split", default="validation")


def template(
    context: str, question: str, answerA: str, answerB: str, answerC: str
) -> ChatTemplate:
    return [
        ChatTurn(
            role="user",
            message=f"""Given this context and question, judge the best answer. Answer in a single letter, "A", "B", or "C", without quotes.

context: {context}
question: {question}

A: {answerA}
B: {answerB}
C: {answerC}
""",
        ),
    ]


@evaluation("siqa")
class SIQAEval(RolloutEvaluation):
    """SIQA evaluation using validation split."""

    CONFIG = SIQAEvalConfig

    def __init__(self) -> None:
        self.ds = load_dataset("lighteval/siqa", split=configure(self.CONFIG).split)
        self.encoder = get_tokenizer()

    @property
    def name(self) -> str:
        return "siqa"

    def max_new_tokens(self, inference: Any) -> int:
        """Only need 1 token for A/B/C answer."""
        return 10

    def get(self, indx: int) -> Tuple[str, str]:
        item = self.ds[indx]
        answer = ["A", "B", "C"][int(item["label"]) - 1]
        prompt = encode_chat_template(
            template(
                item["context"],
                item["question"],
                item["answerA"],
                item["answerB"],
                item["answerC"],
            ),
            self.encoder,
            prompt=True,
            tokenize=False,
        )
        return prompt, answer

    def __len__(self) -> int:
        return len(self.ds)

    def clean(self, y_hat: str) -> str:
        chats: ChatTemplate = decode_chat_template(y_hat)
        assistant_msgs = []
        for i in chats:
            if i.role == "assistant":
                assistant_msgs.append(i.message.strip())
        if not assistant_msgs:
            return ""
        return assistant_msgs[0].strip().upper()

    def check(self, y: str, y_hat: str) -> bool:
        return y.strip().upper() == y_hat.strip().upper()
