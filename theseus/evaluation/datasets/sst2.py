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
class SST2EvalConfig:
    split: str = field("eval/sst2/split", default="validation")


def template(sentence: str) -> ChatTemplate:
    return [
        ChatTurn(
            role="user",
            message=f"""Classify the sentiment of the following sentence; respond with "positive" or "negative", not including quotes.

sentence: {sentence}
""",
        ),
    ]


@evaluation("sst2")
class SST2Eval(RolloutEvaluation):
    """SST-2 sentiment evaluation using validation split."""

    CONFIG = SST2EvalConfig

    def __init__(self) -> None:
        self.ds = load_dataset("stanfordnlp/sst2", split=configure(self.CONFIG).split)
        self.encoder = get_tokenizer()

    @property
    def name(self) -> str:
        return "sst2"

    def max_new_tokens(self, inference: Any) -> int:
        """Only need ~2 tokens for positive/negative answer."""
        return 15

    def get(self, indx: int) -> Tuple[str, str]:
        item = self.ds[indx]
        answer = "positive" if item["label"] == 1 else "negative"
        prompt = encode_chat_template(
            template(item["sentence"]),
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
        return assistant_msgs[0].strip().lower()

    def check(self, y: str, y_hat: str) -> bool:
        return y.strip().lower() == y_hat.strip().lower()
