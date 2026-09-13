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
class AlpacaEvalConfig:
    split: str = field("eval/alpaca/split", default="train")


def template(instruction: str, input_text: str) -> ChatTemplate:
    if input_text:
        return [
            ChatTurn(role="system", message=instruction),
            ChatTurn(role="user", message=input_text),
        ]
    return [
        ChatTurn(role="user", message=instruction),
    ]


@evaluation("alpaca")
class AlpacaEval(RolloutEvaluation):
    """Stanford Alpaca instruction-following evaluation."""

    CONFIG = AlpacaEvalConfig

    def __init__(self) -> None:
        self.ds = load_dataset("tatsu-lab/alpaca", split=configure(self.CONFIG).split)
        self.encoder = get_tokenizer()

    @property
    def name(self) -> str:
        return "alpaca"

    def max_new_tokens(self, inference: Any) -> int:
        return 512

    def get(self, indx: int) -> Tuple[str, str]:
        item = self.ds[indx]
        prompt = encode_chat_template(
            template(item["instruction"], item["input"]),
            self.encoder,
            prompt=True,
            tokenize=False,
        )
        return prompt, item["output"]

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
        return assistant_msgs[0].strip()

    def check(self, y: str, y_hat: str) -> bool:
        return y.strip() == y_hat.strip()
