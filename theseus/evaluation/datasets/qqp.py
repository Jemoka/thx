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
class QQPEvalConfig:
    split: str = field("eval/qqp/split", default="validation")


def template(q1: str, q2: str) -> ChatTemplate:
    return [
        ChatTurn(
            role="user",
            message=f"""Are these two questions paraphrases of each other? Answer with "yes" or "no", not including quotes.

question 1: {q1}
question 2: {q2}
""",
        ),
    ]


@evaluation("qqp")
class QQPEval(RolloutEvaluation):
    """QQP evaluation using validation split."""

    CONFIG = QQPEvalConfig

    def __init__(self) -> None:
        self.ds = load_dataset(
            "nyu-mll/glue", "qqp", split=configure(self.CONFIG).split
        )
        self.encoder = get_tokenizer()

    @property
    def name(self) -> str:
        return "qqp"

    def max_new_tokens(self, inference: Any) -> int:
        """Only need 1 token for yes/no answer."""
        return 10

    def get(self, indx: int) -> Tuple[str, str]:
        item = self.ds[indx]
        answer = "yes" if item["label"] else "no"
        prompt = encode_chat_template(
            template(item["question1"], item["question2"]),
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
