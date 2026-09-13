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
class CFQEvalConfig:
    split: str = field("eval/cfq/split", default="test")
    config: str = field("eval/cfq/config", default="mcd1")


def template(question: str) -> ChatTemplate:
    return [
        ChatTurn(
            role="user",
            message=(
                "Translate the following natural language question into a "
                "SPARQL query. Respond with only the SPARQL query, nothing else.\n\n"
                f"Question: {question}"
            ),
        ),
    ]


@evaluation("cfq")
class CFQEval(RolloutEvaluation):
    """CFQ compositional generalization evaluation using test split."""

    CONFIG = CFQEvalConfig

    def __init__(self) -> None:
        args = configure(self.CONFIG)
        self.ds = load_dataset(
            "google-research-datasets/cfq", args.config, split=args.split
        )
        self.encoder = get_tokenizer()

    @property
    def name(self) -> str:
        return "cfq"

    def max_new_tokens(self, inference: Any) -> int:
        return 200

    def get(self, indx: int) -> Tuple[str, str]:
        item = self.ds[indx]
        prompt = encode_chat_template(
            template(item["question"]),
            self.encoder,
            prompt=True,
            tokenize=False,
        )
        return prompt, item["query"]

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
