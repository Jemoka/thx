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
class MNLIEvalConfig:
    split: str = field("eval/mnli/split", default="validation_matched")


def template(premise: str, hypothesis: str) -> ChatTemplate:
    return [
        ChatTurn(
            role="user",
            message=f"""Does the hypothesis entail the premise? Please respond only with "entailment", "contradiction", or "neutral", not including quotes.

premise: {premise}
hypothesis: {hypothesis}
""",
        ),
    ]


@evaluation("mnli")
class MNLIEval(RolloutEvaluation):
    """MNLI evaluation using validation_matched split."""

    CONFIG = MNLIEvalConfig

    def __init__(self) -> None:
        self.ds = load_dataset("nyu-mll/multi_nli", split=configure(self.CONFIG).split)
        self.encoder = get_tokenizer()

    @property
    def name(self) -> str:
        return "mnli"

    def max_new_tokens(self, inference: Any) -> int:
        """Need ~3 tokens for entailment/neutral/contradiction."""
        return 20

    def get(self, indx: int) -> Tuple[str, str]:
        item = self.ds[indx]
        premise = item["premise"]
        hypothesis = item["hypothesis"]

        if item["label"] == 0:
            answer = "entailment"
        elif item["label"] == 1:
            answer = "neutral"
        else:
            answer = "contradiction"

        prompt = encode_chat_template(
            template(premise, hypothesis),
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
