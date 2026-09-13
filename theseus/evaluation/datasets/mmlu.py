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


def template(question: str, choices: list[str]) -> ChatTemplate:
    choices_text = "\n".join(f"{chr(65 + i)}: {c}" for i, c in enumerate(choices))
    return [
        ChatTurn(
            role="user",
            message=(
                "Answer the following multiple-choice question by selecting "
                "the correct option.\n\n"
                f"Question: {question}\n\n"
                f"{choices_text}\n\n"
                "Answer with only the letter (A, B, C, or D):"
            ),
        ),
    ]


@evaluation("mmlu")
class MMLUEval(RolloutEvaluation):
    """MMLU evaluation (validation split, all subjects)."""

    def __init__(self) -> None:
        self.ds = load_dataset("cais/mmlu", "all", split="validation")
        self.encoder = get_tokenizer()

    @property
    def name(self) -> str:
        return "mmlu"

    def max_new_tokens(self, inference: Any) -> int:
        return 10

    def get(self, indx: int) -> Tuple[str, str]:
        item = self.ds[indx]
        choices: list[str] = item["choices"]
        answer = chr(65 + int(item["answer"]))
        prompt = encode_chat_template(
            template(item["question"], choices),
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
        return assistant_msgs[0].strip().upper()[:1]

    def check(self, y: str, y_hat: str) -> bool:
        return y.strip().upper() == y_hat.strip().upper()
