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


def template(context: str, question: str) -> ChatTemplate:
    return [
        ChatTurn(
            role="user",
            message=(
                "Read the following passage and answer the question. "
                "Respond with only the answer extracted from the passage.\n\n"
                f"Passage: {context}\n\n"
                f"Question: {question}"
            ),
        ),
    ]


@evaluation("squad")
class SQuADEval(RolloutEvaluation):
    """SQuAD v1.1 extractive QA evaluation (validation split)."""

    def __init__(self) -> None:
        self.ds = load_dataset("rajpurkar/squad", split="validation")
        self.encoder = get_tokenizer()

    @property
    def name(self) -> str:
        return "squad"

    def max_new_tokens(self, inference: Any) -> int:
        return 50

    def get(self, indx: int) -> Tuple[str, str]:
        item = self.ds[indx]
        answer = item["answers"]["text"][0] if item["answers"]["text"] else ""
        prompt = encode_chat_template(
            template(item["context"], item["question"]),
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
        return y.strip().lower() in y_hat.strip().lower()
