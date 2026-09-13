"""ARC-Challenge rollout evaluation (Clark et al., 2018).

Grade-school science multiple choice; we use the validation split.
Questions have variable choice counts and labels (sometimes A-D, sometimes 1-4).
We map answerKey to a letter index into the presented choices.
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


def template(question: str, choices: list[str]) -> ChatTemplate:
    choices_text = "\n".join(f"{chr(65 + i)}: {c}" for i, c in enumerate(choices))
    return [
        ChatTurn(
            role="user",
            message=(
                "Answer the following multiple-choice science question.\n\n"
                f"Question: {question}\n\n"
                f"{choices_text}\n\n"
                "Answer with only the letter:"
            ),
        ),
    ]


@evaluation("arc_challenge")
class ARCChallengeEval(RolloutEvaluation):
    """ARC-Challenge rollout evaluation (validation split)."""

    def __init__(self) -> None:
        self.ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="validation")
        self.encoder = get_tokenizer()

    @property
    def name(self) -> str:
        return "arc_challenge"

    def max_new_tokens(self, inference: Any) -> int:
        return 10

    def get(self, indx: int) -> Tuple[str, str]:
        item = self.ds[indx]
        labels: list[str] = item["choices"]["label"]
        texts: list[str] = item["choices"]["text"]
        # Map answerKey ("A" or "1" etc.) to position, then to a letter A..Z
        answer_key = str(item["answerKey"]).strip()
        try:
            position = labels.index(answer_key)
        except ValueError:
            position = 0
        answer = chr(65 + position)
        prompt = encode_chat_template(
            template(item["question"], texts),
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
