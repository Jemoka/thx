"""BIG-Bench Hard rollout evaluation (Suzgun et al., 2022).

BBH is a curated set of 27 subtasks from BIG-Bench with mixed answer styles.
We load all subtasks (via the per-subtask configs of ``lukaemon/bbh``),
concatenate them into one flat dataset, and tag each example with its subtask
so the cleaner/grader can dispatch on answer style.

Answer-style buckets:
* ``letter_choice`` — task asks for "(A)" / "(B)" / ... Pick the parenthesized
  letter from the rollout.
* ``yes_no`` — boolean tasks; grade after normalizing case.
* ``valid_invalid`` — formal-fallacies-style; grade after lower-casing.
* ``free_text`` — everything else; equality on stripped text.
"""

from typing import Any, Tuple

from datasets import concatenate_datasets, load_dataset

from theseus.data.datasets import ChatTemplate, ChatTurn
from theseus.evaluation.base import RolloutEvaluation
from theseus.registry import evaluation
from theseus.data.tokenizer import (
    decode_chat_template,
    encode_chat_template,
    get_tokenizer,
)


BBH_SUBTASKS = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
]

_LETTER_CHOICE_TASKS = {
    "date_understanding",
    "disambiguation_qa",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
}
_YES_NO_TASKS = {
    "causal_judgement",
    "navigate",
    "sports_understanding",
    "web_of_lies",
}
_FALLACY_TASKS = {"formal_fallacies"}
_BOOL_TASKS = {"boolean_expressions"}


def template(question: str) -> ChatTemplate:
    return [
        ChatTurn(
            role="user",
            message=(
                "Solve the following reasoning problem. Respond with only "
                "the final answer.\n\n"
                f"{question}"
            ),
        ),
    ]


@evaluation("bbh")
class BBHEval(RolloutEvaluation):
    """BIG-Bench Hard rollout evaluation."""

    def __init__(self) -> None:
        per_subtask = []
        for subtask in BBH_SUBTASKS:
            ds = load_dataset("lukaemon/bbh", subtask, split="test")
            ds = ds.add_column("subtask", [subtask] * len(ds))
            per_subtask.append(ds)
        self.ds = concatenate_datasets(per_subtask)
        self.encoder = get_tokenizer()

    @property
    def name(self) -> str:
        return "bbh"

    def max_new_tokens(self, inference: Any) -> int:
        return 64

    def get(self, indx: int) -> Tuple[str, str]:
        item = self.ds[indx]
        question = item["input"]
        # Encode the subtask in the gold so check() can dispatch on it without
        # needing the index passed through.  Format: "<subtask>\x1f<target>".
        target = f"{item['subtask']}\x1f{item['target']}"
        prompt = encode_chat_template(
            template(question),
            self.encoder,
            prompt=True,
            tokenize=False,
        )
        return prompt, target

    def __len__(self) -> int:
        return len(self.ds)

    def clean(self, y_hat: str) -> str:
        chats: ChatTemplate = decode_chat_template(y_hat)
        for turn in chats:
            if turn.role == "assistant":
                return turn.message.strip()
        return ""

    def check(self, y: str, y_hat: str) -> bool:
        if "\x1f" in y:
            subtask, target = y.split("\x1f", 1)
        else:
            subtask, target = "", y

        target = target.strip()
        pred = y_hat.strip()

        if subtask in _LETTER_CHOICE_TASKS:
            # Targets are formatted as "(A)" / "(B)" etc.; pick the first
            # parenthesized letter from the prediction.
            import re

            m = re.search(r"\(([A-Z])\)", pred)
            pred_norm = f"({m.group(1)})" if m else pred.upper()
            return pred_norm == target.upper()

        if subtask in _YES_NO_TASKS:
            return (
                target.strip().lower() in pred.lower().split()
                or pred.lower().startswith(target.strip().lower())
            )

        if subtask in _FALLACY_TASKS or subtask in _BOOL_TASKS:
            return target.strip().lower() == pred.strip().lower()

        # free-text default
        return target.strip().lower() == pred.strip().lower()
