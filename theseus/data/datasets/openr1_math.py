import re
from dataclasses import dataclass

from datasets import load_dataset

from theseus.config import configure, field
from theseus.data.datasets import ChatTemplate, ChatTemplateDataset, ChatTurn
from theseus.registry import dataset


_TASK_HEADER_RE = re.compile(r"^##\s*Task\s+\S+\s*\n+", re.IGNORECASE)
_SOLUTION_HEADER_RE = re.compile(r"^##\s*Solution\.?\s*\n+", re.IGNORECASE)


def _strip_header(text: str, pattern: re.Pattern[str]) -> str:
    m = pattern.match(text.lstrip())
    if not m:
        return text
    return text.lstrip()[m.end() :]


def template(problem: str, solution: str) -> ChatTemplate:
    return [
        ChatTurn(role="user", message=_strip_header(problem, _TASK_HEADER_RE)),
        ChatTurn(
            role="assistant", message=_strip_header(solution, _SOLUTION_HEADER_RE)
        ),
    ]


@dataclass
class OpenR1MathConfig:
    split: str = field("data/openr1_math/split", default="train")


@dataset("openr1_math")
class OpenR1Math(ChatTemplateDataset):
    """open-r1/OpenR1-Math-220k math-reasoning IFT dataset.

    Many rows are scraped from competition packets with the literal
    headers ``## Task X.Y.Z.`` (problem) and ``## Solution.`` (solution);
    these are dataset-specific framing that doesn't belong in the prompt,
    so we strip them at load time.
    """

    CONFIG = OpenR1MathConfig

    def __init__(self) -> None:
        self.ds = load_dataset(
            "open-r1/OpenR1-Math-220k", split=configure(self.CONFIG).split
        )

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> ChatTemplate:
        item = self.ds[idx]
        return template(item["problem"], item["solution"])
