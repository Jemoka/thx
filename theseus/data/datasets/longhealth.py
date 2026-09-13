import json
import tempfile
from pathlib import Path

from theseus.data.datasets import ChatTemplate, ChatTemplateDataset, ChatTurn
from theseus.registry import dataset


_BENCHMARK_URL = (
    "https://raw.githubusercontent.com/kbressem/LongHealth/main/data/benchmark_v5.json"
)


def _load_benchmark() -> list[dict[str, object]]:
    """Download and parse the LongHealth benchmark JSON.

    The JSON is a dict mapping patient IDs to patient records.
    """
    import urllib.request

    cache_dir = Path(tempfile.gettempdir()) / "theseus_longhealth"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "benchmark_v5.json"

    if not cache_file.exists():
        urllib.request.urlretrieve(_BENCHMARK_URL, cache_file)  # noqa: S310

    with open(cache_file) as f:
        data = json.load(f)

    if isinstance(data, dict):
        result: list[dict[str, object]] = list(data.values())
        return result
    result_list: list[dict[str, object]] = data
    return result_list


def _build_context(patient: dict[str, object]) -> str:
    """Concatenate all clinical texts for a patient into a single context."""
    texts = patient.get("texts", {})
    parts: list[str] = []
    if isinstance(texts, dict):
        for key in sorted(texts.keys()):
            parts.append(str(texts[key]))
    return "\n\n".join(parts)


def template(
    context: str, question: str, choices: dict[str, str], answer: str
) -> ChatTemplate:
    choices_text = "\n".join(f"{k}: {v}" for k, v in sorted(choices.items()))
    return [
        ChatTurn(
            role="user",
            message=(
                "Read the following clinical document and answer the "
                "question by selecting A, B, C, D, or E.\n\n"
                f"Document:\n{context}\n\n"
                f"Question: {question}\n\n"
                f"{choices_text}\n\n"
                "Answer with only the letter (A, B, C, D, or E):"
            ),
        ),
        ChatTurn(role="assistant", message=answer),
    ]


@dataset("longhealth")
class LongHealth(ChatTemplateDataset):
    """LongHealth: QA benchmark with long clinical documents.

    20 fictional patient cases with 20 multiple-choice questions each
    (400 total).  Downloaded from the official GitHub repository.
    """

    def __init__(self) -> None:
        data = _load_benchmark()
        self.items: list[tuple[str, str, dict[str, str], str]] = []
        for patient in data:
            context = _build_context(patient)
            questions = patient.get("questions", [])
            if not isinstance(questions, list):
                continue
            for q in questions:
                if not isinstance(q, dict):
                    continue
                choices = {
                    "A": str(q.get("answer_a", "")),
                    "B": str(q.get("answer_b", "")),
                    "C": str(q.get("answer_c", "")),
                    "D": str(q.get("answer_d", "")),
                    "E": str(q.get("answer_e", "")),
                }
                self.items.append(
                    (
                        context,
                        str(q.get("question", "")),
                        choices,
                        str(q.get("correct", "")),
                    )
                )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> ChatTemplate:
        context, question, choices, answer = self.items[idx]
        return template(context, question, choices, answer)
