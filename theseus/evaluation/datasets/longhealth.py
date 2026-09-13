import json
import tempfile
from pathlib import Path
from typing import Tuple

from theseus.evaluation.base import PerplexityComparisonEvaluation
from theseus.registry import evaluation


_BENCHMARK_URL = (
    "https://raw.githubusercontent.com/kbressem/LongHealth/main/data/benchmark_v5.json"
)


def _load_benchmark() -> list[dict[str, object]]:
    """Download and parse the LongHealth benchmark JSON."""
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
    texts = patient.get("texts", {})
    parts: list[str] = []
    if isinstance(texts, dict):
        for key in sorted(texts.keys()):
            parts.append(str(texts[key]))
    return "\n\n".join(parts)


@evaluation("longhealth")
class LongHealthEval(PerplexityComparisonEvaluation):
    """LongHealth long-context clinical QA via per-choice perplexity comparison."""

    def __init__(self) -> None:
        data = _load_benchmark()
        self.items: list[tuple[str, str, list[str], int]] = []
        for patient in data:
            context = _build_context(patient)
            questions = patient.get("questions", [])
            if not isinstance(questions, list):
                continue
            for q in questions:
                if not isinstance(q, dict):
                    continue
                continuations = [
                    str(q.get("answer_a", "")),
                    str(q.get("answer_b", "")),
                    str(q.get("answer_c", "")),
                    str(q.get("answer_d", "")),
                    str(q.get("answer_e", "")),
                ]
                correct = str(q.get("correct", "")).strip().upper()
                if not correct:
                    continue
                correct_idx = ord(correct) - ord("A")
                self.items.append(
                    (
                        context,
                        str(q.get("question", "")),
                        continuations,
                        correct_idx,
                    )
                )

    @property
    def name(self) -> str:
        return "longhealth"

    def get(self, indx: int) -> Tuple[str, list[str], int]:
        context, question, continuations, correct_idx = self.items[indx]
        prefix = f"Document:\n{context}\n\nQuestion: {question}\n\nAnswer: "
        return prefix, continuations, correct_idx

    def __len__(self) -> int:
        return len(self.items)
