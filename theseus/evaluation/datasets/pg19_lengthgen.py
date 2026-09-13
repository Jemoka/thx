"""Variable-length PG-19 evaluations for length generalization.

Registers evaluations at several context lengths so we can measure
how perplexity changes as sequence length increases beyond the
training block size.
"""

from datasets import load_dataset

from theseus.evaluation.base import PerplexityEvaluation
from theseus.registry import evaluation


def _make_pg19_lengthgen_class(
    eval_name: str, max_chars: int, num_samples: int = 100
) -> type:
    """Factory for length-specific PG-19 perplexity evaluations."""

    class _PG19LengthGenEval(PerplexityEvaluation):
        __doc__ = f"PG-19 perplexity at ~{eval_name} context length."

        def __init__(self) -> None:
            ds = load_dataset("sedthh/gutenberg_english", split="train", streaming=True)
            self.items: list[str] = []
            for item in ds:
                text = item.get("TEXT", "")
                if text and len(text) >= max_chars:
                    # Truncate to target length
                    self.items.append(text[:max_chars])
                if len(self.items) >= num_samples:
                    break

        @property
        def name(self) -> str:
            return eval_name

        def __len__(self) -> int:
            return len(self.items)

        def get(self, indx: int) -> str:
            return self.items[indx]

    _PG19LengthGenEval.__name__ = f"PG19LengthGen_{eval_name}"
    _PG19LengthGenEval.__qualname__ = f"PG19LengthGen_{eval_name}"
    return _PG19LengthGenEval


# Approximate chars per token ~ 4 for cl100k_base on English prose, so target token counts map to:
# 2k tokens -> ~8k chars, 4k -> ~16k, 8k -> ~32k, 16k -> ~64k, 32k -> ~128k
_EVAL_SPECS = [
    ("pg19_2k_ppl", 8_000),
    ("pg19_4k_ppl", 16_000),
    ("pg19_8k_ppl", 32_000),
    ("pg19_16k_ppl", 64_000),
    ("pg19_32k_ppl", 128_000),
]

# Register each evaluation
for _name, _chars in _EVAL_SPECS:
    _cls = _make_pg19_lengthgen_class(_name, _chars)
    evaluation(_name)(_cls)
