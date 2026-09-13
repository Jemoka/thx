"""Injected sequence memorization evaluation.

Measures how well the model has memorized the 100 injected sequences
from the ``pile_injected`` training dataset.  Uses perplexity on the
injected texts — lower perplexity (higher 1/ppl score) indicates
stronger memorization.
"""

from theseus.data.datasets.pile_injected import INJECTED_TEXTS
from theseus.evaluation.base import PerplexityEvaluation
from theseus.registry import evaluation


@evaluation("pile_injected_ppl")
class PileInjectedEval(PerplexityEvaluation):
    """Perplexity evaluation on injected sequences.

    Returns 1/perplexity (higher is better) — a perfectly memorized
    sequence would have very low perplexity and thus high score.
    """

    def __init__(self) -> None:
        self.items = list(INJECTED_TEXTS)

    @property
    def name(self) -> str:
        return "pile_injected_ppl"

    def __len__(self) -> int:
        return len(self.items)

    def get(self, indx: int) -> str:
        return self.items[indx]
