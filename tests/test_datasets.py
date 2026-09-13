"""Tests for new benchmark datasets and evaluations."""

import pytest


class TestPileInjected:
    """Tests for the injected Pile dataset."""

    def test_injected_texts_reproducible(self):
        """Same seed always produces the same injected sequences."""
        from theseus.data.datasets.pile_injected import (
            _generate_injected_texts,
            INJECTED_TEXTS,
        )

        texts_a = _generate_injected_texts(n_sequences=100, seed=42)
        texts_b = _generate_injected_texts(n_sequences=100, seed=42)
        assert texts_a == texts_b
        assert texts_a == list(INJECTED_TEXTS)

    def test_injected_texts_count(self):
        from theseus.data.datasets.pile_injected import INJECTED_TEXTS

        assert len(INJECTED_TEXTS) == 100

    def test_injected_texts_nonempty(self):
        from theseus.data.datasets.pile_injected import INJECTED_TEXTS

        for text in INJECTED_TEXTS:
            assert len(text) > 100  # each should be substantial

    def test_injection_positions_sorted(self):
        from theseus.data.datasets.pile_injected import INJECTION_POSITIONS

        assert INJECTION_POSITIONS == sorted(INJECTION_POSITIONS)
        assert len(INJECTION_POSITIONS) == 100

    def test_different_seeds_different_texts(self):
        from theseus.data.datasets.pile_injected import _generate_injected_texts

        texts_a = _generate_injected_texts(n_sequences=10, seed=42)
        texts_b = _generate_injected_texts(n_sequences=10, seed=99)
        assert texts_a != texts_b


class TestPileInjectedEval:
    """Tests for the injected sequence memorization evaluation."""

    def test_eval_uses_same_texts(self):
        from theseus.data.datasets.pile_injected import INJECTED_TEXTS
        from theseus.evaluation.datasets.pile_injected import PileInjectedEval

        ev = PileInjectedEval()
        assert len(ev) == len(INJECTED_TEXTS)
        for i in range(len(ev)):
            assert ev.get(i) == INJECTED_TEXTS[i]

    def test_eval_name(self):
        from theseus.evaluation.datasets.pile_injected import PileInjectedEval

        ev = PileInjectedEval()
        assert ev.name == "pile_injected_ppl"


class TestPG19LengthGen:
    """Tests for variable-length PG-19 evaluations."""

    def test_eval_names_registered(self):
        from theseus.registry import EVALUATIONS

        for name in [
            "pg19_2k_ppl",
            "pg19_4k_ppl",
            "pg19_8k_ppl",
            "pg19_16k_ppl",
            "pg19_32k_ppl",
        ]:
            assert name in EVALUATIONS, f"{name} not registered"
