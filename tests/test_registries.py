"""Registry sanity tests — verify all expected datasets, evaluations,
and jobs are registered.

Migrated from scripts/test_new_datasets.py (registry portions).
"""

import pytest


class TestDatasetRegistry:
    def test_core_datasets_registered(self):
        from theseus.registry import DATASETS

        expected = [
            "fineweb", "pile", "pes2o", "pg19", "ccaligned",
            "mnli", "qqp", "sst2", "siqa", "mmlu", "squad",
            "cfq", "clutrr", "harmfulqa", "longhealth", "mtob",
            "pile_detoxify", "pile_injected",
            "alpaca", "bbq", "fever", "longbench", "winogrande",
        ]
        for name in expected:
            assert name in DATASETS, f"'{name}' not in DATASETS registry"


class TestEvaluationRegistry:
    def test_core_evals_registered(self):
        from theseus.registry import EVALUATIONS

        expected = [
            "alpaca", "mnli", "qqp", "sst2", "siqa", "mmlu", "squad",
            "cfq", "clutrr", "longhealth", "mtob",
            "pile_ppl", "pes2o_ppl", "pg19_ppl", "tinystories_ppl", "fineweb_ppl",
            "mnli_ppl", "qqp_ppl", "sst2_ppl", "siqa_ppl",
            "pile_injected_ppl",
            "pg19_2k_ppl", "pg19_4k_ppl", "pg19_8k_ppl", "pg19_16k_ppl", "pg19_32k_ppl",
        ]
        for name in expected:
            assert name in EVALUATIONS, f"'{name}' not in EVALUATIONS registry"


class TestJobRegistry:

    def test_base_jobs_registered(self):
        from theseus.registry import JOBS

        expected = ["gpt/train/pretrain"]
        for name in expected:
            assert name in JOBS, f"'{name}' not in JOBS registry"


def test_all_builtin_jobs_use_canonical_api_and_build_config():
    from omegaconf import DictConfig
    from theseus.config import build
    from theseus.job import BasicJob
    from theseus.registry import JOBS

    for name, job in JOBS.items():
        if not job.__module__.startswith("theseus."):
            continue
        assert issubclass(job, BasicJob), name
        assert isinstance(build(*job.config()), DictConfig), name
