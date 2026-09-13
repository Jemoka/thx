"""Optional groups are needed only when their associated feature is used."""

import importlib
import subprocess
import sys
from types import SimpleNamespace

import pytest


def test_registries_without_optional_groups():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
for name in (
    'wikipedia', 'transformers', 'tokenizers', 'coverage', 'coveralls',
    'pytest', 'playwright', 'pytest_cov', 'ruff', 'pre_commit', 'mypy',
    'ipdb', 'xprof', 'zensical', 'mkdocstrings', 'pymdownx',
):
    sys.modules[name] = None
from theseus.registry import DATASETS, EVALUATIONS, JOBS
assert 'fever' in DATASETS
assert 'fever' in EVALUATIONS
assert 'gpt/train/pretrain' in JOBS
""",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "module_name,class_name",
    [
        ("theseus.data.datasets.fever", "FEVER"),
        ("theseus.evaluation.datasets.fever", "FEVEREval"),
    ],
)
def test_fever_only_requires_wikipedia_for_uncached_evidence(
    monkeypatch, module_name, class_name
):
    module = importlib.import_module(module_name)
    monkeypatch.setitem(sys.modules, "wikipedia", None)
    dataset = object.__new__(getattr(module, class_name))
    dataset._wiki_cache = {"Cached": "Existing summary"}
    assert dataset._get_evidence_text([]) is None
    assert dataset._get_evidence_text([[[0, 0, None]]]) is None
    assert dataset._get_evidence_text([[[0, 0, "Cached"]]]) == "Existing summary"
    with pytest.raises(ModuleNotFoundError, match="uv sync --group fever"):
        dataset._get_evidence_text([[[0, 0, "Uncached"]]])


@pytest.mark.parametrize(
    "module_name",
    ["theseus.data.datasets.fever", "theseus.evaluation.datasets.fever"],
)
def test_fever_uses_installed_wikipedia(monkeypatch, module_name):
    module = importlib.import_module(module_name)
    calls = []

    def summary(title, *, sentences):
        calls.append((title, sentences))
        return "Evidence"

    monkeypatch.setitem(sys.modules, "wikipedia", SimpleNamespace(summary=summary))
    assert module.get_wikipedia_summary("Example_-LRB-person-RRB-") == "Evidence"
    assert calls == [("Example (person)", 3)]


@pytest.mark.parametrize(
    "module_name,class_name",
    [
        ("llama", "Llama"),
        ("marin", "Marin"),
        ("qwen", "Qwen"),
        ("qwen_3_5", "Qwen3_5"),
        ("qwen_3_5_moe", "Qwen3_5MoE"),
        ("gpt_neox", "GPTNeoX"),
    ],
)
def test_pretrained_requires_huggingface(monkeypatch, module_name, class_name):
    module = importlib.import_module(f"theseus.model.models.contrib.{module_name}")
    # Compute dependencies are outside this audit; do not load CUDA torch on CPU.
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ModuleNotFoundError, match="uv sync --group huggingface"):
        getattr(module, class_name).from_pretrained("unused/no-download")


def test_tokenizer_only_requires_selected_backend(monkeypatch):
    from theseus.data.tokenizer import _build_tokenizer_cached

    monkeypatch.setitem(sys.modules, "transformers", None)
    assert _build_tokenizer_cached("trivial", "unused", True, False) is not None
    with pytest.raises(ModuleNotFoundError, match="uv sync --group huggingface"):
        _build_tokenizer_cached("huggingface", "unused/no-download", True, False)
