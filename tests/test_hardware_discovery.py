"""Reported GPU names resolve consistently across hardware discovery paths."""

from types import SimpleNamespace
import subprocess

import jax
import pytest

from theseus.base.chip import SUPPORTED_CHIPS
from theseus.base.hardware import local, match


@pytest.mark.parametrize(
    "name,key",
    [
        ("NVIDIA H100 80GB HBM3", "h100"),
        ("NVIDIA H200", "h200"),
        ("NVIDIA A100-SXM4-80GB", "a100-sxm4-80gb"),
        ("NVIDIA A100-PCIE-40GB", "a100-pcie-40gb"),
        ("NVIDIA RTX A6000", "a6000"),
        ("NVIDIA RTX 6000 Ada Generation", "ada6000"),
        ("NVIDIA RTX PRO 6000 Blackwell Server Edition", "b6000"),
        ("NVIDIA L40S", "l40s"),
        ("NVIDIA L40", "l40"),
        ("NVIDIA GeForce RTX 5090", "rtx5090"),
        ("NVIDIA Unknown 80GB HBM3", None),
        ("NVIDIA H1000", None),
        ("NVIDIA RTX 4090", None),
    ],
)
def test_match(name: str, key: str | None) -> None:
    expected = SUPPORTED_CHIPS[key] if key is not None else None
    assert match(name) == expected


@pytest.mark.parametrize(
    "key", [k for k in SUPPORTED_CHIPS if not k.startswith("tpu-") and k != "cpu"]
)
def test_catalog_gpu_names(key: str) -> None:
    assert match(SUPPORTED_CHIPS[key].display_name) == SUPPORTED_CHIPS[key]


def test_local_falls_back_to_nvidia_smi(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = SimpleNamespace(
        platform="gpu", device_kind="CUDA device", process_index=0
    )
    monkeypatch.setattr(jax, "devices", lambda: [device])
    monkeypatch.setattr(jax, "process_count", lambda: 1)
    monkeypatch.setattr(jax, "process_index", lambda: 0)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="0, NVIDIA Mystery 81920, 81920\n"
        ),
    )

    hardware = local(str(tmp_path), "-")

    assert hardware.chip is not None
    assert hardware.chip.name == "mystery-81920"
    assert hardware.chip.memory == 81920 * 1024**2
