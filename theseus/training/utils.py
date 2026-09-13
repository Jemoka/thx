"""
Small utilities useful during training.
"""

from typing import Any, Dict, Mapping

import jax
from flax.traverse_util import flatten_dict

from theseus.base import Topology


def scalar_metadata(collection: Mapping[str, Any]) -> Dict[str, jax.Array]:
    """Flatten a Flax ``scalars`` collection into trainer metadata."""

    metadata: Dict[str, jax.Array] = {}
    flattened = flatten_dict(collection)  # type: ignore[no-untyped-call]
    for path, emitted in flattened.items():
        name = "/".join(path)
        if name in ("intermediates", "plots"):
            raise ValueError(f"Scalar metric name {name!r} is reserved")
        if name in metadata:
            raise ValueError(f"Scalar metric name {name!r} is ambiguous")
        value = emitted[-1]
        if value.ndim != 0:
            raise ValueError(
                f"Scalar metric {name!r} must be rank zero, got shape {value.shape}"
            )
        metadata[name] = value
    return metadata


def estimate_per_device_batch_size(
    chip_memory: int,
    total_params_millions: float,
    shards: int,
    block_size: int,
    vram_calib_factor: float,
) -> int:
    """Estimate max per-device batch size based on VRAM.

    Memory breakdown (per param, bf16 training with AdamW):
    - Params: 2 bytes (bf16)
    - Gradients: 4 bytes (fp32 for accumulation)
    - Master weights: 4 bytes (fp32 copy for optimizer)
    - Optimizer m: 4 bytes (fp32 first moment)
    - Optimizer v: 4 bytes (fp32 second moment)
    Total: ~18 bytes/param (all sharded by tensor parallelism)

    Activations: scales with batch * seq * sqrt(params)

    Args:
        chip_memory: bytes of VRAM per device
        total_params_millions: model parameters in millions
        shards: tensor parallel shards
        block_size: sequence length
        vram_calib_factor: user-tuned calibration for activation memory

    Returns:
        Estimated batch size (at least 1)
    """
    raise NotImplementedError(
        "for some reason even with *180 this hilariously underestimates memory usage"
    )

    params_per_shard = total_params_millions * 1e6 / shards

    fixed_memory = params_per_shard * 180
    usable_memory = chip_memory * 0.5 - fixed_memory

    # Activation memory per sample (empirical scaling)
    bytes_per_sample = vram_calib_factor * block_size * (params_per_shard**0.5)

    estimated = int(usable_memory / bytes_per_sample)
    return max(1, estimated)


def find_accumulation_steps(
    batch_size: int, per_device_batch_size: int, topology: Topology
) -> tuple[int, int]:
    """Finds the largest per-device batch size and corresponding number of gradient accumulation steps

    Args:
        batch_size (int): Global batch size
        per_device_batch_size (int): Maximum per-device batch size
        topology (Topology): Topology object containing replica information

    Returns:
        Tuple[int, int]: per-device batch size and number of gradient accumulation steps
    """

    replicas = topology.replicas

    for bs in reversed(range(1, per_device_batch_size + 1)):
        if batch_size % (bs * replicas) == 0:
            return bs, batch_size // (bs * replicas)
    raise ValueError(
        f"No grad_acc found for global_batch_size {batch_size} and max_batch_size {per_device_batch_size} and dp_replicate {replicas}"
    )
