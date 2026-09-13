"""Export a Theseus BackbonedTrainer (or BackbonedContrastiveTrainer) checkpoint
to HuggingFace format.

Usage:
    uv run scripts/theseus_to_hf.py SUFFIX --root /path/to/root --name job_name -o /out

SUFFIX is the checkpoint suffix, e.g. 'final/9216', 'best', or 'boundary_0_1'.
"""

import sys
import click
import numpy as np
from pathlib import Path

import jax
import torch
from loguru import logger
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from theseus.base.job import ExecutionSpec
from theseus.job import CheckpointedJob, RestoreableJob
from theseus.registry import JOBS

# Per-implementation conversion functions.
# All take (params, n_layers, hf_cfg) — Qwen ignores hf_cfg for now.
from theseus.model.models.contrib.qwen import _to_hf_state_dict as _qwen_to_hf
from theseus.model.models.contrib.llama import _to_hf_state_dict as _llama_to_hf
from theseus.model.models.contrib.gpt_neox import _to_hf_state_dict as _gpt_neox_to_hf


def _call_to_hf(impl: str, params, n_layers: int, hf_cfg):
    if impl == "qwen":
        return _qwen_to_hf(params, n_layers)
    elif impl == "llama":
        return _llama_to_hf(params, n_layers, hf_cfg)
    elif impl == "gpt_neox":
        return _gpt_neox_to_hf(params, n_layers, hf_cfg)
    else:
        logger.error(f"No _to_hf_state_dict for backbone '{impl}'")
        sys.exit(1)


@click.command()
@click.argument("suffix")
@click.option(
    "--root",
    required=True,
    help="Checkpoint root directory (passed to ExecutionSpec.local)",
)
@click.option("--name", required=True, help="Job name (subfolder inside project/group)")
@click.option("-o", "--output", required=True, help="Output directory for the HF model")
@click.option(
    "--project", default=None, show_default=True, help="Project name (default: general)"
)
@click.option(
    "--group", default=None, show_default=True, help="Group name (default: default)"
)
@click.option(
    "--export-base",
    is_flag=True,
    default=False,
    help="Export the frozen reference (base) model instead of the trained policy (DPO only).",
)
def main(suffix, root, name, output, project, group, export_base):
    """Export a Theseus backbone checkpoint to HuggingFace format."""

    import os

    # disable wandb
    os.environ["WANDB_MODE"] = "disabled"

    # build ExecutionSpec to locate the checkpoint, even though we won't actually run a job here
    spec = ExecutionSpec.local(root, name=name, project=project, group=group)

    # Pre-load config.yaml to find the concrete job class, since RestoreableJob is abstract
    ckpt_path = CheckpointedJob._get_checkpoint_path(spec, suffix)
    raw_cfg = OmegaConf.load(ckpt_path / "config.yaml")
    job_cls = JOBS.get(str(raw_cfg.job)) if "job" in raw_cfg else None
    if job_cls is None:
        logger.error(
            f"Could not find job class '{raw_cfg.get('job')}' in registry. "
            f"Available: {list(JOBS.keys())}"
        )
        sys.exit(1)
    if not issubclass(job_cls, RestoreableJob):
        logger.error(f"Job class '{job_cls.__name__}' is not a RestoreableJob.")
        sys.exit(1)

    # Extract backbone info from the plain YAML cfg before struct mode kicks in
    impl = str(OmegaConf.select(raw_cfg, "architecture.backbone.implementation"))
    weights = str(OmegaConf.select(raw_cfg, "architecture.backbone.weights"))

    logger.info(f"Restoring {job_cls.__name__} from checkpoint '{suffix}' …")
    job, _ = job_cls.from_checkpoint(suffix, spec)

    # Select which params to export
    if export_base:
        if not hasattr(job.state, "base"):
            logger.error(
                "--export-base requires a DPO checkpoint (ContrastiveTrainState)"
            )
            sys.exit(1)
        params = job.state.base
        logger.info("Exporting: reference (base) model")
    else:
        params = job.state.params
        logger.info("Exporting: trained policy model")

    # Convert to HF state dict
    hf_cfg = AutoConfig.from_pretrained(weights)
    n_layers = hf_cfg.num_hidden_layers

    logger.info(f"Converting {n_layers}-layer {impl} params to HF state dict …")
    sd = _call_to_hf(impl, params, n_layers, hf_cfg)

    # Load into HF model
    torch_sd = {k: torch.from_numpy(np.array(jax.device_get(v))) for k, v in sd.items()}
    hf_model = AutoModelForCausalLM.from_config(hf_cfg)
    missing, unexpected = hf_model.load_state_dict(torch_sd, strict=False)
    if missing:
        n = len(missing)
        logger.warning(f"{n} missing key(s): {missing[:3]}{'…' if n > 3 else ''}")
    if unexpected:
        n = len(unexpected)
        logger.warning(f"{n} unexpected key(s): {unexpected[:3]}{'…' if n > 3 else ''}")

    # Save model + tokenizer
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    hf_model.save_pretrained(out)

    tok = AutoTokenizer.from_pretrained(weights)
    tok.save_pretrained(out)

    logger.info(f"Saved to : {out}")


if __name__ == "__main__":
    main()
