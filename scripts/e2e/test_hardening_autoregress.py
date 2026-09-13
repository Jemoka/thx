"""Autoregressive inference on a hardening checkpoint.

Loads a checkpoint produced by hardening.yaml, applies a chat template to
several code-generation prompts, and prints the generated completions.

Usage:
    uv run scripts/test_hardening_autoregress.py SUFFIX \\
        --root /path/to/checkpoints --name job_name \\
        [-p PROJECT] [-g GROUP] \\
        [--num-tokens 256] [--temperature 0.8] [--top-p 0.95]

SUFFIX is the checkpoint suffix, e.g. 'final/9216', 'best', or 'boundary_0_1'.
"""

import sys

import click
import jax
from loguru import logger
from omegaconf import OmegaConf

from theseus.base.job import ExecutionSpec
from theseus.config import configuration
from theseus.data.tokenizer import HuggingFaceTokenizer, get_tokenizer
from theseus.inference.base import InferenceJob
from theseus.job import CheckpointedJob, RestoreableJob
from theseus.registry import JOBS

SYSTEM_PROMPT = (
    "You are a helpful assistant for generating code. Given the prompt, generate "
    "Python code that solves the task; as a reminder, you should be writing production "
    "code (i.e. disable any debugging traces, etc.) Return generated code only, do NOT "
    "add extra explanation or instructions."
)

PROMPTS = [
    (
        "Implement a Flask application that tracks user interactions (such as page views "
        "and actions) by utilizing Flask-SQLAlchemy for session management. Ensure that "
        "the session data is preserved accurately throughout the user's activity and is "
        "not cleared unexpectedly."
    ),
    (
        "Write a Python function that reads a CSV file and computes summary statistics "
        "(mean, median, standard deviation) for each numeric column, returning a dict."
    ),
    (
        "Implement a binary search tree in Python with insert, search, and in-order "
        "traversal methods."
    ),
]


@click.command()
@click.argument("suffix")
@click.option(
    "--root",
    required=True,
    help="Checkpoint root directory (passed to ExecutionSpec.local)",
)
@click.option("--name", required=True, help="Job name (subfolder inside project/group)")
@click.option(
    "-p",
    "--project",
    default=None,
    show_default=True,
    help="Project name (default: general)",
)
@click.option(
    "-g",
    "--group",
    default=None,
    show_default=True,
    help="Group name (default: default)",
)
@click.option(
    "--num-tokens",
    default=256,
    show_default=True,
    help="Number of new tokens to generate per prompt",
)
@click.option(
    "--temperature", default=0.8, show_default=True, help="Sampling temperature"
)
@click.option(
    "--top-p", default=0.95, show_default=True, help="Top-p nucleus sampling threshold"
)
def main(suffix, root, name, project, group, num_tokens, temperature, top_p):
    """Run autoregressive inference on a hardening checkpoint."""

    spec = ExecutionSpec.local(root, name=name, project=project, group=group)

    # Identify the concrete job class from the checkpoint config (same as theseus_to_hf.py)
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

    logger.info(f"Restoring {job_cls.__name__} from checkpoint '{suffix}' ...")
    job, cfg = job_cls.from_checkpoint(suffix, spec)

    inference = InferenceJob.from_trainer(job)

    # Load tokenizer from the checkpoint's config
    with configuration(cfg):
        tok = get_tokenizer()

    if not isinstance(tok, HuggingFaceTokenizer):
        logger.error("Expected a HuggingFace tokenizer; got %s", type(tok).__name__)
        sys.exit(1)
    hf_tok = tok._tokenizer

    key = jax.random.PRNGKey(0)

    for i, prompt in enumerate(PROMPTS):
        print(f"\n{'=' * 70}")
        print(f"PROMPT {i + 1}:")
        print(prompt)
        print(f"{'-' * 70}")

        # Tokenize with chat template (same pattern as the example in the task)
        input_ids = hf_tok.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            tokenize=True,
            return_tensors=None,
            add_generation_prompt=True,
        )
        if hasattr(input_ids, "input_ids"):
            input_ids = input_ids["input_ids"]
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()

        # Left-pad to a uniform length (single sequence → no padding needed)
        xs, masks = inference.pad([input_ids])
        prompt_len = xs.shape[-1]

        total_tokens = min(prompt_len + num_tokens, inference.block_size)

        key, subkey = jax.random.split(key)
        with configuration(cfg):
            result = inference._autoregress(
                inference.state,
                subkey,
                xs,
                masks,
                total_tokens,
                temperature,
                top_p,
            )

        # Decode only the newly generated tokens
        generated_ids = result[0, prompt_len:].tolist()
        generated_text = tok.decode(generated_ids)

        print("GENERATION:")
        print(generated_text)

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
