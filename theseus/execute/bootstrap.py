"""Generate a self-contained remote execution bootstrap."""

import base64
from pathlib import Path

from theseus.execute.bundle import bundle


_BOOTSTRAP_TEMPLATE = Path(__file__).with_suffix(".sh")
_REPOSITORY_PLACEHOLDER = "__REPO_BUNDLE__"

__all__ = ["generate"]


def generate() -> str:
    """Return the bootstrap script with the current repository embedded."""
    script = _BOOTSTRAP_TEMPLATE.read_text().replace(
        _REPOSITORY_PLACEHOLDER,
        base64.b64encode(bundle()).decode("ascii"),
    )
    if _REPOSITORY_PLACEHOLDER in script:
        raise ValueError("Repository placeholder remains unhydrated")
    return script
