"""Pack repository state for remote execution."""

import os
import subprocess
import tempfile
from pathlib import Path


COMMON_EXTENSIONS = frozenset(
    {
        ".bash",
        ".c",
        ".cc",
        ".conf",
        ".cpp",
        ".css",
        ".cu",
        ".cuh",
        ".fish",
        ".go",
        ".h",
        ".hpp",
        ".html",
        ".ini",
        ".j2",
        ".java",
        ".jinja",
        ".jinja2",
        ".js",
        ".json",
        ".jsonl",
        ".jsx",
        ".lock",
        ".md",
        ".proto",
        ".py",
        ".pyi",
        ".rs",
        ".rst",
        ".sh",
        ".sql",
        ".tcss",
        ".toml",
        ".ts",
        ".tsx",
        ".txt",
        ".xml",
        ".yaml",
        ".yml",
        ".zsh",
    }
)

__all__ = ["bundle"]


def bundle(dirty: bool = True) -> bytearray:
    """Return a reproducible gzip-compressed Git archive of the repository.

    A clean bundle archives ``HEAD``. A dirty bundle overlays all tracked
    working-tree changes and source-like untracked files onto ``HEAD``. Dirty
    staging uses temporary index and object databases, leaving the repository's
    index and object store unchanged.
    """
    root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    timestamp = subprocess.run(
        ["git", "log", "-1", "--format=%ct", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    archive = ["git", "archive", "--format=tar.gz", f"--mtime=@{timestamp}"]

    if not dirty:
        result = subprocess.run(
            [*archive, "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
        )
        return bytearray(result.stdout)

    objects = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-path", "objects"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    with tempfile.TemporaryDirectory(prefix="theseus-bundle-") as temporary:
        temporary_objects = Path(temporary, "objects")
        temporary_objects.mkdir()
        environment = os.environ.copy()
        alternates = environment.get("GIT_ALTERNATE_OBJECT_DIRECTORIES")
        environment.update(
            {
                "GIT_INDEX_FILE": str(Path(temporary, "index")),
                "GIT_OBJECT_DIRECTORY": str(temporary_objects),
                "GIT_ALTERNATE_OBJECT_DIRECTORIES": os.pathsep.join(
                    path for path in (objects, alternates) if path
                ),
            }
        )

        subprocess.run(
            ["git", "read-tree", "HEAD"],
            cwd=root,
            env=environment,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "add", "--update", "--", "."],
            cwd=root,
            env=environment,
            check=True,
            capture_output=True,
        )
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=root,
            env=environment,
            check=True,
            capture_output=True,
        ).stdout.split(b"\0")
        source_files = [
            path
            for path in untracked
            if path and Path(os.fsdecode(path)).suffix.lower() in COMMON_EXTENSIONS
        ]
        if source_files:
            subprocess.run(
                [
                    "git",
                    "--literal-pathspecs",
                    "add",
                    "--pathspec-from-file=-",
                    "--pathspec-file-nul",
                ],
                cwd=root,
                env=environment,
                input=b"\0".join(source_files) + b"\0",
                check=True,
                capture_output=True,
            )
        tree = subprocess.run(
            ["git", "write-tree"],
            cwd=root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        result = subprocess.run(
            [*archive, tree],
            cwd=root,
            env=environment,
            check=True,
            capture_output=True,
        )
        return bytearray(result.stdout)
