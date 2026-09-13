"""Kubernetes transport and immutable publication to a shared PVC."""

import io
import json
import shlex
import subprocess
import tarfile
import uuid
from time import monotonic
from dataclasses import dataclass

from loguru import logger

from theseus.execute.provider.utils import RunResult
from theseus.execute.provider.volcano.config import VolcanoConfig
from theseus.execute.provider.volcano.manifest import manifest


@dataclass(frozen=True)
class Kubernetes:
    """Run kubectl against one explicitly configured namespace."""

    host: VolcanoConfig

    def run(
        self, *args: str, data: bytes | None = None, timeout: float = 30.0
    ) -> RunResult:
        command = ["kubectl", "--namespace", self.host.namespace]
        if self.host.kubeconfig:
            command.extend(("--kubeconfig", self.host.kubeconfig))
        if self.host.context:
            command.extend(("--context", self.host.context))
        started = monotonic()
        # Omit payloads and shell commands that may contain configuration.
        operation = args[0] if args else "version"
        logger.debug(
            "KUBECTL | {} in namespace {} (timeout {}s)",
            operation,
            self.host.namespace,
            timeout,
        )
        try:
            result = subprocess.run(
                [*command, *args], input=data, capture_output=True, timeout=timeout
            )
            logger.debug(
                "KUBECTL | {} exited {} after {:.1f}s",
                operation,
                result.returncode,
                monotonic() - started,
            )
            return RunResult(
                result.returncode,
                result.stdout.decode(errors="replace"),
                result.stderr.decode(errors="replace"),
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            logger.debug(
                "KUBECTL | {} failed after {:.1f}s ({})",
                operation,
                monotonic() - started,
                type(error).__name__,
            )
            return RunResult(-1, "", str(error))

    def publish(
        self, directory: str, files: dict[str, str], timeout: float = 120.0
    ) -> RunResult:
        """Stream a dispatch through a temporary helper; always remove the helper."""
        helper = f"theseus-upload-{uuid.uuid4().hex[:12]}"
        resources: dict[str, dict[str, str]] = {}
        for key, value in self.host.helper_resources.items():
            section, resource = key.split(".", 1)
            resources.setdefault(section, {})[resource] = value
        job = manifest(self.host, helper, "exec sleep 600", resources, loader=True)
        logger.debug("VOLCANO | creating upload helper {}", helper)
        created = self.run(
            "create", "-f", "-", data=json.dumps(job).encode(), timeout=timeout
        )
        if not created.ok:
            return created
        try:
            # Volcano names pods deterministically: job-task-index.
            pod = f"{helper}-worker-0"
            logger.debug("VOLCANO | waiting for upload pod {} to become ready", pod)
            ready = self.run(
                "wait",
                "--for=create",
                f"pod/{pod}",
                f"--timeout={int(timeout)}s",
                timeout=timeout + 5,
            )
            if ready.ok:
                ready = self.run(
                    "wait",
                    "--for=condition=Ready",
                    f"pod/{pod}",
                    f"--timeout={int(timeout)}s",
                    timeout=timeout + 5,
                )
            if not ready.ok:
                return ready
            payload = io.BytesIO()
            with tarfile.open(fileobj=payload, mode="w:gz") as archive:
                for name, text in files.items():
                    content = text.encode()
                    entry = tarfile.TarInfo(name)
                    entry.size = len(content)
                    entry.mode = 0o700 if name.endswith(".sh") else 0o600
                    archive.addfile(entry, io.BytesIO(content))
            staging = shlex.quote(f"{directory}.partial-{uuid.uuid4().hex}")
            destination = shlex.quote(directory)
            # Publish the completed directory atomically; a reused nonce must
            # match the dispatch that was published first. Flush file contents before
            # rename and directory metadata before another pod mounts the PVC.
            command = (
                f"set -eu; umask 077; mkdir -p {staging}; "
                f"trap {shlex.quote('rm -rf -- ' + staging)} EXIT; tar -xzf - -C {staging}; sync; "
                f"if mv -T {staging} {destination} 2>/dev/null; then :; "
                f"else cmp {staging}/dispatch.json {destination}/dispatch.json; fi; "
                f"chmod -R go-rwx {destination}; sync"
            )
            logger.debug("VOLCANO | uploading dispatch to {}", directory)
            return self.run(
                "exec",
                "-i",
                pod,
                "--",
                "sh",
                "-c",
                command,
                data=payload.getvalue(),
                timeout=timeout,
            )
        finally:
            logger.debug("VOLCANO | removing upload helper {}", helper)
            removed = self.run(
                "delete",
                "jobs.batch.volcano.sh",
                helper,
                "--ignore-not-found",
                "--wait=false",
            )
            if not removed.ok:
                logger.warning(
                    "VOLCANO | could not remove upload helper {}: {}",
                    helper,
                    removed.stderr,
                )
