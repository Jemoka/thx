"""Run a packed declarative execution."""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, cast

import jax
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from theseus.base import ExecutionSpec, Node, Topology
from theseus.config import configuration
from theseus.execute.combobulator import Step
from theseus.execute.dispatch import DispatchSpec
from theseus.execute.run.lineage import Lineage
from theseus.job import BasicJob


@dataclass
class Runner:
    """Execute a validated dispatch specification in declared step order."""

    dispatch: DispatchSpec
    spec: ExecutionSpec
    config: DictConfig
    lineage: Lineage

    @classmethod
    def new(
        cls,
        dispatch: DispatchSpec,
        spec: ExecutionSpec,
        overrides: DictConfig | None = None,
    ) -> "Runner":
        """Apply config overrides and prepare a dispatch runner."""
        config = cast(
            DictConfig,
            OmegaConf.merge(
                dispatch.job.config,
                overrides if overrides is not None else {},
            ),
        )
        OmegaConf.set_struct(config, True)
        missing = OmegaConf.missing_keys(config)
        if missing:
            raise ValueError(f"Execution config has missing values: {sorted(missing)}")
        if not dispatch.job.steps:
            raise ValueError("DispatchSpec.job must contain at least one step")

        return cls(
            dispatch=dispatch,
            spec=spec,
            config=config,
            lineage=Lineage(spec.hardware, dispatch.nonce),
        )

    def run(self) -> bool:
        """Run every step, or stop successfully when a base rule has no match."""
        previous_tag: str | None = None
        for index, step in enumerate(self.dispatch.job.steps):
            tag = f"{index}:{step.job_name}"
            state = self.lineage.state(tag)
            if state.finished:
                logger.info("RUN | skipping finished job {}", step.job_name)
                previous_tag = tag
                continue
            if state.checkpoint is not None:
                self._run_step(step, state.checkpoint, tag, resume=True)
                previous_tag = tag
                continue

            base = (
                self.lineage.resolve(step.base, previous_tag)
                if step.base is not None
                else None
            )
            if step.base is not None and base is None:
                logger.info(
                    "RUN | base rule for step {} ({}) matched no checkpoint; stopping",
                    index,
                    step.job_name,
                )
                return False
            if step.resume and base is None:
                raise ValueError(
                    f"Step {index} ({step.job_name}) cannot resume without a base"
                )

            self._run_step(step, base, tag, resume=step.resume)
            previous_tag = tag
        return True

    def _run_step(
        self,
        step: Step,
        base: Node | None,
        tag: str,
        resume: bool,
    ) -> None:
        action = "resuming" if resume else "branching" if base else "starting"
        logger.info("RUN | {} {}", action, step.job_name)
        spec = self.spec.model_copy(update={"tag": tag})
        job: BasicJob[Any] | None = None
        try:
            with configuration(self.config):
                job = step.job(spec, base=base)
                job(resume=resume)
        except BaseException:
            if job is not None:
                job.finish()
            raise


def main(argv: Sequence[str] | None = None) -> None:
    """Load and execute a DispatchSpec JSON file."""
    #### runtime setup ####
    parser = argparse.ArgumentParser(prog="python -m theseus.execute.run")
    parser.add_argument("dispatch", type=Path, help="DispatchSpec model JSON")
    parser.add_argument("overrides", nargs="*", help="config.path=value")
    arguments = parser.parse_args(argv)

    malformed = [override for override in arguments.overrides if "=" not in override]
    if malformed:
        raise ValueError(f"Overrides must use field=value syntax: {malformed}")
    overrides = OmegaConf.from_dotlist(arguments.overrides)

    logger.remove()
    logger.add(
        lambda message: sys.stderr.write(message),
        format=(
            "<cyan>{time:YYYY-MM-DD HH:mm:ss}</cyan> |"
            "<level>{level: ^8}</level>| "
            "<magenta>({name}:{line})</magenta> <level>{message}</level>"
        ),
        level="INFO",
        colorize=True,
        enqueue=True,
        filter=lambda record: record["extra"].get("task", "") != "plot",
    )

    try:
        #### parse dispatch spec ####
        dispatch = DispatchSpec.model_validate_json(arguments.dispatch.read_text())
        if dispatch.hardware is None or not dispatch.hardware.hosts:
            raise ValueError("DispatchSpec.hardware must contain at least one host")
        if len(dispatch.hardware.hosts) > 1 and not jax.distributed.is_initialized():
            machine_index = int(
                os.environ.get(
                    "THESEUS_MACHINE_INDEX",
                    os.environ.get("SLURM_PROCID", "0"),
                )
            )
            if not 0 <= machine_index < len(dispatch.hardware.hosts):
                raise ValueError(
                    f"Machine index {machine_index} is outside "
                    "DispatchSpec.hardware.hosts"
                )
            machine = dispatch.hardware.hosts[machine_index]
            local_devices = sum(machine.resources.values()) or None
            explicit_processes = os.environ.get("THESEUS_PROCESS_COUNT")
            jax.distributed.initialize(
                coordinator_address=os.environ.get("THESEUS_COORDINATOR_ADDRESS"),
                num_processes=(
                    int(explicit_processes) if explicit_processes is not None else None
                ),
                process_id=(machine_index if explicit_processes is not None else None),
                local_device_ids=(
                    tuple(range(local_devices)) if local_devices is not None else None
                ),
            )

        #### construction execution spec ####
        topology = (
            Topology.new(dispatch.hardware.chip, shard=dispatch.job.sharding)
            if dispatch.hardware.chip is not None
            else None
        )
        spec = ExecutionSpec(
            name=dispatch.name,
            project=dispatch.project,
            group=dispatch.group,
            hardware=dispatch.hardware,
            topology=topology,
            distributed=jax.process_count() > 1,
            execution_id=dispatch.nonce,
        )
        runner = Runner.new(dispatch, spec, overrides)

        #### dispatch! ####
        completed = runner.run()
    except Exception:
        logger.exception("RUN | execution failed")
        raise

    if completed:
        logger.info("RUN | execution completed")
    else:
        logger.info("RUN | execution stopped because its base was unavailable")


if __name__ == "__main__":
    main()
