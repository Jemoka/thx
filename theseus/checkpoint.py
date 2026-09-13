"""
Orbax's CheckpointManager is tad goofed. We ungoof it.
"""

import errno
import os
import re
import time
from pathlib import Path
from typing import Any

import jax
import orbax.checkpoint as ocp
from jax.experimental import multihost_utils
from loguru import logger

from theseus.base import PyTree


_RESTORE_ATTEMPTS = 6
_RESTORE_RETRY_DELAY_SECONDS = 5


class CheckpointManager:
    """Save and restore full checkpoint paths with Orbax."""

    def __init__(self) -> None:
        self.checkpointer = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())

    def close(self) -> None:
        """Wait for outstanding saves and close the underlying checkpointer."""
        self.checkpointer.close()

    def save(self, tree: PyTree[Any], path: str | os.PathLike[str]) -> None:
        """Start saving a pytree to a complete checkpoint path.

        Waits for any previous save to finish before starting the new asynchronous
        save. All JAX processes must call this method collectively.

        Args:
            tree: Pytree to checkpoint.
            path: Complete checkpoint path.
        """
        checkpoint = Path(path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        logger.debug("CKMGR | saving checkpoint to {}", checkpoint)
        self.checkpointer.wait_until_finished()
        logger.debug("CKMGR | checkpointer ready")
        multihost_utils.sync_global_devices("ckpmgr:pre")
        logger.debug("CKMGR | attempting save")
        self.checkpointer.save(str(checkpoint), args=ocp.args.StandardSave(tree))
        logger.debug("CKMGR | save done on {}", jax.process_index())
        multihost_utils.sync_global_devices("ckpmgr:post")

    def restore(
        self,
        path: str | os.PathLike[str],
        template: PyTree[Any],
        allow_partial: bool = True,
    ) -> tuple[PyTree[Any], PyTree[bool] | None]:
        """Restore or surgically merge a pytree from a checkpoint.

        All JAX processes must call this method collectively.

        Args:
            path: Complete checkpoint path.
            template: Destination pytree defining structure, shapes, dtypes, and
                shardings.
            allow_partial: Whether destination leaves absent from, or incompatible
                with, the checkpoint may be initialized by the caller.

        Returns:
            The restored pytree and a destination-shaped Boolean pytree marking
            leaves that require initialization. The mask is ``None`` when every
            destination leaf was restored.
        """
        checkpoint = Path(path)
        logger.debug("CKMGR | restoring checkpoint from {}", checkpoint)
        with ocp.PyTreeCheckpointer() as restore:
            multihost_utils.sync_global_devices("ckpmgr:pre")
            abstract = jax.tree.map(ocp.utils.to_shape_dtype_struct, template)
            args = ocp.checkpoint_utils.construct_restore_args(abstract)
            transforms = None
            partial = None
            if allow_partial:
                logger.debug("CKMGR | partial restore enabled")
                metadata = restore.metadata(str(checkpoint)).item_metadata
                source_tree = ocp.tree.to_flat_dict(
                    metadata.tree,
                    sep="/",
                    keep_empty_nodes=True,
                )
                destination_tree = ocp.tree.to_flat_dict(
                    abstract,
                    sep="/",
                    keep_empty_nodes=True,
                )
                exact = source_tree.keys() == destination_tree.keys() and all(
                    getattr(source_tree[name], "shape", None)
                    == getattr(leaf, "shape", None)
                    for name, leaf in destination_tree.items()
                )
                if exact:
                    logger.debug(
                        "CKMGR | checkpoint matches target; restoring without transforms"
                    )
                else:
                    source = ocp.tree.to_flat_dict(metadata.tree, sep="/")
                    destination = ocp.tree.to_flat_dict(abstract, sep="/")
                    # Any transformations tree, including an empty one, selects
                    # Orbax's target-shaped model-surgery path.
                    transforms = {}
                    partial_flat = {}
                    for name, leaf in destination.items():
                        stored = source.get(name)
                        source_shape = getattr(stored, "shape", None)
                        target_shape = getattr(leaf, "shape", None)
                        initialize = stored is None or (
                            source_shape is not None
                            and target_shape is not None
                            and tuple(source_shape) != tuple(target_shape)
                        )
                        partial_flat[name] = initialize
                        if initialize:
                            transforms[re.escape(name)] = ocp.Transform(
                                use_fallback=True
                            )
                    initialized = sum(partial_flat.values())
                    logger.debug(
                        "CKMGR | model surgery restoring {} leaves and initializing {}",
                        len(partial_flat) - initialized,
                        initialized,
                    )
                    if any(partial_flat.values()):
                        partial = ocp.tree.from_flat_dict(
                            partial_flat,
                            target=abstract,
                            sep="/",
                        )

            restore_arguments = ocp.args.PyTreeRestore(
                item=abstract,
                restore_args=args,
                transforms=transforms,
            )
            for attempt in range(_RESTORE_ATTEMPTS):
                try:
                    tree = restore.restore(
                        str(checkpoint),
                        args=restore_arguments,
                    )
                    break
                # Orbax raises the built-in base Exception for failed array reads.
                # Retry only its exact wrapper or an explicit EIO below.
                except Exception as error:
                    message = str(error)
                    transient = (
                        isinstance(error, OSError) and error.errno == errno.EIO
                    ) or "os_error_code='5'" in message
                    transient = transient or (
                        type(error) is Exception
                        and message.startswith(
                            "Encountered error while reading array index:"
                        )
                    )
                    if not transient or attempt + 1 == _RESTORE_ATTEMPTS:
                        raise
                    delay = _RESTORE_RETRY_DELAY_SECONDS * 2**attempt
                    logger.warning(
                        "CKMGR | transient checkpoint I/O failure; retrying in "
                        "{} seconds ({}/{})",
                        delay,
                        attempt + 1,
                        _RESTORE_ATTEMPTS,
                    )
                    time.sleep(delay)
            multihost_utils.sync_global_devices("ckpmgr:post")
            logger.info("CKMGR | restored checkpoint from {}", checkpoint)
            return tree, partial
