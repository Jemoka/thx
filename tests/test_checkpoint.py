import errno
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import pytest
from flax.training import train_state

from theseus.checkpoint import CheckpointManager


def apply_model(variables, inputs):
    return inputs @ variables["params"]["dense"]["kernel"]


@pytest.fixture
def state():
    params = {
        "embed": jnp.arange(64, dtype=jnp.float32).reshape(8, 8),
        "dense": {
            "kernel": jnp.arange(64, dtype=jnp.float32).reshape(8, 8) / 64,
            "bias": jnp.arange(8, dtype=jnp.float32),
        },
    }
    state = train_state.TrainState.create(
        apply_fn=apply_model,
        params=params,
        tx=optax.adamw(learning_rate=1e-3),
    )
    gradients = jax.tree.map(jnp.ones_like, state.params)
    return state.apply_gradients(grads=gradients)


@pytest.fixture
def abstract_state(state):
    return jax.eval_shape(lambda tree: tree, state)


def assert_tree_equal(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


def merge_surgery(restored, target, partial):
    initialized = jax.tree.map(
        lambda value, needed: value if needed else None,
        target,
        partial,
    )
    return jax.tree.map(
        lambda old, new, needed: new if needed else old,
        restored,
        initialized,
        partial,
        is_leaf=lambda value: value is None,
    )


def test_restore_train_state_from_abstract_template(tmp_path, state, abstract_state):
    manager = CheckpointManager()
    checkpoint = tmp_path / "step-1"
    manager.save(state, checkpoint)
    manager.checkpointer.wait_until_finished()

    assert all(
        isinstance(leaf, jax.ShapeDtypeStruct)
        for leaf in jax.tree.leaves(abstract_state)
    )
    with patch(
        "theseus.checkpoint.ocp.args.PyTreeRestore",
        wraps=ocp.args.PyTreeRestore,
    ) as restore_args:
        restored, partial = manager.restore(checkpoint, abstract_state)

    assert partial is None
    assert restore_args.call_args.kwargs["transforms"] is None
    assert isinstance(restored, train_state.TrainState)
    assert_tree_equal(restored, state)


def test_restore_train_state_from_concrete_template(tmp_path, state):
    manager = CheckpointManager()
    checkpoint = tmp_path / "step-1"
    manager.save(state, checkpoint)
    manager.checkpointer.wait_until_finished()

    restored, partial = manager.restore(checkpoint, state)

    assert partial is None
    assert_tree_equal(restored, state)


@pytest.mark.parametrize(
    "error",
    (
        OSError(errno.EIO, "Input/output error"),
        ValueError("Input/output error [os_error_code='5']"),
        Exception(
            "Encountered error while reading array index: (slice(0, 1, 1),). "
            "See full TensorStore details"
        ),
    ),
)
def test_restore_retries_transient_io_errors(tmp_path, state, abstract_state, error):
    manager = CheckpointManager()
    checkpoint = tmp_path / "step-1"
    manager.save(state, checkpoint)
    manager.checkpointer.wait_until_finished()
    restore = ocp.PyTreeCheckpointer.restore
    attempts = 0

    def fail_once(checkpointer, *args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise error
        return restore(checkpointer, *args, **kwargs)

    with (
        patch.object(ocp.PyTreeCheckpointer, "restore", new=fail_once),
        patch("theseus.checkpoint.time.sleep") as sleep,
    ):
        restored, partial = manager.restore(checkpoint, abstract_state)

    assert attempts == 2
    sleep.assert_called_once_with(5)
    assert partial is None
    assert_tree_equal(restored, state)


@pytest.mark.parametrize(
    "error",
    (
        ValueError("invalid checkpoint"),
        RuntimeError("invalid runtime state"),
        Exception("unclassified checkpoint failure"),
    ),
)
def test_restore_does_not_retry_other_errors(tmp_path, state, abstract_state, error):
    manager = CheckpointManager()
    checkpoint = tmp_path / "step-1"
    manager.save(state, checkpoint)
    manager.checkpointer.wait_until_finished()

    with (
        patch.object(
            ocp.PyTreeCheckpointer,
            "restore",
            side_effect=error,
        ) as restore,
        patch("theseus.checkpoint.time.sleep") as sleep,
        pytest.raises(type(error), match=str(error)),
    ):
        manager.restore(checkpoint, abstract_state)

    restore.assert_called_once()
    sleep.assert_not_called()


def test_save_does_not_create_latest_pointer(tmp_path, state):
    manager = CheckpointManager()
    checkpoint = tmp_path / "step-1"

    manager.save(state, checkpoint)
    manager.checkpointer.wait_until_finished()

    assert checkpoint.exists()
    assert not (tmp_path / "latest").exists()


def test_abstract_template_reaches_orbax_without_concrete_arrays(
    tmp_path, state, abstract_state
):
    manager = CheckpointManager()
    checkpoint = tmp_path / "step-1"
    manager.save(state, checkpoint)
    manager.checkpointer.wait_until_finished()
    saw_concrete_array = False

    def observe_template(leaf):
        nonlocal saw_concrete_array
        saw_concrete_array |= isinstance(leaf, jax.Array)
        return leaf

    with patch(
        "theseus.checkpoint.ocp.utils.to_shape_dtype_struct",
        side_effect=observe_template,
    ):
        _, partial = manager.restore(checkpoint, abstract_state)

    assert all(
        isinstance(leaf, jax.ShapeDtypeStruct)
        for leaf in jax.tree.leaves(abstract_state)
    )
    assert partial is None
    assert not saw_concrete_array


def test_restore_merges_checkpoint_into_initialized_target(tmp_path):
    source_params = {
        "blocks_0": {"weight": jnp.full((2, 2), 1.0)},
        "blocks_1": {"weight": jnp.full((2, 2), 2.0)},
        "removed": jnp.full((2,), 3.0),
    }
    target_params = {
        "blocks_0": {"weight": jnp.full((2, 2), -1.0)},
        "blocks_1": {"weight": jnp.full((2, 2), -2.0)},
        "new_mlp": {"weight": jnp.full((2, 2), 9.0)},
    }
    tx = optax.adamw(learning_rate=1e-3)
    source = train_state.TrainState.create(
        apply_fn=apply_model,
        params=source_params,
        tx=tx,
    ).apply_gradients(grads=jax.tree.map(jnp.ones_like, source_params))
    target = train_state.TrainState.create(
        apply_fn=apply_model,
        params=target_params,
        tx=tx,
    )
    template = jax.eval_shape(lambda: target)
    checkpoint = tmp_path / "surgery"
    manager = CheckpointManager()
    manager.save(source, checkpoint)
    manager.checkpointer.wait_until_finished()

    restored, partial = manager.restore(checkpoint, template)

    assert partial is not None
    assert partial.params["blocks_0"]["weight"] is False
    assert partial.params["blocks_1"]["weight"] is False
    assert partial.params["new_mlp"]["weight"] is True
    assert isinstance(
        restored.params["new_mlp"]["weight"],
        jax.ShapeDtypeStruct,
    )
    restored = merge_surgery(restored, target, partial)
    np.testing.assert_array_equal(
        restored.params["blocks_0"]["weight"],
        source.params["blocks_0"]["weight"],
    )
    np.testing.assert_array_equal(
        restored.params["blocks_1"]["weight"],
        source.params["blocks_1"]["weight"],
    )
    np.testing.assert_array_equal(
        restored.params["new_mlp"]["weight"],
        target.params["new_mlp"]["weight"],
    )
    assert "removed" not in restored.params
    np.testing.assert_array_equal(
        restored.opt_state[0].mu["blocks_0"]["weight"],
        source.opt_state[0].mu["blocks_0"]["weight"],
    )
    np.testing.assert_array_equal(
        restored.opt_state[0].mu["blocks_1"]["weight"],
        source.opt_state[0].mu["blocks_1"]["weight"],
    )
    np.testing.assert_array_equal(
        restored.opt_state[0].mu["new_mlp"]["weight"],
        target.opt_state[0].mu["new_mlp"]["weight"],
    )


def test_restore_uses_target_for_changed_shapes_and_optimizer(tmp_path):
    source = train_state.TrainState.create(
        apply_fn=apply_model,
        params={"weight": jnp.ones((2,))},
        tx=optax.adamw(learning_rate=1e-3),
    )
    target = train_state.TrainState.create(
        apply_fn=apply_model,
        params={"weight": jnp.full((3,), 7.0)},
        tx=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=1e-3),
        ),
    )
    template = jax.eval_shape(lambda: target)
    checkpoint = tmp_path / "changed"
    manager = CheckpointManager()
    manager.save(source, checkpoint)
    manager.checkpointer.wait_until_finished()

    restored, partial = manager.restore(checkpoint, template)

    assert partial is not None
    assert partial.params["weight"] is True
    assert isinstance(restored.params["weight"], jax.ShapeDtypeStruct)
    restored = merge_surgery(restored, target, partial)
    np.testing.assert_array_equal(restored.params["weight"], target.params["weight"])
    assert jax.tree.structure(restored.opt_state) == jax.tree.structure(
        target.opt_state
    )
    for actual, expected in zip(
        jax.tree.leaves(restored.opt_state),
        jax.tree.leaves(target.opt_state),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
