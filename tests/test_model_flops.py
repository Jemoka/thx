"""Analytical counts follow bound children and their execution multiplicity."""

import jax
import jax.numpy as jnp
import pytest

from theseus.config import build, configuration, configure
from theseus.model.attention.grouped import GroupedSelfAttention
from theseus.model.module import Module
from theseus.model.layers.mlp import MLP, QwenMLP
from theseus.model.models.base import GPT
from theseus.model.models.thoughtbubbles import Thoughtbubbles


def test_unknown_flops():
    assert Module().flops(8) == 0.0


@pytest.mark.parametrize("kind,projections", [(MLP, 2), (QwenMLP, 3)])
@pytest.mark.parametrize("intermediate", [24, 40])
def test_mlp_flops(kind, projections, intermediate):
    model = kind(n_embd=16, intermediate_size=intermediate)
    assert model.flops(8) == 6 * projections * 8 * 16 * intermediate




@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("head_dim", [4, 8])
def test_grouped_attention_flops(gated, head_dim):
    model = GroupedSelfAttention(
        n_embd=16, n_head=4, n_kv_head=2,
        head_dim_override=head_dim, q_output_gate=gated,
    )
    # Q and output use four heads, K/V two each, optional gate four more.
    projection_width = (12 + 4 * gated) * head_dim
    assert model.flops(8) == 6 * 8 * 16 * projection_width + 12 * 8**2 * 4 * head_dim
