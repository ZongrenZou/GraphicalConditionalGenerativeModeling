from typing import Sequence, Tuple

import jax.numpy as jnp
from flax import linen as nn


class VelocityMLP(nn.Module):
    """v_theta(t, z, x): concat [z, t, x] -> MLP -> R^d."""

    hidden_dims: Sequence[int]
    dim_out: int

    @nn.compact
    def __call__(self, t, z, x):
        if t.ndim == 0:
            t = t[None] if z.ndim == 1 else jnp.broadcast_to(t, (z.shape[0],))
        t = t[..., None]

        if z.ndim == 1:
            z = z[None, :]
        if x.ndim == 1:
            x = x[None, :]

        h = jnp.concatenate([z, t, x], axis=-1)
        for width in self.hidden_dims:
            h = nn.silu(nn.Dense(width)(h))
        out = nn.Dense(self.dim_out)(h)
        return out
