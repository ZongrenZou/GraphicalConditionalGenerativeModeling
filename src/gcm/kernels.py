from functools import partial

import jax
import jax.numpy as jnp


def kernel_fn(x, z, xp, zp, lx, lz, active_modes, is_periodic=None):
    if not isinstance(lx, list):
        lx = x.shape[1] * [lx]
    if is_periodic is None:
        is_periodic = x.shape[1] * [False]

    term_z = jnp.array(1.0)
    for i in range(zp.shape[1]):
        term_z = term_z * (
            1 + jnp.exp(-((z[:, i : i + 1] - zp[:, i : i + 1].T) ** 2) / 2 / lz**2)
        )

    term_x = jnp.array(1.0)
    for i in range(xp.shape[1]):
        if is_periodic[i]:
            diff = x[:, i : i + 1] - xp[:, i : i + 1].T
            term_x = term_x * (
                1 + jnp.exp(-2 * jnp.sin(diff / 2) ** 2 / lx[i] ** 2) * active_modes[i]
            )
        else:
            term_x = term_x * (
                1
                + jnp.exp(
                    -((x[:, i : i + 1] - xp[:, i : i + 1].T) ** 2) / 2 / lx[i] ** 2
                )
                * active_modes[i]
            )
    return term_z * term_x


@jax.jit
def kernel_factors(x, z, xp, zp, lx, lz, active_modes):
    active_modes = jnp.asarray(active_modes)
    zd = z[:, None, :] - zp[None, :, :]
    term_z = jnp.prod(1 + jnp.exp(-(zd**2) / (2 * lz**2)), axis=-1)
    xd = x[:, None, :] - xp[None, :, :]
    term_x = 1 + jnp.exp(-(xd**2) / (2 * lx**2)) * active_modes
    return term_z, term_x


@partial(jax.jit, static_argnames=("index",))
def prune_step(X, Z, Y, lx, lz, log_gamma, active_modes, index=-1):
    gamma = jnp.exp(log_gamma)
    Kz, Kx = kernel_factors(X, Z, X, Z, lx, lz, active_modes)
    K = Kz * jnp.prod(Kx, axis=-1)
    n = K.shape[0]
    L = jnp.linalg.cholesky(K + gamma * jnp.eye(n))
    yb = jnp.linalg.solve(L.T, jnp.linalg.solve(L, Y))
    signal2 = yb.T @ K @ yb
    noise2 = yb.T @ (gamma * jnp.eye(n)) @ yb

    Delta = K[..., None] * (1.0 - 1.0 / Kx)
    G = yb @ yb.T
    quads = jnp.sum(Delta * G[..., None], axis=(0, 1))
    j_idx = jnp.arange(active_modes.shape[0])
    if index >= 0:
        valid = (j_idx != index) & (active_modes == 1.0)
    else:
        valid = active_modes == 1.0
    activations = jnp.where(valid, quads, 1e12)
    ratio = noise2 / (noise2 + signal2)
    return ratio, activations
