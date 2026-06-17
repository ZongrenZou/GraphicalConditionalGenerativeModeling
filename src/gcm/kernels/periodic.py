import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp


def kernel_fn(x, z, xp, zp, lx, lz, active_modes, is_periodic=None):
    if not isinstance(lx, list):
        lx = x.shape[1] * [lx]
    if is_periodic is None:
        is_periodic = x.shape[1] * [False]

    term_z = 1
    for i in range(zp.shape[1]):
        term_z = term_z * (
            1 + jnp.exp(-((z[:, i : i + 1] - zp[:, i : i + 1].T) ** 2) / 2 / lz**2)
        )

    term_x = 1
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
