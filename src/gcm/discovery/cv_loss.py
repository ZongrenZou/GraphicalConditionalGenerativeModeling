import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from gcm.kernels.per_dim import kernel_fn


@jax.jit
def loss_function(_params, data, active_modes):
    lx = _params["lx"]
    lz = _params["lz"]
    gamma = jnp.exp(_params["log_gamma"])

    x, z, y = data

    K = kernel_fn(x, z, x, z, lx, lz, active_modes)
    y_regressed = K @ jnp.linalg.solve(K + gamma * np.eye(K.shape[0]), y)
    error = y - y_regressed

    H = K @ jnp.linalg.inv(K + gamma * np.eye(K.shape[0]))
    H = jnp.linalg.diagonal(H)

    error = error.reshape([-1, y.shape[1]])
    H = H.reshape([-1, 1])
    H = jnp.tile(H, [1, error.shape[1]])
    cv = jnp.mean((error / (1 - H)) ** 2, axis=0)
    cv = jnp.sum(cv)
    return cv
