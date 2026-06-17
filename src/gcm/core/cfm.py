from dataclasses import dataclass, field
from typing import Sequence

import diffrax as dfx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from flax.training import train_state
from tqdm import trange

from gcm.core.velocity_mlp import VelocityMLP


@dataclass
class ModelCfg:
    hidden_dims: Sequence[int] = (64, 64)


@dataclass
class TrainCfg:
    lr: float = 2e-4
    batch_size: int = 256
    steps: int = 50_000
    log_every: int = 500
    seed: int = 0
    epochs: int | None = field(default=None, repr=False)

    @property
    def num_steps(self) -> int:
        return self.epochs if self.epochs is not None else self.steps


def make_cfm_batch(key, Xb, Yb):
    key_t, key_z0 = jr.split(key)
    B, d = Yb.shape
    t = jr.uniform(key_t, (B,))
    z0 = jr.normal(key_z0, (B, d))
    z_t = (1.0 - t)[:, None] * z0 + t[:, None] * Yb
    u_t = Yb - z0
    return t, z_t, Xb, u_t


def loss_fn(params, model, key, Xb, Yb):
    t, z_t, xb, u_t = make_cfm_batch(key, Xb, Yb)
    v_pred = model.apply({"params": params}, t, z_t, xb)
    mse = jnp.mean(jnp.sum((v_pred - u_t) ** 2, axis=-1))
    return mse


def create_train_state(rng, model, d_in: int, k_in: int, cfg: TrainCfg):
    t0 = jnp.array(0.0)
    z0 = jnp.zeros((1, d_in))
    x0 = jnp.zeros((1, k_in))
    params = model.init(rng, t0, z0, x0)["params"]
    tx = optax.adam(cfg.lr)
    return train_state.TrainState(
        step=0, apply_fn=model.apply, params=params, tx=tx, opt_state=tx.init(params)
    )


def train_cfm_flax(key, X_train, Y_train, model_cfg, train_cfg):
    N, k = X_train.shape
    d = Y_train.shape[1]

    model = VelocityMLP(hidden_dims=tuple(model_cfg.hidden_dims), dim_out=d)
    key, init_key = jr.split(key)
    state = create_train_state(init_key, model, d_in=d, k_in=k, cfg=train_cfg)

    @jax.jit
    def step(state, key, Xb, Yb):
        batch_loss, grads = jax.value_and_grad(loss_fn)(
            state.params, model, key, Xb, Yb
        )
        state = state.apply_gradients(grads=grads)
        return state, batch_loss

    n_batches = N // train_cfg.batch_size
    if n_batches == 0:
        raise ValueError(
            f"batch_size ({train_cfg.batch_size}) must be <= dataset size ({N})"
        )

    pbar = trange(1, train_cfg.num_steps + 1, desc="train")
    for _it in pbar:
        idx = np.random.choice(N, N, replace=False)
        loss = 0.0

        for i in range(n_batches):
            idx_batch = idx[i * train_cfg.batch_size : (i + 1) * train_cfg.batch_size]
            X_batch = jnp.array(X_train[idx_batch])
            Y_batch = jnp.array(Y_train[idx_batch])
            key, _ = jr.split(key, 2)
            state, batch_loss = step(state, key, X_batch, Y_batch)
            loss += batch_loss

        pbar.set_postfix(loss=f"{float(loss / n_batches):.4f}")

    return state.params, model


def sample(
    params,
    x,
    model,
    *,
    z_dim: int | None = None,
    reshape_output: bool = True,
    seed: int | None = None,
    squeeze_time: bool = False,
):
    def velocity(params, model, t, z, x):
        out = model.apply({"params": params}, t, z, x)
        return out.reshape(-1) if reshape_output else out

    term = dfx.ODETerm(lambda t, y, args: velocity(params, model, t, y, args))

    @jax.jit
    def solve_single(z0i, x):
        sol = dfx.diffeqsolve(
            term,
            dfx.Tsit5(),
            t0=0.0,
            t1=1.0,
            dt0=None,
            y0=z0i,
            args=x,
            saveat=dfx.SaveAt(t1=True),
            stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-5),
            max_steps=1_000_000,
        )
        return sol.ys

    if z_dim is None:
        z_dim = int(model.dim_out)
    if seed is not None:
        z0 = jr.normal(jr.PRNGKey(seed), (x.shape[0], z_dim))
    else:
        z0 = np.random.normal(size=[x.shape[0], z_dim])
    z0 = jnp.array(z0)
    zT = jax.vmap(solve_single, in_axes=(0, 0))(z0, x)
    zT = jnp.asarray(zT)
    if squeeze_time:
        zT = zT[:, 0, :]
    return zT, z0
