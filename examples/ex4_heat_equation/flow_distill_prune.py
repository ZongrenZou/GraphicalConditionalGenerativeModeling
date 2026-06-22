import gcm.core as models
import gcm.kernels as utils
from gcm.discovery import discover_vectorized

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import scipy.io as sio


if __name__ == "__main__":
    data = sio.loadmat("./data/data.mat")
    x_data = data["X"]
    u_data = data["U"]
    y_data = data["X_next"] - data["X"]

    x_data = np.concatenate([x_data, u_data], axis=-1)
    x_mu = np.mean(x_data, axis=0)
    x_sd = np.std(x_data, axis=0)
    X_train = (x_data - x_mu) / x_sd

    seed = 81763263  # for reproducibility
    np.random.seed(seed)

    # prune only x1 and x10
    for index in [1, 10]:
        y_mu = np.mean(y_data[:, index - 1 : index], axis=0)
        y_sd = np.std(y_data[:, index - 1 : index], axis=0)
        Y_train = (y_data[:, index - 1 : index] - y_mu) / y_sd

        key = np.random.randint(0, 1_000_000_000)
        key = jr.PRNGKey(key)

        ############ Training ############
        params, model = models.train_cfm_flax(
            key,
            X_train,
            Y_train,
            model_cfg=models.ModelCfg(
                hidden_dims=[64, 64, 64, 64],
            ),
            train_cfg=models.TrainCfg(
                lr=1e-4,
                batch_size=1_000,
                steps=1_000,
            ),
        )

        ############ Sampling ############
        M = x_data.shape[0]
        n_repeats = 100
        x = np.tile(x_data, [n_repeats, 1]).reshape([n_repeats * M, x_data.shape[1]])
        z_samples, z0_samples = models.sample(
            params,
            x=(x - x_mu) / x_sd,
            model=model,
            reshape_output=False,
        )
        y_samples = z_samples[:, 0, :] * y_sd + y_mu
        x_samples = x
        z_samples = z0_samples

        ############ Pruning ############
        prune_params = {
            "lx": 1.0,
            "lz": 1.0,
            "log_gamma": jnp.log(1.0),
        }
        n_modes = 12
        n_train = 5_000

        idx = np.random.choice(x_data.shape[0], x_data.shape[0], replace=False)
        x_train = x_samples[idx[:n_train]]
        z_train = z_samples[idx[:n_train]]
        y_train = y_samples[idx[:n_train]]

        X = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
        Y = (y_train - np.mean(y_train, axis=0)) / np.std(y_train, axis=0)
        Z = (z_train - np.mean(z_train, axis=0)) / np.std(z_train, axis=0)

        mode_names = [rf"$x_{{{i}}}$" for i in range(1, n_modes - 1)] + [r"$u_1$", r"$u_2$"]
        discover_vectorized(
            index,
            X,
            Y,
            Z,
            prune_params,
            n_modes,
            prune_step=utils.prune_step,
            target_name=rf"$\Delta x_{{{index}}}$",
            names=mode_names,
            fig_path=f"./figs/ratios_{index}.png",
            figsize=(12, 5),
        )

    print("End main.")
