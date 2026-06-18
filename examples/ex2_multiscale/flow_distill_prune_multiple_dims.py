import jax.numpy as jnp
import jax.random as jr
import numpy as np
import scipy.io as sio

import gcm.core as models
import gcm.kernels as utils
from gcm.discovery import discover_vectorized

from data.data import ensure_data


if __name__ == "__main__":
    for m in [20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200]:
        data = sio.loadmat(ensure_data(m))
        x_data = data["Xs"]
        y_data = data["dXs"][:, 0:1]

        x_mu = np.mean(x_data, axis=0)
        x_sd = np.std(x_data, axis=0)
        y_mu = np.mean(y_data, axis=0)
        y_sd = np.std(y_data, axis=0)
        X_train = (x_data - x_mu) / x_sd
        Y_train = (y_data - y_mu) / y_sd

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
                steps=2_000,
            ),
        )

        ############ Sampling ############
        z_samples, z0_samples = models.sample(
            params,
            x=(x_data - x_mu) / x_sd,
            model=model,
            reshape_output=False,
        )
        y_samples = z_samples[:, 0, :] * y_sd + y_mu
        x_samples = x_data
        z_samples = z0_samples

        ############ Pruning ############
        prune_params = {
            "lx": 5.0,
            "lz": 1.0,
            "log_gamma": jnp.log(1.0),
        }
        M = 2000

        idx = np.random.choice(x_data.shape[0], x_data.shape[0], replace=False)
        x_train = x_samples[idx[:M]]
        z_train = z_samples[idx[:M]]
        y_train = y_samples[idx[:M]]

        X = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
        Y = (y_train - np.mean(y_train, axis=0)) / np.std(y_train, axis=0)
        Z = z_train

        discover_vectorized(
            0,
            X,
            Y,
            Z,
            prune_params,
            m,
            prune_step=utils.prune_step,
            prune_index=0,
            fig_path=f"./figs/ratios_m_{m}_1.png",
            figsize=(30, 5),
        )

    print("End main.")
