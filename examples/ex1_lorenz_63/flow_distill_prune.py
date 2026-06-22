import jax.random as jr
import numpy as np
import scipy.io as sio

import gcm.core as models
from gcm.discovery import discover_product


if __name__ == "__main__":
    data = sio.loadmat("./data/data.mat")
    x_data = data["X"]
    y_data = data["dX"]

    ## normalize the data
    x_mu = np.mean(x_data, axis=0)
    x_sd = np.std(x_data, axis=0)
    y_mu = np.mean(y_data, axis=0)
    y_sd = np.std(y_data, axis=0)
    X_train = (x_data - x_mu) / x_sd
    Y_train = (y_data - y_mu) / y_sd

    for run_idx in range(10):
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
                epochs=1_000,
            ),
        )

        ############ Sampling ############
        n_draws = 100
        sample_x: np.ndarray = np.tile(x_data, [n_draws, 1]).reshape(
            [n_draws * x_data.shape[0], x_data.shape[1]]
        )

        z_samples, z0_samples = models.sample(
            params,
            x=(sample_x - x_mu) / x_sd,
            model=model,
        )
        y_samples = np.asarray(z_samples)[:, 0, :] * y_sd + y_mu

        ############ Pruning ############
        prune_params = {
            "lx": 1.0,
            "lz": 1.0,
            "log_gamma": 0.0,
        }
        n_vars = 3
        n_train = 5_000

        idx = np.random.choice(sample_x.shape[0], sample_x.shape[0], replace=False)
        x_train = sample_x[idx[:n_train]]
        z_train = z0_samples[idx[:n_train]]
        y_train = y_samples[idx[:n_train]]

        X = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
        Y = (y_train - np.mean(y_train, axis=0)) / np.std(y_train, axis=0)
        Z = (z_train - np.mean(z_train, axis=0)) / np.std(z_train, axis=0)

        for index in (1, 2, 3):
            discover_product(
                index,
                X,
                Y,
                Z,
                prune_params,
                n_vars,
                y_col_slice=index - 1,
                fig_path=f"./figs/ratios_{index}_{run_idx}.png",
                output_path=f"./outputs/ratios_{index}_{run_idx}.txt",
            )

    print("End main.")
