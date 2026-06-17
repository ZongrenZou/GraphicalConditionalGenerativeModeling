import jax.random as jr
import numpy as np
import scipy.io as sio

import gcm.core as models

from common import build_oil_features, prune_oil_gpr


if __name__ == "__main__":
    data = sio.loadmat("./data/data_oil_gpr.mat")["data"]
    x_data, y_data = build_oil_features(data[:, 0:1], data[:, 1:2])

    x_mu = np.mean(x_data, axis=0)
    x_sd = np.std(x_data, axis=0)
    y_mu = np.mean(y_data, axis=0)
    y_sd = np.std(y_data, axis=0)
    X_train = (x_data - x_mu) / x_sd
    Y_train = (y_data - y_mu) / y_sd

    key = np.random.randint(0, 1_000_000_000)
    key = jr.PRNGKey(key)

    params, model = models.train_cfm_flax(
        key,
        X_train,
        Y_train,
        model_cfg=models.ModelCfg(hidden_dims=[64, 64]),
        train_cfg=models.TrainCfg(lr=1e-4, batch_size=1_000, steps=2_000),
    )

    n_repeats = 100
    x_samples = np.tile(x_data, [n_repeats, 1])
    z_samples, z0_samples = models.sample(
        params,
        x=(x_samples - x_mu) / x_sd,
        model=model,
        reshape_output=False,
    )
    y_samples = z_samples[:, 0, :] * y_sd + y_mu

    prune_oil_gpr(
        x_samples,
        z0_samples,
        y_samples,
        fig_path="./figs/ratios_phase_1986_2026.png",
    )
    print("End main.")
