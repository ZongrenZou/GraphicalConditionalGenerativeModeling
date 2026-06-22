from pathlib import Path

import gcm.core as models
from gcm.core.ckpt_io import save_model

import jax.random as jr
import numpy as np
import scipy.io as sio


if __name__ == "__main__":
    M = 95_000

    data = sio.loadmat("./data/pendulum_data.mat")
    theta = data["theta"].reshape([500, 200])
    theta_next = data["theta_next"].reshape([500, 200])
    u = data["u"].reshape([500, 200])
    y_data = theta_next - theta

    x_data = np.stack(
        [
            np.sin(theta[:, 9:]),
            np.sin(theta[:, 8:-1]),
            np.sin(theta[:, 7:-2]),
            np.sin(theta[:, 6:-3]),
            np.sin(theta[:, 5:-4]),
            np.sin(theta[:, 4:-5]),
            np.sin(theta[:, 3:-6]),
            np.sin(theta[:, 2:-7]),
            np.sin(theta[:, 1:-8]),
            np.sin(theta[:, :-9]),
            np.cos(theta[:, 9:]),
            np.cos(theta[:, 8:-1]),
            np.cos(theta[:, 7:-2]),
            np.cos(theta[:, 6:-3]),
            np.cos(theta[:, 5:-4]),
            np.cos(theta[:, 4:-5]),
            np.cos(theta[:, 3:-6]),
            np.cos(theta[:, 2:-7]),
            np.cos(theta[:, 1:-8]),
            np.cos(theta[:, :-9]),
            u[:, 9:],
            u[:, 8:-1],
            u[:, 7:-2],
            u[:, 6:-3],
            u[:, 5:-4],
            u[:, 4:-5],
            u[:, 3:-6],
            u[:, 2:-7],
            u[:, 1:-8],
            u[:, :-9],
        ],
        axis=-1,
    )
    y_data = y_data[:, 9:]

    x_data = x_data.reshape([-1, 30])[:M]
    y_data = y_data.reshape([-1, 1])[:M]

    seed = 81763263  # for reproducibility
    np.random.seed(seed)

    ## normalize the data
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
            steps=1_000,
        ),
    )

    save_model(
        ckpt_dir="./checkpoints/cfm_all",
        params=params,
        model=model,
        model_cfg=models.ModelCfg(hidden_dims=[64, 64, 64, 64]),
        extras={
            "x_mu": x_mu,
            "x_sd": x_sd,
            "y_mu": y_mu,
            "y_sd": y_sd,
        },
    )

    ############ Sampling ############
    x_data = x_data.reshape([-1, 30])
    M = x_data.shape[0]
    N = 10
    x = np.tile(x_data, [N, 1]).reshape([N * M, x_data.shape[1]])

    z_samples, z0_samples = models.sample(
        params,
        x=(x - x_mu) / x_sd,
        model=model,
    )
    y_samples = z_samples[:, 0, :] * y_sd + y_mu

    samples_path = Path("./outputs/samples.mat")
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(
        samples_path,
        {
            "y_samples": y_samples,
            "z_samples": z0_samples,
            "x_samples": x,
            "x_mu": x_mu,
            "x_sd": x_sd,
            "y_mu": y_mu,
            "y_sd": y_sd,
        },
    )

    print("End main.")
