import jax.numpy as jnp
import jax.random as jr
import numpy as np
import scipy.io as sio

import gcm.core as models
from gcm.core.ckpt_io import save_model
from gcm.discovery import discover_product


LUNAR_VAR_NAMES = [
    "x",
    "y",
    "vx",
    "vy",
    "angle",
    "va",
    "leg 1",
    "leg 2",
    "a_1",
    "a_2",
]


if __name__ == "__main__":
    data = sio.loadmat("./data/lunar_lander_impulse_data.mat")
    actions = data["actions"]
    states = data["states"]
    delta_vx = data["delta_vx"].reshape([-1, 1])
    delta_vy = data["delta_vy"].reshape([-1, 1])
    delta_va = data["delta_angvel"].reshape([-1, 1])

    x_data = np.concatenate(
        [states, actions],
        axis=-1,
    )
    ## choose the output
    # y_data = delta_vx
    # y_data = delta_vy
    y_data = delta_va

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
        # ckpt_dir="./checkpoints/cfm_vx",
        # ckpt_dir="./checkpoints/cfm_vy",
        ckpt_dir="./checkpoints/cfm_va",
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
    z_samples, z0_samples = models.sample(
        params,
        x=(x_data - x_mu) / x_sd,
        model=model,
    )
    y_samples = z_samples[:, 0, :] * y_sd + y_mu
    x_samples = x_data
    z_samples = z0_samples

    ############ Pruning ############
    prune_params = {
        "lx": 10 * [2.0],
        "lz": 1.0,
        "log_gamma": jnp.log(1.0),
    }
    n_vars = 10
    n_train = 5000

    idx = np.random.choice(x_data.shape[0], x_data.shape[0], replace=False)
    x_train = x_samples[idx[:n_train]]
    z_train = z_samples[idx[:n_train]]
    y_train = y_samples[idx[:n_train]]

    X = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
    Y = (y_train - np.mean(y_train, axis=0)) / np.std(y_train, axis=0)
    Z = z_train

    discover_product(
        0,
        X,
        Y,
        Z,
        prune_params,
        n_vars,
        target_name=r"$\Delta v_a$",
        names=LUNAR_VAR_NAMES,
        fig_path="./figs/ratios_v_a.png",
        tight_layout=True,
    )

    print("End main.")
