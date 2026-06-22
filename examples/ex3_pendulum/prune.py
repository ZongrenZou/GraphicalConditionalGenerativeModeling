import argparse
from pathlib import Path

import gcm.core as models
import gcm.kernels as utils
from gcm.core.ckpt_io import load_model

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

SAMPLES_PATH = Path("./outputs/samples.mat")
CKPT_DIR = Path("./checkpoints/cfm_all")
DATA_PATH = Path("./data/pendulum_data.mat")


def _build_pendulum_inputs(M: int = 95_000):
    data = sio.loadmat(DATA_PATH)
    theta = data["theta"].reshape([500, 200])
    theta_next = data["theta_next"].reshape([500, 200])
    u = data["u"].reshape([500, 200])

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
    y_data = (theta_next - theta)[:, 9:]
    x_data = x_data.reshape([-1, 30])[:M]
    y_data = y_data.reshape([-1, 1])[:M]
    return x_data, y_data


def _ensure_samples_mat():
    if SAMPLES_PATH.exists():
        return sio.loadmat(SAMPLES_PATH)

    if not CKPT_DIR.joinpath("params.msgpack").exists():
        raise FileNotFoundError(
            f"{SAMPLES_PATH} not found and no checkpoint at {CKPT_DIR}. "
            "Run flow_distill.py first."
        )

    print(f"{SAMPLES_PATH} not found — generating from {CKPT_DIR} ...")
    params, model, extras = load_model(CKPT_DIR, models.VelocityMLP)
    x_mu, x_sd = extras["x_mu"], extras["x_sd"]
    y_mu, y_sd = extras["y_mu"], extras["y_sd"]

    x_data, _ = _build_pendulum_inputs()
    N = 10
    x = np.tile(x_data, [N, 1]).reshape([N * x_data.shape[0], x_data.shape[1]])

    z_samples, z0_samples = models.sample(
        params,
        x=(x - x_mu) / x_sd,
        model=model,
    )
    y_samples = z_samples[:, 0, :] * y_sd + y_mu

    SAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(
        SAMPLES_PATH,
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
    print(f"Saved {SAMPLES_PATH}")
    return sio.loadmat(SAMPLES_PATH)


def discover(
    index,
    *,
    N_groups,
    N_vars,
    groups,
    group_names,
    params0,
    is_periodic,
    _X,
    _Y,
    _Z,
):
    X = _X
    Y = _Y
    Z = _Z

    active_modes = N_vars * [1.0]
    target_name = r"$\Delta\theta$"

    pruned_names = []
    ratios = []

    for i in range(N_groups + 1):
        params = params0.copy()
        lx = params["lx"]
        lz = params["lz"]
        gamma = jnp.exp(params["log_gamma"])

        K = utils.kernel_fn(X, Z, X, Z, lx, lz, active_modes, is_periodic)
        L = jnp.linalg.cholesky(K + gamma * jnp.eye(K.shape[0]))
        yb = jnp.linalg.solve(L.T, jnp.linalg.solve(L, Y))

        signal2 = yb.T @ K @ yb
        noise2 = yb.T @ (gamma * np.eye(K.shape[0])) @ yb

        group_activations = N_groups * [1e12]
        for g in range(N_groups):
            var_indices = groups[g]
            if all(active_modes[j] == 0 for j in var_indices):
                continue

            _active_modes = active_modes.copy()
            for j in var_indices:
                _active_modes[j] = 0

            _K = K - utils.kernel_fn(X, Z, X, Z, lx, lz, _active_modes, is_periodic)
            rkhs_norm2 = yb.T @ _K @ yb
            print(f"  Group {group_names[g]} activation: {rkhs_norm2.reshape([]):.6f}")
            group_activations[g] = rkhs_norm2.reshape([])

        if i < N_groups:
            prune_g = np.argmin(group_activations)
            print(f"Pruning group: {group_names[prune_g]}")
            for j in groups[prune_g]:
                active_modes[j] = 0
            pruned_names += [group_names[prune_g]]

        ratios += [noise2 / (noise2 + signal2)]

    ratios = [v.reshape([]) for v in ratios]

    fig, ax = plt.subplots(1, 1, figsize=[12, 5])
    plt.suptitle("Finding ancestors of " + target_name)
    ax.plot(pruned_names, np.array(ratios[1:]) - np.array(ratios[:-1]), "k-o")
    ax.set_title("Increment of noise-to-signal ratio per pruned group")
    ax.set_xlabel("Pruned group")
    plt.tight_layout()
    fig.savefig(f"./figs/ratios_{index}.png", dpi=200)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=1)
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--M", type=int, default=5_000)
    args = parser.parse_args()

    N_vars = 20
    N_groups = args.N
    M = args.M

    groups = {k: [k, k + 10] for k in range(N_groups)}
    group_names = [
        r"$(\theta_t, u_t)$",
        r"$(\theta_{t-1}, u_{t-1})$",
        r"$(\theta_{t-2}, u_{t-2})$",
        r"$(\theta_{t-3}, u_{t-3})$",
        r"$(\theta_{t-4}, u_{t-4})$",
        r"$(\theta_{t-5}, u_{t-5})$",
        r"$(\theta_{t-6}, u_{t-6})$",
        r"$(\theta_{t-7}, u_{t-7})$",
        r"$(\theta_{t-8}, u_{t-8})$",
        r"$(\theta_{t-9}, u_{t-9})$",
    ]

    params0 = {
        "lx": N_vars * [1.0],
        "lz": 10.0,
        "log_gamma": jnp.log(1.0),
    }
    print("Number of groups: ", N_groups)
    print("Number of data: ", M)

    data = _ensure_samples_mat()
    z_data = data["z_samples"]
    x_data = data["x_samples"]
    y_data = data["y_samples"]

    x_data = x_data.reshape([-1, x_data.shape[-1]])
    z_data = z_data.reshape([-1, z_data.shape[-1]])
    y_data = y_data.reshape([-1, y_data.shape[-1]])

    sin_cols = x_data[:, 0:10]
    cos_cols = x_data[:, 10:20]
    u_cols = x_data[:, 20:30]
    theta_cols = np.arctan2(sin_cols, cos_cols)
    x_data_theta = np.concatenate([theta_cols, u_cols], axis=1)
    is_periodic = [True] * 10 + [False] * 10

    idx = np.random.choice(x_data_theta.shape[0], x_data_theta.shape[0], replace=False)
    x_train = x_data_theta[idx[:M]]
    z_train = z_data[idx[:M]]
    y_train = y_data[idx[:M]]

    x_mu = np.mean(x_train, axis=0)
    x_sd = np.std(x_train, axis=0)
    y_mu = np.mean(y_train, axis=0)
    y_sd = np.std(y_train, axis=0)

    _X = (x_train - x_mu) / x_sd
    _Y = (y_train - y_mu) / y_sd
    _Z = z_train

    discover(
        args.index,
        N_groups=N_groups,
        N_vars=N_vars,
        groups=groups,
        group_names=group_names,
        params0=params0,
        is_periodic=is_periodic,
        _X=_X,
        _Y=_Y,
        _Z=_Z,
    )
    print("End main.")


if __name__ == "__main__":
    main()
