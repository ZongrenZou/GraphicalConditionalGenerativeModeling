import numpy as np

from gcm.discovery import discover_product

OIL_GPR_PRUNE_NAMES = [
    "GPR at\n one day",
    "GPR at\n one week",
    "GPR at\n two weeks",
    "GPR at\n three weeks",
    "GPR at\n four weeks",
]
OIL_TARGET_NAME = (
    r"return of the oil price at step n: $\log(P_{n+1}) - \log(P_n)$"
)


def build_oil_features(oil: np.ndarray, gpr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r_oil = np.log(oil[1:]) - np.log(oil[:-1])
    log_gpr = np.log(gpr[:-1])

    y_data = r_oil[20:, :]
    x_data = np.concatenate(
        [
            r_oil[19:-1, :],
            r_oil[15:-5, :],
            r_oil[10:-10, :],
            r_oil[5:-15, :],
            r_oil[0:-20, :],
            log_gpr[20:, :],
            log_gpr[16:-4, :],
            log_gpr[11:-9, :],
            log_gpr[6:-14, :],
            log_gpr[1:-19, :],
        ],
        axis=-1,
    )
    return x_data, y_data


def prune_oil_gpr(
    x_samples: np.ndarray,
    z_samples: np.ndarray,
    y_samples: np.ndarray,
    *,
    fig_path: str,
) -> None:
    n_prunable = 5
    n_train = 10_000
    prune_params = {
        "lx": 5.0,
        "lz": 5.0,
        "log_gamma": 1.0,
    }

    not_pruned_x = x_samples[:, :n_prunable]
    x_samples = np.concatenate([x_samples[:, n_prunable:], not_pruned_x], axis=1)

    idx = np.random.choice(x_samples.shape[0], x_samples.shape[0], replace=False)
    x_train = x_samples[idx[:n_train]]
    z_train = z_samples[idx[:n_train]]
    y_train = y_samples[idx[:n_train]]
    x_val = x_samples[idx[n_train : 2 * n_train]]
    z_val = z_samples[idx[n_train : 2 * n_train]]
    y_val = y_samples[idx[n_train : 2 * n_train]]

    X = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
    Y = (y_train - np.mean(y_train, axis=0)) / np.std(y_train, axis=0)
    Z = z_train
    X_val = (x_val - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
    Y_val = (y_val - np.mean(y_train, axis=0)) / np.std(y_train, axis=0)
    Z_val = z_val

    discover_product(
        1,
        X,
        Y,
        Z,
        prune_params,
        n_prunable,
        n_kernel_modes=10,
        target_name=OIL_TARGET_NAME,
        names=OIL_GPR_PRUNE_NAMES,
        fig_path=fig_path,
        X_val=X_val,
        Y_val=Y_val,
        Z_val=Z_val,
        normalize_activation=True,
        tight_layout=True,
    )
