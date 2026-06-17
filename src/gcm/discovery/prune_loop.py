import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt


def discover_product(
    index,
    X,
    Y,
    Z,
    params,
    N,
    run_idx,
    *,
    kernel_fn,
    target_name: str | None = None,
    names: list[str] | None = None,
    fig_path: str | None = None,
    output_path: str | None = None,
    y_col_slice: slice | int | None = None,
):
    if y_col_slice is not None:
        if isinstance(y_col_slice, int):
            Y = Y[:, y_col_slice : y_col_slice + 1]
        else:
            Y = Y[:, y_col_slice]

    active_modes = N * [1.0]
    if names is None:
        names = ["x{}".format(str(i + 1)) for i in range(N)]
    if target_name is None:
        target_name = "dx{}/dt".format(str(index))

    pruned_names = []
    ratios = []

    for i in range(N + 1):
        lx = params["lx"]
        lz = params["lz"]
        gamma = jnp.exp(params["log_gamma"])

        K = kernel_fn(X, Z, X, Z, lx, lz, active_modes)
        yb = jnp.linalg.solve(K + gamma * np.eye(K.shape[0]), Y)

        signal2 = yb.T @ K @ yb
        noise2 = yb.T @ (gamma * np.eye(K.shape[0])) @ yb

        activations = N * [1e12]
        for j in range(N):
            if active_modes[j] == 1:
                _active_modes = active_modes.copy()
                _active_modes[j] = 0
            else:
                continue
            _K = 1 * K - kernel_fn(X, Z, X, Z, lx, lz, _active_modes)
            rkhs_norm2 = yb.T @ _K @ yb
            print(
                "The {}th variable's activation: ".format(str(j)),
                rkhs_norm2.reshape([]),
            )
            activations[j] = rkhs_norm2.reshape([])

        if i < N:
            idx = np.argmin(activations)
            print("The {}th mode has the lowest energy.".format(str(idx)))
            active_modes[idx] = 0
            pruned_names += [names[idx]]

        ratios += [noise2 / (noise2 + signal2)]

    ratios = [v.reshape([]) for v in ratios]

    if fig_path is not None:
        fig, ax = plt.subplots(1, 2, figsize=[12, 5])
        plt.suptitle("Finding ancestors of " + target_name)
        ax[0].plot(ratios, "k-o")
        ax[0].set_title("The noise-to-signal ratio")
        ax[1].plot(pruned_names, np.array(ratios[1:]) - np.array(ratios[:-1]), "k-o")
        ax[1].set_title("The increment of the noise-to-signal ratio")
        ax[1].set_xlabel("Pruned variables")
        fig.savefig(fig_path, dpi=200)
        plt.close()

    if output_path is not None:
        np.savetxt(output_path, np.array(ratios))

    return ratios, pruned_names


def discover_vectorized(
    index,
    X,
    Y,
    Z,
    params,
    m,
    *,
    prune_step,
    target_name: str | None = None,
    names: list[str] | None = None,
    fig_path: str | None = None,
    prune_index: int = -1,
):
    active_modes = jnp.ones(m)
    if names is None:
        names = ["$x_{" + str(i + 1) + "}$" for i in range(m)]
    if target_name is None:
        target_name = "$\Delta x_{" + str(index + 1) + "}$"

    pruned_names = []
    ratios = []
    Xj = jnp.asarray(X)
    Zj = jnp.asarray(Z)
    Yj = jnp.asarray(Y)

    for i in range(m):
        lx = params["lx"]
        lz = params["lz"]
        log_gamma = params["log_gamma"]

        ratio, activations = prune_step(
            Xj, Zj, Yj, lx, lz, log_gamma, active_modes, prune_index
        )

        if i < m - 1:
            idx = int(jnp.argmin(activations))
            print("The {}th mode has the lowest energy.".format(str(idx)))
            active_modes = active_modes.at[idx].set(0.0)
            pruned_names += [names[idx]]

        ratios += [ratio]

    ratios = [v.reshape([]) for v in ratios]

    if fig_path is not None:
        fig, ax = plt.subplots(1, 2, figsize=[10, 5])
        plt.suptitle("Finding ancestors of " + target_name)
        ax[0].plot(ratios, "k-o")
        ax[0].set_title("The noise-to-signal ratio")
        ax[1].plot(pruned_names, np.array(ratios[1:]) - np.array(ratios[:-1]), "k-o")
        ax[1].set_title("The increment of the noise-to-signal ratio")
        ax[1].set_xlabel("Pruned variables")
        plt.tight_layout()
        fig.savefig(fig_path, dpi=200)
        plt.close()

    return ratios, pruned_names
