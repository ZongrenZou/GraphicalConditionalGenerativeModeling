import argparse
from dataclasses import dataclass
from pathlib import Path

import gcm.core as models
import gcm.kernels as utils
from gcm.discovery import loss_function

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


@dataclass
class PruneCfg:
    num_input_modes: int = 21
    sample_pairs: int = 2_000
    lz: float = 10.0
    gamma: float = 1.0


def safe_std(x: np.ndarray) -> np.ndarray:
    sd = np.std(x, axis=0)
    sd[sd == 0] = 1.0
    return sd


def load_training_data(data_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = sio.loadmat(data_path)
    actions = data["actions"]
    states = data["obs"]
    deltas = data["deltas"]

    x_data = np.concatenate([states[:, :7], states[:, 10:], actions], axis=-1)
    y_data = np.concatenate([deltas[:, :7], deltas[:, 10:]], axis=-1)
    return x_data, y_data


def discover_ancestors(
    x_data: np.ndarray,
    z_data: np.ndarray,
    y_data: np.ndarray,
    output_index: int,
    prune_cfg: PruneCfg,
    output_dir: Path,
):
    sample_count = min(prune_cfg.sample_pairs, x_data.shape[0] // 2)
    if sample_count < 2:
        raise ValueError(
            "Need at least four sampled rows to build train/validation pruning splits."
        )

    idx = np.random.permutation(x_data.shape[0])

    x_train = x_data[idx[:sample_count]]
    z_train = z_data[idx[:sample_count]]
    y_train = y_data[idx[:sample_count], output_index : output_index + 1]
    x_val = x_data[idx[sample_count : 2 * sample_count]]
    z_val = z_data[idx[sample_count : 2 * sample_count]]
    y_val = y_data[
        idx[sample_count : 2 * sample_count], output_index : output_index + 1
    ]

    x_mu = np.mean(x_train, axis=0)
    x_sd = safe_std(x_train)
    y_mu = np.mean(y_train, axis=0)
    y_sd = safe_std(y_train)

    x_train_norm = (x_train - x_mu) / x_sd
    y_train_norm = (y_train - y_mu) / y_sd
    x_val_norm = (x_val - x_mu) / x_sd
    y_val_norm = (y_val - y_mu) / y_sd

    params = {
        "lx": prune_cfg.num_input_modes * [10.0],
        "lz": prune_cfg.lz,
        "log_gamma": jnp.log(prune_cfg.gamma),
    }
    active_modes = prune_cfg.num_input_modes * [1.0]
    variable_names = [
        "x1",
        "x2",
        "x3",
        "x4",
        "x5",
        "x6",
        "x7",
        "x8",
        "x9",
        "x10",
        "x11",
        "x12",
        "x13",
        "x14",
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a6",
        "a7",
    ]

    pruned_names = []
    ratios = []
    losses_train = []
    losses_val = []

    for step_idx in range(prune_cfg.num_input_modes + 1):
        data = (x_train_norm, z_train, y_train_norm)
        loss = loss_function(params, data, active_modes)

        lx = params["lx"]
        lz = params["lz"]
        gamma = jnp.exp(params["log_gamma"])

        kernel_train = utils.kernel_fn(
            x_train_norm,
            z_train,
            x_train_norm,
            z_train,
            lx,
            lz,
            active_modes,
        )
        chol = jnp.linalg.cholesky(
            kernel_train + gamma * jnp.eye(kernel_train.shape[0])
        )
        yb = jnp.linalg.solve(chol.T, jnp.linalg.solve(chol, y_train_norm))

        kernel_val = utils.kernel_fn(
            x_val_norm,
            z_val,
            x_train_norm,
            z_train,
            lx,
            lz,
            active_modes,
        )
        y_pred = kernel_val @ yb
        train_loss = jnp.mean((y_train_norm - kernel_train @ yb) ** 2)
        val_loss = jnp.mean((y_val_norm - y_pred) ** 2)
        losses_train.append(float(train_loss))
        losses_val.append(float(val_loss))

        signal2 = yb.T @ kernel_train @ yb
        noise2 = yb.T @ (gamma * np.eye(kernel_train.shape[0])) @ yb
        ratios.append(float((noise2 / (noise2 + signal2)).reshape(())))

        print(
            f"output {output_index + 1:02d}, prune step {step_idx:02d}: "
            f"cv_loss={float(loss):.6f}, train={float(train_loss):.6f}, val={float(val_loss):.6f}",
            flush=True,
        )

        if step_idx == prune_cfg.num_input_modes:
            continue

        activations = prune_cfg.num_input_modes * [1e12]
        for feature_idx in range(prune_cfg.num_input_modes):
            if active_modes[feature_idx] != 1:
                continue
            reduced_modes = active_modes.copy()
            reduced_modes[feature_idx] = 0
            kernel_delta = kernel_train - utils.kernel_fn(
                x_train_norm,
                z_train,
                x_train_norm,
                z_train,
                lx,
                lz,
                reduced_modes,
            )
            rkhs_norm2 = yb.T @ kernel_delta @ yb
            activations[feature_idx] = float(rkhs_norm2.reshape(()))

        remove_idx = int(np.argmin(activations))
        active_modes[remove_idx] = 0
        pruned_names.append(variable_names[remove_idx])
        print(
            f"output {output_index + 1:02d}: pruned {variable_names[remove_idx]} at step {step_idx:02d}",
            flush=True,
        )

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Finding ancestors of output_{output_index + 1}")
    ax[0].plot(ratios, "k-o")
    ax[0].set_title("Noise-to-signal ratio")
    ax[0].set_xlabel("Pruning step")
    ax[1].plot(pruned_names, np.array(ratios[1:]) - np.array(ratios[:-1]), "k-o")
    ax[1].set_title("Increment of ratio")
    ax[1].set_xlabel("Pruned variables")
    plt.tight_layout()
    fig_path = output_dir / "figs" / f"ratios_{output_index + 1:02d}.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    txt_path = output_dir / "outputs" / f"ratios_{output_index + 1:02d}.txt"
    np.savetxt(txt_path, np.asarray(ratios))

    return {
        "output_index": output_index + 1,
        "pruned_names": pruned_names,
        "ratios": ratios,
        "losses_train": losses_train,
        "losses_val": losses_val,
        "figure_path": str(fig_path),
        "ratios_path": str(txt_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("./data/reacher_init.mat"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("."))
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=1_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=81763263)
    parser.add_argument("--prune-samples", type=int, default=2_000)
    args = parser.parse_args()

    output_root = args.output_root
    (output_root / "outputs").mkdir(parents=True, exist_ok=True)
    (output_root / "figs").mkdir(parents=True, exist_ok=True)

    x_data, y_data = load_training_data(args.data_path)
    x_mu = np.mean(x_data, axis=0)
    x_sd = safe_std(x_data)
    y_mu = np.mean(y_data, axis=0)
    y_sd = safe_std(y_data)
    x_train = (x_data - x_mu) / x_sd
    y_train = (y_data - y_mu) / y_sd

    key = jr.PRNGKey(args.seed)
    params, model = models.train_cfm_flax(
        key,
        x_train,
        y_train,
        model_cfg=models.ModelCfg(hidden_dims=(128, 128, 128, 128)),
        train_cfg=models.TrainCfg(
            lr=args.lr,
            batch_size=args.batch_size,
            steps=args.steps,
        ),
    )

    n_repeats = 20
    x_data = np.tile(x_data, [n_repeats, 1])
    x_norm = np.tile(x_train, [n_repeats, 1])

    z_samples_norm, z0_samples = models.sample(
        params,
        x_norm,
        model,
        seed=args.seed + 1,
        squeeze_time=True,
    )
    y_samples = np.asarray(z_samples_norm) * y_sd + y_mu
    z0_samples = np.asarray(z0_samples)

    prune_cfg = PruneCfg(sample_pairs=args.prune_samples)
    prune_summaries = []
    for output_index in range(y_samples.shape[1]):
        prune_summaries.append(
            discover_ancestors(
                x_data=x_data,
                z_data=z0_samples,
                y_data=y_samples,
                output_index=output_index,
                prune_cfg=prune_cfg,
                output_dir=output_root,
            )
        )

    summary_path = output_root / "outputs" / "prune_summary.npz"
    np.savez(
        summary_path,
        pruned_names=np.array(
            [",".join(item["pruned_names"]) for item in prune_summaries], dtype=object
        ),
        ratios=np.array(
            [np.asarray(item["ratios"]) for item in prune_summaries], dtype=object
        ),
        losses_train=np.array(
            [np.asarray(item["losses_train"]) for item in prune_summaries], dtype=object
        ),
        losses_val=np.array(
            [np.asarray(item["losses_val"]) for item in prune_summaries], dtype=object
        ),
    )
    summary_txt_path = output_root / "outputs" / "prune_summary.txt"
    with summary_txt_path.open("w", encoding="utf-8") as fh:
        for item in prune_summaries:
            fh.write(f"output_{item['output_index']:02d}\n")
            fh.write(f"pruned_order: {', '.join(item['pruned_names'])}\n")
            fh.write(f"ratios: {item['ratios']}\n")
            fh.write("\n")

    print(f"Saved pruning summary to {summary_path}")
    print(f"Saved pruning text summary to {summary_txt_path}")
    print("End main.")


if __name__ == "__main__":
    main()
