from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import scipy.io as sio
from dataclasses import dataclass


@dataclass
class L96TwoScaleConfig:
    """
    Two-scale Lorenz-96 configuration.

    Equations (k = 0..K-1, j = 0..J-1; all indices periodic):
      dX_k/dt = X_{k-1}(X_{k+1} - X_{k-2}) - X_k + F - (h*c/b) * sum_j Y_{j,k}
      dY_{j,k}/dt = c*b*Y_{j+1,k}(Y_{j-1,k} - Y_{j+2,k}) - c*Y_{j,k} + (h*c/b) * X_k
    """

    K: int = 36
    J: int = 10
    F: float = 10.0
    h: float = 1.0
    c: float = 10.0
    b: float = 10.0
    dt: float = 0.005
    steps: int = 20000
    store_every: int = 10
    spinup: int = 2000


def _rhs(
    X: np.ndarray, Y: np.ndarray, cfg: L96TwoScaleConfig
) -> Tuple[np.ndarray, np.ndarray]:
    Xm1 = np.roll(X, 1)
    Xp1 = np.roll(X, -1)
    Xm2 = np.roll(X, 2)
    dX = Xm1 * (Xp1 - Xm2) - X + cfg.F

    coupling_scale = (cfg.h * cfg.c) / cfg.b
    dX -= coupling_scale * Y.sum(axis=0)

    Yjp1 = np.roll(Y, -1, axis=0)
    Yjm1 = np.roll(Y, 1, axis=0)
    Yjp2 = np.roll(Y, -2, axis=0)

    dY = cfg.c * cfg.b * (Yjp1 * (Yjm1 - Yjp2)) - cfg.c * Y
    dY += coupling_scale * X[np.newaxis, :]

    return dX, dY


def _rk4_step(
    X: np.ndarray, Y: np.ndarray, cfg: L96TwoScaleConfig
) -> Tuple[np.ndarray, np.ndarray]:
    dt = cfg.dt

    k1x, k1y = _rhs(X, Y, cfg)
    k2x, k2y = _rhs(X + 0.5 * dt * k1x, Y + 0.5 * dt * k1y, cfg)
    k3x, k3y = _rhs(X + 0.5 * dt * k2x, Y + 0.5 * dt * k2y, cfg)
    k4x, k4y = _rhs(X + dt * k3x, Y + dt * k3y, cfg)

    Xn = X + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
    Yn = Y + (dt / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)
    return Xn, Yn


def simulate(
    cfg: L96TwoScaleConfig,
    X0: Optional[np.ndarray] = None,
    Y0: Optional[np.ndarray] = None,
    seed: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    if X0 is None:
        X = cfg.F + 0.01 * rng.standard_normal(cfg.K)
    else:
        X = np.array(X0, dtype=float, copy=True)
        assert X.shape == (cfg.K,)

    if Y0 is None:
        Y = 0.01 * rng.standard_normal((cfg.J, cfg.K))
    else:
        Y = np.array(Y0, dtype=float, copy=True)
        assert Y.shape == (cfg.J, cfg.K)

    for _ in range(cfg.spinup):
        X, Y = _rk4_step(X, Y, cfg)

    n_kept_steps = max(cfg.steps, 0)
    stride = max(cfg.store_every, 1)
    T = (n_kept_steps // stride) + 1

    Xs = np.empty((T, cfg.K), dtype=float)
    Ys = np.empty((T, cfg.J, cfg.K), dtype=float)
    ts = np.empty(T, dtype=float)

    Xs[0] = X
    Ys[0] = Y
    ts[0] = 0.0

    write_idx = 1
    time_accum = 0.0
    for n in range(1, n_kept_steps + 1):
        X, Y = _rk4_step(X, Y, cfg)
        time_accum += cfg.dt
        if (n % stride) == 0:
            Xs[write_idx] = X
            Ys[write_idx] = Y
            ts[write_idx] = time_accum
            write_idx += 1

    return ts, Xs, Ys


def default_config(m: int) -> L96TwoScaleConfig:
    return L96TwoScaleConfig(
        K=m,
        J=10,
        F=10.0,
        h=0.1,
        c=10.0,
        b=10.0,
        dt=0.001,
        steps=100_000,
        store_every=10,
        spinup=2000,
    )


def generate_dataset(m: int, *, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Simulate two-scale L96 and return flattened (Xs, dXs) arrays."""
    cfg = default_config(m)
    _, Xs, _ = simulate(cfg, seed=seed)
    dXs = Xs[1:] - Xs[:-1]
    Xs = Xs[:-1]
    return Xs.reshape(-1, m), dXs.reshape(-1, m)


def save_dataset(m: int, output_dir: Path, *, seed: int = 42) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"data_{m}.mat"
    Xs, dXs = generate_dataset(m, seed=seed)
    sio.savemat(path, {"Xs": Xs, "dXs": dXs})
    return path


def ensure_data(m: int, data_dir: Path | str = "./data", *, seed: int = 42) -> Path:
    data_dir = Path(data_dir)
    path = data_dir / f"data_{m}.mat"
    if not path.exists():
        print(f"{path} not found — generating (m={m}, this may take a while)...")
        save_dataset(m, data_dir, seed=seed)
        print(f"Saved {path}")
    return path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate multiscale L96 datasets.")
    parser.add_argument(
        "m",
        nargs="+",
        type=int,
        help="Slow-variable counts, e.g. 10 20 30",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    args = parser.parse_args()
    for m in args.m:
        path = save_dataset(m, args.output_dir)
        print(f"Wrote {path}")
