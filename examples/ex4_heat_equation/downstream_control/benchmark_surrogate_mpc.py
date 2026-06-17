import json
import time
from dataclasses import dataclass
from pathlib import Path
from flax import linen as nn
from flax import serialization
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import jaxopt
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat
from scipy.linalg import solve_discrete_are

CONTROL_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ExperimentCfg:
    n: int = 10
    kappa: float = 0.01
    dt: float = 0.2
    sigma: float = 0.2
    T: int = 100
    H: int = 20
    S: int = 64
    m: int = 2
    u_bound: float = 0.1
    num_trials: int = 50
    seed: int = 1322
    plot_path: str = "./figs/benchmark_surrogate_mpc.png"
    all_traj_plot_path: str = "./figs/benchmark_surrogate_mpc_all_traj.png"
    mpc_mean_cum_total_cost_path: str = "./figs/benchmark_mpc_mean_cum_total_cost.png"
    results_mat_path: str = "./figs/benchmark_surrogate_mpc_results.mat"


CFG = ExperimentCfg()

UNPRUNED_X0_STUDENT = "./checkpoints/cfm_1_student"
UNPRUNED_XN_STUDENT = "./checkpoints/cfm_10_student"
PRUNED_X0_STUDENT = "./checkpoints/cfm_1_pruned_student"
PRUNED_XN_STUDENT = "./checkpoints/cfm_10_pruned_student"


class StudentMLP(nn.Module):
    hidden_dims: tuple[int, ...]
    dim_out: int

    @nn.compact
    def __call__(self, x, z):
        if x.ndim == 1:
            x = x[None, :]
        if z.ndim == 1:
            z = z[None, :]
        h = jnp.concatenate([x, z], axis=-1)
        for width in self.hidden_dims:
            h = nn.silu(nn.Dense(width)(h))
        out = nn.Dense(self.dim_out)(h)
        return out if out.shape[0] > 1 else out[0]


def make_student_predictor(ckpt_dir: str):
    ckpt_path = (CONTROL_DIR / ckpt_dir).resolve()
    student_cfg = json.loads((ckpt_path / "student_cfg.json").read_text())
    params = serialization.msgpack_restore((ckpt_path / "params.msgpack").read_bytes())
    extras = dict(np.load(ckpt_path / "extras.npz", allow_pickle=False))

    model = StudentMLP(
        hidden_dims=tuple(student_cfg["hidden_dims"]),
        dim_out=int(student_cfg["out_dim"]),
    )
    y_mu = jnp.asarray(extras["y_mu"])
    y_sd = jnp.asarray(extras["y_sd"])

    def predict_single(x_cond, z0):
        pred = model.apply({"params": params}, x_cond, z0)
        return pred.reshape(y_mu.shape) * y_sd + y_mu

    return jax.jit(predict_single), jax.jit(jax.vmap(predict_single, in_axes=(0, 0)))


predict_unpruned_x0_single, predict_unpruned_x0_batch = make_student_predictor(
    UNPRUNED_X0_STUDENT
)
predict_unpruned_xn_single, predict_unpruned_xn_batch = make_student_predictor(
    UNPRUNED_XN_STUDENT
)
predict_pruned_x0_single, predict_pruned_x0_batch = make_student_predictor(
    PRUNED_X0_STUDENT
)
predict_pruned_xn_single, predict_pruned_xn_batch = make_student_predictor(
    PRUNED_XN_STUDENT
)


def build_system_matrices(n=10, kappa=0.01, dt=0.5):
    h = 1.0 / (n - 1)
    r = kappa * dt / h**2
    assert r <= 0.5, f"Stability violated: r={r:.4f} > 0.5"

    a_tilde = np.zeros((n, n))
    for i in range(1, n - 1):
        a_tilde[i, i - 1] = 1.0
        a_tilde[i, i] = -2.0
        a_tilde[i, i + 1] = 1.0
    a_tilde[0, 0] = -2.0
    a_tilde[0, 1] = 2.0
    a_tilde[n - 1, n - 2] = 2.0
    a_tilde[n - 1, n - 1] = -2.0

    a = np.eye(n) + r * a_tilde
    b_tilde = np.zeros((n, 2))
    b_tilde[0, 0] = 1.0
    b_tilde[n - 1, 1] = 1.0
    b = (2.0 * dt / h) * b_tilde
    return jnp.array(a), jnp.array(b)


def solve_modified_dare(a, b, q, r, sigma, max_iter=10000, tol=1e-12):
    a_np = np.array(a)
    b_np = np.array(b)
    q_np = np.array(q)
    r_np = np.array(r)

    p = solve_discrete_are(a_np, b_np, q_np, r_np)
    for _ in range(max_iter):
        apba = a_np.T @ p @ b_np
        schur = r_np + b_np.T @ p @ b_np
        p_new = (
            a_np.T @ p @ a_np
            + sigma**2 * np.diag(np.diag(p))
            - apba @ np.linalg.solve(schur, apba.T)
            + q_np
        )
        p_new = (p_new + p_new.T) / 2.0
        err = np.max(np.abs(p_new - p))
        p = p_new
        if err < tol:
            break

    schur = r_np + b_np.T @ p @ b_np
    k = np.linalg.solve(schur, b_np.T @ p @ a_np)
    return jnp.array(k), jnp.array(p)


@jax.jit
def true_step(a, b, sigma, x, u, key):
    w = jr.normal(key, shape=x.shape)
    return a @ x + b @ u + sigma * x * w


def simulate_lqr(k, a, b, sigma, x0, T, key_dynamics):
    x_traj = [x0]
    u_traj = []
    x = x0

    for _ in range(T):
        u = -k @ x
        key_dynamics, subkey = jr.split(key_dynamics)
        x = true_step(a, b, sigma, x, u, subkey)
        x_traj.append(x)
        u_traj.append(u)

    return jnp.stack(x_traj), jnp.stack(u_traj)


def make_known_mpc(a, b, H, S, q, sigma, n, u_bound, m=2, maxiter=300):
    r = 0.1 * jnp.eye(m)

    def rollout_cost(u_flat, x0, w_scenarios):
        u_seq = u_flat.reshape(H, m)

        def step_batch(x_batch, inputs):
            u_k, w_k = inputs
            u_k = jnp.clip(u_k, -u_bound, u_bound)
            cost_k = jnp.sum((x_batch @ q) * x_batch, axis=1) + u_k @ r @ u_k
            x_next = x_batch @ a.T + u_k @ b.T + sigma * x_batch * w_k
            return x_next, cost_k

        x0_batch = jnp.broadcast_to(x0, (S, n))
        x_final, costs = jax.lax.scan(
            step_batch, x0_batch, (u_seq, w_scenarios.swapaxes(0, 1))
        )
        terminal = jnp.sum((x_final @ q) * x_final, axis=1)
        return jnp.mean(jnp.sum(costs, axis=0) + terminal)

    @jax.jit
    def solve(x0, key, u_init):
        key, subkey = jr.split(key)
        w_scenarios = jr.normal(subkey, shape=(S, H, n))

        def obj(u):
            return jax.value_and_grad(rollout_cost)(u, x0, w_scenarios)

        solver = jaxopt.LBFGS(
            fun=obj,
            value_and_grad=True,
            maxiter=maxiter,
            jit=True,
            unroll=False,
        )
        result = solver.run(u_init.flatten())
        u_opt = jnp.clip(result.params.reshape(H, m), -u_bound, u_bound)
        return u_opt, u_opt[0], key

    return solve


def make_unpruned_surrogate_mpc(a, b, H, S, q, sigma, n, u_bound, m=2, maxiter=300):
    r = 0.1 * jnp.eye(m)

    def rollout_cost(u_flat, x0, w_scenarios):
        u_seq = u_flat.reshape(H, m)

        def step_batch(x_batch, inputs):
            u_k, w_k = inputs
            u_k = jnp.clip(u_k, -u_bound, u_bound)
            u_batch = jnp.broadcast_to(u_k, (S, m))
            cost_k = jnp.sum((x_batch @ q) * x_batch, axis=1) + u_k @ r @ u_k
            x_next = x_batch @ a.T + u_k @ b.T + sigma * x_batch * w_k

            x_cond = jnp.concatenate([x_batch, u_batch], axis=1)
            delta_left = predict_unpruned_x0_batch(x_cond, w_k[:, 0:1]).reshape(S)
            delta_right = predict_unpruned_xn_batch(x_cond, w_k[:, -1:]).reshape(S)
            x_next = x_next.at[:, 0].set(x_batch[:, 0] + delta_left)
            x_next = x_next.at[:, -1].set(x_batch[:, -1] + delta_right)
            return x_next, cost_k

        x0_batch = jnp.broadcast_to(x0, (S, n))
        x_final, costs = jax.lax.scan(
            step_batch, x0_batch, (u_seq, w_scenarios.swapaxes(0, 1))
        )
        terminal = jnp.sum((x_final @ q) * x_final, axis=1)
        return jnp.mean(jnp.sum(costs, axis=0) + terminal)

    @jax.jit
    def solve(x0, key, u_init):
        key, subkey = jr.split(key)
        w_scenarios = jr.normal(subkey, shape=(S, H, n))

        def obj(u):
            return jax.value_and_grad(rollout_cost)(u, x0, w_scenarios)

        solver = jaxopt.LBFGS(
            fun=obj,
            value_and_grad=True,
            maxiter=maxiter,
            jit=True,
            unroll=False,
        )
        result = solver.run(u_init.flatten())
        u_opt = jnp.clip(result.params.reshape(H, m), -u_bound, u_bound)
        return u_opt, u_opt[0], key

    return solve


def make_pruned_surrogate_mpc(a, b, H, S, q, sigma, n, u_bound, m=2, maxiter=300):
    r = 0.1 * jnp.eye(m)

    def rollout_cost(u_flat, x0, w_scenarios):
        u_seq = u_flat.reshape(H, m)

        def step_batch(x_batch, inputs):
            u_k, w_k = inputs
            u_k = jnp.clip(u_k, -u_bound, u_bound)
            u_batch = jnp.broadcast_to(u_k, (S, m))
            cost_k = jnp.sum((x_batch @ q) * x_batch, axis=1) + u_k @ r @ u_k
            x_next = x_batch @ a.T + u_k @ b.T + sigma * x_batch * w_k

            x0_cond = jnp.concatenate([x_batch[:, 0:2], u_batch[:, 0:1]], axis=1)
            xn_cond = jnp.concatenate([x_batch[:, 8:10], u_batch[:, 1:2]], axis=1)
            delta_left = predict_pruned_x0_batch(x0_cond, w_k[:, 0:1]).reshape(S)
            delta_right = predict_pruned_xn_batch(xn_cond, w_k[:, -1:]).reshape(S)
            x_next = x_next.at[:, 0].set(x_batch[:, 0] + delta_left)
            x_next = x_next.at[:, -1].set(x_batch[:, -1] + delta_right)
            return x_next, cost_k

        x0_batch = jnp.broadcast_to(x0, (S, n))
        x_final, costs = jax.lax.scan(
            step_batch, x0_batch, (u_seq, w_scenarios.swapaxes(0, 1))
        )
        terminal = jnp.sum((x_final @ q) * x_final, axis=1)
        return jnp.mean(jnp.sum(costs, axis=0) + terminal)

    @jax.jit
    def solve(x0, key, u_init):
        key, subkey = jr.split(key)
        w_scenarios = jr.normal(subkey, shape=(S, H, n))

        def obj(u):
            return jax.value_and_grad(rollout_cost)(u, x0, w_scenarios)

        solver = jaxopt.LBFGS(
            fun=obj,
            value_and_grad=True,
            maxiter=maxiter,
            jit=True,
            unroll=False,
        )
        result = solver.run(u_init.flatten())
        u_opt = jnp.clip(result.params.reshape(H, m), -u_bound, u_bound)
        return u_opt, u_opt[0], key

    return solve


def simulate_mpc(controller, a, b, sigma, x0, T, H, m, key_dynamics, key_mpc):
    x_traj = [x0]
    u_traj = []
    x = x0
    u_init = jnp.zeros((H, m))
    for _ in range(T):
        u_opt, u0, key_mpc = controller(x, key_mpc, u_init)
        u_init = jnp.concatenate([u_opt[1:], u_opt[-1:]], axis=0)

        key_dynamics, subkey = jr.split(key_dynamics)
        x = true_step(a, b, sigma, x, u0, subkey)
        x_traj.append(x)
        u_traj.append(u0)

    return jnp.stack(x_traj), jnp.stack(u_traj)


def compute_cost(x_traj, u_traj, q, r):
    state_cost = float(sum(x @ q @ x for x in x_traj[:-1]))
    control_cost = float(sum(u @ r @ u for u in u_traj))
    return state_cost + control_cost, state_cost, control_cost


def summarize(name, results, baseline_total_costs):
    total_costs = np.array([r["total_cost"] for r in results], dtype=float)
    state_costs = np.array([r["state_cost"] for r in results], dtype=float)
    control_costs = np.array([r["control_cost"] for r in results], dtype=float)
    terminal_norms = np.array([r["terminal_norm"] for r in results], dtype=float)
    wall_times = np.array([r["wall_time"] for r in results], dtype=float)
    relative_gap = (total_costs - baseline_total_costs) / baseline_total_costs

    print(f"\n{name}")
    print(
        f"  mean total cost     : {total_costs.mean():.4f} +/- {total_costs.std():.4f}"
    )
    print(f"  mean state cost     : {state_costs.mean():.4f}")
    print(f"  mean control cost   : {control_costs.mean():.4f}")
    print(f"  mean terminal ||x|| : {terminal_norms.mean():.4f}")
    if name == "LQR":
        print("  mean wall time      : n/a")
    else:
        print(
            f"  mean wall time      : {wall_times.mean():.4f}s total "
            f"({(wall_times / CFG.T).mean():.4f}s/step, full closed-loop sim)"
        )
    if name not in {"LQR", "MPC-known"}:
        print(
            f"  relative cost gap   : {relative_gap.mean() * 100:.2f}% +/- {relative_gap.std() * 100:.2f}%"
        )


def export_results_mat(
    results: dict[str, list[dict]],
    path: Path,
    cfg: ExperimentCfg,
    a,
    b,
    q,
    r,
) -> None:
    """Save all trial trajectories and scalars to a MATLAB-readable .mat file.

    For each controller prefix, ``{prefix}_cum_total_cost`` has shape (num_trials, T):
    row ``i`` is the cumulative running cost (state + control) through time steps
    ``0..t`` for trial ``i`` (same convention as ``compute_cost``, no terminal x_T term).
    """
    md: dict[str, object] = {
        "meta_n": np.int32(cfg.n),
        "meta_kappa": np.float64(cfg.kappa),
        "meta_dt": np.float64(cfg.dt),
        "meta_sigma": np.float64(cfg.sigma),
        "meta_T": np.int32(cfg.T),
        "meta_H": np.int32(cfg.H),
        "meta_S": np.int32(cfg.S),
        "meta_m": np.int32(cfg.m),
        "meta_u_bound": np.float64(cfg.u_bound),
        "meta_num_trials": np.int32(cfg.num_trials),
        "meta_seed": np.int32(cfg.seed),
        "A_sys": np.asarray(a, dtype=np.float64),
        "B_sys": np.asarray(b, dtype=np.float64),
        "Q_cost": np.asarray(q, dtype=np.float64),
        "R_cost": np.asarray(r, dtype=np.float64),
    }
    q_np = np.asarray(q, dtype=np.float64)
    r_np = np.asarray(r, dtype=np.float64)
    for name, trials in results.items():
        prefix = name.replace("-", "_")
        x_stack = np.stack([t["x_traj"] for t in trials], axis=0)
        u_stack = np.stack([t["u_traj"] for t in trials], axis=0)
        state_per_step = np.sum(
            (x_stack[:, :-1, :] @ q_np) * x_stack[:, :-1, :], axis=2
        )
        ctrl_per_step = np.sum((u_stack @ r_np) * u_stack, axis=2)
        cum_total_cost = np.cumsum(state_per_step + ctrl_per_step, axis=1)

        md[f"{prefix}_x_traj"] = x_stack
        md[f"{prefix}_u_traj"] = u_stack
        md[f"{prefix}_cum_total_cost"] = cum_total_cost
        md[f"{prefix}_total_cost"] = np.array(
            [t["total_cost"] for t in trials], dtype=np.float64
        )
        md[f"{prefix}_state_cost"] = np.array(
            [t["state_cost"] for t in trials], dtype=np.float64
        )
        md[f"{prefix}_control_cost"] = np.array(
            [t["control_cost"] for t in trials], dtype=np.float64
        )
        md[f"{prefix}_terminal_norm"] = np.array(
            [t["terminal_norm"] for t in trials], dtype=np.float64
        )
        md[f"{prefix}_wall_time"] = np.array(
            [t["wall_time"] for t in trials], dtype=np.float64
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    savemat(str(path), md, do_compression=True)


def main():
    cfg = CFG
    a, b = build_system_matrices(n=cfg.n, kappa=cfg.kappa, dt=cfg.dt)
    q = jnp.eye(cfg.n)
    r = 0.1 * jnp.eye(cfg.m)
    k_lqr, _ = solve_modified_dare(a, b, q, r, sigma=cfg.sigma)

    known_controller = make_known_mpc(
        a, b, cfg.H, cfg.S, q, cfg.sigma, cfg.n, cfg.u_bound, m=cfg.m
    )
    unpruned_controller = make_unpruned_surrogate_mpc(
        a, b, cfg.H, cfg.S, q, cfg.sigma, cfg.n, cfg.u_bound, m=cfg.m
    )
    pruned_controller = make_pruned_surrogate_mpc(
        a, b, cfg.H, cfg.S, q, cfg.sigma, cfg.n, cfg.u_bound, m=cfg.m
    )

    controller_map = {
        "MPC-known": known_controller,
        "MPC-surrogate-unpruned": unpruned_controller,
        "MPC-surrogate-pruned": pruned_controller,
    }
    results = {"LQR": []}
    results.update({name: [] for name in controller_map})

    master_key = jr.PRNGKey(cfg.seed)
    for trial_idx in range(cfg.num_trials):
        master_key, trial_key = jr.split(master_key)
        x0_key, dyn_key, plan_known_key, plan_unpruned_key, plan_pruned_key = jr.split(
            trial_key, 5
        )
        x0 = jr.normal(x0_key, shape=(cfg.n,))
        controller_keys = {
            "MPC-known": (dyn_key, plan_known_key),
            "MPC-surrogate-unpruned": (dyn_key, plan_unpruned_key),
            "MPC-surrogate-pruned": (dyn_key, plan_pruned_key),
        }

        x_traj, u_traj = simulate_lqr(k_lqr, a, b, cfg.sigma, x0, cfg.T, dyn_key)
        total_cost, state_cost, control_cost = compute_cost(x_traj, u_traj, q, r)
        results["LQR"].append(
            {
                "x_traj": np.asarray(x_traj),
                "u_traj": np.asarray(u_traj),
                "total_cost": total_cost,
                "state_cost": state_cost,
                "control_cost": control_cost,
                "terminal_norm": float(jnp.linalg.norm(x_traj[-1])),
                "wall_time": 0.0,
            }
        )
        print(
            f"trial={trial_idx + 1:02d}/{cfg.num_trials:02d} "
            f"{'LQR':24s} total_cost={total_cost:.4f}"
        )

        for name, controller in controller_map.items():
            key_dynamics, key_mpc = controller_keys[name]
            t0 = time.time()
            x_traj, u_traj = simulate_mpc(
                controller,
                a,
                b,
                cfg.sigma,
                x0,
                cfg.T,
                cfg.H,
                cfg.m,
                key_dynamics,
                key_mpc,
            )
            wall_time = time.time() - t0
            total_cost, state_cost, control_cost = compute_cost(x_traj, u_traj, q, r)
            results[name].append(
                {
                    "x_traj": np.asarray(x_traj),
                    "u_traj": np.asarray(u_traj),
                    "total_cost": total_cost,
                    "state_cost": state_cost,
                    "control_cost": control_cost,
                    "terminal_norm": float(jnp.linalg.norm(x_traj[-1])),
                    "wall_time": wall_time,
                }
            )
            print(
                f"trial={trial_idx + 1:02d}/{cfg.num_trials:02d} "
                f"{name:24s} total_cost={total_cost:.4f} wall_time={wall_time:.2f}s"
            )

    baseline_total_costs = np.array(
        [r["total_cost"] for r in results["MPC-known"]], dtype=float
    )
    for name in ["LQR", *controller_map.keys()]:
        summarize(name, results[name], baseline_total_costs)

    out_mat = (CONTROL_DIR / cfg.results_mat_path).resolve()
    export_results_mat(results, out_mat, cfg, a, b, q, r)

    t_grid = np.arange(cfg.T + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    style_map = {
        "LQR": ("LQR", "-", "k"),
        "MPC-known": ("Known", "--", "b"),
        "MPC-surrogate-unpruned": ("Surrogate-unpruned", "--", "r"),
        "MPC-surrogate-pruned": ("Surrogate-pruned", ":", "g"),
    }

    for name, (label, linestyle, color) in style_map.items():
        x_stack = np.stack([r["x_traj"] for r in results[name]], axis=0)
        u_stack = np.stack([r["u_traj"] for r in results[name]], axis=0)
        mean_state_norm = np.linalg.norm(x_stack, axis=2).mean(axis=0)
        mean_control_norm = np.linalg.norm(u_stack, axis=2).mean(axis=0)
        mean_cum_state_cost = np.mean(
            np.cumsum(
                np.sum(
                    (x_stack[:, :-1, :] @ np.asarray(q)) * x_stack[:, :-1, :], axis=2
                ),
                axis=1,
            ),
            axis=0,
        )

        axes[0].semilogy(
            t_grid,
            mean_state_norm,
            label=label,
            linewidth=2,
            linestyle=linestyle,
            color=color,
        )
        axes[1].plot(
            np.arange(cfg.T),
            mean_cum_state_cost,
            label=label,
            linewidth=2,
            linestyle=linestyle,
            color=color,
        )
        axes[2].semilogy(
            np.arange(cfg.T),
            mean_control_norm,
            label=label,
            linewidth=2,
            linestyle=linestyle,
            color=color,
        )

    axes[0].set_title("Mean state norm")
    axes[1].set_title("Mean cumulative state cost")
    axes[2].set_title("Mean control norm")
    for ax in axes:
        ax.legend()

    plt.tight_layout()
    out = (CONTROL_DIR / cfg.plot_path).resolve()
    plt.savefig(out, dpi=150)
    plt.close()

    fig_all, axes_all = plt.subplots(1, 2, figsize=(14, 4))
    color_lookup = {name: f"C{idx}" for idx, name in enumerate(style_map)}
    for name, (label, linestyle, color) in style_map.items():
        x_stack = np.stack([r["x_traj"] for r in results[name]], axis=0)
        u_stack = np.stack([r["u_traj"] for r in results[name]], axis=0)
        state_norms = np.linalg.norm(x_stack, axis=2)
        control_norms = np.linalg.norm(u_stack, axis=2)

        for idx in range(state_norms.shape[0]):
            axes_all[0].semilogy(
                t_grid,
                state_norms[idx],
                color=color_lookup[name],
                alpha=0.18,
                linestyle=linestyle,
                linewidth=1,
            )
            axes_all[1].semilogy(
                np.arange(cfg.T),
                control_norms[idx],
                color=color_lookup[name],
                alpha=0.18,
                linestyle=linestyle,
                linewidth=1,
            )

        axes_all[0].semilogy(
            t_grid,
            state_norms.mean(axis=0),
            color=color_lookup[name],
            linewidth=2.5,
            linestyle=linestyle,
            label=label,
        )
        axes_all[1].semilogy(
            np.arange(cfg.T),
            control_norms.mean(axis=0),
            color=color_lookup[name],
            linewidth=2.5,
            linestyle=linestyle,
            label=label,
        )

    axes_all[0].set_title("All state-norm trajectories")
    axes_all[1].set_title("All control-norm trajectories")
    for ax in axes_all:
        ax.legend()

    plt.tight_layout()
    out_all = (CONTROL_DIR / cfg.all_traj_plot_path).resolve()
    plt.savefig(out_all, dpi=150)
    plt.close()

    mpc_only_style = {
        "MPC-known": ("Known dynamics", "-", "k"),
        "MPC-surrogate-unpruned": ("Surrogate (unpruned)", "-.", "g"),
        "MPC-surrogate-pruned": ("Surrogate (pruned)", "--", "r"),
    }
    q_np = np.asarray(q)
    r_np = np.asarray(r)
    fig_mpc, ax_mpc = plt.subplots()
    for name, (label, linestyle, color) in mpc_only_style.items():
        x_stack = np.stack([r["x_traj"] for r in results[name]], axis=0)
        u_stack = np.stack([r["u_traj"] for r in results[name]], axis=0)
        state_per_step = np.sum(
            (x_stack[:, :-1, :] @ q_np) * x_stack[:, :-1, :], axis=2
        )
        ctrl_per_step = np.sum((u_stack @ r_np) * u_stack, axis=2)
        mean_cum_total = np.mean(
            np.cumsum(state_per_step + ctrl_per_step, axis=1), axis=0
        )
        ax_mpc.plot(
            np.arange(cfg.T),
            mean_cum_total,
            label=label,
            linewidth=2.5,
            linestyle=linestyle,
            color=color,
        )
    # ax_mpc.set_ylim([39, 41])
    ax_mpc.set_xlabel("time step")
    ax_mpc.set_ylabel("mean cumulative cost")
    ax_mpc.set_title("Mean cumulative total cost (state + control), MPC only")
    ax_mpc.legend()
    ax_mpc.grid(True, alpha=0.3)
    plt.tight_layout()
    out_mpc_cum = (CONTROL_DIR / cfg.mpc_mean_cum_total_cost_path).resolve()
    plt.savefig(out_mpc_cum, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
