import jax
import jax.numpy as jnp
import numpy as np
import diffrax as dfx


def sample(params, x, model, *, z_dim: int | None = None, reshape_output: bool = True):
    def velocity(params, model, t, z, x):
        out = model.apply({"params": params}, t, z, x)
        return out.reshape(-1) if reshape_output else out

    term = dfx.ODETerm(lambda t, y, args: velocity(params, model, t, y, args))

    @jax.jit
    def solve_single(z0i, x):
        sol = dfx.diffeqsolve(
            term,
            dfx.Tsit5(),
            t0=0.0,
            t1=1.0,
            dt0=None,
            y0=z0i,
            args=x,
            saveat=dfx.SaveAt(t1=True),
            stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-5),
            max_steps=1_000_000,
        )
        return sol.ys

    if z_dim is None:
        z_dim = int(model.dim_out)
    z0 = np.random.normal(size=[x.shape[0], z_dim])
    z0 = jnp.array(z0)
    zT = jax.vmap(solve_single, in_axes=(0, 0))(z0, x)
    return zT, z0
