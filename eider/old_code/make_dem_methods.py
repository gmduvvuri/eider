import numpy as np
from numpy.polynomial.chebyshev import chebval
import jax
from tinygp import kernels, GaussianProcess
from jax import numpy as jnp
from functools import partial


# x_gp_len = 20
# x_gp = jnp.linspace(-1, 1, x_gp_len)
x_gp = None
x_arr_len = 200
x_arr = jnp.linspace(-1, 1, x_arr_len)


def make_dem_cheby(x_arr, coeffs):
    return 10.0 ** (chebval(x_arr, coeffs))


def make_dem_gp(
    params,
    y,
    gofnt_matrix,
    temp,
    flux_weighting,
    build_gp_mean,
    build_gp_sample,
    build_gp_dict,
    n_cheby_params,
    n_kernel_params,
    dict_names,
    mode="sample",
):
    coeffs = params[:n_cheby_params]
    dem_mean = make_dem_cheby(coeffs)
    dem_ys = y / (
        flux_weighting * np.trapz(gofnt_matrix, temp, axis=1)
    )
    gp_xs = get_gp_xs(dem_mean, gofnt_matrix, len(y))
    cheb_ys = chebval(gp_xs, coeffs)
    gp_ys = jnp.array(jnp.log10(dem_ys) - cheb_ys)
    param_dict = build_gp_dict(
        params[-n_kernel_params:],
        gp_xs,
        gp_ys,
        dict_names,
    )
    if mode == "sample":
        arr, gp_var = build_gp_sample(param_dict)
        psi_std = 10.0**(chebval(x_arr, coeffs) + jnp.sqrt(gp_var))
    elif mode == "mean":
        arr = build_gp_mean(param_dict)
    psi_model = 10.0 ** (chebval(x_arr, coeffs) + arr)
    if mode == "sample":
        return psi_model, psi_std
    else:
        return psi_model


def build_matern32(params):
    amp_matern = jnp.exp(params["log_amp_matern"])
    matern_scale = params["matern_scale"]
    gp_xs = params["gp_xs"]
    gp_ys = params["gp_ys"]
    k_tot = amp_matern * kernels.Matern32(
        matern_scale
    )
    gp = GaussianProcess(k_tot, gp_xs, diag=0.01, mean=0.0)
    return gp, gp_ys


@jax.jit
def build_matern32_sample(param_dict):
    gp, gp_ys = build_matern32(param_dict)
    gp_log_prob, cond_gp = gp.condition(gp_ys, x_arr)
    return cond_gp.sample(jax.random.PRNGKey(1)), cond_gp.variance


@jax.jit
def build_matern32_mean(param_dict):
    gp, gp_ys = build_matern32(param_dict)
    gp_log_prob, cond_gp = gp.condition(gp_ys, x_arr)
    return cond_gp.mean


def build_const_gp(params):
    amp_const = jnp.exp(params["log_amp_const"])
    gp_xs = params["gp_xs"]
    gp_ys = params["gp_ys"]
    k_tot = kernels.Constant(amp_const)
    gp = GaussianProcess(k_tot, gp_xs, diag=0.01, mean=0.0)
    return gp, gp_ys


@jax.jit
def build_const_gp_sample(param_dict):
    gp, gp_ys = build_const_gp(param_dict)
    gp_log_prob, cond_gp = gp.condition(gp_ys, x_arr)
    return cond_gp.sample(jax.random.PRNGKey(1)), cond_gp.variance


@jax.jit
def build_const_gp_mean(param_dict):
    gp, gp_ys = build_const_gp(param_dict)
    gp_log_prob, cond_gp = gp.condition(gp_ys, x_arr)
    return cond_gp.mean


def build_const_matern32(params):
    amp_const = jnp.exp(params["log_amp_const"])
    amp_matern = jnp.exp(params["log_amp_matern"])
    matern_scale = params["matern_scale"]
    gp_xs = params["gp_xs"]
    gp_ys = params["gp_ys"]
    k_matern = amp_matern * kernels.Matern32(
        matern_scale
    )
    # k_const = kernels.Constant(amp_const)
    k_tot = k_matern  # + k_const
    gp = GaussianProcess(k_tot, gp_xs, diag=amp_const, mean=0.0)
    return gp, gp_ys


@jax.jit
def build_const_matern32_sample(param_dict):
    gp, gp_ys = build_const_matern32(param_dict)
    gp_log_prob, cond_gp = gp.condition(gp_ys, x_arr)
    return cond_gp.sample(jax.random.PRNGKey(1)), cond_gp.variance


@jax.jit
def build_const_matern32_mean(param_dict):
    gp, gp_ys = build_const_matern32(param_dict)
    gp_log_prob, cond_gp = gp.condition(gp_ys, x_arr)
    return cond_gp.mean


def build_rational_quadratic(params):
    amp = jnp.exp(params["log_amp"])
    scale = params["scale"]
    alpha = jnp.exp(params["log_alpha"])
    gp_xs = params["gp_xs"]
    gp_ys = params["gp_ys"]
    k_tot = amp * kernels.RationalQuadratic(scale=scale, alpha=alpha)
    gp = GaussianProcess(k_tot, gp_xs, diag=0.01, mean=0.0)
    return gp, gp_ys


@jax.jit
def build_rational_quadratic_sample(param_dict):
    gp, gp_ys = build_rational_quadratic(param_dict)
    gp_log_prob, cond_gp = gp.condition(gp_ys, x_arr)
    return cond_gp.sample(jax.random.PRNGKey(1)), cond_gp.variance


@jax.jit
def build_rational_quadratic_mean(param_dict):
    gp, gp_ys = build_rational_quadratic(param_dict)
    gp_log_prob, cond_gp = gp.condition(gp_ys, x_arr)
    return cond_gp.mean


def build_expsquared(params):
    amp = jnp.exp(params["log_amp"])
    scale = params["scale"]
    gp_xs = params["gp_xs"]
    gp_ys = params["gp_ys"]
    k_tot = amp * kernels.ExpSquared(
        scale
    )
    gp = GaussianProcess(k_tot, gp_xs, diag=0.01, mean=0.0)
    return gp, gp_ys


@jax.jit
def build_expsquared_sample(param_dict):
    gp, gp_ys = build_expsquared(param_dict)
    gp_log_prob, cond_gp = gp.condition(gp_ys, x_arr)
    return cond_gp.sample(jax.random.PRNGKey(1)), cond_gp.variance


@jax.jit
def build_expsquared_mean(param_dict):
    gp, gp_ys = build_expsquared(param_dict)
    gp_log_prob, cond_gp = gp.condition(gp_ys, x_arr)
    return cond_gp.mean


def build_matern52(params):
    amp_matern = jnp.exp(params["log_amp_matern"])
    matern_scale = params["matern_scale"]
    gp_xs = params["gp_xs"]
    gp_ys = params["gp_ys"]
    k_tot = amp_matern * kernels.Matern52(
        matern_scale
    )
    gp = GaussianProcess(k_tot, gp_xs, diag=0.01, mean=0.0)
    return gp, gp_ys


@jax.jit
def build_matern52_sample(param_dict):
    gp, gp_ys = build_matern52(param_dict)
    gp_log_prob, cond_gp = gp.condition(gp_ys, x_arr)
    return cond_gp.sample(jax.random.PRNGKey(1)), cond_gp.variance


@jax.jit
def build_matern52_mean(param_dict):
    gp, gp_ys = build_matern52(param_dict)
    gp_log_prob, cond_gp = gp.condition(gp_ys, x_arr)
    return cond_gp.mean


def build_const_matern52(params):
    amp_const = jnp.exp(params["log_amp_const"])
    amp_matern = jnp.exp(params["log_amp_matern"])
    matern_scale = params["matern_scale"]
    gp_xs = params["gp_xs"]
    gp_ys = params["gp_ys"]
    k_matern = amp_matern * kernels.Matern52(
        matern_scale
    )
    k_const = kernels.Constant(amp_const)
    k_tot = k_matern + k_const
    gp = GaussianProcess(k_tot, gp_xs, diag=1e-3, mean=0.0)
    return gp, gp_ys


@jax.jit
def build_const_matern52_sample(param_dict):
    gp, gp_ys = build_const_matern52(param_dict)
    gp_log_prob, cond_gp = gp.condition(gp_ys, x_arr)
    return cond_gp.sample(jax.random.PRNGKey(1)), cond_gp.variance


@jax.jit
def build_const_matern52_mean(param_dict):
    gp, gp_ys = build_const_matern52(param_dict)
    gp_log_prob, cond_gp = gp.condition(gp_ys, x_arr)
    return cond_gp.mean


def build_gp_dict(
    params, gp_xs, gp_ys, dict_names
):
    param_dict = {
        "gp_xs": gp_xs,
        "gp_ys": gp_ys,
    }
    for param, name in zip(params, dict_names):
        param_dict[name] = param
    return param_dict


@partial(jax.jit, static_argnames=["y_len"])
def get_gp_xs(dem_mean, gofnt_matrix, y_len):
    return jnp.array(
        [
            x_arr[np.argmax(dem_mean * gofnt_matrix[i])]
            for i in range(y_len)
        ]
    )


def make_dem_matern32(
    params,
    y,
    gofnt_matrix,
    temp,
    flux_weighting,
    n_cheby_params,
    n_kernel_params,
    mode,
):
    psi_model = make_dem_gp(
        params,
        y,
        gofnt_matrix,
        temp,
        flux_weighting,
        build_matern32_mean,
        build_matern32_sample,
        build_gp_dict,
        n_cheby_params,
        n_kernel_params,
        ['log_amp_matern', 'matern_scale'],
        mode,
    )
    return psi_model


def make_dem_const_gp(
    params,
    y,
    gofnt_matrix,
    temp,
    flux_weighting,
    n_cheby_params,
    n_kernel_params,
    mode,
):
    psi_model = make_dem_gp(
        params,
        y,
        gofnt_matrix,
        temp,
        flux_weighting,
        build_const_gp_mean,
        build_const_gp_sample,
        build_gp_dict,
        n_cheby_params,
        n_kernel_params,
        ['log_amp_const'],
        mode
    )
    return psi_model


def make_dem_const_matern32(
    params,
    y,
    gofnt_matrix,
    temp,
    flux_weighting,
    n_cheby_params,
    n_kernel_params,
    mode,
):
    psi_model = make_dem_gp(
        params,
        y,
        gofnt_matrix,
        temp,
        flux_weighting,
        build_const_matern32_mean,
        build_const_matern32_sample,
        build_gp_dict,
        n_cheby_params,
        n_kernel_params,
        ['log_amp_const', 'log_amp_matern', 'matern_scale'],
        mode,
    )
    return psi_model


def make_dem_rational_quadratic(
    params,
    y,
    gofnt_matrix,
    temp,
    flux_weighting,
    n_cheby_params,
    n_kernel_params,
    mode,
):
    psi_model = make_dem_gp(
        params,
        y,
        gofnt_matrix,
        temp,
        flux_weighting,
        build_rational_quadratic_mean,
        build_rational_quadratic_sample,
        build_gp_dict,
        n_cheby_params,
        n_kernel_params,
        ['log_amp', 'scale', 'log_alpha'],
        mode,
    )
    return psi_model


def make_dem_expsquared(
    params,
    y,
    gofnt_matrix,
    temp,
    flux_weighting,
    n_cheby_params,
    n_kernel_params,
    mode,
):
    psi_model = make_dem_gp(
        params,
        y,
        gofnt_matrix,
        temp,
        flux_weighting,
        build_expsquared_mean,
        build_expsquared_sample,
        build_gp_dict,
        n_cheby_params,
        n_kernel_params,
        ['log_amp', 'scale'],
        mode,
    )
    return psi_model


def make_dem_matern52(
    params,
    y,
    gofnt_matrix,
    temp,
    flux_weighting,
    n_cheby_params,
    n_kernel_params,
    mode,
):
    psi_model = make_dem_gp(
        params,
        y,
        gofnt_matrix,
        temp,
        flux_weighting,
        build_matern52_mean,
        build_matern52_sample,
        build_gp_dict,
        n_cheby_params,
        n_kernel_params,
        ['log_amp_matern', 'matern_scale'],
        mode,
    )
    return psi_model


def make_dem_const_matern52(
    params,
    y,
    gofnt_matrix,
    temp,
    flux_weighting,
    n_cheby_params,
    n_kernel_params,
    mode,
):
    psi_model = make_dem_gp(
        params,
        y,
        gofnt_matrix,
        temp,
        flux_weighting,
        build_const_matern52_mean,
        build_const_matern52_sample,
        build_gp_dict,
        n_cheby_params,
        n_kernel_params,
        ['log_amp_const', 'log_amp_matern', 'matern_scale'],
        mode,
    )
    return psi_model
