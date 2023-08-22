import numpy as np
import emcee
import multiprocessing
from numpy.polynomial.chebyshev import chebval
from typing import List, Callable, Any, Union
from gofnt_routines import do_gofnt_matrix_integral
from make_dem_methods import make_dem_cheby, make_dem_const_gp
from jax import numpy as jnp
import os
# from matplotlib import pyplot as plt


os.environ["OMP_NUM_THREADS"] = "1"


def test_within_interval(input, lower_bound, upper_bound):
    if input < lower_bound:
        return False
    elif input > upper_bound:
        return False
    return True


def ln_prior_cutoff_dem(
    params: List[float],
    psi_low: float = 0.0,
    psi_high: float = 26.0,
) -> float:
    """Apply a uniform prior between 10**+/- 2 for coefficients used in
    Chebyshev polynomial model for DEM. The c_0 coefficient is sampled
    uniformly between two limits and then gets an exponential penalty
    such that the probability is 1/e less likely at limits psi_low and psi_high
    which are set by reasonable physical assumptions for the DEM.
    The temperature interval is 1e8 K, the path-length varies between
    0.01 R_sun and 10 R_sun (1e8 and 1e11 cm), and the n_e varies between
    1e8 and 1e13 cm^-3

    Keyword arguments:
    :param params: List of coefficients for Chebyshev polynomial in format used
    by numpy chebyshev polynomial followed by the predicted flux factor
    uncertainty.
    :type params: list.

    Returns:
    :returns: float -- 0 if prior is satisfied, -np.inf otherwise.

    """
    flux_factor = params[-1]
    coeffs = params[:-1]
    lp = 0.0

    if not(test_within_interval(coeffs[0], psi_low, psi_high)):
        # print('Leading coefficient is bad')
        return -np.inf

    if not(test_within_interval(flux_factor, -1, 1)):
        # print("Flux Factor out of range")
        return -np.inf
    elif chebval(-1.0, coeffs) <= chebval(-0.99, coeffs):
        # print("DEM does not turn up at low temperature end")
        return -np.inf
    elif chebval(1.0, coeffs) >= chebval(0.99, coeffs):
        # print("DEM does not turn down at high temperature end")
        return -np.inf
    else:
        return lp


def ln_prob_flux_sigma_dem(
    params: List[float],
    y: np.ndarray,
    yerr: Union[np.ndarray, float],
    log_temp: np.ndarray,
    temp: np.ndarray,
    gofnt_matrix: np.ndarray,
    flux_weighting: float,
) -> float:
    """Evaluate the likelihood with an additional variance term associated
    with the uncertainty of the predicted flux from the model
    Keyword arguments:
    :param params: List of coefficients for Chebyshev polynomial in format used
    by numpy chebyshev polynomial followed by the predicted flux factor
    uncertainty.
    :type params: list.

    :param y: Bin-integral of observed spectrum.
    :type y: np.ndarray.

    :param yerr: Error on y, either an array or constant float.
    :type yerr: Union[np.ndarray, float].

    :param log_temp: log10(Temperature  array for ChiantiPy emissivities)
    :type log_temp: np.ndarray.

    :param temp: Temperature array for ChiantiPy emissivities.
    :type temp: np.ndarray.

    :param gofnt_matrix: Contribution matrix along wavelength bin and
    temperature arrays.
    :type gofnt_matrix: np.ndarray.

    :param flux_weighting: Weight the intensity integral to map to
    some observable or desired quantity
    (flux received by exoplanet, surface flux of star etc.)
    :type flux_weighting: float.

    Returns:
    :returns: float -- -1/2 Chi-squared by comparing bin integral of observed
    spectrum to DEM integrated spectrum

    """
    flux_factor = 10.0 ** (params[-1])
    coeffs = params[:-1]
    x_arr = np.linspace(-1, 1, len(temp))
    psi_model = make_dem_cheby(x_arr, coeffs)
    if np.nanmin(psi_model) <= 0:
        return -np.inf
    model = do_gofnt_matrix_integral(
        psi_model, gofnt_matrix, temp, flux_weighting
    )
    var_term = ((flux_factor * model) ** 2) + (yerr ** 2)
    lead_term = np.log(1.0 / np.sqrt(2.0 * np.pi * var_term))
    inv_var = 1.0 / var_term
    val = np.sum(
        lead_term - (0.5 * ((((y - model) ** 2) * inv_var)))
    )
    if np.isfinite(val):
        return val
    return -np.inf


def ln_prior_matern(params, psi_low=17.0, psi_high=26.0):
    lp = 0.0
    for param in params:
        if np.isfinite(param):
            pass
        else:
            return -np.inf
    coeff_0 = params[0]

    if not(test_within_interval(coeff_0, psi_low, psi_high)):
        return -np.inf
    amp = np.exp(params[-2])
    scale = params[-1]

    if not(test_within_interval(scale, 0.01, 2.0)):
        # print('bad scale')
        return -np.inf
    if not(test_within_interval(amp, 1e-4 * coeff_0, 1e0 * coeff_0)):
        # print('bad amp')
        return -np.inf
    return lp


def ln_prior_const_gp(params, psi_low=17.0, psi_high=26.0):
    lp = 0.0
    for param in params:
        if np.isfinite(param):
            pass
        else:
            return -np.inf
    coeff_0 = params[0]

    if not(test_within_interval(coeff_0, psi_low, psi_high)):
        return -np.inf
    amp = np.exp(params[-1])

    if not(test_within_interval(amp, 1e-4 * coeff_0, 1e0 * coeff_0)):
        return -np.inf
    return lp


def ln_prior_const_matern(params, psi_low=17.0, psi_high=26.0):
    lp = 0.0
    for param in params:
        if np.isfinite(param):
            pass
        else:
            return -np.inf
    coeff_0 = params[0]
    amp_const = np.exp(params[-3])
    amp_matern = np.exp(params[-2])
    scale = params[-1]

    if not(test_within_interval(coeff_0, psi_low, psi_high)):
        return -np.inf
    if not(test_within_interval(scale, 0.01, 2.0)):
        return -np.inf
    if not(test_within_interval(amp_const, 1e-8 * coeff_0, 1e4 * coeff_0)):
        return -np.inf
    if not(test_within_interval(amp_matern, 1e-4 * coeff_0, 1e4 * coeff_0)):
        return -np.inf
    return lp


def ln_prior_rational_quadratic(params, psi_low=17.0, psi_high=26.0):
    lp = 0.0
    for param in params:
        if np.isfinite(param):
            pass
        else:
            return -np.inf
    coeff_0 = params[0]
    amp = np.exp(params[-3])
    scale = params[-2]
    alpha = np.exp(params[-1])

    if not(test_within_interval(coeff_0, psi_low, psi_high)):
        return -np.inf
    if not(test_within_interval(scale, 0.01, 2.0)):
        return -np.inf
    if not(test_within_interval(amp, 1e-4 * coeff_0, 1e-1 * coeff_0)):
        return -np.inf
    if not(test_within_interval(alpha, 1e-5, 1e5)):
        return -np.inf
    return lp


def ln_prob_gp(
    make_dem_gp,
    params,
    y,
    yerr,
    temp,
    gofnt_matrix,
    flux_weighting,
    n_cheby_params,
    n_kernel_params,
):
    psi_model, psi_std = make_dem_gp(
        params,
        y,
        gofnt_matrix,
        temp,
        flux_weighting,
        n_cheby_params,
        n_kernel_params,
        "sample",
    )
    model = do_gofnt_matrix_integral(
        psi_model, gofnt_matrix, temp, flux_weighting
    )
    model_std = do_gofnt_matrix_integral(psi_std, gofnt_matrix, temp, flux_weighting)
    var_term = yerr ** 2 + (model_std ** 2)
    lead_term = np.log(1.0 / np.sqrt(2.0 * np.pi * var_term))
    inv_var = 1.0 / var_term
    val = np.sum(
        lead_term - (0.5 * ((((y - model) ** 2) * inv_var)))
    )
    if np.isfinite(val):
        return val
    # print('bad model')
    return -np.inf


def ln_likelihood_cheby(
    params: List[float],
    y: np.ndarray,
    yerr: Union[np.ndarray, float],
    log_temp: np.ndarray,
    temp: np.ndarray,
    gofnt_matrix: np.ndarray,
    flux_weighting: float,
    ln_prob_func: Callable = ln_prob_flux_sigma_dem,
    ln_prior_func: Callable = ln_prior_cutoff_dem,
) -> float:
    """Combine a defined prior and probability function to determine the
    ln_likelihood of a model given two types of data: the bin-integral of an
    observed spectrum or the intensities of individual lines.

    Keyword arguments:
    :param params: List of coefficients for Chebyshev polynomial in format used
    by numpy chebyshev polynomial followed by the predicted flux factor
    uncertainty.
    :type params: list.

    :param y: Bin-integral of observed spectrum.
    :type y: np.ndarray.

    :param yerr: Error on y, either an array or constant float.
    :type yerr: Union[np.ndarray, float].

    :param log_temp: log10(Temperature  array for ChiantiPy emissivities)
    :type log_temp: np.ndarray.

    :param temp: Temperature array for ChiantiPy emissivities.
    :type temp: np.ndarray.

    :param wave_arr: Wavelength array with bin centers.
    :type wave_arr: np.ndarray.

    :param gofnt_matrix: Contribution matrix along wavelength bin and
    temperature arrays.
    :type gofnt_matrix: np.ndarray.

    :param flux_weighting: Weight the intensity integral to map to
    some observable or desired quantity
    (flux received by exoplanet, surface flux of star etc.)
    :type flux_weighting: float.

    Returns:
    :returns: float -- ln_likelihood from comparing bin integral of observed
    spectrum to DEM integrated spectrum or individual line intensities to DEM
    integrated line intensities.

    """
    params = jnp.array(params)

    lp = ln_prior_func(params)
    if np.isfinite(lp):
        return lp + ln_prob_func(
            params,
            y,
            yerr,
            log_temp,
            temp,
            gofnt_matrix,
            flux_weighting,
        )
    return -np.inf


def ln_likelihood_gp(
    params: List[float],
    y: np.ndarray,
    yerr: Union[np.ndarray, float],
    temp: np.ndarray,
    gofnt_matrix: np.ndarray,
    flux_weighting: float,
    make_dem_gp: Callable = make_dem_const_gp,
    n_cheby_params: int = 6,
    n_kernel_params: int = 3,
    ln_prob_func: Callable = ln_prob_gp,
    ln_prior_func: Callable = ln_prior_const_gp,
) -> float:
    """Combine a defined prior and probability function to determine the
    ln_likelihood of a model given two types of data: the bin-integral of an
    observed spectrum or the intensities of individual lines.

    Keyword arguments:
    :param params: Array of knot y-values for GP interpolation followed by
                   the amplitude and scale of the Matern32 GP kernel.
    :type params: np.ndarray

    :param knot_locs: Array of knot x-values for GP interpolation
    :type knot_locs: np.ndarray.

    :param y: Bin-integral of observed spectrum.
    :type y: np.ndarray.

    :param yerr: Error on y, either an array or constant float.
    :type yerr: Union[np.ndarray, float].

    :param temp: Temperature array for ChiantiPy emissivities.
    :type temp: np.ndarray.

    :param gofnt_matrix: Contribution matrix along wavelength bin and
    temperature arrays.
    :type gofnt_matrix: np.ndarray.

    :param flux_weighting: Weight the intensity integral to map to
    some observable or desired quantity
    (flux received by exoplanet, surface flux of star etc.)
    :type flux_weighting: float.

    Returns:
    :returns: float -- ln_likelihood from comparing bin integral of observed
    spectrum to DEM integrated spectrum or individual line intensities to DEM
    integrated line intensities.

    """

    lp = ln_prior_func(params)
    if np.isfinite(lp):
        return lp + ln_prob_func(
            make_dem_gp,
            params,
            y,
            yerr,
            temp,
            gofnt_matrix,
            flux_weighting,
            n_cheby_params,
            n_kernel_params,
        )
    return -np.inf


def fit_emcee(
    init_pos: np.ndarray,
    likelihood_func: Callable,
    likelihood_args: List[Any],
    n_walkers: int,
    burn_in_steps: int,
    production_steps: int,
    init_spread: float = 1e-1,
    second_spread: float = 1e-2,
    double_burn: bool = True,
    thread_num: int = multiprocessing.cpu_count(),
    count_print: bool = True,
    count_num: int = 100,
    progress_file: str = "temp_backend.h5",
):
    """Run the emcee sampler with a given likelihood function
    and return the flatchain samples, flatchain ln_probability, and the sampler
    object.

    Keyword arguments:
    :param init_pos: Initial values for the model parameters.
    : type init_pos: np.ndarray.

    :param likelihood_func: Likelihood function for the model.
    :type likelihood_func: function.

    :param likelihood_args: Arguments required for the likelihood function.
    :type likelihood_args: list.

    :param n_walkers: Number of walkers for the emcee sampler.
    :type n_walkers: int.

    :param burn_in_steps: Number of steps for the burn-in phase.
    :type burn_in_steps: int.

    :param production_steps: Number of steps for the production phase.
    :type production_steps: int.

    :param init_spread: Multiplicative factor by which to scramble the initial
    position of the walkers. (default 1e-3)
    :type init_spread: float.

    :param second_spread: Multiplicative factor by which to scramble the
    highest likelihood position after the burn-in phase. (default 1e-4)
    :type second_spread: float.

    :param double_burn: Whether or not to do a second burn-in phase. Treated
    identically to the initial. (default True)
    :type double_burn: bool

    :param thread_num: Number of threads for the emcee sampler to use.
    (default cpu_count)
    :type thread_num: int.

    :param count_print: Whether or not to print progress messages during
    production.
    :type count_print: bool.

    :param count_num: Interval for print messages.
    :type count_num: int.

    Returns:
    :returns: np.ndarray -- Reshaped sampler positions, collapsed along walker
    axis (see emcee documentation).

    :returns: np.ndarray -- ln_probability values for walker positions aligned
    to the flatchain.

    :returns: emcee.sampler -- emcee sampler object, refer to emcee
    documentation.

    """

    ndim = len(init_pos)
    pos = [
        init_pos
        + init_spread * np.random.randn(ndim) * init_pos
        for i in range(n_walkers)
    ]
    print(
        "Starting ln_likelihood is: ",
        likelihood_func(init_pos, *likelihood_args),
    )
    print("Initializing walkers")
    with multiprocessing.Pool(thread_num) as pool:
        if progress_file is not None:
            backend = emcee.backends.HDFBackend(progress_file)
            store_backend = True
        else:
            backend = None
            store_backend = False
        autocorr = np.empty(
            burn_in_steps + burn_in_steps + production_steps
        )
        old_tau = np.inf
        sampler = emcee.EnsembleSampler(
            n_walkers,
            ndim,
            likelihood_func,
            args=likelihood_args,
            pool=pool,
            backend=backend,
        )
        iter_index = sampler.iteration
        print("Starting burn-in")
        if double_burn:
            burn_in_steps *= 2
        p0, prob, _ = sampler.run_mcmc(
            pos, burn_in_steps, store=store_backend, tune=True,
        )
        p0 = [
            p0[np.argmax(prob)]
            + second_spread
            * np.random.randn(ndim)
            * p0[np.argmax(prob)]
            for i in range(n_walkers)
        ]
        print("Starting production")
        for sample in sampler.sample(
            p0,
            iterations=production_steps,
            progress=count_print,
            store=store_backend,
            tune=True,
        ):
            if sampler.iteration % count_num:
                continue
            tau = sampler.get_autocorr_time(tol=0)
            print('Tau estimate: ', tau)
            autocorr[iter_index] = np.mean(tau)
            iter_index += 1
            converged = np.all(tau * 100 < sampler.iteration)
            if converged:
                print('> 100 autocorr times')
            converged &= np.all(
                np.abs(old_tau - tau) / tau < 0.05
            )
            if converged:
                print("Converged at iter: ", sampler.iteration)
                break
            old_tau = tau
    if store_backend:
        reader = emcee.backends.HDFBackend(
            progress_file, read_only=True
        )
        flatchain = reader.get_chain(flat=True)
        flatlnprob = reader.get_log_prob(flat=True)
    else:
        flatchain = sampler.flatchain
        flatlnprob = sampler.flatlnprobability
    return flatchain, flatlnprob
