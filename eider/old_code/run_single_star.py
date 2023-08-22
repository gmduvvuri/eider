import os.path
import corner
import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy import constants as const
from astropy.table import Table
from jax import numpy as jnp
from astropy.io import fits

from data_prep import (
    generate_constant_R_wave_arr,
    generate_spectrum_cheby,
    generate_spectrum_gp,
    resample_spectrum,
)
from dem_plots import (
    compare_ion,
    compare_spec,
    display_fig,
    plot_dem,
    plot_spectrum,
    plot_emissivities,
)
from fitting import fit_emcee
from gofnt_routines import (
    generate_ion_gofnts,
    parse_ascii_table_CHIANTI,
    resample_gofnt_matrix,
)

jax.config.update("jax_enable_x64", True)


def generate_flux_weighting(star_name, star_dist, star_rad):
    flux_weighting = (
        (
            np.pi
            * u.sr
            * (star_rad**2.0)
            * (1.0 / (star_dist**2.0))
        ).to(u.sr)
    ).value
    np.save("flux_weighting_" + star_name, [flux_weighting])
    return flux_weighting


def get_best_gofnt_matrix_press(
    abundance, press, abund_type="sol0", mode="r100"
):
    gofnt_dir = "../../../../gofnt_dir_coarse/"
    gofnt_root = "gofnt_w1_w1500_t4_t8_n200_" + mode + "_p"
    gofnt_matrix = np.load(
        gofnt_dir
        + gofnt_root
        + str(int(np.log10(press)))
        + "_"
        + abund_type
        + ".npy"
    )
    gofnt_matrix *= 10.0**abundance
    return gofnt_matrix


def get_line_data_gofnts(
    star_name, line_table_file, abundance, temp, dens, bin_width
):
    line_table = parse_ascii_table_CHIANTI(line_table_file)
    gofnt_lines, flux, err, names = generate_ion_gofnts(
        line_table, abundance, bin_width, temp, dens
    )
    np.save("gofnt_lines_" + star_name + ".npy", gofnt_lines)
    np.save("ion_fluxes_" + star_name + ".npy", flux)
    np.save("ion_errs_" + star_name + ".npy", err)
    np.save("ion_names_" + star_name + ".npy", names)
    return gofnt_lines, flux, err, names


def get_spectrum_data_gofnt(
    star_name, data_npy_file, gofnt_matrix
):
    wave, wave_bins, flux, err = np.load(
        data_npy_file, allow_pickle=True
    )
    flux *= wave_bins
    err *= wave_bins
    wave_old, _ = generate_constant_R_wave_arr(1, 1500, 100)
    gofnt_spectrum = resample_gofnt_matrix(
        gofnt_matrix, wave, wave_bins, wave_old
    )
    temp = np.logspace(4, 8, 200)
    gofnt_ints = np.trapz(gofnt_spectrum, temp)
    print(gofnt_ints)
    mask = np.where((gofnt_ints >= 3e-19))
    print(gofnt_ints[mask])

    plt.errorbar(
        wave,
        flux,
        yerr=err,
        label="data",
        drawstyle="steps-mid",
        color="b",
    )

    plt.plot(
        wave[mask],
        flux[mask],
        marker="o",
        ls="",
        label="Selected Constraints",
    )
    plt.xlabel(r"Wavelength [$\textrm{\AA}$]")
    plt.ylabel(
        r"Flux Density [erg s$^{-1}$ cm$^{-2}$ $\textrm{\AA}^{-1}$]"
    )
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig("spectrum_data_constraints.pdf")
    plt.show()

    np.save(
        "gofnt_spectrum_" + star_name + ".npy",
        gofnt_spectrum[mask[0], :],
    )
    np.save("spectrum_fluxes_" + star_name + ".npy", flux[mask])
    np.save("spectrum_errs_" + star_name + ".npy", err[mask])
    np.save("spectrum_waves_" + star_name + ".npy", wave[mask])
    np.save(
        "spectrum_bins_" + star_name + ".npy", wave_bins[mask]
    )
    return gofnt_spectrum[mask[0], :], flux[mask], err[mask]


def get_star_data_gofnt_press(
    star_name,
    abundance,
    press,
    line_table_file=None,
    data_npy_file=None,
    bin_width=1.0,
):
    big_gofnt = get_best_gofnt_matrix_press(abundance, press)
    temp = np.logspace(4, 8, 200)
    dens = press / temp
    if line_table_file is not None:
        if os.path.isfile("gofnt_lines_" + star_name + ".npy"):
            gofnt_lines = np.load(
                "gofnt_lines_" + star_name + ".npy"
            )
            flux = np.load("ion_fluxes_" + star_name + ".npy")
            err = np.load("ion_errs_" + star_name + ".npy")
        else:
            gofnt_lines, flux, err, _ = get_line_data_gofnts(
                star_name,
                line_table_file,
                abundance,
                temp,
                dens,
                bin_width,
            )
        line_flux = flux
        line_err = err
    else:
        gofnt_lines = None
    if data_npy_file is not None:
        if os.path.isfile(
            "gofnt_spectrum_" + star_name + ".npy"
        ):
            gofnt_spectrum = np.load(
                "gofnt_spectrum_" + star_name + ".npy"
            )
            flux = np.load(
                "spectrum_fluxes_" + star_name + ".npy"
            )
            err = np.load("spectrum_errs_" + star_name + ".npy")
        else:
            gofnt_spectrum, flux, err = get_spectrum_data_gofnt(
                star_name, data_npy_file, big_gofnt
            )
        spectrum_flux = flux
        spectrum_err = err
    else:
        gofnt_spectrum = None
    if gofnt_lines is None:
        if gofnt_spectrum is None:
            print(
                "Where is this star's data to do anything with?"
            )
        else:
            gofnt_matrix = gofnt_spectrum
            np.save("gofnt_" + star_name + ".npy", gofnt_matrix)
            np.save("flux_" + star_name + ".npy", flux)
            np.save("err_" + star_name + ".npy", err)
    elif gofnt_spectrum is None:
        gofnt_matrix = gofnt_lines
        np.save("gofnt_" + star_name + ".npy", gofnt_matrix)
        np.save("flux_" + star_name + ".npy", flux)
        np.save("err_" + star_name + ".npy", err)
    else:
        gofnt_matrix = np.append(
            gofnt_spectrum, gofnt_lines, axis=0
        )
        flux = np.append(spectrum_flux, line_flux)
        err = np.append(spectrum_err, line_err)
        np.save("gofnt_" + star_name + ".npy", gofnt_matrix)
        np.save("flux_" + star_name + ".npy", flux)
        np.save("err_" + star_name + ".npy", err)
    return gofnt_matrix, flux, err


def run_mcmc_single_star(
    init_pos,
    gofnt_matrix,
    flux,
    err,
    flux_weighting,
    star_name,
    ln_prior_func,
    ln_prob_func,
    ln_likelihood_func,
    make_dem_gp,
    n_cheby_params,
    n_kernel_params,
    n_walkers=24,
    burn_in_steps=1000,
    production_steps=10000,
    thread_num=6,
    count_num=2000,
    double_burn=True,
):
    temp = np.logspace(4, 8, 200)
    if dem_method == "gp":
        samples, lnprob = fit_emcee(
            init_pos=init_pos,
            likelihood_func=ln_likelihood_func,
            likelihood_args=[
                jnp.array(flux),
                jnp.array(err),
                jnp.array(temp),
                jnp.array(gofnt_matrix),
                flux_weighting,
                make_dem_gp,
                n_cheby_params,
                n_kernel_params,
                ln_prob_func,
                ln_prior_func,
            ],
            n_walkers=n_walkers,
            burn_in_steps=burn_in_steps,
            production_steps=production_steps,
            thread_num=thread_num,
            count_num=count_num,
            double_burn=double_burn,
            progress_file="backend_" + star_name + ".h5",
        )
    elif dem_method == "cheby":
        samples, lnprob = fit_emcee(
            init_pos=init_pos,
            likelihood_func=ln_likelihood_func,
            likelihood_args=[
                jnp.array(flux),
                jnp.array(err),
                jnp.array(log_temp),
                jnp.array(temp),
                jnp.array(gofnt_matrix),
                flux_weighting,
                ln_prob_func,
                ln_prior_func,
            ],
            n_walkers=n_walkers,
            burn_in_steps=burn_in_steps,
            production_steps=production_steps,
            thread_num=thread_num,
            count_num=count_num,
            double_burn=double_burn,
            progress_file="backend_" + star_name + ".h5",
        )
    else:
        print("Invalid DEM Method")
        return None
    np.save("samples_" + star_name, samples)
    np.save("lnprob_" + star_name, lnprob)
    return samples, lnprob


def generate_spectrum_data_npy(star_name, xray_fname):
    xdata = fits.getdata(xray_fname)
    xray_wave = xdata["WAVELENGTH"]
    xray_flux = xdata["FLUX"]
    xray_err = xdata["ERROR"]
    xray_bins = xdata["WAVELENGTH1"] - xdata["WAVELENGTH0"]

    plt.errorbar(
        xray_wave,
        xray_flux,
        yerr=xray_err,
        label="data",
        drawstyle="steps-mid",
        color="b",
    )

    xray_output_axis = np.arange(5, 62, 1)
    xray_bins = 3 * np.ones_like(xray_output_axis)
    flux_unit = u.erg / (u.s * u.AA * (u.cm**2))
    input_spec_tuple = (
        xray_wave * u.AA,
        xray_flux * flux_unit,
        xray_err,
    )
    xray_wave, xray_flux, xray_err = resample_spectrum(
        input_spec_tuple, xray_output_axis * u.AA
    )
    mask = np.where(
        np.isfinite(xray_flux)
        & (np.isfinite(xray_err))
    )
    xray_flux = xray_flux[mask].value
    xray_err = np.array(xray_err[mask])
    xray_wave = xray_wave[mask].value
    xray_bins = xray_bins[mask]

    plt.errorbar(
        xray_wave,
        xray_flux,
        yerr=xray_err,
        label="resampled",
        drawstyle="steps-mid",
        color="orange",
    )

    plt.xlabel(r"Wavelength [$\textrm{\AA}$]")
    plt.ylabel(
        r"Flux Density [erg s$^{-1}$ cm$^{-2}$ $\textrm{\AA}^{-1}$]"
    )
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig("spectrum_data.pdf")
    plt.show()
    print("Loaded X-ray Data")
    data_npy_arr = np.array(
        [
            xray_wave,
            xray_bins,
            xray_flux,
            xray_err,
        ],
        dtype=float,
    )
    np.save(
        star_name + "_spectrum_data.npy",
        data_npy_arr,
    )
    return star_name + "_spectrum_data.npy"


def load_gofnt(
    star_name, abundance, press, line_table_file, data_npy_file
):
    if os.path.isfile("gofnt_" + star_name + ".npy"):
        gofnt_matrix = np.load("gofnt_" + star_name + ".npy")
        flux = np.load("flux_" + star_name + ".npy")
        err = np.load("err_" + star_name + ".npy")
    else:
        out = get_star_data_gofnt_press(
            star_name,
            abundance,
            press,
            line_table_file,
            data_npy_file=data_npy_file,
        )
        gofnt_matrix, flux, err = out
    return gofnt_matrix, flux, err


def load_samples(star_name):
    samples = np.load("samples_" + star_name + ".npy")
    lnprob = np.load("lnprob_" + star_name + ".npy")
    return samples, lnprob


def do_dem_plot(
    star_name,
    samples,
    lnprob,
    flux,
    gofnt_matrix,
    temp,
    flux_weighting,
    title_name,
    dem_method="cheby",
    make_dem_gp=None,
    n_cheby_params=6,
    n_kernel_params=3,
):
    if os.path.isfile("dem_" + star_name + ".pdf"):
        pass
    else:
        if os.path.isfile("gofnt_lines_" + star_name + ".npy"):
            ion_names = np.load(
                "ion_names_" + star_name + ".npy"
            )
            ion_gofnts = np.load(
                "gofnt_lines_" + star_name + ".npy"
            )
            ion_fluxes = np.load(
                "ion_fluxes_" + star_name + ".npy"
            )
        else:
            ion_names = None
            ion_gofnts = None
            ion_fluxes = None
        print(ion_names)
        g = plot_dem(
            samples,
            lnprob,
            flux,
            gofnt_matrix,
            temp,
            flux_weighting,
            "b",
            "cornflowerblue",
            0.1,
            500,
            "MCMC Samples",
            "Best-fit DEM model",
            title_name,
            ion_names=ion_names,
            ion_gofnts=ion_gofnts,
            ion_fluxes=ion_fluxes,
            dem_method=dem_method,
            make_dem_gp=make_dem_gp,
            n_cheby_params=n_cheby_params,
            n_kernel_params=n_kernel_params,
        )
        g = display_fig(g, "dem_" + star_name, mode="pdf")
        plt.clf()


def do_corner_plot(
    star_name,
    samples,
    dem_method="cheby",
    n_cheby_params=None,
    n_kernel_params=None,
):
    if os.path.isfile("corner_" + star_name + ".pdf"):
        pass
    elif dem_method == "gp":
        h = corner.corner(
            samples[:, :n_cheby_params],
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12},
            plot_contours=True,
        )
        h = display_fig(h, "corner_" + star_name, mode="pdf")
        plt.clf()

        h2 = corner.corner(
            samples[:, -n_kernel_params:],
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12},
            plot_contours=True,
        )
        h2 = display_fig(
            h2, "corner_kernel_" + star_name, mode="pdf"
        )
        plt.clf()
    else:
        h = corner.corner(
            samples,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12},
            plot_contours=True,
        )
        h = display_fig(h, "corner_" + star_name, mode="pdf")


def load_ion_dem_npys(star_name):
    ion_names = np.load("ion_names_" + star_name + ".npy")
    ion_gofnts = np.load("gofnt_lines_" + star_name + ".npy")
    ion_fluxes = np.load("ion_fluxes_" + star_name + ".npy")
    ion_errs = np.load("ion_errs_" + star_name + ".npy")
    dems = np.load("spectrum_" + star_name + "_dems.npy")
    return ion_names, ion_gofnts, ion_fluxes, ion_errs, dems


def full_run_press(
    star_name_root,
    title_name,
    press,
    abundance,
    line_table_file,
    data_npy_file,
    init_pos,
    flux_weighting,
    dem_method,
    ln_prior_func,
    ln_prob_func,
    ln_likelihood_func,
    make_dem_gp,
    n_cheby_params,
    n_kernel_params,
    n_walkers,
    burn_in_steps,
    production_steps,
    thread_num,
    count_num,
    double_burn,
):
    star_name = (
        star_name_root + "_p" + str(int(np.log10(press)))
    )
    gofnt_all = get_best_gofnt_matrix_press(
        abundance, press, mode="b1"
    )
    wave_all = np.arange(1, 1501, 1.0)
    bin_all = np.ones_like(wave_all)
    temp = np.logspace(4, 8, 200)
    out = load_gofnt(
        star_name,
        abundance,
        press,
        line_table_file,
        data_npy_file=data_npy_file,
    )
    gofnt_matrix, flux, err = out
    if os.path.isfile("samples_" + star_name + ".npy"):
        samples, lnprob = load_samples(star_name)
    else:
        samples, lnprob = run_mcmc_single_star(
            init_pos,
            gofnt_matrix,
            flux,
            err,
            flux_weighting,
            star_name,
            ln_prior_func=ln_prior_func,
            ln_prob_func=ln_prob_func,
            ln_likelihood_func=ln_likelihood_func,
            make_dem_gp=make_dem_gp,
            n_cheby_params=n_cheby_params,
            n_kernel_params=n_kernel_params,
            n_walkers=n_walkers,
            burn_in_steps=burn_in_steps,
            production_steps=production_steps,
            thread_num=thread_num,
            count_num=count_num,
            double_burn=double_burn,
        )
    do_dem_plot(
        star_name,
        samples,
        lnprob,
        flux,
        gofnt_matrix,
        temp,
        flux_weighting,
        title_name,
        dem_method=dem_method,
        make_dem_gp=make_dem_gp,
        n_cheby_params=n_cheby_params,
        n_kernel_params=n_kernel_params,
    )
    do_corner_plot(
        star_name,
        samples,
        dem_method=dem_method,
        n_cheby_params=n_cheby_params,
        n_kernel_params=n_kernel_params,
    )
    if os.path.isfile("spectrum_" + star_name + ".fits"):
        spectrum_table = Table.read(
            "spectrum_" + star_name + ".fits"
        )
    else:
        spectrum_name = "spectrum_" + star_name
        if dem_method == "cheby":
            spectrum_table, _ = generate_spectrum_cheby(
                spectrum_name,
                samples,
                lnprob,
                gofnt_all,
                flux_weighting,
                wave_all,
                bin_all,
            )
        elif dem_method == "gp":
            spectrum_table, _ = generate_spectrum_gp(
                spectrum_name,
                samples,
                lnprob,
                make_dem_gp,
                flux,
                gofnt_matrix,
                n_cheby_params,
                n_kernel_params,
                gofnt_all,
                flux_weighting,
                wave_all,
                bin_all,
            )
    if os.path.isfile("spectrum_" + star_name + ".pdf"):
        pass
    else:
        spec_fig = plot_spectrum(
            "spectrum_" + star_name + ".fits", title_name
        )
        spec_fig = display_fig(
            spec_fig, "spectrum_" + star_name, mode="pdf"
        )

    if os.path.isfile("gofnt_lines_" + star_name + ".npy"):
        out = load_ion_dem_npys(star_name)
        (
            ion_names,
            ion_gofnts,
            ion_fluxes,
            ion_errs,
            dems,
        ) = out
        if os.path.isfile("compare_ion_" + star_name + ".pdf"):
            pass
        else:
            compare_ion_fig = compare_ion(
                ion_gofnts,
                ion_names,
                ion_fluxes,
                ion_errs,
                dems,
                samples,
                flux_weighting,
                temp,
                title_name,
                dem_method=dem_method,
                make_dem_gp=make_dem_gp,
                n_cheby_params=n_cheby_params,
                n_kernel_params=n_kernel_params,
            )
            compare_ion_fig = display_fig(
                compare_ion_fig,
                "compare_ion_" + star_name,
                mode="pdf",
            )
            emissivity_line_fig = plot_emissivities(
                temp, ion_gofnts, title_name, 0.5, "b"
            )
            emissivity_line_fig = display_fig(
                emissivity_line_fig,
                "emissivity_line_" + star_name,
                mode="pdf",
            )
    else:
        pass
    if os.path.isfile("gofnt_spectrum_" + star_name + ".npy"):
        spec_out = np.load(
            star_name_root + "_spectrum_data.npy"
        )
        spec_wave, spec_bins, spec_flux, spec_err = spec_out
        spec_gofnt = resample_gofnt_matrix(
            gofnt_all, spec_wave, spec_bins, wave_all
        )
        filter_spec_gofnt = np.load(
            "gofnt_spectrum_" + star_name + ".npy"
        )
        if os.path.isfile("compare_spec_" + star_name + ".pdf"):
            pass
        else:
            compare_spec_fig = compare_spec(
                spec_gofnt,
                spec_wave,
                spec_bins,
                spec_flux,
                spec_err,
                samples,
                flux_weighting,
                temp,
                title_name,
                dem_method=dem_method,
                make_dem_gp=make_dem_gp,
                n_cheby_params=n_cheby_params,
                n_kernel_params=n_kernel_params,
            )
            compare_spec_fig = display_fig(
                compare_spec_fig,
                "compare_spec_" + star_name,
                mode="pdf",
            )
            emissivity_spec_fig = plot_emissivities(
                temp, filter_spec_gofnt, title_name, 0.5, "b"
            )
            emissivity_spec_fig = display_fig(
                emissivity_spec_fig,
                "emissivity_spec_" + star_name,
                mode="pdf",
            )
    if os.path.isfile("emissivity_all" + star_name + ".pdf"):
        pass
    else:
        emissivity_all_fig = plot_emissivities(
            temp, gofnt_matrix, title_name, 0.5, "b"
        )
        emissivity_all_fig = display_fig(
            emissivity_all_fig,
            "emissivity_all_" + star_name,
            mode="pdf",
        )


if __name__ == "__main__":
    star_name_root = "eps_eri"
    xray_fname = "../hlsp_muscles_xmm_epic_v-eps-eri_multi_v22_component-spec.fits"

    generate_spectrum_data_npy(star_name_root, xray_fname)

    wave, _, flux, err = np.load(
        star_name_root + "_spectrum_data.npy"
    )

    temp = np.logspace(4, 8, 200)
    log_temp = np.log10(temp)
    title_name = r"Eps Eri"
    abundance = 0.0
    star_rad = 0.735 * u.Rsun
    star_dist = 3.2198 * u.pc
    flux_weighting = generate_flux_weighting(
        star_name_root, star_dist, star_rad
    )
    dem_method = "cheby"
    n_cheby_params = 6
    n_kernel_params = 0
    n_walkers = 25
    burn_in_steps = 2000
    production_steps = 200000
    thread_num = 4
    count_num = 3000
    double_burn = True

    if dem_method == "cheby":
        from fitting import (
            ln_prob_flux_sigma_dem as ln_prob_func,
        )
        from fitting import ln_prior_cutoff_dem as ln_prior_func
        from fitting import (
            ln_likelihood_cheby as ln_likelihood_func,
        )

        make_dem_gp = None
    elif dem_method == "gp":
        from fitting import ln_prior_matern as ln_prior_func
        from fitting import ln_prob_gp as ln_prob_func
        from fitting import (
            ln_likelihood_gp as ln_likelihood_func,
        )
        from make_dem_methods import (
            make_dem_matern32 as make_dem_gp,
        )

    init_pos = jnp.array(
        [
            22.49331207,
            -3.31678227,
            -0.49848262,
            -1.27244452,
            -0.93897052,
            -0.67235648,
            np.log10(0.3),
        ]
    )
    press_list = [
        1e17,
    ]

    line_table_file = star_name_root + "_linetable.ascii"
    data_npy_file = star_name_root + "_spectrum_data.npy"
    for press in press_list:
        full_run_press(
            star_name_root,
            title_name,
            press,
            abundance,
            line_table_file,
            data_npy_file,
            init_pos,
            flux_weighting,
            dem_method,
            ln_prior_func=ln_prior_func,
            ln_prob_func=ln_prob_func,
            ln_likelihood_func=ln_likelihood_func,
            make_dem_gp=make_dem_gp,
            n_cheby_params=n_cheby_params,
            n_kernel_params=n_kernel_params,
            n_walkers=n_walkers,
            burn_in_steps=burn_in_steps,
            production_steps=production_steps,
            thread_num=thread_num,
            count_num=count_num,
            double_burn=double_burn,
        )
