import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import roman
from scipy.integrate import cumtrapz
from astropy.io import fits
from astropy import units as u
from astropy import constants as const
from astropy.table import Table, setdiff
from gofnt_routines import do_gofnt_matrix_integral
from make_dem_methods import make_dem_cheby
from matplotlib.gridspec import (
    GridSpec,
    GridSpecFromSubplotSpec,
)
from pandas import read_csv
from run_single_star import get_best_gofnt_matrix_press


sns.set_context("notebook")
plt.rc("text", usetex=True)
plt.rc("font", family="serif")

temp = np.logspace(4, 8, 200)
gofnt_all = get_best_gofnt_matrix_press(0.0, 1e17, mode="b1")

main_color = "b"
sample_alpha = 0.2
main_label = "Best-fit"
low_y = 19
high_y = 24.5
sample_num = 1000
wave_low = 30
wave_upp = 1100
log_low = 5e-18
log_upp = 1e-11
par_color = "cornflowerblue"
main_marker = "*"
pop_marker = "."
pop_color = "0.2"
pop_alpha = sample_alpha / 3
sample_marker = "s"
sample_color = "rebeccapurple"
subsample_marker = "o"
subsample_color = "r"
subsample_alpha = 0.8
temp_low = 2750
temp_upp = 6200
ion_flux_low = 1e-15
ion_flux_high = 1e-11
rot_low = 1e-1
rot_upp = 250

star_name = r"Eps \ Eri"
star_root = "eps_eri"
press_str = "p17"
star_press_root = star_root + "_" + press_str
star_temp = 5084
star_mass = 0.82
star_rad = 0.735
star_dist = 3.212
star_rot = 11.4

spec_fits = "spectrum_" + star_press_root + ".fits"
samples = np.load("samples_" + star_press_root + ".npy")
lnprob = np.load("lnprob_" + star_press_root + ".npy")
flux_arr = np.load("flux_" + star_press_root + ".npy")
gofnt_arr = np.load("gofnt_" + star_press_root + ".npy")
flux_weighting = np.load("flux_weighting_" + star_root + ".npy")
dems = np.load("spectrum_" + star_press_root + "_dems.npy")

ion_names = np.load("ion_names_" + star_press_root + ".npy")
ion_gofnts = np.load("gofnt_lines_" + star_press_root + ".npy")
ion_flux_arr = np.load("ion_fluxes_" + star_press_root + ".npy")
ion_errs = np.load("ion_errs_" + star_press_root + ".npy")
spec_waves = np.load("spectrum_waves_" + star_press_root + ".npy")
spec_flux_arr = np.load("spectrum_fluxes_" + star_press_root + ".npy")
spec_err_arr = np.load("spectrum_errs_" + star_press_root + ".npy")
spec_gofnts = np.load("gofnt_spectrum_" + star_press_root + ".npy")

n_cheby_params = np.shape(samples)[1] - 1
earth_weight = ((star_dist * (u.pc / u.AU)).value)**2

temp_str = str(star_temp) + " K"
mass_str = str(star_mass) + r"$\, M_\odot$"
rad_str = "{:.2f}".format(star_rad) + r"$\, R_\odot$"
rot_str = str(star_rot) + r"$\, \rm{days}$"
dist_str = "{:.1f}".format(star_dist) + r"$\,$pc"

star_title = r"$\bf{" + star_name + r"}$: "
star_title += temp_str + "; "
star_title += mass_str + "; "
star_title += rad_str + "; "
star_title += rot_str + "; "
star_title += dist_str

fig = plt.figure(figsize=(6.5, 8))
gs = GridSpec(4, 3, figure=fig, hspace=0.15)
dem_ax = fig.add_subplot(gs[:2, :2])
spec_log_ax = fig.add_subplot(gs[2:, :2])
fit_ax = fig.add_subplot(gs[:3, 2])
par_ax = fig.add_subplot(gs[3, 2])

fig.suptitle(star_title)


def plot_dem(
    samples,
    lnprob,
    flux_arr,
    gofnt_matrix,
    temp,
    flux_weighting,
    main_color="b",
    alpha=0.2,
    sample_num=1000,
    main_label="Best-fit Model",
    low_y=19.0,
    high_y=26.0,
    ion_names=None,
    ion_gofnts=None,
    ion_fluxes=None,
    dem_method="cheby",
    make_dem_gp=None,
    n_cheby_params=6,
    n_kernel_params=3,
):
    x_arr = np.linspace(-1, 1, len(temp))
    if dem_method == "cheby":
        psi_model = make_dem_cheby(
            x_arr, samples[np.argmax(lnprob)][:-1]
        )
    elif dem_method == "gp":
        psi_model = make_dem_gp(
            samples[np.argmax(lnprob)],
            flux_arr,
            gofnt_matrix,
            temp,
            flux_weighting,
            n_cheby_params,
            n_kernel_params,
            "mean",
        )
    total_samples = np.random.choice(len(samples), sample_num)
    psi_ys = flux_arr / (
        flux_weighting * np.trapz(gofnt_matrix, temp)
    )
    temp_lows = np.min(temp) * np.ones((len(gofnt_matrix)))
    temp_upps = np.max(temp) * np.ones_like(temp_lows)
    temp_lows = 1e4 * np.ones_like(psi_ys)
    temp_upps = 1e8 * np.ones_like(temp_lows)
    for i in range(len(flux_arr)):
        gofnt_cumtrapz = cumtrapz(gofnt_matrix[i], temp)
        low_index = np.argmin(
            np.abs(gofnt_cumtrapz - (0.16 * gofnt_cumtrapz[-1]))
        )
        upp_index = np.argmin(
            np.abs(gofnt_cumtrapz - (0.84 * gofnt_cumtrapz[-1]))
        )
        temp_lows[i] = temp[low_index + 1]
        temp_upps[i] = temp[upp_index + 1]
    all_psi = np.zeros((sample_num, len(psi_model)))
    for i in range(0, sample_num):
        s = samples[total_samples[i]]
        if dem_method == "cheby":
            temp_psi = make_dem_cheby(x_arr, s[:-1])
        elif dem_method == "gp":
            s = samples[total_samples[i]]
            temp_psi, _ = make_dem_gp(
                s,
                flux_arr,
                gofnt_matrix,
                temp,
                flux_weighting,
                n_cheby_params,
                n_kernel_params,
                "sample",
            )
        all_psi[i] = temp_psi
    dem_ax.loglog(
        temp,
        psi_model,
        color=main_color,
        label=main_label,
        ls="--",
    )
    dem_ax.fill_between(
        temp,
        np.percentile(all_psi, 16, axis=0),
        np.percentile(all_psi, 84, axis=0),
        color=main_color,
        alpha=alpha,
    )
    dem_ax.hlines(
        psi_ys,
        temp_lows,
        temp_upps,
        label="Flux Constraints",
        colors="k",
        zorder=100,
    )
    if ion_names is not None:
        dem_xs = np.array(
            [
                temp[np.argmax(ion_gofnts[i])]
                for i in range(len(ion_names))
            ]
        )
        dem_ys = np.copy(ion_fluxes)
        dem_ys /= do_gofnt_matrix_integral(
            np.ones_like(temp), ion_gofnts, temp, flux_weighting
        )
        for i in range(len(ion_names)):
            ion_name = ion_names[i].split("_")
            new_name = ion_name[0].capitalize() + " "
            new_name += roman.toRoman(int(ion_name[1]))
            dem_ax.text(
                dem_xs[i] * 0.9,
                dem_ys[i] * 1.05,
                new_name,
                fontsize="x-small",
            )
    dem_ax.set_ylim(10.0**low_y, 10.0**high_y)
    dem_ax.set_xlabel("Temperature [K]")
    y_label = r"$\Psi(T) = N_e N_{\mathrm{H}} \frac{ds}{dT}$ "
    y_label += r"[$\rm{cm}^{-5}\, \rm{K}^{-1}$]"
    dem_ax.set_ylabel(y_label)
    return plt.gcf()


_ = plot_dem(
    samples,
    lnprob,
    flux_arr,
    gofnt_arr,
    temp,
    flux_weighting,
    main_color,
    sample_alpha,
    sample_num,
    main_label,
    low_y,
    high_y,
    ion_names,
    ion_gofnts,
    ion_flux_arr,
    dem_method="cheby",
    make_dem_gp=None,
    n_cheby_params=n_cheby_params,
    n_kernel_params=3,
)


def plot_spectrum(
    spec_fits,
    alpha,
    color,
    log_low,
    log_upp,
    wave_low,
    wave_upp,
    flux_weight,
):
    hdu = fits.open(spec_fits)
    wave = hdu[1].data["Wavelength"]
    flux = hdu[1].data["Flux_density"]
    upp = flux + hdu[1].data["Upper_Error_84"]
    low = flux - hdu[1].data["Lower_Error_16"]
    spec_log_ax.set_xlim(wave_low, wave_upp)
    spec_log_ax.set_ylim(log_low, log_upp)
    spec_log_ax.semilogy(
        wave,
        flux * flux_weight,
        drawstyle="steps-mid",
        color=color,
    )
    spec_log_ax.fill_between(
        wave,
        low * flux_weight,
        upp * flux_weight,
        color=color,
        alpha=alpha,
        step="mid",
    )
    spec_log_ax.set_xlabel(r"Wavelength [$\mathrm{\AA}$]")
    spec_log_ax.set_ylabel(
        r"Flux Density [erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$]"
    )
    return plt.gcf()


plot_spectrum(
    spec_fits,
    sample_alpha,
    main_color,
    log_low,
    log_upp,
    wave_low,
    wave_upp,
    earth_weight,
)


def compare_ion(
    y_axis_labels,
    flux,
    err,
    gofnt,
    samples,
    earth_weight,
    temp,
    main_color,
    ion_flux_low,
    ion_flux_high,
    sample_num=10000,
):
    model_array = np.zeros((sample_num, len(flux)))
    rand_indices = np.random.choice(
        range(np.shape(samples)[0]), size=sample_num
    )
    x_arr = np.linspace(-1, 1, len(temp))
    for i, rand_index in enumerate(rand_indices):
        sample = samples[rand_index]
        coeffs = sample[:-1]
        s_factor = 10.0 ** sample[-1]
        psi = make_dem_cheby(x_arr, coeffs)
        model = do_gofnt_matrix_integral(
            psi, gofnt, temp, flux_weighting
        )
        model_array[i, :] = np.random.normal(
            loc=model, scale=(s_factor * model)
        )

    model_med = np.nanmedian(model_array, axis=0)
    model_low = np.percentile(model_array, 16, axis=0)
    model_upp = np.percentile(model_array, 84, axis=0)
    err_asym = [
        (model_med - model_low) * earth_weight,
        (model_upp - model_med) * earth_weight,
    ]
    fit_ax.errorbar(
        model_med * earth_weight,
        y_axis_labels,
        xerr=err_asym,
        color=main_color,
        ls="",
        marker="x",
        label="Model Prediction",
    )
    fit_ax.errorbar(
        flux * earth_weight,
        y_axis_labels,
        xerr=err * earth_weight,
        color="k",
        marker=".",
        ls="",
        label="Data",
    )
    fit_ax.set_xlabel(
        r"Flux [erg s$^{-1}$ cm$^{-2}$]",
    )
    fit_ax.set_xscale("log")
    fit_ax.set_xlim(ion_flux_low, ion_flux_high)
    fit_ax.tick_params(axis="y", which="major", labelsize=8)
    return None


_, med, _ = dems
ion_temp_fs = temp[np.argmax(med * ion_gofnts, axis=1)]
spec_temp_fs = temp[np.argmax(med * spec_gofnts, axis=1)]

temp_fs = np.append(ion_temp_fs, spec_temp_fs)

y_axis_names = []

spec_energs = ((const.h * const.c) / (spec_waves * u.AA)).to(
    u.keV
)

for i in range(len(ion_names)):
    ion_name = ion_names[i].split("_")
    new_name = ion_name[0].capitalize() + " "
    new_name += roman.toRoman(int(ion_name[1]))
    y_axis_names.append(new_name)

y_axis_names = np.append(
    y_axis_names,
    [
        "{:.2f}".format(spec_energs[i])
        for i in range(len(spec_energs))
    ],
)

y_axis_names = np.array(y_axis_names)[np.argsort(temp_fs)]
fit_flux = np.append(ion_flux_arr, spec_flux_arr)[
    np.argsort(temp_fs)
]

fit_err = np.append(ion_errs, spec_err_arr)[np.argsort(temp_fs)]
fit_gofnt = np.append(ion_gofnts, spec_gofnts, axis=0)[
    np.argsort(temp_fs), :
]

compare_ion(
    y_axis_names[::-1],
    fit_flux[::-1],
    fit_err[::-1],
    fit_gofnt[::-1],
    samples,
    earth_weight,
    temp,
    main_color,
    ion_flux_low,
    ion_flux_high,
)


def plot_sample(
    temp,
    rot,
    main_color,
    main_marker,
    pop_marker,
    pop_color,
    pop_alpha,
    sample_marker,
    sample_color,
    sample_alpha,
    subsample_marker,
    subsample_color,
    subsample_alpha,
    temp_low=2750,
    temp_upp=6200,
    rot_low=1e-1,
    rot_upp=250,
):
    old_sample_table = Table.read("../../collate_products/star_table.tex")
    subsample_table = Table.read("../../collate_products/subsample_table.tex")
    sample_table = setdiff(old_sample_table, subsample_table)
    sample_table.sort("T_eff")
    all_table = Table.read(
        "../../collate_products/teffrot_sampleNewton.csv",
        format="csv",
        comment="#",
    )
    all_table2 = read_csv(
        "../../collate_products/McQuillan2014table1.dat",
        delim_whitespace=True,
        usecols=[1, 4],
        names=["T_eff", "P_rot"],
        dtype={"T_eff": int, "P_rot": np.float64},
    )
    par_ax.scatter(
        all_table["Teff (K)"],
        all_table["Rotation Period (days)"],
        marker=pop_marker,
        color=pop_color,
        alpha=pop_alpha,
        s=5,
        zorder=-1
    )
    par_ax.scatter(
        all_table2["T_eff"],
        all_table2["P_rot"],
        marker=pop_marker,
        color=pop_color,
        alpha=pop_alpha,
        s=5,
        zorder=-1
    )
    par_ax.plot(
        sample_table["T_eff"],
        sample_table["P_rot"],
        ls="",
        marker=sample_marker,
        color=sample_color,
        alpha=sample_alpha,
        zorder=5,
        markersize=5,
    )
    par_ax.plot(
        subsample_table["T_eff"],
        subsample_table["P_rot"],
        ls="",
        marker=subsample_marker,
        color=subsample_color,
        markerfacecolor="none",
        alpha=subsample_alpha,
        zorder=10,
        markersize=5,
    )
    par_ax.scatter(
        temp,
        rot,
        marker=main_marker,
        color=main_color,
        zorder=100,
        s=60
    )
    par_ax.set_yscale("log")
    par_ax.set_xlabel(
        r"$T_{\textrm{eff}}$ [K]",
        fontsize="small"
    )
    par_ax.set_ylabel(
        r" $P_{\textrm{rot}}$ [days]",
        rotation=270,
        fontsize="small"
    )
    par_ax.set_xlim(temp_low, temp_upp)
    par_ax.set_ylim(rot_low, rot_upp)
    return None


plot_sample(
    star_temp,
    star_rot,
    par_color,
    main_marker,
    pop_marker,
    pop_color,
    pop_alpha,
    sample_marker,
    sample_color,
    sample_alpha,
    subsample_marker,
    subsample_color,
    subsample_alpha,
    temp_low,
    temp_upp,
    rot_low,
    rot_upp,
)

dem_ax.tick_params(
    bottom=False, top=True, labeltop=True, labelbottom=False
)
dem_ax.xaxis.set_ticks_position("top")
dem_ax.xaxis.set_label_position("top")
fit_ax.xaxis.set_label_position("top")
par_ax.yaxis.set_label_position("right")
fit_ax.tick_params(
    top=True,
    labeltop=True,
    bottom=False,
    labelbottom=False,
    left=False,
    labelleft=False,
    right=True,
    labelright=True,
)
par_ax.tick_params(
    left=False, labelleft=False, right=True, labelright=True
)
plt.savefig(star_press_root + "_summary_fig.pdf")
plt.show()
