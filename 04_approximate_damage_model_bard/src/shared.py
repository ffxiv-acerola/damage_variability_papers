import matplotlib as mpl
import numpy as np
from ffxiv_stats import Rate
from scipy.signal import fftconvolve

BARD_RATE = Rate(3000, 2000, 100)


def set_custom_style():
    """Set custom color scheme for matplotlib."""
    # Define custom colors
    custom_colors = ["#5842ff", "#ff4000", "#00b368"]

    # Set the color cycle to use these colors
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=custom_colors)

    # Return the custom colors in case needed directly
    return custom_colors


def create_custom_cmap(name="custom_cmap"):
    """Create a custom colormap from the custom colors."""
    custom_colors = ["#5842ff", "#ff4000", "#00b368"]
    custom_cmap = mpl.colors.LinearSegmentedColormap.from_list(name, custom_colors)

    return custom_cmap


def thousands_formatter(x, pos):
    return f"{int(x / 1000)}K" if x >= 1000 else str(int(x))


def relative_d2(potency: int, reference_potency: int = 220, reference_d2: int = 10000):
    return int(potency / reference_potency * reference_d2)


def mixture_distribution(rotation_list, weights):
    supp_min = min([r.rotation_dps_support[0] for r in rotation_list])
    supp_max = max([r.rotation_dps_support[-1] for r in rotation_list])

    mixture_support = np.arange(supp_min, supp_max + 1, step=1)
    mixture_pmf = np.zeros(shape=mixture_support.shape)

    for idx, r in enumerate(rotation_list):
        mixture_slice = r.rotation_dps_support - supp_min
        mixture_pmf[mixture_slice] += r.rotation_dps_distribution * weights[idx]

    return mixture_pmf, mixture_support


def convolve_sequences(rotation_pdf_list):
    rotation_distribution = fftconvolve(
        rotation_pdf_list[0][0],
        rotation_pdf_list[1][0],
    )

    if len(rotation_pdf_list) > 2:
        for a in range(2, len(rotation_pdf_list)):
            rotation_distribution = fftconvolve(
                rotation_distribution, rotation_pdf_list[a][0]
            )

    support_min = sum([a[1][0] for a in rotation_pdf_list])
    support_max = sum([a[1][-1] for a in rotation_pdf_list])

    supp = np.arange(support_min, support_max + 1, step=1)

    rotation_distribution /= np.trapezoid(rotation_distribution, supp)
    return rotation_distribution, supp


def pmf_mean(support, pmf):
    return np.trapezoid(support * pmf, support)


def pmf_var(support, pmf):
    return np.trapezoid(support**2 * pmf, support) - pmf_mean(support, pmf) ** 2


def pmf_std(support, pmf):
    return pmf_var(support, pmf) ** (0.5)
