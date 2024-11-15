import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import quantity_support
from matplotlib.colors import LogNorm
from pyirf.binning import join_bin_lo_hi

from .vis_utils import (
    find_columnwise_stats,
    get_2d_hist_from_table,
    get_bin_centers,
    get_x_bin_values_with_rebinning,
)

__all__ = [
    "plot_2d_irf_table",
    "plot_2d_table_with_col_stats",
    "plot_2d_table_col_stats",
    "plot_hist2d",
    "plot_hist2d_as_contour",
    "plot_irf_table",
]

quantity_support()


def plot_2d_irf_table(
    ax, table, column, x_prefix, y_prefix, x_label=None, y_label=None, **mpl_args
):
    mat_vals, xbins, ybins = get_2d_hist_from_table(x_prefix, y_prefix, table, column)

    if not x_label:
        x_label = x_prefix
    if not y_label:
        y_label = y_prefix
    plot = plot_hist2d(
        ax, mat_vals, xbins, ybins, xlabel=x_label, ylabel=y_label, **mpl_args
    )
    plt.colorbar(plot)
    return ax


def plot_2d_table_with_col_stats(
    ax,
    table,
    column,
    x_prefix,
    y_prefix,
    num_rebin=4,
    stat_kind=2,
    quantiles=[0.2, 0.8],
    x_label=None,
    y_label=None,
    density=False,
    mpl_args={
        "histo": {"xscale": "log"},
        "stats": {"color": "firebrick"},
    },
):
    """
    Function to draw 2d histogram along with columnwise statistics
    the plotted errorbars shown depending on stat_kind:
    0 -> mean + standard deviation
    1 -> median + standard deviation
    2 -> median + user specified quantiles around median (default 0.1 to 0.9)
    """

    mat_vals, xbins, ybins = get_2d_hist_from_table(x_prefix, y_prefix, table, column)
    xcent = get_bin_centers(xbins)
    rebin_x, rebin_xcent, rebin_hist, density = get_x_bin_values_with_rebinning(
        num_rebin, xbins, xcent, mat_vals, density
    )

    plot = plot_hist2d(
        ax,
        rebin_hist,
        rebin_x,
        ybins,
        xlabel=x_label,
        ylabel=y_label,
        **mpl_args["histo"],
    )
    plt.colorbar(plot)

    ax = plot_2d_table_col_stats(
        ax,
        table,
        column,
        x_prefix,
        y_prefix,
        num_rebin,
        stat_kind,
        quantiles,
        x_label,
        y_label,
        density,
        mpl_args=mpl_args,
        lbl_prefix="",
    )
    return ax


def plot_2d_table_col_stats(
    ax,
    table,
    column,
    x_prefix,
    y_prefix,
    num_rebin=4,
    stat_kind=2,
    quantiles=[0.2, 0.8],
    x_label=None,
    y_label=None,
    density=False,
    lbl_prefix="",
    mpl_args={"xscale": "log"},
):
    """
    Function to draw columnwise statistics of 2d hist
    the content values shown depending on stat_kind:
    0 -> mean + standard deviation
    1 -> median + standard deviation
    2 -> median + user specified quantiles around median (default 0.1 to 0.9)
    """

    mat_vals, xbins, ybins = get_2d_hist_from_table(x_prefix, y_prefix, table, column)
    xcent = get_bin_centers(xbins)
    rebin_x, rebin_xcent, rebin_hist, density = get_x_bin_values_with_rebinning(
        num_rebin, xbins, xcent, mat_vals, density
    )

    stats = find_columnwise_stats(rebin_hist, ybins, quantiles, density)

    sel = stats[:, 0] > 0
    if stat_kind == 1:
        y_idx = 0
        err = stats[sel, 2]
        label = "mean + std"
    if stat_kind == 2:
        y_idx = 1
        err = stats[sel, 2]
        label = "median + std"
    if stat_kind == 3:
        y_idx = 1
        err = np.zeros_like(stats[:, 3:])
        err[sel, 0] = stats[sel, 1] - stats[sel, 3]
        err[sel, 1] = stats[sel, 4] - stats[sel, 1]
        err = err[sel, :].T
        label = f"median + IRQ[{quantiles[0]:.2f},{quantiles[1]:.2f}]"

    ax.errorbar(
        x=rebin_xcent[sel],
        y=stats[sel, y_idx],
        yerr=err,
        label=f"{lbl_prefix} {label}",
    )
    if "xscale" in mpl_args:
        ax.set_xscale(mpl_args["xscale"])

    ax.legend(loc="best")

    return ax


def plot_irf_table(
    ax, table, column, prefix=None, lo_name=None, hi_name=None, label=None, **mpl_args
):
    if isinstance(column, str):
        vals = np.squeeze(table[column])
    else:
        vals = column

    if prefix:
        lo = table[f"{prefix}_LO"]
        hi = table[f"{prefix}_HI"]
    elif hi_name and lo_name:
        lo = table[lo_name]
        hi = table[hi_name]
    else:
        raise ValueError(
            "Either prefix or both `lo_name` and `hi_name` has to be given"
        )
    if not label:
        label = column

    bins = np.squeeze(join_bin_lo_hi(lo, hi))
    ax.stairs(vals, bins, label=label, **mpl_args)


def plot_hist2d_as_contour(
    ax,
    hist,
    xedges,
    yedges,
    xlabel,
    ylabel,
    levels=5,
    xscale="linear",
    yscale="linear",
    norm="log",
    cmap="Reds",
):
    if norm == "log":
        norm = LogNorm(vmax=hist.max())
    xg, yg = np.meshgrid(xedges[1:], yedges[1:])
    out = ax.contour(xg, yg, hist.T, norm=norm, cmap=cmap, levels=levels)
    ax.set(xscale=xscale, xlabel=xlabel, yscale=yscale, ylabel=ylabel)
    return out


def plot_hist2d(
    ax,
    hist,
    xedges,
    yedges,
    xlabel,
    ylabel,
    xscale="linear",
    yscale="linear",
    norm="log",
    cmap="viridis",
    colorbar=False,
):
    if isinstance(hist, u.Quantity):
        hist = hist.value

    if norm == "log":
        norm = LogNorm(vmax=hist.max())

    xg, yg = np.meshgrid(xedges, yedges)
    out = ax.pcolormesh(xg, yg, hist, norm=norm, cmap=cmap)
    ax.set(xscale=xscale, xlabel=xlabel, yscale=yscale, ylabel=ylabel)
    if colorbar:
        plt.colorbar(out)
    return out
