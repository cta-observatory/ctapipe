import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from astropy.visualization import quantity_support
from matplotlib.colors import LogNorm
from pyirf.binning import join_bin_lo_hi

quantity_support()


def plot_2D_irf_table(
    ax, table, column, x_prefix, y_prefix, x_label=None, y_label=None, **mpl_args
):
    x_lo_name, x_hi_name = f"{x_prefix}_LO", f"{x_prefix}_HI"
    y_lo_name, y_hi_name = f"{y_prefix}_LO", f"{y_prefix}_HI"

    xbins = np.hstack((table[x_lo_name][0], table[x_hi_name][0][-1]))

    ybins = np.hstack((table[y_lo_name][0], table[y_hi_name][0][-1]))
    if not x_label:
        x_label = x_prefix
    if not y_label:
        y_label = y_prefix
    if isinstance(column, str):
        mat_vals = np.squeeze(table[column])
    else:
        mat_vals = column

    plot = plot_hist2D(
        ax, mat_vals, xbins, ybins, xlabel=x_label, ylabel=y_label, **mpl_args
    )
    plt.colorbar(plot)
    return ax


def rebin_x_2D_hist(hist, xbins, x_cent, num_bins_merge=3):
    num_y, num_x = hist.shape
    if (num_x) % num_bins_merge == 0:
        rebin_x = xbins[::num_bins_merge]
        rebin_xcent = x_cent.reshape(-1, num_bins_merge).mean(axis=1)
        rebin_hist = hist.reshape(num_y, -1, num_bins_merge).sum(axis=2)
        return rebin_x, rebin_xcent, rebin_hist
    else:
        raise ValueError(
            f"Could not merge {num_bins_merge} along axis of dimension {num_x}"
        )


def find_columnwise_stats(table, col_bins, percentiles, density=False):
    tab = np.squeeze(table)
    out = np.ones((tab.shape[1], 5)) * -1
    # This loop over the columns seems unavoidable,
    # so having a reasonable number of bins in that
    # direction is good
    for idx, col in enumerate(tab.T):
        if (col > 0).sum() == 0:
            continue
        col_est = st.rv_histogram((col, col_bins), density=density)
        out[idx, 0] = col_est.mean()
        out[idx, 1] = col_est.median()
        out[idx, 2] = col_est.std()
        out[idx, 3] = col_est.ppf(percentiles[0])
        out[idx, 4] = col_est.ppf(percentiles[1])
    return out


def plot_2D_table_with_col_stats(
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
    """Function to draw 2d histogram along with columnwise statistics
    the plotted errorbars shown depending on stat_kind:
        0 -> mean + standard deviation
        1 -> median + standard deviation
        2 -> median + user specified quantiles around median (default 0.1 to 0.9)
    """
    x_lo_name, x_hi_name = f"{x_prefix}_LO", f"{x_prefix}_HI"
    y_lo_name, y_hi_name = f"{y_prefix}_LO", f"{y_prefix}_HI"

    xbins = np.hstack((table[x_lo_name][0], table[x_hi_name][0][-1]))

    ybins = np.hstack((table[y_lo_name][0], table[y_hi_name][0][-1]))

    xcent = np.convolve(
        [0.5, 0.5], np.hstack((table[x_lo_name][0], table[x_hi_name][0][-1])), "valid"
    )
    if not x_label:
        x_label = x_prefix
    if not y_label:
        y_label = y_prefix
    if isinstance(column, str):
        mat_vals = np.squeeze(table[column])
    else:
        mat_vals = column

    if num_rebin > 1:
        rebin_x, rebin_xcent, rebin_hist = rebin_x_2D_hist(
            mat_vals, xbins, xcent, num_bins_merge=num_rebin
        )
        density = False
    else:
        rebin_x, rebin_xcent, rebin_hist = xbins, xcent, mat_vals

    stats = find_columnwise_stats(rebin_hist, ybins, quantiles, density)
    plot = plot_hist2D(
        ax,
        rebin_hist,
        rebin_x,
        ybins,
        xlabel=x_label,
        ylabel=y_label,
        **mpl_args["histo"],
    )
    plt.colorbar(plot)

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
        label=label,
        **mpl_args["stats"],
    )
    ax.legend(loc="best")

    return ax


def plot_2D_table_col_stats(
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
    """Function to draw columnwise statistics of 2D hist
    the content values shown depending on stat_kind:
        0 -> mean + standard deviation
        1 -> median + standard deviation
        2 -> median + user specified quantiles around median (default 0.1 to 0.9)
    """
    x_lo_name, x_hi_name = f"{x_prefix}_LO", f"{x_prefix}_HI"
    y_lo_name, y_hi_name = f"{y_prefix}_LO", f"{y_prefix}_HI"

    xbins = np.hstack((table[x_lo_name][0], table[x_hi_name][0][-1]))

    ybins = np.hstack((table[y_lo_name][0], table[y_hi_name][0][-1]))

    xcent = np.convolve(
        [0.5, 0.5], np.hstack((table[x_lo_name][0], table[x_hi_name][0][-1])), "valid"
    )
    if not x_label:
        x_label = x_prefix
    if not y_label:
        y_label = y_prefix
    if isinstance(column, str):
        mat_vals = np.squeeze(table[column])
    else:
        mat_vals = column

    if num_rebin > 1:
        rebin_x, rebin_xcent, rebin_hist = rebin_x_2D_hist(
            mat_vals, xbins, xcent, num_bins_merge=num_rebin
        )
        density = False
    else:
        rebin_xcent, rebin_hist = xcent, mat_vals

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


def plot_hist2D_as_contour(
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


def plot_hist2D(
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
):
    if isinstance(hist, u.Quantity):
        hist = hist.value

    if norm == "log":
        norm = LogNorm(vmax=hist.max())

    xg, yg = np.meshgrid(xedges, yedges)
    out = ax.pcolormesh(xg, yg, hist, norm=norm, cmap=cmap)
    ax.set(xscale=xscale, xlabel=xlabel, yscale=yscale, ylabel=ylabel)
    return out
