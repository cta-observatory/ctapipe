import numpy as np
import scipy.stats as st


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


def rebin_x_2d_hist(hist, xbins, x_cent, num_bins_merge=3):
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


def get_2d_hist_from_table(x_prefix, y_prefix, table, column):
    x_lo_name, x_hi_name = f"{x_prefix}_LO", f"{x_prefix}_HI"
    y_lo_name, y_hi_name = f"{y_prefix}_LO", f"{y_prefix}_HI"

    xbins = np.hstack((table[x_lo_name][0], table[x_hi_name][0][-1]))
    ybins = np.hstack((table[y_lo_name][0], table[y_hi_name][0][-1]))

    if isinstance(column, str):
        mat_vals = np.squeeze(table[column])
    else:
        mat_vals = column

    return mat_vals, xbins, ybins


def get_bin_centers(bins):
    return np.convolve(bins, [0.5, 0.5], mode="valid")


def get_x_bin_values_with_rebinning(num_rebin, xbins, xcent, mat_vals, density):
    if num_rebin > 1:
        rebin_x, rebin_xcent, rebin_hist = rebin_x_2d_hist(
            mat_vals, xbins, xcent, num_bins_merge=num_rebin
        )
        density = False
    else:
        rebin_x, rebin_xcent, rebin_hist = xbins, xcent, mat_vals

    return rebin_x, rebin_xcent, rebin_hist, density
