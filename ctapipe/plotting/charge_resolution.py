import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from matplotlib import pyplot as plt
import os
from ctapipe.core import Component, Provenance
from traitlets import Unicode, Int


def sum_errors(array):
    """
    Simple sum of squares to combine errors

    Parameters
    ----------
    array : ndarray

    Returns
    -------
    float
    """
    return np.sqrt(np.sum(np.power(array, 2)) / array.size)


def bin_dataframe(df, n_bins):
    """
    Assign a "bin" column to the dataframe to indicate which bin the true
    charges belong to.

    Bins are assigned in log space.

    Parameters
    ----------
    df : pd.DataFrame
    n_bins : int
        Number of bins to allow in range

    Returns
    -------
    pd.DataFrame
    """
    true = df['true'].values
    min_ = true.min()
    max_ = true.max()
    bins = np.geomspace(min_, max_, n_bins)
    bins = np.append(bins, 10**(np.log10(bins[-1]) +
                                np.diff(np.log10(bins))[0]))
    df['bin'] = np.digitize(true, bins, right=True) - 1

    return df


class ChargeResolutionPlotter(Component):

    output_path = Unicode(
        '', help='Output path to save the plot.'
    ).tag(config=True)

    n_bins = Int(
        40,
        help='Number of bins for collecting true charges and combining '
             'their resolution'
    ).tag(config=True)

    def __init__(self, config=None, tool=None, **kwargs):
        """
        Plots the charge resolution HDF5 file obtained from
        `ctapipe.analysis.camera.charge_resolution`.

        Also contains the equation from which the charge resolution
        requirement is defined, which can be plotted alongside the charge
        resolution curve.

        `ctapipe.tools.plot_charge_resolution` demonstrated the use of this
        component.

        Parameters
        ----------
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        tool : ctapipe.core.Tool
            Tool executable that is calling this component.
            Passes the correct logger to the component.
            Set to None if no Tool to pass.
        kwargs
        """
        super().__init__(config=config, tool=tool, **kwargs)
        self._df_pixel = None
        self._df_camera = None

        self.fig = plt.figure(figsize=(12, 7.42))
        self.ax = self.fig.add_subplot(111)

        self.ax.set_xlabel("True Charge (p.e.)")
        self.ax.set_ylabel(
            r"Fractional Charge Resolution $\frac{{\sigma_Q}}{{Q}}$"
        )

        if not self.output_path:
            raise ValueError("Output path must be specified")

    def _set_file(self, path):
        """
        Reads the charge resolution DataFrames from the file and assigns it to
        this class ready for plotting.

        Parameters
        ----------
        path : str
            Path to the charge resolution HDF5 file
        """
        with pd.HDFStore(path, 'r') as store:
            self._df_pixel = store['charge_resolution_pixel']
            self._df_camera = store['charge_resolution_camera']

    def _plot(self, x, y, **kwargs):
        """
        Plot the given points onto the figure

        Parameters
        ----------
        x : ndarray
        y : ndarray
        xerr : ndarray
        yerr : ndarray
        label : str
        """
        defaults = dict(
            mew=1, capsize=1, elinewidth=0.5, markersize=2,
            linewidth=0.5, fmt='.'
        )
        kwargs = {**defaults, **kwargs}
        (_, caps, _) = self.ax.errorbar(x, y, **kwargs)
        for cap in caps:
            cap.set_markeredgewidth(0.5)

    def plot_average(self, path, label='', **kwargs):
        """
        Plot the average and standard deviation of the charge resolution
        across the pixels of the camera.

        Parameters
        ----------
        path : str
            Path to the charge resolution HDF5 file
        label : str
            Label for the figure's legend
        kwargs
        """
        self._set_file(path)
        df_binned = bin_dataframe(self._df_pixel, self.n_bins)
        agg = {'charge_resolution': ['mean', 'std'], 'true': 'mean'}
        df_agg = df_binned.groupby(['bin']).agg(agg)
        x = df_agg['true']['mean'].values
        y = df_agg['charge_resolution']['mean'].values
        yerr = df_agg['charge_resolution']['std'].values
        self._plot(x, y, yerr=yerr, label=label, **kwargs)

    def plot_pixel(self, path, pixel, label='', **kwargs):
        """
        Plot a single pixel's charge resolution.

        The yerr represents the amount of entries.

        Parameters
        ----------
        path : str
            Path to the charge resolution HDF5 file
        pixel : int
            Pixel index to plot
        label : str
            Label for the figure's legend
        kwargs
        """
        self._set_file(path)
        df_p = self._df_pixel.loc[self._df_pixel['pixel'] == pixel]
        df_binned = bin_dataframe(df_p, self.n_bins)
        agg = {'charge_resolution': 'mean', 'true': 'mean', 'n': 'sum'}
        df_agg = df_binned.groupby(['bin']).agg(agg)
        x = df_agg['true'].values
        y = df_agg['charge_resolution'].values
        yerr = 1 / np.sqrt(df_agg['n'].values)
        self._plot(x, y, yerr=yerr, label=label, **kwargs)

    def plot_camera(self, path, label='', **kwargs):
        """
        Plot the charge resolution for the entire camera.

        The yerr represents the amount of entries.

        Parameters
        ----------
        path : str
            Path to the charge resolution HDF5 file
        label : str
            Label for the figure's legend
        kwargs
        """
        self._set_file(path)
        df_binned = bin_dataframe(self._df_camera, self.n_bins)
        agg = {'charge_resolution': 'mean', 'true': 'mean', 'n': 'sum'}
        df_agg = df_binned.groupby(['bin']).agg(agg)
        x = df_agg['true'].values
        y = df_agg['charge_resolution'].values
        yerr = 1 / np.sqrt(df_agg['n'].values)
        self._plot(x, y, yerr=yerr, label=label, **kwargs)

    def _finish(self):
        """
        Perform the finishing touches to the figure before saving.
        """
        self.ax.set_xscale('log')
        self.ax.get_xaxis().set_major_formatter(
            FuncFormatter(lambda x, _: f'{x:g}'))
        self.ax.set_yscale('log')
        self.ax.get_yaxis().set_major_formatter(
            FuncFormatter(lambda y, _: f'{y:g}'))

    def save(self):
        """
        Save the figure to the path defined by the `output_path` trait
        """
        self._finish()

        output_dir = os.path.dirname(self.output_path)
        if not os.path.exists(output_dir):
            print(f"Creating directory: {output_dir}")
            os.makedirs(output_dir)

        self.fig.savefig(self.output_path, bbox_inches='tight')
        print(f"Figure saved to: {self.output_path}")
        Provenance().add_output_file(self.output_path)

        plt.close(self.fig)

    @staticmethod
    def limit_curves(q, nsb, t_w, n_e, sigma_g, enf):
        """
        Equation for calculating the Goal and Requirement curves, as defined
        in SCI-MC/121113.
        https://portal.cta-observatory.org/recordscentre/Records/SCI/
        SCI-MC/measurment_errors_system_performance_1YQCBC.pdf

        Parameters
        ----------
        q : ndarray
            Number of photoeletrons (variable).
        nsb : float
            Number of NSB photons.
        t_w : float
            Effective signal readout window size.
        n_e : float
            Electronic noise
        sigma_g : float
            Multiplicative errors of the gain.
        enf : float
            Excess noise factor.
        """
        sigma_0 = np.sqrt(nsb * t_w + n_e**2)
        sigma_enf = 1 + enf
        sigma_q = np.sqrt(sigma_0**2 + sigma_enf**2 * q + sigma_g**2 * q**2)
        return sigma_q / q

    @staticmethod
    def requirement(q):
        """
        CTA requirement curve.

        Parameters
        ----------
        q : ndarray
            Number of photoeletrons
        """
        nsb = 0.125
        t_w = 15
        n_e = 0.87
        sigma_g = 0.1
        enf = 0.2
        defined_npe = 1000
        lc = __class__.limit_curves
        requirement = lc(q, nsb, t_w, n_e, sigma_g, enf)
        requirement[q > defined_npe] = np.nan

        return requirement

    @staticmethod
    def poisson(q):
        """
        Poisson limit curve.

        Parameters
        ----------
        q : ndarray
            Number of photoeletrons
        """
        poisson = np.sqrt(q) / q
        return poisson

    def plot_requirement(self, q):
        """
        Plot the CTA requirement curve onto the figure.

        Parameters
        ----------
        q : ndarray
            Charges to evaluate the requirement curve at
        """
        req = self.requirement(q)
        self.ax.plot(q, req, '--', color='black', label="Requirement")

    def plot_poisson(self, q):
        """
        Plot the Poisson limit curve onto the figure.

        Parameters
        ----------
        q : ndarray
            Charges to evaluate the limit at
        """
        poisson = self.poisson(q)
        self.ax.plot(q, poisson, '--', color='grey', label="Poisson")


class ChargeResolutionWRRPlotter(ChargeResolutionPlotter):
    def __init__(self, config=None, tool=None, **kwargs):
        """
        Plots the charge resolution similarly to ChargeResolutionPlotter, with
        the values divided by the CTA requirement.

        Parameters
        ----------
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        tool : ctapipe.core.Tool
            Tool executable that is calling this component.
            Passes the correct logger to the component.
            Set to None if no Tool to pass.
        kwargs
        """
        super().__init__(config=config, tool=tool, **kwargs)
        self.ax.set_xlabel("True Charge (p.e.)")
        self.ax.set_ylabel(r"$\frac{{\sigma_Q}}{{Q}}$ / Requirement")

    def _plot(self, x, y, **kwargs):
        y = y / self.requirement(x)
        if 'yerr' in kwargs:
            kwargs['yerr'] /= self.requirement(x)
        super()._plot(x, y, **kwargs)

    def plot_requirement(self, q):
        req = self.requirement(q)
        req /= self.requirement(q)
        self.ax.plot(q, req, '--', color='black', label="Requirement")

    def plot_poisson(self, q):
        poisson = self.poisson(q)
        poisson /= self.requirement(q)
        self.ax.plot(q, poisson, '--', color='grey', label="Poisson")

    def _finish(self):
        self.ax.set_xscale('log')
        self.ax.get_xaxis().set_major_formatter(
            FuncFormatter(lambda x, _: f'{x:g}'))
