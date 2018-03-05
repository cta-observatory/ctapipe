from os.path import dirname, exists, splitext, basename
from os import makedirs
from math import ceil, floor
from matplotlib import pyplot as plt
from math import log10
import warnings
import numpy as np
from matplotlib.colors import LogNorm
from traitlets import Dict, List, Unicode
from ctapipe.core import Tool, Component
from ctapipe.analysis.camera.chargeresolution import ChargeResolutionCalculator


class ChargeResolutionVariationPlotter(Component):
    output_path = Unicode(None, allow_none=True,
                          help='Output path to save the '
                               'plot.').tag(config=True)

    def __init__(self, config=None, tool=None, **kwargs):
        """
        Calculator of charge resolution.

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
        reductor : ctapipe.calib.camera.reductors.Reductor
            The reductor to use to reduce the waveforms in the event.
            By default no data volume reduction is applied, and the dl0 samples
            will equal the r1 samples.
        kwargs
        """
        super().__init__(config=config, parent=tool, **kwargs)

        try:
            if self.output_path is None:
                raise ValueError
        except ValueError:
            self.log.exception('Please specify an output path')
            raise

        self.fig = plt.figure(figsize=(20, 8))
        self.ax_l = self.fig.add_subplot(121)
        self.ax_r = self.fig.add_subplot(122)

        self.fig.subplots_adjust(left=0.05, right=0.95, wspace=0.6)

        self.legend_handles = []
        self.legend_labels = []

    def plot_hist(self, hist, xedges, yedges):

        hist[hist == 0.0] = np.nan

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111)
        ax.set_title(splitext(basename(self.output_path))[0])
        x, y = np.meshgrid(xedges, yedges)
        x = np.power(10, x)
        y = np.power(10, y)
        hist_mask = np.ma.masked_where(np.isnan(hist), hist)
        im = ax.pcolormesh(x, y, hist_mask, norm=LogNorm())
        cb = plt.colorbar(im)
        ax.set_aspect('equal')
        ax.grid()
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'True Charge $Q_T$ (p.e.)')
        ax.set_ylabel(r'Measured Charge $Q_M$ (p.e.)')
        cb.ax.set_ylabel("Count")

        line = np.linspace(*ax.get_xlim(), 100)
        ax.plot(line, line, c='0.75', ls='--')

        # Add minor ticks
        lmin = floor(log10(hist_mask.min()))
        lmax = ceil(log10(hist_mask.max()))
        logticks = np.tile(np.arange(lmin, 10), lmax) * (
            np.power(10, np.arange(lmax * 10) // 10))
        logticks = im.norm(logticks[(logticks != 0) &
                                    (logticks >= hist_mask.min()) &
                                    (logticks <= hist_mask.max())])
        cb.ax.yaxis.set_ticks(logticks, minor=True)
        cb.ax.yaxis.set_tick_params(which='minor', length=5)
        cb.ax.tick_params(length=10)

        output_dir = dirname(self.output_path)
        if not exists(output_dir):
            self.log.info("[output] Creating directory: {}".format(output_dir))
            makedirs(output_dir)
        self.log.info("[output] {}".format(self.output_path))
        warnings.filterwarnings("ignore", module="matplotlib")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.savefig(self.output_path, bbox_inches='tight')


class ChargeResolutionVariationViewer(Tool):
    name = "ChargeResolutionVariationViewer"
    description = "Plot the charge resolution from " \
                  "ChargeResolutionCalculator objects restored via " \
                  "pickled dictionaries."

    input_path = Unicode(None, allow_none=True,
                         help='Path to the hdf5 file produced from'
                              'ChargeResolutionCalculator.save()'
                              '').tag(config=True)

    aliases = Dict(dict(f='ChargeResolutionVariationViewer.input_path',
                        O='ChargeResolutionVariationPlotter.output_path',
                        ))
    classes = List([ChargeResolutionCalculator,
                    ChargeResolutionVariationPlotter
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calculator = None
        self.plotter = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        self.calculator = ChargeResolutionCalculator(**kwargs)
        self.plotter = ChargeResolutionVariationPlotter(**kwargs)

    def start(self):
        self.calculator.load(self.input_path)

    def finish(self):
        hist = self.calculator.variation_hist
        xedges = self.calculator.variation_xedges
        yedges = self.calculator.variation_yedges
        self.plotter.plot_hist(hist, xedges, yedges)


def main():
    exe = ChargeResolutionVariationViewer()
    exe.run()


if __name__ == '__main__':
    main()
