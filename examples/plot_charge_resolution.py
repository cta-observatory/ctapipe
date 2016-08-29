import argparse
import glob
import os
from astropy import log
from os.path import basename, splitext, dirname
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from math import log10, floor, ceil
import warnings

from ctapipe.utils.datasets import get_path
from ctapipe.io.files import InputFile, origin_list
from ctapipe.calib.camera.calibrators import calibration_parameters, \
    calibrate_source, calibration_arguments
from ctapipe.analysis.camera.chargeresolution import ChargeResolution


def argparsing():
    parser = argparse.ArgumentParser(
        description='Display each event in the file')
    parser.add_argument('-f', '--file', dest='input_paths',
                        default=[get_path('gamma_test.simtel.gz')], nargs='*',
                        help='path to the input files to be combined for a '
                             'single charge resolution. '
                             'Default = gamma_test.simtel.gz')
    parser.add_argument('-O', '--origin', dest='origin', action='store',
                        default='hessio', help='origin of the file: {}'
                        .format(origin_list()))
    parser.add_argument('-t', '--telescope', dest='tel', action='store',
                        type=int, default=None, nargs='*',
                        help='list of telecopes to be included. '
                             'Default = All')
    parser.add_argument('-o', '--output', dest='output_path', action='store',
                        default=None,
                        help='path to store a pdf output of the plots. '
                             'default = display on screen instead')
    parser.add_argument('--comparison', dest='comparison', action='store',
                        default=None,
                        help='output path for a True Charge versus Measured'
                             'Charge graph. Default = do not plot graph')
    parser.add_argument('-M', '--maxpe', dest='maxpe', action='store',
                        default=None, type=float,
                        help='maximum pe to calculate the charge resolution'
                             ' up to. Default = maximum pe in file')
    parser.add_argument('--maxpeplot', dest='maxpeplot', action='store',
                        default=None, type=float,
                        help='maximum pe to plot up to. Default = maxpe')
    parser.add_argument('-B', '--binning', dest='binning', action='store',
                        default="log",
                        help='binning of the charge resoltion graph: '
                             '"none" = no binning, "normal" = bin, '
                             '"log" = bin in log space. Default = log')
    parser.add_argument('--normalx', dest='normalx', action='store_true',
                        default=False,
                        help='Use a normal x axis instead of the defualt log'
                             'axis')
    parser.add_argument('--normaly', dest='normaly', action='store_true',
                        default=False,
                        help='Use a normal y axis instead of the defualt log'
                             'axis')
    parser.add_argument('-E', '--example', dest='example', action='store',
                        default=None,
                        help='output path for an example runcard text file')
    parser.add_argument('-R', '--runcard', dest='runcard', action='store',
                        default=None,
                        help='path to a runcard text file with arguments that '
                             'override command line arguments. This run card '
                             'can allow complex combinations of files and '
                             'calibrations to compare the charge resolution '
                             'against each other.')

    calibration_arguments(parser)

    logger_detail = parser.add_mutually_exclusive_group()
    logger_detail.add_argument('-q', '--quiet', dest='quiet',
                               action='store_true', default=False,
                               help='Quiet mode')
    logger_detail.add_argument('-v', '--verbose', dest='verbose',
                               action='store_true', default=False,
                               help='Verbose mode')
    logger_detail.add_argument('-d', '--debug', dest='debug',
                               action='store_true', default=False,
                               help='Debug mode')

    args = parser.parse_args()

    if args.quiet:
        log.setLevel(40)
    if args.verbose:
        log.setLevel(20)
    if args.debug:
        log.setLevel(10)

    # if args.example is not None:
    #     example_runcard(args.example)
    if args.runcard is None:
        name = splitext(basename(args.input_paths[0]))[0]
        args.chargeres_names = [name]
        chargeres_args = {name: args}
        plot_args = {name: args}
    else:
        chargeres_args = {}
        plot_args = {}
        chargeres_cmdline = {}
        plot_cmdline = {}
        plot_chargeres_names = {}
        current = None
        name = ''
        runcard = open(args.runcard)
        for line in runcard:
            if line.strip() and not line.startswith('#'):
                argument = line.split()[0]
                if argument == '[chargeres]':
                    name = line.split()[1:][0]
                    chargeres_cmdline[name] = []
                    current = chargeres_cmdline[name]
                    continue
                elif argument == '[plot]':
                    name = line.split()[1:][0]
                    plot_cmdline[name] = []
                    current = plot_cmdline[name]
                    continue
                elif argument == '[chargeres_names]':
                    plot_chargeres_names[name] = line.split()[1:]
                    continue
                current.extend(line.split())

        for name, cmdline in chargeres_cmdline.items():
            chargeres_args[name] = parser.parse_args(cmdline)
        for name, cmdline in plot_cmdline.items():
            plot_args[name] = parser.parse_args(cmdline)
            plot_args[name].chargeres_names = plot_chargeres_names[name]

    # Check all chargeres_names exist
    for plot_name, args in plot_args.items():
        for name in args.chargeres_names:
            try:
                if name not in chargeres_args:
                    raise IndexError
            except IndexError:
                log.exception("[chargeres_names] For plot: {}, no chargeres "
                              "has the name: {}".format(plot_name, name))

    # Check all binning is valid
    possible = ['none', 'normal', 'log']
    test = [args.binning in possible for args in plot_args.values()]
    try:
        if not all(test):
            raise IndexError
    except IndexError:
        log.exception("[binning] Only options are: {}".format(possible))
        raise

    return chargeres_args, plot_args


def calculate_charge_resolutions(chargeres_args):
    run = 0
    num_runs = len(chargeres_args)
    chargeres_dict = {}
    for name, args in chargeres_args.items():
        run += 1
        log.info("[run] Calculating charge resolution {}/{}"
                 .format(run, num_runs))

        # Print event/args values
        log.info("[files] {}".format(args.input_paths))
        telescopes = "All" if args.tel is None else args.tel
        log.info("[origin] {}".format(args.origin))
        log.info("[telescopes] {}".format(telescopes))

        # Obtain InputFiles (to check files exist)
        input_file_list = []
        for filepath in args.input_paths:
            tilde_filepath = os.path.expanduser(filepath)
            for globfile in glob.glob(tilde_filepath):
                input_file_list.append(InputFile(globfile, args.origin))

        # Find maxpe
        maxpe = args.maxpe
        if maxpe is None:
            maxpe = 0
            for input_file in input_file_list:
                file_maxpe = input_file.find_max_true_npe(args.tel, 100)
                if file_maxpe > maxpe:
                    maxpe = file_maxpe
        log.info("[maxpe] {}".format(maxpe))

        params = calibration_parameters(args)

        chargeres = ChargeResolution(maxpe)

        for input_file in input_file_list:
            source = input_file.read(100)
            calibrated_source = calibrate_source(source, params)
            chargeres.add_source(calibrated_source, args.tel)

        # Plot comparison graphs
        if args.comparison is not None:
            hist = chargeres.variation_hist
            hist[hist == 0.0] = np.nan
            xedges = chargeres.variation_xedges
            yedges = chargeres.variation_yedges

            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111)
            ax.set_title(name)
            x, y = np.meshgrid(xedges, yedges)
            x = np.power(10, x)
            y = np.power(10, y)
            hist_mask = np.ma.masked_where(np.isnan(hist), hist)
            im = ax.pcolormesh(x, y, hist_mask, norm=LogNorm(),
                               cmap=plt.cm.viridis)
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

            args.comparison = os.path.expanduser(args.comparison)
            output_dir = dirname(args.comparison)
            if not os.path.exists(output_dir):
                log.info("[output] Creating directory: {}".format(output_dir))
                os.makedirs(output_dir)
            log.info("[output] {}".format(args.comparison))
            warnings.filterwarnings("ignore", module="matplotlib")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fig.savefig(args.comparison, bbox_inches='tight')

        chargeres_dict[name] = chargeres

    log.info('[run] Charge resolution calculations finished')

    return chargeres_dict


def plot_charge_resolutions(chargeres_dict, plot_args):
    run = 0
    for name, args in plot_args.items():
        run += 1
        log.info('[run] Plotting block: {}/{}'.format(run, len(plot_args)))

        handles = []
        labels = []

        fig = plt.figure(figsize=(20, 8))
        ax_l = fig.add_subplot(121)
        ax_r = fig.add_subplot(122)

        fig.subplots_adjust(left=0.05, right=0.95, wspace=0.6)
        fig.suptitle(name)

        max_x_charge = 0
        for chargeres_name in args.chargeres_names:
            chargeres = chargeres_dict[chargeres_name]
            if args.binning == 'none':
                x_charge, res, error, scaled_res, scaled_error = \
                    chargeres.get_charge_resolution()
            elif args.binning == 'normal':
                x_charge, res, error, scaled_res, scaled_error = \
                    chargeres.get_binned_charge_resolution(False)
            else:
                x_charge, res, error, scaled_res, scaled_error = \
                    chargeres.get_binned_charge_resolution(True)

            eb_l = ax_l.errorbar(x_charge, res, yerr=error, marker='x',
                                 linestyle="None")
            ax_r.errorbar(x_charge, scaled_res, yerr=scaled_error,
                          marker='x', linestyle="None")

            handles.append(eb_l)
            labels.append(chargeres_name)

            current_max = x_charge[np.invert(np.isnan(x_charge))].max()
            if max_x_charge < current_max:
                max_x_charge = current_max

        # Get requirement and goal curves
        x = np.logspace(log10(0.9), log10(max_x_charge*1.1), 100)
        requirement = ChargeResolution.requirement(x)
        goal = ChargeResolution.goal(x)
        poisson = ChargeResolution.poisson(x)

        r_p, = ax_l.plot(x, requirement, 'r', ls='--')
        g_p, = ax_l.plot(x, goal, 'g', ls='--')
        p_p, = ax_l.plot(x, poisson, c='0.75', ls='--')
        ax_r.plot(x, requirement / goal, 'r')
        ax_r.plot(x, goal / goal, 'g')
        ax_r.plot(x, poisson / goal, c='0.75', ls='--')

        handles.append(r_p)
        labels.append("Requirement")
        handles.append(g_p)
        labels.append("Goal")
        handles.append(p_p)
        labels.append("Poisson")

        if not args.normalx:
            ax_l.set_xscale('log')
            ax_r.set_xscale('log')
        if not args.normaly:
            ax_l.set_yscale('log')

        ax_l.set_xlabel(r'True Charge $Q_T$ (p.e.)')
        ax_l.set_ylabel('Charge Resolution')
        ax_r.set_xlabel(r'True Charge $Q_T$ (p.e.)')
        ax_r.set_ylabel('Charge Resolution/Goal')

        ax_l.legend(handles, labels, bbox_to_anchor=(1.02, 1.), loc=2,
                    borderaxespad=0., fontsize=10)

        ax_l.set_xlim(0.9, max_x_charge*1.1)
        ax_r.set_xlim(0.9, max_x_charge*1.1)
        if max_x_charge > 2000:
            ax_r.set_xlim(0.9, 2000*1.1)
        if args.maxpeplot is not None:
            ax_l.set_xlim(0.9, args.maxpeplot)
            ax_r.set_xlim(0.9, args.maxpeplot)

        warnings.filterwarnings("ignore", module="matplotlib")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if args.output_path is None:
                plt.show()
            else:
                args.output_path = os.path.expanduser(args.output_path)
                output_dir = dirname(args.output_path)
                if not os.path.exists(output_dir):
                    log.info(
                        "[output] Creating directory: {}".format(output_dir))
                    os.makedirs(output_dir)
                log.info("[output] {}".format(args.output_path))
                fig.savefig(args.output_path, bbox_inches='tight')


def main():
    script = os.path.splitext(os.path.basename(__file__))[0]
    log.info("[SCRIPT] {}".format(script))

    chargeres_args, plot_args = argparsing()
    chargeres_dict = calculate_charge_resolutions(chargeres_args)
    plot_charge_resolutions(chargeres_dict, plot_args)

    log.info("[COMPLETE]")


if __name__ == '__main__':
    main()
