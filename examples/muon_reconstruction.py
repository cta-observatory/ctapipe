# from calibration_pipeline import display_telescope

import argparse
import os

from astropy import log
from astropy.table import Table

from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
# from calibration_pipeline import display_telescope
from ctapipe.calib.camera.r1 import HessioR1Calibrator
from ctapipe.image.muon.muon_diagnostic_plots import plot_muon_efficiency, \
    plot_muon_event
from ctapipe.image.muon.muon_reco_functions import analyze_muon_event
from ctapipe.io.hessio import hessio_event_source
from ctapipe.utils import get_dataset

"""
Example to load raw data (hessio format),
calibrate and reconstruct muon ring
parameters
"""


def display_muon_plot(event):
    for tid in event['TelIds']:
        idx = event['TelIds'].index(tid)
        if event['MuonIntensityParams'][idx]:
            print("MUON:", event['MuonRingParams'][idx].run_id, event['MuonRingParams'][idx].event_id,
                  event['MuonIntensityParams'][idx].impact_parameter, event['MuonIntensityParams'][idx].ring_width, "mu_eff=",
                  event['MuonIntensityParams'][idx].optical_efficiency_muon)
    pass


def main():
    script = os.path.splitext(os.path.basename(__file__))[0]
    log.info("[SCRIPT] {}".format(script))

    parser = argparse.ArgumentParser(
        description='Display each event in the file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--file', dest='input_path', action='store',
                        default=get_dataset('gamma_test.simtel.gz'),
                        help='path to the input file')

    parser.add_argument('-O', '--origin', dest='origin', action='store',
                        # was .origin_list()
                        default='hessio', help='origin of the file')
    parser.add_argument('-D', dest='display', action='store_true',
                        default=False, help='display the camera events')
    parser.add_argument('-t', '--telescope', dest='tel_FIXME', action='store',
                        type=int, default=None,
                        help='telecope to view. Default = All')
    parser.add_argument('--pdf', dest='output_path', action='store',
                        default=None,
                        help='path to store a pdf output of the plots')
    parser.add_argument('--calib-help', dest='calib_help', action='store_true',
                        default=False,
                        help='display the arguments used for the camera '
                             'calibration')

    args, excess_args = parser.parse_known_args()

    # params, unknown_args = calibration_parameters(excess_args,
    #                                              args.origin,
    #                                              args.calib_help)
    r1 = HessioR1Calibrator(None, None)
    dl0 = CameraDL0Reducer(None, None)
    dl1 = CameraDL1Calibrator(None, None)

    log.debug("[file] Reading file")
    # input_file = InputFile(args.input_path, args.origin)
    # source = input_file.read()
    source = hessio_event_source(args.input_path)


    muoneff = []
    impactp = []
    ringwidth = []
    plot_dict = {'MuonEff': muoneff, 'ImpactP': impactp, 'RingWidth': ringwidth}

    numev = 0

    for event in source:
        print("Event Number", numev)
        r1.calibrate(event)
        dl0.reduce(event)
        dl1.calibrate(event)

        muon_evt = analyze_muon_event(event)

        numev += 1
        # Test display #Flag 1 for true (wish to display)
        # plot_muon_event(event,muon_evt)
        # display_telescope(muon_evt, muon_evt[0].tel_id, 1, geom_dict, pp, fig)
        #if muon_evt[0] is not None and muon_evt[1] is not None:
        if not muon_evt['MuonIntensityParams']: #No telescopes contained a good muon
            continue
        else:
            plot_muon_event(event, muon_evt, args)

            for tid in muon_evt['TelIds']:
                idx = muon_evt['TelIds'].index(tid)
                if not muon_evt['MuonIntensityParams'][idx]:
                    continue

                plot_dict['MuonEff'].append(muon_evt['MuonIntensityParams'][idx].optical_efficiency_muon)
                plot_dict['ImpactP'].append(muon_evt['MuonIntensityParams'][idx].impact_parameter.value)
                plot_dict['RingWidth'].append(muon_evt['MuonIntensityParams'][idx].ring_width.value)

            display_muon_plot(muon_evt)
            # Store and or Plot muon parameters here

            # if numev > 50: #for testing purposes - kill early
            #    break

    t = Table([muoneff, impactp, ringwidth],
              names=('MuonEff', 'ImpactP', 'RingWidth'),
              meta={'name': 'muon analysis results'})
    t['ImpactP'].unit = 'm'
    t['RingWidth'].unit = 'deg'
    #    print('plotdict',plot_dict)

    #t.write(str(args.output_path) + '_muontable.fits', overwrite=True)  # NEED
    # this to overwrite

    #plot_muon_efficiency(args.output_path)

    log.info("[COMPLETE]")


if __name__ == '__main__':
    main()
