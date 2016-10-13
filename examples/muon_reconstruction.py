import argparse
from ctapipe.utils.datasets import get_path
import os
from astropy import log
from ctapipe.io.files import InputFile
from calibration_pipeline import display_telescope
from ctapipe.calib.camera.calibrators import calibration_parameters, \
    calibrate_source
from matplotlib import pyplot as plt
from ctapipe.image.muon.muon_reco_functions import analyze_muon_source

"""
Example to load raw data (hessio format),
calibrate and reconstruct muon ring
parameters
"""

def display_muon_plot(event):
    print("MUON:",event.run_id,event.event_id,event.impact_parameter,event.width,event.efficiency)
    pass

def main():
    script = os.path.splitext(os.path.basename(__file__))[0]
    log.info("[SCRIPT] {}".format(script))

    parser = argparse.ArgumentParser(
        description='Display each event in the file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--file', dest='input_path', action='store',
                        default=get_path('gamma_test.simtel.gz'),
                        help='path to the input file')

    parser.add_argument('-O', '--origin', dest='origin', action='store',
                        choices=InputFile.origin_list(),
                        default='hessio', help='origin of the file')
    parser.add_argument('-D', dest='display', action='store_true',
                        default=False, help='display the camera events')
    parser.add_argument('-t', '--telescope', dest='tel', action='store',
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

    params, unknown_args = calibration_parameters(excess_args,
                                                  args.origin,
                                                  args.calib_help)

    log.debug("[file] Reading file")
    input_file = InputFile(args.input_path, args.origin)
    source = input_file.read()

    geom_dict = {}

    calibrated_source = calibrate_source(source, params, geom_dict)

    muons = analyze_muon_source(calibrated_source, params, geom_dict) # Function that receive muons and make a look iver the muon event
    
    fig = plt.figure(figsize=(16, 7))
    if args.display:
        plt.show(block=False)
    pp = PdfPages(args.output_path) if args.output_path is not None else None
    for muon_evt in muons:
            #display_telescope(cal_evt, tel_id, args.display, geom_dict, pp, fig)
            #display_muon_plot(muon_evt) 
            print("M:",muon_evt.run_id,muon_evt.event_id)
            
            print("CAL:",cal_evt.dl1.run_id,cal_evt.dl1.event_id)
            
    if pp is not None:
        pp.close()

    log.info("[COMPLETE]")

if __name__ == '__main__':
    main()
