import argparse
from ctapipe.utils.datasets import get_path
import os
import numpy as np
from astropy import log
from ctapipe.io.files import InputFile
from calibration_pipeline import display_telescope
from ctapipe.calib.camera.calibrators import calibration_parameters, \
    calibrate_source
from matplotlib import colors, pyplot as plt
from ctapipe.image.muon.muon_reco_functions import analyze_muon_source
from ctapipe.image.muon.muon_diagnostic_plots import plot_muon_efficiency
from ctapipe.plotting.camera import CameraPlotter

from IPython import embed

"""
Example to load raw data (hessio format),
calibrate and reconstruct muon ring
parameters
"""

def display_muon_plot(event):
    print("MUON:",event[0].run_id,event[0].event_id,event[1].impact_parameter,event[1].ring_width,"mu_eff=",event[1].optical_efficiency_muon)
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

    muons = analyze_muon_source(calibrated_source, params, geom_dict, args) # Function that receive muons and make a look over the muon event    

    #fig = plt.figure(figsize=(16, 7))
    #if args.display:
    #    plt.show(block=False)
    #pp = PdfPages(args.output_path) if args.output_path is not None else None

    #colorbar = None
    #Display events before muon analysis (also do this later)
    #for cal_evt in calibrated_source:
     #   tel_id = 1 #True for muon simulations with only one tel simulated
     #   #display_telescope(evt, evt.dl0.tel[tel_id], 1, geom_dict, pp, fig)    
     #   npads = 1
    #    # Only create two pads if there is timing information extracted
    #    # from the calibration
    #    ax1 = fig.add_subplot(1, npads, 1)
    #    plotter = CameraPlotter(cal_evt,geom_dict)
    #    signals = cal_evt.dl1.tel[tel_id].pe_charge
    #    camera1 = plotter.draw_camera(tel_id,signals,ax1)
    #    
    #    cmaxmin = (max(signals) - min(signals))
    #    cmap_charge = colors.LinearSegmentedColormap.from_list(
    #        'cmap_c', [(0 / cmaxmin, 'darkblue'),
    #                   (np.abs(min(signals)) / cmaxmin, 'black'),
    #                   (2.0 * np.abs(min(signals)) / cmaxmin, 'blue'),
    #                   (2.5 * np.abs(min(signals)) / cmaxmin, 'green'),
    #                   (1, 'yellow')])
    #    camera1.pixels.set_cmap(cmap_charge)
    #    if not colorbar:
    #        camera1.add_colorbar(ax=ax1, label=" [photo-electrons]")
    #        colorbar = camera1.colorbar
    #    else:
    #        camera1.colorbar = colorbar
    #        camera1.update(True)
    #    ax1.set_title("CT {} ({}) - Mean pixel charge"
    #                  .format(tel_id, geom_dict[tel_id].cam_id))
        
    #    plt.pause(0.1)
    #    if pp is not None:
    #        pp.savefig(fig)
                    
    #
    for muon_evt in muons:
        #Test display #Flag 1 for true (wish to display)
        # display_telescope(muon_evt, muon_evt[0].tel_id, 1, geom_dict, pp, fig)    
        if muon_evt[0] is not None and muon_evt[1] is not None:
            display_muon_plot(muon_evt) 
            #Store and or Plot muon parameters here
    #if pp is not None:
    #    pp.close()

    plot_muon_efficiency(muons)

    log.info("[COMPLETE]")

if __name__ == '__main__':
    main()
