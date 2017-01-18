import argparse
from ctapipe.utils.datasets import get_path
import os
import numpy as np
from astropy import log
from astropy.table import Table
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
                         #was .origin_list()
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
                    
    #

    plot_dict = {}
    muoneff = []
    impactp = []
    ringwidth = []
    plot_dict = {'MuonEff':muoneff,'ImpactP':impactp,'RingWidth':ringwidth}


    for muon_evt in muons:
        #Test display #Flag 1 for true (wish to display)
        # display_telescope(muon_evt, muon_evt[0].tel_id, 1, geom_dict, pp, fig)    
        if muon_evt[0] is not None and muon_evt[1] is not None:
            
            plot_dict['MuonEff'].append(muon_evt[1].optical_efficiency_muon)
            plot_dict['ImpactP'].append(muon_evt[1].impact_parameter.value)
            plot_dict['RingWidth'].append(muon_evt[1].ring_width.value)
            
            display_muon_plot(muon_evt) 
            #Store and or Plot muon parameters here


    t = Table([muoneff, impactp, ringwidth], names=('MuonEff','ImpactP','RingWidth'), meta={'name': 'muon analysis results'})
    t['ImpactP'].unit = 'm'
    t['RingWidth'].unit = 'deg'
    #    print('plotdict',plot_dict)

    t.write(args.output_path+'_muontable.fits',overwrite=True)

    #plot_muon_efficiency(plot_dict,args.output_path)
    plot_muon_efficiency(args.output_path)

    log.info("[COMPLETE]")

if __name__ == '__main__':
    main()
