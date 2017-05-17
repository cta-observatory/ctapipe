#!/usr/bin/env python3

import argparse
import logging
import sys

import numpy as np
from matplotlib import colors, pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from ctapipe import visualization
from ctapipe.instrument import InstrumentDescription as ID
from ctapipe.io.hessio import hessio_event_source
from ctapipe.utils import get_dataset
from ctapipe.instrument import CameraGeometry

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
if __debug__:
    logger.setLevel(logging.DEBUG)

fig = plt.figure(figsize=(20, 10))
cmaps = [plt.cm.jet, plt.cm.winter,
         plt.cm.ocean, plt.cm.bone, plt.cm.gist_earth, plt.cm.hot,
         plt.cm.cool, plt.cm.coolwarm]
gain_label=("HG","LG")

def get_cmap(signals):
    cmaxmin = (max(signals)-min(signals))
    cmap = colors.LinearSegmentedColormap.from_list(
        'cmap_c', [(0, 'black'),
                   (0.1, 'darkblue'),
                   (0.5, 'blue'),
                   (0.97, 'red'),
                   (1., 'yellow')])
    return cmap    
    
def display_telescope(calibrated_data,cam,tel_id,pdffilename):
    logger.info("Tel_ID %d\n"%tel_id)
    pp = PdfPages("%s"%pdffilename)
    global fig
    for event in calibrated_data:
        tels = list(event.r0.tels_with_data)
        logger.debug(tels)
        # Select telescope
        if tel_id not in tels:
            continue

        fig.clear()

        # geom = cam['CameraTable_VersionFeb2016_TelID%s'%tel_id]
        geom = CameraGeometry.guess(*event.inst.pixel_pos[tel_id],
                                       event.inst.optical_foclen[tel_id])
        # Select number of pads to display. It depends on the numberr of gains:
        nchan = event.inst.num_channels[tel_id]

        plt.suptitle("EVENT {} {:.1e} TeV @({:.1f},{:.1f})deg @{:.1f} m".format(
            event.r0.event_id, event.mc.energy,
            event.mc.alt, event.mc.az,
            np.sqrt(pow(event.mc.core_x, 2) +
                    pow(event.mc.core_y, 2))))
        print("\t draw cam {} (gains={})...".format(tel_id,nchan))
        ax=[]
        disp=[]
        signals=[]
        npads=0
        for i in range(nchan):
            npads += 1
            # Display the camera charge (HG/LG)
            ax.append(plt.subplot(nchan, 2, npads))
            disp.append(visualization.CameraDisplay(geom, ax=ax[-1],
                                                    title="CT{} [{} ADC cts]".format(tel_id,gain_label[i])))
            disp[-1].pixels.set_antialiaseds(False)
            signals.append(event.r0.tel[tel_id].adc_sums[i])
            disp[-1].image = signals[-1]
            disp[-1].pixels.set_cmap(get_cmap(disp[-1].image))
            disp[-1].add_colorbar(label=" [{} ADC cts]".format(gain_label[i]))

            # Display the camera charge for significant pixels (HG/LG)
            npads += 1
            ax.append(plt.subplot(nchan, 2, npads))
            disp.append(visualization.CameraDisplay(geom, ax=ax[-1],
                                                    title="CT{} [{} ADC cts]".format(tel_id,gain_label[i])))
            disp[-1].pixels.set_antialiaseds(False)
            signals.append(event.r0.tel[tel_id].adc_sums[i])
            m = (get_zero_sup_mode(tel_id) & 0x001) or (get_significant(tel_id) & 0x020)
            disp[-1].image = signals[-1]*(m/m.max())
            disp[-1].pixels.set_cmap(get_cmap(disp[-1].image))
            disp[-1].add_colorbar(label=" [{} ADC cts]".format(gain_label[i]))

            
        plt.pause(0.1)
        pp.savefig(fig)

    pp.close()

if __name__ == '__main__':

    # Declare and parse command line option
    parser = argparse.ArgumentParser(
        description='Tel_id, pixel id and number of event to compute.')
    parser.add_argument('--f', dest='filename',
                        required=False,
                        default=get_dataset('gamma_test_large.simtel.gz'),
                        help='filename <MC file name>')
    parser.add_argument('--d', dest='display', action='store_true',
                        required=False,help='display the camera events')
    parser.add_argument('--tel', dest='tel_id', action='store',
                        required=False, default=None,
                        help='Telescope ID to display')
    parser.add_argument('--pdf', dest='pdffilename', action='store',
                        required=False, default="./tmp.pdf",
                        help='PDF output filename')

    args = parser.parse_args()

    if args.display:
        plt.show(block=False)

    tel, cam, opt = ID.load(args.filename)
    # Load the DL0 events into the generator source
    source = hessio_event_source(args.filename)

    print(args.tel_id)
    # Display the calibrated results into camera displays
    if args.display:
        if args.tel_id is not None:
            display_telescope(source,cam,int(args.tel_id),args.pdffilename)
        else:
            logger.error("[event] please specify one telescope "
                      "(given in the file name) ")

            exit()
 

    logger.info("[COMPLETE]")
    sys.stdout.flush()

