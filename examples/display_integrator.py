"""
Create a plot of where the integration window lays on the trace for the pixel
with the highest charge, its neighbours, and the pixel with the lowest max
charge and its neighbours. Also shows a disgram of which pixels count as a
neighbour, the camera image for the max charge timeslice, the true pe camera
image, and a calibrated camera image

Example Command:
p examples/display_integrator.py -f ~/Software/outputs/sim_telarray/
meudon_gamma/simtel_runmeudon_gamma_30tel_30deg_19.gz -O hessio -e 1 -t 8
--integrator 5 --integration-window 7 3
"""

import argparse
import os
from astropy import log
import numpy as np
from matplotlib import pyplot as plt

from ctapipe.calib.camera.calibrators import calibration_arguments, \
    calibration_parameters
from ctapipe.calib.camera.calibrators import calibrate_event
from ctapipe.io import CameraGeometry
from ctapipe.io.files import InputFile, origin_list
from ctapipe.plotting.camera import CameraPlotter


def get_source(filepath):
    from ctapipe.utils.datasets import get_path
    from ctapipe.io.hessio import hessio_event_source
    source = hessio_event_source(get_path(filepath))
    return source


def main():
    script = os.path.splitext(os.path.basename(__file__))[0]
    log.info("[SCRIPT] {}".format(script))

    parser = argparse.ArgumentParser(description='Create a gif of an event')
    parser.add_argument('-f', '--file', dest='input_path', action='store',
                        required=True, help='path to the input file')
    parser.add_argument('-O', '--origin', dest='origin', action='store',
                        required=True, help='origin of the file: {}'
                        .format(origin_list()))
    parser.add_argument('-o', '--output', dest='output_dir', action='store',
                        default=None,
                        help='path of the output directory to store the '
                             'images (default = input file directory)')
    parser.add_argument('-e', '--event', dest='event_req', action='store',
                        required=True, type=int,
                        help='event index to plot (not id!)')
    parser.add_argument('--id', dest='event_id_f', action='store_true',
                        default=False, help='-e will specify event_id instead '
                                            'of index')
    parser.add_argument('-t', '--telescope', dest='tel', action='store',
                        type=int, default=None, help='telecope to view')
    parser.add_argument('-c', '--channel', dest='chan', action='store',
                        type=int, default=0,
                        help='channel to view (default = 0 (HG))')

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

    telid = args.tel
    chan = args.chan

    log.debug("[file] Reading file")
    input_file = InputFile(args.input_path, args.origin)
    event = input_file.get_event(args.event_req, args.event_id_f)

    # Print event/args values
    log.info("[event_index] {}".format(event.count))
    log.info("[event_id] {}".format(event.dl0.event_id))
    log.info("[telescope] {}".format(telid))
    log.info("[channel] {}".format(chan))

    params = calibration_parameters(args)

    # Create a dictionary to store any geoms in
    geom = CameraGeometry.guess(*event.meta.pixel_pos[telid],
                                event.meta.optical_foclen[telid])
    geom_dict = {telid: geom}
    calibrated_event = calibrate_event(event, params, geom_dict)

    # Select telescope
    tels = list(calibrated_event.dl0.tels_with_data)
    if telid is None or telid not in tels:
        log.error("[event] please specify one of the following telescopes "
                  "for this event: {}".format(tels))
        exit()

    # Extract required images
    data_ped = calibrated_event.dl1.tel[telid].pedestal_subtracted_adc[chan]
    true_pe = calibrated_event.mc.tel[telid].photo_electrons
    measured_pe = calibrated_event.dl1.tel[telid].pe_charge[chan]
    max_time = np.unravel_index(np.argmax(data_ped), data_ped.shape)[1]
    max_charges = np.max(data_ped, axis=1)
    max_pixel = int(np.argmax(max_charges))
    min_pixel = int(np.argmin(max_charges))

    # Get Neighbours
    max_pixel_nei = geom.neighbors[max_pixel]
    min_pixel_nei = geom.neighbors[min_pixel]

    # Get Windows
    windows = calibrated_event.dl1.tel[telid].integration_window[chan]
    length = np.sum(windows, axis=1)
    start = np.argmax(windows, axis=1)
    end = start + length - 1

    # Draw figures
    ax_max_nei = {}
    ax_min_nei = {}
    fig_waveforms = plt.figure(figsize=(24, 10))
    fig_waveforms.subplots_adjust(hspace=.5)
    fig_camera = plt.figure(figsize=(30, 24))

    ax_max_pix = fig_waveforms.add_subplot(4, 2, 1)
    ax_min_pix = fig_waveforms.add_subplot(4, 2, 2)
    ax_max_nei[0] = fig_waveforms.add_subplot(4, 2, 3)
    ax_min_nei[0] = fig_waveforms.add_subplot(4, 2, 4)
    ax_max_nei[1] = fig_waveforms.add_subplot(4, 2, 5)
    ax_min_nei[1] = fig_waveforms.add_subplot(4, 2, 6)
    ax_max_nei[2] = fig_waveforms.add_subplot(4, 2, 7)
    ax_min_nei[2] = fig_waveforms.add_subplot(4, 2, 8)

    ax_img_nei = fig_camera.add_subplot(2, 2, 1)
    ax_img_max = fig_camera.add_subplot(2, 2, 2)
    ax_img_true = fig_camera.add_subplot(2, 2, 3)
    ax_img_cal = fig_camera.add_subplot(2, 2, 4)

    plotter = CameraPlotter(event, geom_dict)

    # Draw max pixel traces
    plotter.draw_waveform(data_ped[max_pixel], ax_max_pix)
    ax_max_pix.set_title("(Max) Pixel: {}".format(max_pixel))
    ax_max_pix.set_ylabel("Amplitude-Ped (ADC)")
    max_ylim = ax_max_pix.get_ylim()
    plotter.draw_waveform_positionline(start[max_pixel], ax_max_pix)
    plotter.draw_waveform_positionline(end[max_pixel], ax_max_pix)
    for i, ax in ax_max_nei.items():
        if len(max_pixel_nei) > i:
            pix = max_pixel_nei[i]
            plotter.draw_waveform(data_ped[pix], ax)
            ax.set_title("(Max Nei) Pixel: {}".format(pix))
            ax.set_ylabel("Amplitude-Ped (ADC)")
            ax.set_ylim(max_ylim)
            plotter.draw_waveform_positionline(start[pix], ax)
            plotter.draw_waveform_positionline(end[pix], ax)

    # Draw min pixel traces
    plotter.draw_waveform(data_ped[min_pixel], ax_min_pix)
    ax_min_pix.set_title("(Min) Pixel: {}".format(min_pixel))
    ax_min_pix.set_ylabel("Amplitude-Ped (ADC)")
    ax_min_pix.set_ylim(max_ylim)
    plotter.draw_waveform_positionline(start[min_pixel], ax_min_pix)
    plotter.draw_waveform_positionline(end[min_pixel], ax_min_pix)
    for i, ax in ax_min_nei.items():
        if len(min_pixel_nei) > i:
            pix = min_pixel_nei[i]
            plotter.draw_waveform(data_ped[pix], ax)
            ax.set_title("(Min Nei) Pixel: {}".format(pix))
            ax.set_ylabel("Amplitude-Ped (ADC)")
            ax.set_ylim(max_ylim)
            plotter.draw_waveform_positionline(start[pix], ax)
            plotter.draw_waveform_positionline(end[pix], ax)

    # Draw cameras
    nei_camera = np.zeros_like(max_charges, dtype=np.int)
    nei_camera[min_pixel_nei] = 2
    nei_camera[min_pixel] = 1
    nei_camera[max_pixel_nei] = 3
    nei_camera[max_pixel] = 4
    camera = plotter.draw_camera(telid, nei_camera, ax_img_nei)
    camera.cmap = plt.cm.jet
    ax_img_nei.set_title("Neighbour Map")
    plotter.draw_camera_pixel_annotation(telid, max_pixel, min_pixel,
                                         ax_img_nei)

    camera = plotter.draw_camera(telid, data_ped[:, max_time], ax_img_max)
    camera.add_colorbar(ax=ax_img_max, label="Amplitude-Ped (ADC)")
    ax_img_max.set_title("Max Timeslice (T = {})".format(max_time))
    plotter.draw_camera_pixel_annotation(telid, max_pixel, min_pixel,
                                         ax_img_max)

    camera = plotter.draw_camera(telid, true_pe, ax_img_true)
    camera.add_colorbar(ax=ax_img_true, label="True Charge (Photo-electrons)")
    ax_img_true.set_title("True Charge")
    plotter.draw_camera_pixel_annotation(telid, max_pixel, min_pixel,
                                         ax_img_true)

    camera = plotter.draw_camera(telid, measured_pe, ax_img_cal)
    camera.add_colorbar(ax=ax_img_cal, label="Calib Charge (Photo-electrons)")
    ax_img_cal.set_title("Charge (integrator={})".format(params['integrator']))
    plotter.draw_camera_pixel_annotation(telid, max_pixel, min_pixel,
                                         ax_img_cal)

    fig_waveforms.suptitle("Integrator = {}".format(params['integrator']))
    fig_camera.suptitle("Camera = {}".format(geom.cam_id))

    # TODO: another figure of all waveforms that have non-zero true charge

    waveform_output_name = "{}_e{}_t{}_c{}_integrator{}_waveform.pdf"\
        .format(input_file.filename, event.count, telid, chan, args.integrator)
    camera_output_name = "{}_e{}_t{}_c{}_integrator{}_camera.pdf"\
        .format(input_file.filename, event.count, telid, chan, args.integrator)
    output_dir = args.output_dir if args.output_dir is not None else \
        input_file.output_directory
    output_dir = os.path.join(output_dir, script)
    if not os.path.exists(output_dir):
        log.info("[output] Creating directory: {}".format(output_dir))
        os.makedirs(output_dir)

    waveform_output_path = os.path.join(output_dir, waveform_output_name)
    log.info("[output] {}".format(waveform_output_path))
    fig_waveforms.savefig(waveform_output_path, format='pdf')

    camera_output_path = os.path.join(output_dir, camera_output_name)
    log.info("[output] {}".format(camera_output_path))
    fig_camera.savefig(camera_output_path, format='pdf')


if __name__ == '__main__':
    main()
