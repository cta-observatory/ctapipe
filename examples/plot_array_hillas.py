"""
Plots the (rough) hillas parameters for each event on an ArrayDisplay
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord, AltAz
from astropy import units as u

from ctapipe.calib import CameraCalibrator
from ctapipe.coordinates import TiltedGroundFrame
from ctapipe.image import hillas_parameters, tailcuts_clean, HillasParameterizationError
from ctapipe.io import event_source
from ctapipe.utils import datasets
from ctapipe.visualization import ArrayDisplay

if __name__ == '__main__':

    # importing data from avaiable datasets in ctapipe
    filename = datasets.get_dataset_path("gamma_test_large.simtel.gz")

    if len(sys.argv) > 1:
        filename = sys.argv[1]

    # reading the Monte Carlo file for LST
    source = event_source(filename)

    # pointing direction of the telescopes
    point_azimuth = {}
    point_altitude = {}

    calib = CameraCalibrator()
    off_angles = []
    first_event = True
    markers = None

    for event in source:

        subarray = event.inst.subarray

        if first_event:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            array_disp = ArrayDisplay(subarray, axes=ax, tel_scale=1.0)
            array_disp.telescopes.set_linewidth(3)
            array_disp.add_labels()
            first_event = False
            hit_pattern = np.zeros(subarray.num_tels)

        if len(event.r0.tels_with_data) < 3:
            continue

        # calibrating the event
        calib(event)
        hillas_dict = {}

        # plot the core position, which must be transformed from the tilted
        # system to the system that the ArrayDisplay is in (default
        # GroundFrame)
        point_dir = SkyCoord(
            *event.mcheader.run_array_direction,
            frame=AltAz()
        )
        tiltedframe = TiltedGroundFrame(pointing_direction=point_dir)
        if markers:
            for marker in markers:
                marker.remove()

        core_coord = SkyCoord(
            x=event.mc.core_x,
            y=event.mc.core_y,
            frame=tiltedframe
        ).transform_to(array_disp.frame)

        markers = ax.plot([core_coord.x.value, ], [core_coord.y.value, ],
                          "r+", markersize=10)

        # plot the hit pattern (triggered tels).
        # first expand the tels_with_data list into a fixed-length vector,
        # then set the value so that the ArrayDisplay shows it as color per
        # telescope.
        tel_idx = event.inst.subarray.tel_indices
        hit_pattern[:] = 0
        mask = [tel_idx[t] for t in event.r0.tels_with_data]
        hit_pattern[mask] = 10.0
        array_disp.values = hit_pattern

        # calculate and plot the hillas params

        for tel_id in event.dl0.tels_with_data:

            # Camera Geometry required for hillas parametrization
            camgeom = subarray.tel[tel_id].camera

            # note the [0] is for channel 0 which is high-gain channel
            image = event.dl1.tel[tel_id].image[0]

            # Cleaning  of the image
            cleaned_image = image.copy()

            # create a clean mask of pixels above the threshold
            cleanmask = tailcuts_clean(
                camgeom, image, picture_thresh=10, boundary_thresh=5
            )
            # set all rejected pixels to zero
            cleaned_image[~cleanmask] = 0

            # Calculate hillas parameters
            try:
                hillas_dict[tel_id] = hillas_parameters(camgeom, cleaned_image)
            except HillasParameterizationError:
                pass  # skip failed parameterization (normally no signal)

        array_disp.set_vector_hillas(hillas_dict, angle_offset=0 * u.deg)

        plt.pause(0.1)  # allow matplotlib to redraw the display

        if len(hillas_dict) < 2:
            continue
