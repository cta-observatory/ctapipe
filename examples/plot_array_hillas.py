"""
Plots the (rough) hillas parameters for each event on an ArrayDisplay
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord, AltAz
from astropy import units as u

from ctapipe.calib import CameraCalibrator
from ctapipe.coordinates import TiltedGroundFrame, MissingFrameAttributeWarning
from ctapipe.image import hillas_parameters, tailcuts_clean, HillasParameterizationError
from ctapipe.image import timing_parameters
from ctapipe.io import EventSource
from ctapipe.utils import datasets
from ctapipe.visualization import ArrayDisplay
import warnings

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=MissingFrameAttributeWarning)

    # importing data from avaiable datasets in ctapipe
    filename = datasets.get_dataset_path("gamma_test_large.simtel.gz")

    if len(sys.argv) > 1:
        filename = sys.argv[1]

    # reading the Monte Carlo file for LST
    source = EventSource(filename)

    # pointing direction of the telescopes
    point_azimuth = {}
    point_altitude = {}

    subarray = source.subarray
    calib = CameraCalibrator(subarray=subarray)
    off_angles = []
    first_event = True
    markers = None

    for event in source:
        if first_event:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            array_disp = ArrayDisplay(subarray, axes=ax, tel_scale=1.0)
            array_disp.telescopes.set_linewidth(3)
            array_disp.add_labels()
            first_event = False
            hit_pattern = np.zeros(subarray.num_tels)

        if len(event.r0.tel.keys()) < 3:
            continue

        # calibrating the event
        calib(event)
        hillas_dict = {}
        timing_dict = {}

        # plot the core position, which must be transformed from the tilted
        # system to the system that the ArrayDisplay is in (default
        # GroundFrame)
        point_dir = SkyCoord(
            az=event.pointing.array_azimuth,
            alt=event.pointing.array_altitude,
            frame=AltAz(),
        )
        tiltedframe = TiltedGroundFrame(pointing_direction=point_dir)
        if markers:
            for marker in markers:
                marker.remove()

        core_coord = SkyCoord(
            x=event.simulation.shower.core_x,
            y=event.simulation.shower.core_y,
            frame=tiltedframe,
        ).transform_to(array_disp.frame)

        markers = ax.plot(
            [core_coord.x.value], [core_coord.y.value], "r+", markersize=10
        )

        # plot the hit pattern (triggered tels).
        # first expand the tel.keys() list into a fixed-length vector,
        # then set the value so that the ArrayDisplay shows it as color per
        # telescope.
        tel_idx = source.subarray.tel_indices
        hit_pattern[:] = 0
        mask = [tel_idx[t] for t in event.r0.tel.keys()]
        hit_pattern[mask] = 10.0
        array_disp.values = hit_pattern

        # calculate and plot the hillas params

        for tel_id in event.dl0.tel.keys():

            # Camera Geometry required for hillas parametrization
            camgeom = subarray.tel[tel_id].camera.geometry

            # note the [0] is for channel 0 which is high-gain channel
            image = event.dl1.tel[tel_id].image
            time = event.dl1.tel[tel_id].peak_time

            # Cleaning  of the image
            cleaned_image = image.copy()

            # create a clean mask of pixels above the threshold
            cleanmask = tailcuts_clean(
                camgeom, image, picture_thresh=10, boundary_thresh=5
            )
            if np.count_nonzero(cleanmask) < 10:
                continue

            # set all rejected pixels to zero
            cleaned_image[~cleanmask] = 0

            # Calculate hillas parameters
            try:
                hillas_dict[tel_id] = hillas_parameters(camgeom, cleaned_image)
            except HillasParameterizationError:
                continue  # skip failed parameterization (normally no signal)

            timing_dict[tel_id] = timing_parameters(
                camgeom, image, time, hillas_dict[tel_id], cleanmask
            ).slope.value

        array_disp.set_vector_hillas(
            hillas_dict, 500, timing_dict, angle_offset=0 * u.deg
        )

        plt.pause(0.1)  # allow matplotlib to redraw the display

        if len(hillas_dict) < 2:
            continue
