"""
Plots the (rough) hillas parameters for each event on an ArrayDisplay
"""

import matplotlib.pyplot as plt

from ctapipe.calib import CameraCalibrator
from ctapipe.image import hillas_parameters, tailcuts_clean, \
    HillasParameterizationError
from ctapipe.io import event_source
from ctapipe.utils import datasets
from ctapipe.visualization import ArrayDisplay


if __name__ == '__main__':

    # importing data from avaiable datasets in ctapipe
    filename = datasets.get_dataset_path("gamma_test_large.simtel.gz")

    # reading the Monte Carlo file for LST
    source = event_source(filename)

    # pointing direction of the telescopes
    point_azimuth = {}
    point_altitude = {}

    calib = CameraCalibrator(eventsource=source)
    off_angles = []
    first_event = True
    markers = None

    for event in source:

        if len(event.r0.tels_with_data) < 2:
            continue

        # calibrating the event
        calib.calibrate(event)
        hillas_dict = {}
        subarray = event.inst.subarray

        if first_event:
            fig, ax = plt.subplots(1,1, figsize=(10, 8))
            array_disp = ArrayDisplay(subarray, axes=ax, tel_scale=4.0)
            array_disp.telescopes.set_linewidth(0)
            first_event = False
            hit_pattern = np.zeros(subarray.num_tels)

        # plot the core pos
        if markers:
            for marker in markers:
                marker.remove()
        core_x = event.mc.core_x.to("m").value
        core_y = event.mc.core_y.to("m").value
        markers = ax.plot([core_x,], [core_y,], "r+", markersize=10)
        tel_ids = sub.tel_id

        # plot the hit pattern (triggered tels):
        hit_pattern[:] = 0
        for itel in list(event.r0.tels_with_data):
            hit_pattern[itel-1] = 10.0 # hack: todo: use tel_index[itel]
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

        array_disp.set_vector_hillas(hillas_dict)

        plt.pause(0.1)  # allow matplotlib to redraw the display

        if len(hillas_dict) < 2:
            continue
