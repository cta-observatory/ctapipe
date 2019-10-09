import astropy.units as u
from astropy.coordinates import SkyCoord, AltAz

from ctapipe.io import event_source
from ctapipe.calib import CameraCalibrator
from ctapipe.image.cleaning import tailcuts_clean, number_of_islands
from ctapipe.image import leakage, hillas_parameters
from ctapipe.image.timing_parameters import timing_parameters
from ctapipe.reco import HillasReconstructor
from ctapipe.utils.datasets import get_dataset_path

import matplotlib.pyplot as plt
from ctapipe.visualization import ArrayDisplay


# unoptimized cleaning levels, copied from
# https://github.com/tudo-astroparticlephysics/cta_preprocessing
cleaning_level = {
    'LSTCam': (3.5, 7.5, 2),  # ?? (3, 6) for Abelardo...
    'FlashCam': (4, 8, 2),  # there is some scaling missing?
    'ASTRICam': (5, 7, 2),
}


input_url = get_dataset_path('gamma_test_large.simtel.gz')
event_source = event_source(input_url)

calibrator = CameraCalibrator()
horizon_frame = AltAz()

reco = HillasReconstructor()

for event in event_source:
    print('Event', event.count)
    calibrator(event)

    # mapping of telescope_id to parameters for stereo reconstruction
    hillas_containers = {}

    time_gradients = {}

    for telescope_id, dl1 in event.dl1.tel.items():
        camera = event.inst.subarray.tels[telescope_id].camera

        image = dl1.image
        peakpos = dl1.pulse_time

        # cleaning
        boundary, picture, min_neighbors = cleaning_level[camera.cam_id]
        clean = tailcuts_clean(
            camera,
            image,
            boundary_thresh=boundary,
            picture_thresh=picture,
            min_number_picture_neighbors=min_neighbors
        )

        # ignore images with less than 5 pixels after cleaning
        if clean.sum() < 5:
            continue

        # image parameters
        hillas_c = hillas_parameters(camera[clean], image[clean])
        leakage_c = leakage(camera, image, clean)
        n_islands, island_ids = number_of_islands(camera, clean)

        timing_c = timing_parameters(
            camera[clean], image[clean], peakpos[clean], hillas_c, clean,
        )

        # store parameters for stereo reconstruction
        hillas_containers[telescope_id] = hillas_c

        # store timegradients for plotting
        # ASTRI has no timing in PROD3b, so we use skewness instead
        if camera.cam_id != 'ASTRICam':
            time_gradients[telescope_id] = timing_c.slope.value
        else:
            time_gradients[telescope_id] = hillas_c.skewness
        print(camera.cam_id, time_gradients[telescope_id])

        # make sure each telescope get's an arrow
        if abs(time_gradients[telescope_id]) < 0.2:
            time_gradients[telescope_id] = 1

    # ignore events with less than two telescopes
    if len(hillas_containers) < 2:
        continue
    array_pointing = SkyCoord(
        az=event.mcheader.run_array_direction[0],
        alt=event.mcheader.run_array_direction[1],
        frame=horizon_frame
    )
    stereo = reco.predict(
        hillas_containers,
        event.inst,
        array_pointing,
    )

    plt.figure()
    angle_offset = event.mcheader.run_array_direction[0]
    disp = ArrayDisplay(event.inst.subarray)

    disp.set_vector_hillas(
        hillas_containers,
        time_gradient=time_gradients,
        angle_offset=angle_offset,
        length=500
    )
    plt.scatter(
        event.mc.core_x, event.mc.core_y,
        s=200, c='k', marker='x', label='True Impact',
    )
    plt.scatter(
        stereo.core_x, stereo.core_y,
        s=200, c='r', marker='x', label='Estimated Impact',
    )

    plt.legend()
    plt.show()
