from astropy.coordinates import SkyCoord
import astropy.units as u
from ctapipe.io import event_source
from ctapipe.image.cleaning import tailcuts_clean
from ctapipe.calib import CameraCalibrator
from ctapipe.utils.datasets import get_dataset_path
import matplotlib.pyplot as plt
import numpy as np

from ctapipe.coordinates import HorizonFrame, CameraFrame, NominalFrame


cleaning_level = {
    'LSTCam': (3.5, 7.5, 2),  # ?? (3, 6) for Abelardo...
    'FlashCam': (4, 8, 2),  # there is some scaling missing?
    'ASTRICam': (5, 7, 2),
}


input_url = get_dataset_path('gamma_test_large.simtel.gz')


with event_source(input_url=input_url) as source:
    calibrator = CameraCalibrator(
        eventsource=source,
    )

    for event in source:

        calibrator.calibrate(event)

        nominal_frame = NominalFrame(
            origin=SkyCoord(alt=70 * u.deg, az=0 * u.deg, frame=HorizonFrame)
        )


        nom_x = []
        nom_y = []
        photons = []

        for tel_id, dl1 in event.dl1.tel.items():
            camera = event.inst.subarray.tels[tel_id].camera
            focal_length = event.inst.subarray.tels[tel_id].optics.equivalent_focal_length
            image = dl1.image[0]

            # telescope mc info
            mc_tel = event.mc.tel[tel_id]

            telescope_pointing = SkyCoord(
                alt=mc_tel['altitude_raw'],
                az=mc_tel['azimuth_raw'],
                unit='rad', frame=HorizonFrame(),
            )
            camera_frame = CameraFrame(
                telescope_pointing=telescope_pointing, focal_length=focal_length
            )

            boundary, picture, min_neighbors = cleaning_level[camera.cam_id]
            clean = tailcuts_clean(
                camera,
                image,
                boundary_thresh=boundary,
                picture_thresh=picture,
                min_number_picture_neighbors=min_neighbors
            )

            cam_coords = SkyCoord(
                camera.pix_x[clean],
                camera.pix_y[clean],
                frame=camera_frame
            )
            nom = cam_coords.transform_to(nominal_frame)
            nom_x.append(nom.x.wrap_at(180 * u.deg).to_value(u.deg))
            nom_y.append(nom.y.to_value(u.deg))
            photons.append(image[clean])

        nom_x = np.concatenate(nom_x)
        nom_y = np.concatenate(nom_y)
        photons = np.concatenate(photons)

        nom_x = np.repeat(nom_x, photons.astype(int))
        nom_y = np.repeat(nom_y, photons.astype(int))

        plt.hexbin(nom_x, nom_y, gridsize=50, extent=[-5, 5, -5, 5])
        plt.gca().set_aspect(1)
        plt.show()
