from astropy.coordinates import SkyCoord, AltAz
import astropy.units as u
from ctapipe.io import event_source
from ctapipe.image.cleaning import tailcuts_clean
from ctapipe.calib import CameraCalibrator
from ctapipe.utils.datasets import get_dataset_path
import matplotlib.pyplot as plt
import numpy as np

from ctapipe.coordinates import CameraFrame, NominalFrame


cleaning_level = {
    'LSTCam': (3.5, 7.5, 2),  # ?? (3, 6) for Abelardo...
    'FlashCam': (4, 8, 2),  # there is some scaling missing?
    'ASTRICam': (5, 7, 2),
}


input_url = get_dataset_path('gamma_test_large.simtel.gz')


with event_source(input_url=input_url) as source:
    calibrator = CameraCalibrator()

    for event in source:

        calibrator(event)

        nominal_frame = NominalFrame(
            origin=SkyCoord(alt=70 * u.deg, az=0 * u.deg, frame=AltAz)
        )

        nom_delta_az = []
        nom_delta_alt = []
        photons = []

        for tel_id, dl1 in event.dl1.tel.items():
            camera = event.inst.subarray.tels[tel_id].camera
            focal_length = event.inst.subarray.tels[tel_id].optics.equivalent_focal_length
            image = dl1.image

            # telescope mc info
            mc_tel = event.mc.tel[tel_id]

            telescope_pointing = SkyCoord(
                alt=mc_tel['altitude_raw'],
                az=mc_tel['azimuth_raw'],
                unit='rad', frame=AltAz(),
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
            nom_delta_az.append(nom.delta_az.to_value(u.deg))
            nom_delta_alt.append(nom.delta_alt.to_value(u.deg))
            photons.append(image[clean])

        nom_delta_az = np.concatenate(nom_delta_az)
        nom_delta_alt = np.concatenate(nom_delta_alt)
        photons = np.concatenate(photons)

        nom_delta_az = np.repeat(nom_delta_az, photons.astype(int))
        nom_delta_alt = np.repeat(nom_delta_alt, photons.astype(int))

        plt.hexbin(nom_delta_az, nom_delta_alt, gridsize=50, extent=[-5, 5, -5, 5])
        plt.xlabel('delta_az / deg')
        plt.ylabel('delta_alt / deg')
        plt.gca().set_aspect(1)
        plt.show()
