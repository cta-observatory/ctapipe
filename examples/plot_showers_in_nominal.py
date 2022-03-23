from astropy.coordinates import SkyCoord, AltAz
import astropy.units as u
from ctapipe.io import EventSource
from ctapipe.image.cleaning import tailcuts_clean
from ctapipe.calib import CameraCalibrator
from ctapipe.utils.datasets import get_dataset_path
import matplotlib.pyplot as plt
import numpy as np
import warnings

from ctapipe.coordinates import CameraFrame, NominalFrame, MissingFrameAttributeWarning


warnings.filterwarnings("ignore", category=MissingFrameAttributeWarning)


cleaning_level = {
    "LSTCam": (3.5, 7.5, 2),  # ?? (3, 6) for Abelardo...
    "FlashCam": (4, 8, 2),  # there is some scaling missing?
    "ASTRICam": (5, 7, 2),
}


input_url = get_dataset_path("gamma_test_large.simtel.gz")


with EventSource(input_url=input_url) as source:
    calibrator = CameraCalibrator(subarray=source.subarray)

    for event in source:

        calibrator(event)

        nominal_frame = NominalFrame(
            origin=SkyCoord(alt=70 * u.deg, az=0 * u.deg, frame=AltAz)
        )

        nom_fov_lon = []
        nom_fov_lat = []
        photons = []

        subarray = source.subarray

        for tel_id, dl1 in event.dl1.tel.items():
            geom = subarray.tels[tel_id].camera.geometry
            focal_length = subarray.tels[tel_id].optics.equivalent_focal_length
            image = dl1.image

            telescope_pointing = SkyCoord(
                alt=event.pointing.tel[tel_id].altitude,
                az=event.pointing.tel[tel_id].azimuth,
                frame=AltAz(),
            )
            camera_frame = CameraFrame(
                telescope_pointing=telescope_pointing, focal_length=focal_length
            )

            boundary, picture, min_neighbors = cleaning_level[geom.camera_name]
            clean = tailcuts_clean(
                geom,
                image,
                boundary_thresh=boundary,
                picture_thresh=picture,
                min_number_picture_neighbors=min_neighbors,
            )

            cam_coords = SkyCoord(
                geom.pix_x[clean], geom.pix_y[clean], frame=camera_frame
            )
            nom = cam_coords.transform_to(nominal_frame)
            nom_fov_lon.append(nom.fov_lon.to_value(u.deg))
            nom_fov_lat.append(nom.fov_lat.to_value(u.deg))
            photons.append(image[clean])

        nom_fov_lon = np.concatenate(nom_fov_lon)
        nom_fov_lat = np.concatenate(nom_fov_lat)
        photons = np.concatenate(photons)

        nom_fov_lon = np.repeat(nom_fov_lon, photons.astype(int))
        nom_fov_lat = np.repeat(nom_fov_lat, photons.astype(int))

        plt.hexbin(nom_fov_lon, nom_fov_lat, gridsize=50, extent=[-5, 5, -5, 5])
        plt.xlabel("fov_lon / deg")
        plt.ylabel("fov_lat / deg")
        plt.gca().set_aspect(1)
        plt.show()
