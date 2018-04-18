""" Create a theta-square plot .

"""

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

from ctapipe.calib import CameraCalibrator
from ctapipe.image import hillas_parameters
from ctapipe.image import tailcuts_clean
from ctapipe.io import event_source
from ctapipe.reco import HillasReconstructor
from ctapipe.utils import datasets, linalg

# importing data from avaiable datasets in ctapipe
filename = datasets.get_dataset_path("gamma_test_large.simtel.gz")

# reading the Monte Carlo file for LST
source = event_source(filename, allowed_tels={1, 2, 3, 4})

# pointing direction of the telescopes
point_azimuth = {}
point_altitude = {}

reco = HillasReconstructor()
calib = CameraCalibrator(r1_product="HESSIOR1Calibrator")
off_angles = []

for event in source:

    # The direction the incident particle. Converting Monte Carlo Shower
    # parameter theta and phi to corresponding to 3 components (x,y,z) of a
    # vector
    shower_azimuth = event.mc.az  # same as in Monte Carlo file i.e. phi
    shower_altitude = np.pi * u.rad / 2 - event.mc.alt  # altitude = 90 - theta
    shower_direction = linalg.set_phi_theta(shower_azimuth, shower_altitude)
    # calibrating the event
    calib.calibrate(event)
    hillas_params = {}
    subarray = event.inst.subarray

    for tel_id in event.dl0.tels_with_data:

        # telescope pointing direction
        point_azimuth[tel_id] = event.mc.tel[tel_id].azimuth_raw * u.rad
        point_altitude[tel_id] = (
            np.pi / 2 - event.mc.tel[tel_id].altitude_raw
        ) * u.rad
        #        print(point_azimuth,point_altitude)

        # Camera Geometry required for hillas parametrization
        camgeom = subarray.tel[tel_id].camera

        # note the [0] is for channel 0 which is high-gain channel
        image = event.dl1.tel[tel_id].image[0]

        # Cleaning  of the image
        cleaned_image = image
        # create a clean mask of pixels above the threshold
        cleanmask = tailcuts_clean(
            camgeom, image, picture_thresh=10, boundary_thresh=5
        )
        # set all rejected pixels to zero
        cleaned_image[~cleanmask] = 0

        # Calulate hillas parameters
        # It fails for empty pixels
        try:
            hillas_params[tel_id] = hillas_parameters(camgeom, cleaned_image)
        except:
            pass

    if len(hillas_params) < 2:
        continue

    reco.get_great_circles(
        hillas_params, event.inst.subarray, point_azimuth, point_altitude
    )

    # fit the gamma's direction of origin
    # return reconstructed direction (3 components) with errors on the values
    reco_direction, reco_dir_err = reco.fit_origin_crosses()

    # In case fit fails to get any real value
    if np.isnan(reco_direction).any():
        continue

    # get angular offset between reconstructed shower direction and MC
    # generated shower direction
    off_angle = linalg.angle(reco_direction, shower_direction)

    # Appending all estimated off angles
    off_angles.append(off_angle.to(u.deg).value)

# calculate theta square
thetasq = []
for i in off_angles:
    thetasq.append(i**2)

# To plot thetasquare The number of events in th data files for LSTCam is not
#  significantly high to give a nice thetasquare plot for gammas One can use
# deedicated MC file for LST get nice plot
plt.figure(figsize=(10, 8))
plt.hist(thetasq, bins=np.linspace(0, 10, 50))
plt.title(r'$\theta^2$ plot')
plt.xlabel(r'$\theta^2$')
plt.ylabel("# of events")
plt.show()
