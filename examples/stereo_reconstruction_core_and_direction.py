"""Example of showers reconstruction."""

# ==============================================================================
#                                IMPORTS
# ==============================================================================

# FROM THE STANDARD LIBRARY
import argparse
import signal

# FROM THIRD-PARTY LIBRARIES
import numpy as np
import astropy.units as u
from astropy.coordinates import AltAz, SkyCoord
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# FROM CTAPIPE
from ctapipe.containers import HillasParametersContainer, ImageParametersContainer
from ctapipe.utils import get_dataset_path
from ctapipe.io import EventSource
from ctapipe.coordinates import TelescopeFrame, NominalFrame, GroundFrame
from ctapipe.calib import CameraCalibrator
from ctapipe.image import (
    mars_cleaning_1st_pass,
    number_of_islands,
    largest_island,
    hillas_parameters,
    timing_parameters,
    HillasParameterizationError,
)
from ctapipe.reco import HillasReconstructor
from ctapipe.visualization import ArrayDisplay

# ==============================================================================
#                            CLASSES & FUNCTIONS
# ==============================================================================


class SignalHandler:
    """ handles ctrl+c signals; set up via
        `signal_handler = SignalHandler()
        `signal.signal(signal.SIGINT, signal_handler)`
        # or for two step interupt:
        `signal.signal(signal.SIGINT, signal_handler.stop_drawing)`
    """

    def __init__(self):
        self.stop = False
        self.draw = True

    def __call__(self, signal, frame):
        if self.stop:
            print("you pressed Ctrl+C again -- exiting NOW")
            exit(-1)
        print("you pressed Ctrl+C!")
        print("exiting after current event")
        self.stop = True

    def stop_drawing(self, signal, frame):
        if self.stop:
            print("you pressed Ctrl+C again -- exiting NOW")
            exit(-1)

        if self.draw:
            print("you pressed Ctrl+C!")
            print("turn off drawing")
            self.draw = False
        else:
            print("you pressed Ctrl+C!")
            print("exiting after current event")
            self.stop = True


signal_handler = SignalHandler()
signal.signal(signal.SIGINT, signal_handler)

# ==============================================================================
#                               INPUT DATA
# ==============================================================================

description = "Example of showers reconstruction. To exit press ctrl-c."
description += "You can use Matplotlib interactive GUI to zoom in."

parser = argparse.ArgumentParser(description=description)

default_test_file = "gamma_LaPalma_baseline_20Zd_180Az_prod3b_test.simtel.gz"
parser.add_argument(
    "--infile",
    type=str,
    default=default_test_file,
    help=f"simtel file to use (default: {default_test_file})",
)
parser.add_argument(
    "--max_events",
    type=int,
    default=None,
    help="maximum number of showers to analyze (default: all showers)",
)
parser.add_argument(
    "--show_pixels", action="store_true", help="Show pixel positions in NominalFrame",
)

args = parser.parse_args()

infile = get_dataset_path(args.infile)
source = EventSource(infile, max_events=args.max_events)

# ==============================================================================
#                         ANALYSIS INITIALIZATION
# ==============================================================================

subarray = source.subarray
calibrator = CameraCalibrator(subarray=subarray)
horizon_frame = AltAz()

image_cleaning_levels = {  # some values are dummy because not yet optimized
    "LSTCam": 6.0,
    "NectarCam": 6.0,
    "FlashCam": 8.0,
    "ASTRICam": 7.0,
    "SCTCam": 5.0,
}


# ==============================================================================
#                               EVENTS LOOP
# ==============================================================================

for event in source:

    print(f"EVENT #{event.count} with ID #{event.index.event_id}...")

    print(f"Telescopes with DL0 data = {list(event.r1.tel.keys())}")

    # ARRAY INFORMATION
    array_pointing = SkyCoord(
        az=event.pointing.array_azimuth,
        alt=event.pointing.array_altitude,
        frame=horizon_frame,
    )

    # Direction reconstruction setup
    reconstructor = HillasReconstructor(subarray=subarray)

    # CALIBRATION
    calibrator(event)

    # INITIALIZATION OF EVENT-WISE INFORMATION

    # Pointing direction of the telescopes
    telescope_pointings = {}

    # Image parametrization
    geometry_TelescopeFrame = {}  # we save it for plotting later
    parametrized_images = {}
    time_gradients = {}

    # ==========================================================================
    #                             TELESCOPE LOOP
    # ==========================================================================

    for tel_id in event.r1.tel.keys():

        # Camera information
        camera = subarray.tel[tel_id].camera
        cam_id = camera.camera_name
        geometry_CameraFrame = camera.geometry

        print(f"Telescope #{tel_id} equipped with {cam_id} camera...")

        # Pointing direction of this telescope
        telescope_pointings[tel_id] = SkyCoord(
            alt=event.pointing.tel[tel_id].altitude,
            az=event.pointing.tel[tel_id].azimuth,
            frame=horizon_frame,
        )

        # IMAGE CLEANING

        # get calibrated data
        calibrated_image = event.dl1.tel[tel_id].image
        peak_times = event.dl1.tel[tel_id].peak_time
        cleaned_image = calibrated_image

        # Tranform camera geometry from CameraFrame to TelescopeFrame
        geometry_TelescopeFrame[tel_id] = geometry_CameraFrame.transform_to(
            TelescopeFrame(telescope_pointing=telescope_pointings[tel_id],)
        )

        # Apply image cleaning to the calibrated image in the TelescopeFrame
        cleaning_mask = mars_cleaning_1st_pass(
            geometry_TelescopeFrame[tel_id],
            calibrated_image,
            picture_thresh=image_cleaning_levels[cam_id],
            boundary_thresh=image_cleaning_levels[cam_id] / 2.0,
            keep_isolated_pixels=False,
            min_number_picture_neighbors=1,
        )
        # set all rejected pixels to zero
        cleaned_image[~cleaning_mask] = 0

        # find islands
        num_islands, labels = number_of_islands(
            geometry_TelescopeFrame[tel_id], cleaning_mask
        )
        if num_islands > 1:  # if more islands survived..
            # ...find the biggest one
            mask_biggest = largest_island(labels)
            cleaned_image[~mask_biggest] = 0
            # overwrite results
            cleaning_mask = mask_biggest

        # IMAGE PARAMETRIZATION

        event.dl1.tel[tel_id].parameters = ImageParametersContainer()

        try:

            print("Attempting image parametrization via Hillas approach...")
            image_parameters = hillas_parameters(
                geometry_TelescopeFrame[tel_id], cleaned_image
            )

            timing = timing_parameters(
                geometry_TelescopeFrame[tel_id],
                calibrated_image,
                peak_times,
                image_parameters,
                cleaning_mask,
            )  # apply timing fit
            # ssts have no timing in prod3b, so we'll use the skewness
            time_gradient = (
                timing.slope.value
                if geometry_TelescopeFrame[tel_id].camera_name != "ASTRICam"
                else image_parameters.skewness
            )
            # time_gradient = timing.slope.value
            # to be sure to get an arrow in the array plot for each telescope
            # might have the wrong direction though
            if abs(time_gradient) < 0.2:
                time_gradient = 1.0

            noGood = False
            parametrized_images[tel_id] = image_parameters
            time_gradients[tel_id] = time_gradient

            print("Image parametrized!")
            print(image_parameters)

            if image_parameters.width == 0 or image_parameters.width == np.nan:
                noGood = True
                event.dl1.tel[tel_id].parameters.hillas = HillasParametersContainer(
                    x = float("nan") * u.deg,
                    y = float("nan") * u.deg,
                    r = float("nan") * u.deg,
                    width = float("nan") * u.deg,
                    length = float("nan") * u.deg
                )
            else:
                event.dl1.tel[tel_id].parameters.hillas = image_parameters
                print(event.dl1.tel[tel_id].parameters.hillas)

        except HillasParameterizationError as e:
            print(f"WARNING: {e}")
            noGood = True
            event.dl1.tel[tel_id].parameters.hillas = HillasParametersContainer(
                x = float("nan") * u.deg,
                y = float("nan") * u.deg,
                r = float("nan") * u.deg,
                width = float("nan") * u.deg,
                length = float("nan") * u.deg
            )

    if len(parametrized_images) < 2:  # discard events with < 2 images
        print("WARNING: Less than 2 images survived the cleaning!")
        print("No Direction reconstruction will be performed.")
        continue
    else:  # otherwise continue with direction reconstruction

        # ======================================================================
        #                       DIRECTION RECONSTRUCTION
        # ======================================================================

        reconstructor(event)

        # ======================================================================
        #                           VISUALIZATION
        # ======================================================================

        # GENERAL SETTINGS

        plt.subplots_adjust(left=0.075, hspace=0.05)

        n_rows = 1
        n_cols = 2

        if source.max_events:
            plt.suptitle(
                f"EVENT #{event.count} of {source.max_events} with ID #{event.index.event_id}"
            )
        else:
            plt.suptitle(f"EVENT #{event.count} with ID #{event.index.event_id}")

        length = 250

        angle_offset = event.pointing.array_azimuth

        core_dict = {
            tel_id: dl1.parameters.core.psi for tel_id, dl1 in event.dl1.tel.items()
        }

        # COORDINATES OF THE IMPACT CORE ON THE GROUND

        ground_frame = GroundFrame()
        simulated_core = SkyCoord(
            x=event.simulation.shower.core_x,
            y=event.simulation.shower.core_y,
            z=0,
            unit=u.m,
            frame=ground_frame,
        )
        reconstructed_core = SkyCoord(
            x=event.dl2.shower["HillasReconstructor"].core_x,
            y=event.dl2.shower["HillasReconstructor"].core_y,
            z=0,
            unit=u.m,
            frame=ground_frame,
        )
        error_core = u.Quantity(
            np.sqrt(
                np.power(reconstructed_core.x - simulated_core.x, 2)
                + np.power(reconstructed_core.y - simulated_core.y, 2)
            ),
            unit=u.m,
        )

        print("COORDINATES OF THE SHOWER'S CORE ON THE GROUND:")
        print(f"- simulated : x = {simulated_core.x}, y = {simulated_core.y}")
        print(
            f"- reconstructed : x = {reconstructed_core.x}, y = {reconstructed_core.y}"
        )
        print(f"- error = {error_core}")

        # plot of the ground frame and core positions

        plt.subplot(n_rows, n_cols, 1, aspect="equal")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")

        array_disp = ArrayDisplay(
            subarray,
            autoupdate=True,
            tel_scale=2.0,
            alpha=0.7,
            title="SHOWER'S CORE POSITION (GROUNDFRAME)",
            radius=None,
        )
        array_disp.add_labels()
        array_disp.set_line_hillas(parametrized_images, core_dict, range=100)

        plt.plot(
            simulated_core.x,
            simulated_core.y,
            "*",
            color="black",
            label="simulated core",
        )
        plt.plot(
            reconstructed_core.x,
            reconstructed_core.y,
            "o",
            color="red",
            alpha=0.5,
            fillstyle="none",
            markersize=5,
            label="reconstructed core",
        )

        plt.legend(loc="best")

        # COORDINATES OF THE SHOWER'S DIRECTION IN THE SKY

        # Transform all positions into the nominal frame
        nominal_frame = NominalFrame(origin=array_pointing)
        simulated_direction = SkyCoord(
            alt=event.simulation.shower.alt,
            az=event.simulation.shower.az,
            frame="altaz",
        ).transform_to(nominal_frame)
        reconstructed_direction = SkyCoord(
            alt=event.dl2.shower["HillasReconstructor"].alt,
            az=event.dl2.shower["HillasReconstructor"].az,
            frame="altaz",
        ).transform_to(
            nominal_frame
        )  # coming from TelescopeFrame, axes are flipped!

        error_direction = simulated_direction.separation(reconstructed_direction)

        print("COORDINATES OF THE SHOWER'S DIRECTION IN THE SKY:")
        print(
            f"- simulated : \
        lon = {simulated_direction.fov_lon}, \
        lat = {simulated_direction.fov_lat}"
        )
        print(
            f"- reconstructed : \
        lon = {reconstructed_direction.fov_lon}, \
        lat = {reconstructed_direction.fov_lat}"
        )
        print(f"- error = {error_direction}")

        # plot of the shower's direction in the nominal frame

        plt.subplot(n_rows, n_cols, 2, aspect=1)
        plt.title("SHOWER'S DIRECTION IN THE SKY (NOMINAL FRAME)")
        plt.xlabel("Field of View Longitude [deg]")
        plt.ylabel("Field of View Latitude [deg]")
        ax = plt.gca()
        cmap = plt.cm.get_cmap("tab20c", len(parametrized_images))

        for i, tel_id in enumerate(parametrized_images.keys()):

            # image = np.zeros_like(parametrized_images[tel_id])

            telescopeframe = TelescopeFrame(
                telescope_pointing=telescope_pointings[tel_id]
            )

            # Plot camera
            nominal_geometry = geometry_TelescopeFrame[tel_id].transform_to(
                nominal_frame
            )
            pixel_coord = SkyCoord(
                fov_lon=nominal_geometry.pix_x,
                fov_lat=nominal_geometry.pix_y,
                frame=nominal_frame,
            )

            # Plot Hillas ellipses

            centerOfgravity_telescopeFrame = SkyCoord(
                fov_lon=parametrized_images[tel_id].x,
                fov_lat=parametrized_images[tel_id].y,
                frame=telescopeframe,
            )
            centerOfgravity = centerOfgravity_telescopeFrame.transform_to(nominal_frame)
            length = u.Quantity(parametrized_images[tel_id].length).value
            width = u.Quantity(parametrized_images[tel_id].width).value
            psi = u.Quantity(parametrized_images[tel_id].psi).to("deg")

            ellipse = Ellipse(
                xy=(centerOfgravity.fov_lon.value, centerOfgravity.fov_lat.value),
                width=length,
                height=width,
                angle=psi.value,
                fill=False,
                linewidth=3,
                color=cmap(i),
                label=f"Tel #{tel_id}",
            )
            ax.add_patch(ellipse)

            if args.show_pixels:
                ax.scatter(
                    x=pixel_coord.fov_lon.deg,
                    y=pixel_coord.fov_lat.deg,
                    s=30,
                    color=cmap(i),
                    alpha=0.25,
                )

        plt.plot(
            simulated_direction.fov_lon,
            simulated_direction.fov_lat,
            "*",
            color="black",
            label="simulated",
        )
        plt.plot(
            reconstructed_direction.fov_lon,
            reconstructed_direction.fov_lat,
            "o",
            alpha=0.5,
            color="red",
            fillstyle="none",
            markersize=5,
            label="reconstructed",
        )

        plt.legend(loc="best", fontsize=10)

        plt.show()
