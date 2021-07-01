"""
Plot examples of reconstructed core position on the ground and hillas circles on the sky.
"""

# FROM THIRD-PARTY LIBRARIES
import numpy as np
import astropy.units as u
from astropy.coordinates import AltAz, SkyCoord
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from traitlets import Bool, Unicode, Int

# FROM CTAPIPE
from ctapipe.core import Tool
from ctapipe.core.traits import classes_with_traits
from ctapipe.containers import HillasParametersContainer, ImageParametersContainer
from ctapipe.utils import get_dataset_path
from ctapipe.io import EventSource
from ctapipe.coordinates import TelescopeFrame, NominalFrame, GroundFrame
from ctapipe.calib import CameraCalibrator
from ctapipe.image import (
    mars_cleaning_1st_pass,
    hillas_parameters,
    timing_parameters,
    HillasParameterizationError,
)
from ctapipe.reco import HillasReconstructor
from ctapipe.visualization import ArrayDisplay

class StereoRecoPlots(Tool):

    name = "ctapipe-examples-display-reconstruction"
    description = Unicode(__doc__)

    use_default_test_file = Bool(
        help="If True use always gamma_LaPalma_baseline_20Zd_180Az_prod3b_test.simtel.gz", default_value=True
    ).tag(config=True)

    show_pixels = Bool(
        help="Show pixel positions in NominalFrame", default_value=True
    ).tag(config=True)

    event_id = Int(
        help="Show a specific event"
    ).tag(config=True)

    aliases = {
        ("i", "input"): "EventSource.input_url",
        ("t", "allowed-tels"): "EventSource.allowed_tels",
        ("m", "max-events"): "EventSource.max_events",
        ("p", "show-pixels"): "StereoRecoPlots.show_pixels",
        ("d", "event-id"): "StereoRecoPlots.event_id",
    }

    classes = (classes_with_traits(EventSource))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup(self):

        if self.use_default_test_file:
            EventSource.input_url.default_value = get_dataset_path("gamma_LaPalma_baseline_20Zd_180Az_prod3b_test.simtel.gz")
        self.event_source = EventSource(parent=self)

        self.calibrate = CameraCalibrator(
            parent=self, subarray=self.event_source.subarray
        )
        self.reconstruct = HillasReconstructor(
            parent=self, subarray=self.event_source.subarray
        )

        self.horizon_frame = AltAz()

    def start(self):
      
        for event in self.event_source:

            # Show only a specific event
            if self.event_id:
                if event.index.event_id != self.event_id:
                    continue

            array_pointing = SkyCoord(
                az=event.pointing.array_azimuth,
                alt=event.pointing.array_altitude,
                frame=self.horizon_frame,
            )
            
            self.calibrate(event)

            telescope_pointings = {}
            geometry_TelescopeFrame = {}  # we save it for plotting later
            parametrized_images = {}
            time_gradients = {}

            for tel_id in event.r1.tel.keys():

                # Camera information
                camera = self.event_source.subarray.tel[tel_id].camera
                cam_id = camera.camera_name
                geometry_CameraFrame = camera.geometry

                # Pointing direction of this telescope
                telescope_pointings[tel_id] = SkyCoord(
                    alt=event.pointing.tel[tel_id].altitude,
                    az=event.pointing.tel[tel_id].azimuth,
                    frame=self.horizon_frame,
                )

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
                    picture_thresh=6,
                    boundary_thresh=3,
                    keep_isolated_pixels=False,
                    min_number_picture_neighbors=1,
                )
                # set all rejected pixels to zero
                cleaned_image[~cleaning_mask] = 0

                # IMAGE PARAMETRIZATION

                event.dl1.tel[tel_id].parameters = ImageParametersContainer()

                try:

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
                
                    time_gradient = timing.slope.value

                    # to be sure to get an arrow in the array plot for each telescope
                    # might have the wrong direction though
                    if abs(time_gradient) < 0.2:
                        time_gradient = 1.0

                    parametrized_images[tel_id] = image_parameters
                    time_gradients[tel_id] = time_gradient

                    self.log.debug("Image parametrized!")
                    self.log.debug(image_parameters)

                    if image_parameters.width == 0 or image_parameters.width == np.nan:
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
                    event.dl1.tel[tel_id].parameters.hillas = HillasParametersContainer(
                        x = float("nan") * u.deg,
                        y = float("nan") * u.deg,
                        r = float("nan") * u.deg,
                        width = float("nan") * u.deg,
                        length = float("nan") * u.deg
                    )

            if len(parametrized_images) < 2:  # discard events with < 2 images
                self.log.warning("Less than 2 images survived the cleaning!")
                self.log.warning("No direction reconstruction will be performed.")
                continue
            else:

                self.reconstruct(event)

                plt.subplots_adjust(left=0.075, hspace=0.05)

                n_rows = 1
                n_cols = 2

                if self.event_source.max_events:
                    plt.suptitle(
                        f"EVENT #{event.count} of {self.event_source.max_events} with ID #{event.index.event_id}"
                    )
                else:
                    plt.suptitle(f"EVENT #{event.count} with ID #{event.index.event_id}")

                length = 250

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
                    x=event.dl2.stereo.geometry["HillasReconstructor"].core_x,
                    y=event.dl2.stereo.geometry["HillasReconstructor"].core_y,
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
                    self.event_source.subarray,
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
                    alt=event.dl2.stereo.geometry["HillasReconstructor"].alt,
                    az=event.dl2.stereo.geometry["HillasReconstructor"].az,
                    frame="altaz",
                ).transform_to(
                    nominal_frame
                )  # coming from TelescopeFrame, axes are flipped!

                error_direction = simulated_direction.separation(reconstructed_direction)

                self.log.info("COORDINATES OF THE SHOWER'S DIRECTION IN THE SKY:")
                self.log.info(
                    f"- simulated : \
                lon = {simulated_direction.fov_lon}, \
                lat = {simulated_direction.fov_lat}"
                )
                self.log.info(
                    f"- reconstructed : \
                lon = {reconstructed_direction.fov_lon}, \
                lat = {reconstructed_direction.fov_lat}"
                )
                self.log.info(f"- error = {error_direction}")

                # plot of the shower's direction in the nominal frame

                plt.subplot(n_rows, n_cols, 2, aspect=1)
                plt.title("SHOWER'S DIRECTION IN THE SKY (NOMINAL FRAME)")
                plt.xlabel("Field of View Longitude [deg]")
                plt.ylabel("Field of View Latitude [deg]")
                ax = plt.gca()
                cmap = plt.cm.get_cmap("tab20c", len(parametrized_images))

                for i, tel_id in enumerate(parametrized_images.keys()):

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

                    cog_telescope_frame = SkyCoord(
                        fov_lon=parametrized_images[tel_id].x,
                        fov_lat=parametrized_images[tel_id].y,
                        frame=telescopeframe,
                    )
                    cog = cog_telescope_frame.transform_to(nominal_frame)
                    length = u.Quantity(parametrized_images[tel_id].length).value
                    width = u.Quantity(parametrized_images[tel_id].width).value
                    psi = u.Quantity(parametrized_images[tel_id].psi).to("deg")

                    ellipse = Ellipse(
                        xy=(cog.fov_lon.value, cog.fov_lat.value),
                        width=length,
                        height=width,
                        angle=psi.value,
                        fill=False,
                        linewidth=3,
                        color=cmap(i),
                        label=f"Tel #{tel_id}",
                    )
                    ax.add_patch(ellipse)

                    if self.show_pixels:

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

            # If a specific event no need to continue
            if self.event_id:
                exit()
    

def main():
    """ run the app """
    tool = StereoRecoPlots()
    tool.run()


if __name__ == "__main__":
    main()