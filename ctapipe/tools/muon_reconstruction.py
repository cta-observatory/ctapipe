from tqdm import tqdm
import numpy as np
from astropy.coordinates import SkyCoord
from ctapipe.containers import TelEventIndexContainer

from ctapipe.calib import CameraCalibrator
from ctapipe.core import Provenance
from ctapipe.core import Tool, ToolConfigurationError
from ctapipe.core import traits
from ctapipe.io import EventSource
from ctapipe.io import HDF5TableWriter
from ctapipe.image.cleaning import TailcutsImageCleaner
from ctapipe.coordinates import TelescopeFrame, CameraFrame
from ctapipe.containers import MuonParametersContainer
from ctapipe.instrument import CameraGeometry

from ctapipe.image.muon import (
    MuonRingFitter,
    MuonIntensityFitter,
    ring_containment,
    ring_completeness,
    intensity_ratio_inside_ring,
    mean_squared_error,
)


class MuonAnalysis(Tool):
    """
    Detect and extract muon ring parameters, and write the muon ring and
    intensity parameters to an output table.

    The resulting output can be read e.g. using for example
    `pandas.read_hdf(filename, 'dl1/event/telescope/parameters/muon')`
    """

    name = "ctapipe-reconstruct-muons"
    description = traits.Unicode(__doc__)

    output = traits.Path(directory_ok=False, help="HDF5 output file name").tag(
        config=True
    )

    completeness_threshold = traits.FloatTelescopeParameter(
        default_value=30.0, help="Threshold for calculating the ``ring_completeness``"
    ).tag(config=True)

    ratio_width = traits.FloatTelescopeParameter(
        default_value=1.5,
        help=(
            "Ring width for intensity ratio"
            " computation as multiple of pixel diameter"
        ),
    ).tag(config=True)

    overwrite = traits.Bool(
        default_value=False, help="If true, overwrite outputfile without asking"
    ).tag(config=True)

    min_pixels = traits.IntTelescopeParameter(
        help=(
            "Minimum number of pixels after cleaning and ring finding"
            "required to process an event"
        ),
        default_value=100,
    ).tag(config=True)

    pedestal = traits.FloatTelescopeParameter(
        help="Pedestal noise rms", default_value=1.1
    ).tag(config=True)

    classes = [
        CameraCalibrator,
        TailcutsImageCleaner,
        EventSource,
        MuonRingFitter,
        MuonIntensityFitter,
    ]

    aliases = {
        "i": "EventSource.input_url",
        "input": "EventSource.input_url",
        "o": "MuonAnalysis.output",
        "output": "MuonAnalysis.output",
        "max-events": "EventSource.max_events",
        "allowed-tels": "EventSource.allowed_tels",
    }

    flags = {
        "overwrite": ({"MuonAnalysis": {"overwrite": True}}, "overwrite output file")
    }

    def setup(self):
        if self.output is None:
            raise ToolConfigurationError("You need to provide an --output file")

        if self.output.exists() and not self.overwrite:
            raise ToolConfigurationError(
                "Outputfile {self.output} already exists, use `--overwrite` to overwrite"
            )

        self.source = EventSource(parent=self)
        subarray = self.source.subarray

        self.calib = CameraCalibrator(subarray=subarray, parent=self)
        self.ring_fitter = MuonRingFitter(parent=self)
        self.intensity_fitter = MuonIntensityFitter(subarray=subarray, parent=self)
        self.cleaning = TailcutsImageCleaner(parent=self, subarray=subarray)
        self.writer = HDF5TableWriter(
            self.output, "", add_prefix=True, parent=self, mode="w"
        )
        self.pixels_in_tel_frame = {}
        self.field_of_view = {}
        self.pixel_widths = {}

        for p in ["min_pixels", "pedestal", "ratio_width", "completeness_threshold"]:
            getattr(self, p).attach_subarray(self.source.subarray)

    def start(self):
        for event in tqdm(self.source, desc="Processing events: "):
            self.process_array_event(event)

    def process_array_event(self, event):
        self.calib(event)

        for tel_id, dl1 in event.dl1.tel.items():
            self.process_telescope_event(event.index, tel_id, dl1)

        self.writer.write(
            "sim/event/subarray/shower", [event.index, event.simulation.shower]
        )

    def process_telescope_event(self, event_index, tel_id, dl1):
        event_id = event_index.event_id

        if self.source.subarray.tel[tel_id].optics.num_mirrors != 1:
            self.log.warn(
                f"Skipping non-single mirror telescope {tel_id}"
                " set --allowed_tels to get rid of this warning"
            )
            return

        self.log.debug(f"Processing event {event_id}, telescope {tel_id}")
        image = dl1.image
        clean_mask = self.cleaning(tel_id, image)

        if np.count_nonzero(clean_mask) <= self.min_pixels.tel[tel_id]:
            self.log.debug(
                f"Skipping event {event_id}-{tel_id}:"
                f" has less then {self.min_pixels.tel[tel_id]} pixels after cleaning"
            )
            return

        x, y = self.get_pixel_coords(tel_id)

        # iterative ring fit.
        # First use cleaning pixels, then only pixels close to the ring
        # three iterations seems to be enough for most rings
        mask = clean_mask
        for i in range(3):
            ring = self.ring_fitter(x, y, image, mask)
            dist = np.sqrt((x - ring.center_x) ** 2 + (y - ring.center_y) ** 2)
            mask = np.abs(dist - ring.radius) / ring.radius < 0.4

        if np.count_nonzero(mask) <= self.min_pixels.tel[tel_id]:
            self.log.debug(
                f"Skipping event {event_id}-{tel_id}:"
                f" Less then {self.min_pixels.tel[tel_id]} pixels on ring"
            )
            return

        if np.isnan(
            [ring.radius.value, ring.center_x.value, ring.center_y.value]
        ).any():
            self.log.debug(
                f"Skipping event {event_id}-{tel_id}: Ring fit did not succeed"
            )
            return

        parameters = self.calculate_muon_parameters(tel_id, image, clean_mask, ring)

        # intensity_fitter does not support a mask yet, set ignored pixels to 0
        image[~mask] = 0

        result = self.intensity_fitter(
            tel_id,
            ring.center_x,
            ring.center_y,
            ring.radius,
            image,
            pedestal=self.pedestal.tel[tel_id],
        )

        self.log.info(
            f"Muon fit: r={ring.radius:.2f}"
            f", width={result.width:.4f}"
            f", efficiency={result.optical_efficiency:.2%}"
        )

        tel_event_index = TelEventIndexContainer(**event_index, tel_id=tel_id)

        self.writer.write(
            "dl1/event/telescope/parameters/muons",
            [tel_event_index, ring, parameters, result],
        )

    def calculate_muon_parameters(self, tel_id, image, clean_mask, ring):
        fov_radius = self.get_fov(tel_id)
        x, y = self.get_pixel_coords(tel_id)

        # add ring containment, not filled in fit
        containment = ring_containment(
            ring.radius, ring.center_x, ring.center_y, fov_radius
        )

        completeness = ring_completeness(
            x,
            y,
            image,
            ring.radius,
            ring.center_x,
            ring.center_y,
            threshold=self.completeness_threshold.tel[tel_id],
        )

        pixel_width = self.get_pixel_width(tel_id)
        intensity_ratio = intensity_ratio_inside_ring(
            x[clean_mask],
            y[clean_mask],
            image[clean_mask],
            ring.radius,
            ring.center_x,
            ring.center_y,
            width=self.ratio_width.tel[tel_id] * pixel_width,
        )

        mse = mean_squared_error(
            x[clean_mask],
            y[clean_mask],
            image[clean_mask],
            ring.radius,
            ring.center_x,
            ring.center_y,
        )

        return MuonParametersContainer(
            containment=containment,
            completeness=completeness,
            intensity_ratio=intensity_ratio,
            mean_squared_error=mse,
        )

    def get_fov(self, tel_id):
        """Guesstimate fov radius for telescope with id `tel_id`"""
        # memoize fov calculation
        if tel_id not in self.field_of_view:
            cam = self.source.subarray.tel[tel_id].camera.geometry
            border = cam.get_border_pixel_mask()

            x, y = self.get_pixel_coords(tel_id)
            self.field_of_view[tel_id] = np.sqrt(x[border] ** 2 + y[border] ** 2).mean()

        return self.field_of_view[tel_id]

    def get_pixel_width(self, tel_id):
        """Guesstimate fov radius for telescope with id `tel_id`"""
        # memoize fov calculation
        if tel_id not in self.pixel_widths:
            x, y = self.get_pixel_coords(tel_id)
            self.pixel_widths[tel_id] = CameraGeometry.guess_pixel_width(x, y)

        return self.pixel_widths[tel_id]

    def get_pixel_coords(self, tel_id):
        """Get pixel coords in telescope frame for telescope with id `tel_id`"""
        # memoize transformation
        if tel_id not in self.pixels_in_tel_frame:
            telescope = self.source.subarray.tel[tel_id]
            cam = telescope.camera.geometry
            camera_frame = CameraFrame(
                focal_length=telescope.optics.equivalent_focal_length,
                rotation=cam.cam_rotation,
            )
            cam_coords = SkyCoord(x=cam.pix_x, y=cam.pix_y, frame=camera_frame)
            tel_coord = cam_coords.transform_to(TelescopeFrame())
            self.pixels_in_tel_frame[tel_id] = tel_coord

        coords = self.pixels_in_tel_frame[tel_id]
        return coords.fov_lon, coords.fov_lat

    def finish(self):
        Provenance().add_output_file(self.output, role="muon_efficiency_parameters")
        self.writer.close()


def main():
    tool = MuonAnalysis()
    tool.run()


if __name__ == "__main__":
    main()
