"""
High level muon analysis  (MuonProcessor Component)
"""
import numpy as np

from ctapipe.containers import (
    ArrayEventContainer,
    MuonCameraContainer,
    MuonParametersContainer,
)
from ctapipe.coordinates import TelescopeFrame
from ctapipe.core import QualityQuery, TelescopeComponent
from ctapipe.core.traits import FloatTelescopeParameter, List, Tuple, Unicode

from .features import (
    intensity_ratio_inside_ring,
    mean_squared_error,
    ring_completeness,
    ring_containment,
)
from .intensity_fitter import MuonIntensityFitter
from .ring_fitter import MuonRingFitter


class DL1Query(QualityQuery):
    """
    For configuring quality checks performed on image parameters
    before rings are fitted.
    """

    quality_criteria = List(
        Tuple(Unicode(), Unicode()),
        default_value=[
            ("min_pixels", "p.morphology.n_pixels > 100"),
            ("min_intensity", "p.hillas.intensity > 500"),
        ],
        help=(
            "list of tuples of ('<description', 'expression string') to accept "
            "(select) a given data value.  E.g. `[('mycut', 'x > 3'),]. "
            "You may use `numpy` as `np` and `astropy.units` as `u`, but no other"
            " modules."
        ),
    ).tag(config=True)


class RingQuery(QualityQuery):
    """
    For configuring quality checks performed on the extracted rings before
    computing efficiency parameters.
    """

    quality_criteria = List(
        Tuple(Unicode(), Unicode()),
        default_value=[
            ("radius_not_nan", "np.isfinite(ring.radius.value)"),
            ("min_pixels", "np.count_nonzero(mask) > 50"),
            ("ring_containment", "parameters.containment > 0.5"),
        ],
        help=(
            "list of tuples of ('<description', 'expression string') to accept "
            "(select) a given data value.  E.g. `[('mycut', 'x > 3'),]. "
            "You may use `numpy` as `np` and `astropy.units` as `u`, but no other"
            " modules."
        ),
    ).tag(config=True)


class MuonProcessor(TelescopeComponent):
    """
    Takes cleaned images and extracts muon rings. Should be run after ImageProcessor.
    """

    completeness_threshold = FloatTelescopeParameter(
        default_value=30.0, help="Threshold for calculating the ``ring_completeness``"
    ).tag(config=True)

    ratio_width = FloatTelescopeParameter(
        default_value=1.5,
        help=(
            "Ring width for intensity ratio"
            " computation as multiple of pixel diameter"
        ),
    ).tag(config=True)

    pedestal = FloatTelescopeParameter(
        help="Pedestal noise rms", default_value=1.1
    ).tag(config=True)

    def __init__(self, subarray, **kwargs):
        super().__init__(subarray, **kwargs)
        self.dl1_query = DL1Query(parent=self)
        self.ring_query = RingQuery(parent=self)
        self.geometries = {
            tel_id: tel.camera.geometry.transform_to(TelescopeFrame())
            for tel_id, tel in self.subarray.tel.items()
        }
        self.fov_radius = {
            tel_id: geometry.guess_radius()
            for tel_id, geometry in self.geometries.items()
        }

        self.ring_fitter = MuonRingFitter(parent=self)

        self.intensity_fitter = MuonIntensityFitter(subarray=subarray, parent=self)

    def __call__(self, event: ArrayEventContainer):
        for tel_id in event.dl1.tel:
            self._process_telescope_event(event, tel_id)

    def _process_telescope_event(self, event, tel_id):
        """
        Extract and process a ring from a single image.

        Parameters
        ----------
        event: ArrayEventContainer
            Collection of all event information
        tel_id: int
            Telescope ID of the instrument that has measured the image
        """
        event_index = event.index
        event_id = event_index.event_id

        if self.subarray.tel[tel_id].optics.n_mirrors != 1:
            self.log.warning(
                f"Skipping non-single mirror telescope {tel_id}"
                " set --allowed_tels to get rid of this warning"
            )
            event.muon.tel[tel_id] = MuonCameraContainer()
            return

        self.log.debug(f"Processing event {event_id}, telescope {tel_id}")
        dl1 = event.dl1.tel[tel_id]
        image = dl1.image
        mask = dl1.image_mask
        if mask is None:
            mask = image > 0

        checks = self.dl1_query(p=dl1.parameters)

        if not all(checks):
            event.muon.tel[tel_id] = MuonCameraContainer()
            return

        geometry = self.geometries[tel_id]
        x = geometry.pix_x
        y = geometry.pix_y

        # iterative ring fit.
        # First use cleaning pixels, then only pixels close to the ring
        # three iterations seems to be enough for most rings
        for _ in range(3):
            ring = self.ring_fitter(x, y, image, mask)
            dist = np.sqrt((x - ring.center_x) ** 2 + (y - ring.center_y) ** 2)
            mask = np.abs(dist - ring.radius) / ring.radius < 0.4

        parameters = self._calculate_muon_parameters(
            tel_id, image, dl1.image_mask, ring
        )

        checks = self.ring_query(parameters=parameters, ring=ring, mask=mask)
        if not all(checks):
            event.muon.tel[tel_id] = MuonCameraContainer(
                parameters=parameters, ring=ring
            )
            return

        result = self.intensity_fitter(
            tel_id,
            ring.center_x,
            ring.center_y,
            ring.radius,
            image,
            mask=mask,
            pedestal=np.full(mask.shape, self.pedestal.tel[tel_id]),
        )

        self.log.info(
            f"Muon fit: r={ring.radius:.2f}"
            f", width={result.width:.4f}"
            f", efficiency={result.optical_efficiency:.2%}"
        )

        event.muon.tel[tel_id] = MuonCameraContainer(
            ring=ring, efficiency=result, parameters=parameters
        )

    def _calculate_muon_parameters(
        self, tel_id, image, clean_mask, ring
    ) -> MuonParametersContainer:
        """
        Calculate features from identified muon rings.

        Parameters
        ----------
        tel_id: int
            Telescope ID of the instrument that has measured the image
        image: np.ndarray
            Image to process
        clean_mask: np.ndarray[bool]
            DL1 Image cleaning mask
        ring: MuonRingContainer
            Collection of the fitted rings parameters

        Returns
        -------
        MuonParametersContainer:
            Collection of the fitted rings containment in the camera,
            completeness, intensity ratio and the pixels MSE around
            the fitted ring.
        """
        if np.isnan(ring.radius.value):
            return MuonParametersContainer()

        geometry = self.geometries[tel_id]
        fov_radius = self.fov_radius[tel_id]
        x = geometry.pix_x
        y = geometry.pix_y

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

        pixel_width = geometry.pixel_width[clean_mask]
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
