import astropy.units as u
import numpy as np

from ...core import TelescopeComponent


class PointingCalibrator(TelescopeComponent):
    """
    Calibrates telescope pointing by evaluating pre-interpolated structural
    pointing and structural displacement containers from event monitoring data.
    """

    def __call__(self, event) -> None:
        """
        Calibrate pointing for all triggered telescopes in an event.

        Parameters
        ----------
        event : ctapipe.containers.DataContainer
            The event to calibrate.
        """
        for tel_id in event.trigger.tels_with_trigger:
            if tel_id not in event.monitoring.tel:
                continue

            mon = event.monitoring.tel[tel_id]
            pointing_container = mon.structure_pointing
            if pointing_container is None:
                self.log.warning(
                    "Structure pointing data not available for telescope %s. "
                    "Skipping pointing calibration for this telescope.",
                    tel_id,
                )
                continue

            if not self._apply_structure_displacement(pointing_container, mon):
                self.log.warning(
                    "Structure displacement data not available for telescope %s. "
                    "Using raw structure pointing.",
                    tel_id,
                )
            event.monitoring.tel[tel_id].pointing = pointing_container

    def _apply_structure_displacement(self, pointing_container, mon_tel) -> bool:
        """
        Apply structural displacement to raw structure pointing.
        """
        displacement = mon_tel.structure_displacement
        if displacement is None:
            self.log.warning(
                "No structure displacement data available, using raw structure pointing."
            )
            return False
        # Combine raw encoder positions with structural displacement offsets
        alt_corr = pointing_container.altitude + displacement.delta_altitude
        az_corr = (pointing_container.azimuth + displacement.delta_azimuth) % (
            2 * np.pi * u.rad
        )
        pointing_container.update(
            azimuth=az_corr,
            altitude=alt_corr,
        )

        return True

    def _apply_camera_displacement(self, event, tel_id):
        """
        Apply the camera displacement to the pointing of the telescope.

        Parameters
        ----------
        event: ctapipe.containers.DataContainer
            The event to calibrate.
        tel_id: int
            The telescope ID to calibrate.
        """
        raise NotImplementedError("_apply_camera_displacement is not yet implemented.")

    def _apply_pointing_correction(self, event, tel_id):
        """
        Apply the pointing correction to the pointing of the telescope.

        Parameters
        ----------
        event: ctapipe.containers.DataContainer
            The event to calibrate.
        tel_id: int
            The telescope ID to calibrate.
        """
        raise NotImplementedError("_apply_pointing_correction is not yet implemented.")
