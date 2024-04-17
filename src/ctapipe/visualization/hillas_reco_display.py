#!/usr/bin/env python3

import astropy.units as u
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.visualization import quantity_support
from matplotlib.collections import PatchCollection
from matplotlib.patches import Ellipse

from ..coordinates import GroundFrame, NominalFrame, TelescopeFrame, TiltedGroundFrame
from ..core import Component, traits
from . import ArrayDisplay


class HillasRecoDisplay(Component):
    """Sky and ground ellipse crossings from a Hillas-parameter-style
    reconstruction.
    """

    trace_points = traits.Bool(False, "Accumulate origin points")

    def __init__(self, subarray, config=None, parent=None, figsize=None, **kwargs):
        super().__init__(config=config, parent=parent, **kwargs)

        self.subarray = subarray
        self.figure, ax = plt.subplots(1, 2, constrained_layout=True, figsize=figsize)
        self.ax_origin, self.ax_impact = ax

        self.figure.suptitle(f"Hillas Reconstruction\n{self.subarray.name}")
        self.ax_origin.set_title("Shower Origin (Nominal)")
        self.ax_origin.set_aspect(1.0)
        self.ax_origin.set_xlabel("FOV Longitude / deg")
        self.ax_origin.set_ylabel("FOV Latitude / deg")
        self.ax_origin.set_xlim(-5, 5)
        self.ax_origin.set_ylim(-5, 5)
        self.ax_origin.grid(True)

        self.array_display = ArrayDisplay(
            subarray=subarray, axes=self.ax_impact, frame=GroundFrame()
        )
        self.ax_impact.set_title("Shower Impact (Tilted)")

        self.origin_ellipses = None
        self.origin_points = None
        self.impact_points = None

    def __call__(self, event):
        """Update the display with a new event.

        Parameters
        ----------
        event: ctapipe.containers.ArrayEventContainer
            event with hillas parameter and HillasReconstructor information
        """

        self.clear()
        self._update_sky(event)
        self._update_impact(event)

    def _update_impact(self, event):
        """
        Update the ground impact display
        """

        pointing_direction = SkyCoord(
            alt=event.pointing.array_altitude,
            az=event.pointing.array_azimuth,
            frame="altaz",
        )

        hillas_dict = {tid: tel.parameters.hillas for tid, tel in event.dl1.tel.items()}
        core_dict = {tid: tel.parameters.core.psi for tid, tel in event.dl1.tel.items()}
        time_gradient_dict = {
            tid: -tel.parameters.hillas.skewness for tid, tel in event.dl1.tel.items()
        }

        self.array_display.set_vector_hillas(
            hillas_dict,
            core_dict,
            length=500,
            time_gradient=time_gradient_dict,
            angle_offset=event.pointing.array_azimuth,
        )

        # Overlay the true point of origin
        true_core_pos = SkyCoord(
            x=event.simulation.shower.core_x,
            y=event.simulation.shower.core_y,
            z=0 * u.m,
            pointing_direction=pointing_direction,
            frame=TiltedGroundFrame,
        ).transform_to(self.array_display.frame)

        with quantity_support():
            self.impact_points = self.ax_impact.scatter(
                true_core_pos.x.to_value("m"),
                true_core_pos.y.to_value("m"),
                marker="+",
                s=50,
                color="red",
            )

        # reco_shower = event.dl2.stereo.geometry["HillasReconstructor"]

    def _update_sky(self, event):
        """
        Update the sky display
        """
        fov_center = SkyCoord(
            az=event.pointing.array_azimuth,
            alt=event.pointing.array_altitude,
            frame="altaz",
        )

        nominal_frame = NominalFrame(origin=fov_center)

        ellipse_list = []

        for tel_id, dl1 in event.dl1.tel.items():
            pointing_direction = SkyCoord(
                alt=event.pointing.tel[tel_id].altitude,
                az=event.pointing.tel[tel_id].azimuth,
                frame="altaz",
            )
            tel_frame = TelescopeFrame(telescope_pointing=pointing_direction)
            lon = dl1.parameters.hillas.fov_lon
            lat = dl1.parameters.hillas.fov_lat

            coord = SkyCoord(lon, lat, frame=tel_frame).transform_to(nominal_frame)

            ellipse_list.append(
                Ellipse(
                    xy=(coord.fov_lon.to_value("deg"), coord.fov_lat.to_value("deg")),
                    width=2 * dl1.parameters.hillas.length.to_value("deg"),
                    height=2 * dl1.parameters.hillas.width.to_value("deg"),
                    angle=dl1.parameters.hillas.psi.to_value("deg"),
                    fill=True,
                    alpha=0.5,
                    color="blue",
                    linewidth=0,
                )
            )

        self.origin_ellipses = PatchCollection(ellipse_list, match_original=True)
        self.ax_origin.add_collection(self.origin_ellipses)

        # Overlay the true point of origin
        true_origin = SkyCoord(
            az=event.simulation.shower.az,
            alt=event.simulation.shower.alt,
            frame="altaz",
        ).transform_to(nominal_frame)

        self.origin_points = self.ax_origin.scatter(
            [
                true_origin.fov_lon.to_value("deg"),
            ],
            [
                true_origin.fov_lat.to_value("deg"),
            ],
            marker="+",
            s=50,
            color="red",
        )

    def clear(
        self,
    ):
        """Removes the existing ellipses and points."""
        if self.origin_ellipses:
            self.origin_ellipses.remove()
            self.origin_ellipses = None

        if self.origin_points:
            if self.trace_points:
                self.origin_points.set_color("grey")
                self.origin_points.set_alpha(0.5)
            else:
                self.origin_points.remove()
                self.origin_points = None

        if self.impact_points:
            self.impact_points.remove()
            self.impact_points = None
