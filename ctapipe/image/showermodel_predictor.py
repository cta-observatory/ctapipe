import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, SkyCoord
from erfa.ufunc import s2p as spherical_to_cartesian

from ctapipe.coordinates import CameraFrame, TelescopeFrame

__all__ = ["ShowermodelPredictor"]


class ShowermodelPredictor:
    def __init__(
        self,
        subarray,
        showermodel=None,
    ):
        """Creates images of a given showermodel for a set of telescopes given some telescope parameters.

        Parameters
        ----------
        subarray: SubarrayDescription
        showermodel: optional, e.g. GaussianShowermodel
            Model description of shower properties
        """
        tel_positions = {}
        tel_solid_angles = {}
        tel_mirror_area = {}
        for tel_id in subarray.tel.keys():
            tel_positions[tel_id] = subarray.positions[tel_id].to_value(u.m)
            geometry = subarray.tel[tel_id].camera.geometry
            tel_solid_angles[tel_id] = geometry.transform_to(
                TelescopeFrame()
            ).pix_area.to_value(u.rad**2)
            tel_mirror_area[tel_id] = subarray.tel[tel_id].optics.mirror_area.to_value(
                u.m**2
            )

        self.tel_positions = tel_positions
        self.tel_solid_angles = tel_solid_angles
        self.tel_mirror_area = tel_mirror_area
        self.showermodel = showermodel
        self.subarray = subarray

    def generate_images(self):
        """Predicts images for telescopes."""
        imgs = {}
        for tel_id, pix_coords_altaz in self.tel_pix_coords_altaz.items():
            area = self.tel_mirror_area[tel_id]
            solid_angle = self.tel_solid_angles[tel_id]
            vec_oc = self._vec_oc[tel_id]
            vec_pointing = self._telescope_axis(pix_coords_altaz[0])
            imgs[tel_id] = self._generate_img(
                area, solid_angle, vec_oc, pix_coords_altaz, vec_pointing
            )

        return imgs

    def _generate_img(self, area, solid_angle, vec_oc, pix_coords_altaz, vec_pointing):
        """Generates one image of the shower for a given telescope.

        Parameters
        ----------
        area : u.Quantity[length**2]
            Area of the telescope mirror
        solid_angle : u.Quantity[Angle**2]
            Solid angle for each pixel as a 1d-quantity of shape (n_pixels)
        vec_oc : u.Quantity[length]
            Vector from optical center of telescope to barycenter of shower
        pix_coords_altaz : u.Quantity[Angle]
            Pointing of the pixels in AltAz as a 1d-quantity of shape (n_pixels)
        vec_pointing : u.Quantity[length]
            Unit vector of the telescope axis/pointing
        """
        pix_content = self._photons(
            area, solid_angle, vec_oc, pix_coords_altaz, vec_pointing
        )
        return pix_content

    @property
    def _vec_oc(self):
        """Calculates vector of optical center of each telescope to the barycenter of the shower."""
        vec = {}
        for tel_id, position in self.tel_positions.items():
            vec[tel_id] = self.showermodel.barycenter + [
                -position[1],
                position[0],
                position[2],
            ]  # Corsika (y,-x,z)-> (x,y,z)
        return vec

    def _vec_los(self, pix_altaz):
        """Calculates unit vector for each pixel pointing/altaz coord along the line of sight.

        Parameters
        ----------
        pix_altaz : u.Quantity[Angle]
            Pointing/AltAz of the pixel along the line of sight as a 1d-quantity of shape (n_pixels)
        """
        vec = spherical_to_cartesian(pix_altaz[:, 1], pix_altaz[:, 0], 1.0)
        return vec

    def _telescope_axis(self, tel_pointing):
        """Calculates the unit vector of the telescope axis.

        Parameters
        ----------
        tel_pointing : u.Quantity[Angle]
            Pointing of the telescope in AltAz
        """
        vec = spherical_to_cartesian(tel_pointing[1], tel_pointing[0], 1.0)
        return vec

    def _photons(self, area, solid_angle, vec_oc, pix_coords_altaz, vec_pointing):
        """Calculates the photons contained in a pixel of interest.
        See https://arxiv.org/pdf/astro-ph/0601373.pdf Equation (1).

        Parameters
        ----------
        area : u.Quantity[length**2]
            Area of the mirror
        solid_angle : u.Quantity[Angle**2]
            Solid angle of the pixels as 1d-quantity of shape (n_pixels)
        vec_oc : u.Quantity[length]
            Vector from optical center of the telescope to barycenter of the shower
        pix_coords_altaz : u.Quantity[Angle]
            Pointing of the pixels in AltAz as a 1d-quantity of shape (n_pixels)
        vec_pointing : u.Quantity[lenght]
            Unit vector along the telescope axis
        """
        vec_los = self._vec_los(pix_coords_altaz)
        epsilon = np.arccos(
            np.clip(
                vec_los @ self.showermodel.vec_shower_axis,
                a_min=-1,
                a_max=1,
            )
        )

        theta = np.arccos(
            np.clip(
                vec_los @ vec_pointing,
                a_min=-1,
                a_max=1,
            )
        )

        photons = (
            area
            * solid_angle
            * self.showermodel.emission_probability(epsilon)
            * np.cos(theta)
            * self.showermodel.photon_integral(vec_oc, vec_los, epsilon)
        )

        return photons

    def pointing(self, event):
        """Set the pointing of the pixels"""
        self.tel_pix_coords_altaz = self._tel_pix_coords_altaz(event)

    def _tel_pix_coords_altaz(self, event):
        """Helper function calculating pixel pointing in AltAz"""
        tel_pix_coords_altaz = {}
        for tel_id in event.dl1.tel.keys():
            geometry = self.subarray.tel[tel_id].camera.geometry
            # (x,y)->(y,x) since this is also in a NorthingEasting frame instead of EastingNorthing similar to tel_positions
            pix_x = geometry.pix_y
            pix_y = geometry.pix_x
            focal_length = self.subarray.tel[tel_id].optics.equivalent_focal_length

            pointing = event.pointing.tel[tel_id]
            altaz = AltAz(az=pointing.azimuth, alt=pointing.altitude)
            camera_frame = CameraFrame(
                focal_length=focal_length, telescope_pointing=altaz
            )

            cam_coords = SkyCoord(x=pix_x, y=pix_y, frame=camera_frame)

            cam_altaz = cam_coords.transform_to(AltAz())
            tel_pix_coords_altaz[tel_id] = np.stack(
                (cam_altaz.alt.to_value(u.rad), cam_altaz.az.to_value(u.rad)), -1
            )

        return tel_pix_coords_altaz
