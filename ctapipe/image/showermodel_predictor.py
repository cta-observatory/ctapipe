import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from numpy.linalg import norm

from ctapipe.coordinates import (
    EastingNorthingFrame,
    GroundFrame,
    altaz_to_righthanded_cartesian,
)

__all__ = ["ShowermodelPredictor"]


class ShowermodelPredictor:
    def __init__(
        self,
        tel_positions,
        tel_pix_coords_altaz,
        tel_solid_angles,
        tel_mirror_area,
        showermodel=None,
    ):
        """Creates images of a given showermodel for a set of telescopes given some telescope parameters.

        Parameters
        ----------
        tel_positions : u.Quantity[length, ndim=2]
            Telescope positions in the GroundFrame of the telescope array as a 2d-quantity of shape (n_telescopes, 3)
        tel_pix_coords_altaz : u.Quantity[Angle, ndim=2]
            AltAz coordinates of the pixels for each telescope as a 2d-quantity of shape (n_telescopes, n_pixels)
        tel_solid_angles : u.Quantity[Angle**2, ndim=2]
            Solid angles of the pixels for each telescope as a 2d-quantity of shape (n_telescopes, n_pixels)
        tel_mirror_area : u.Quantity[length**2]
            Mirror area of each telescope as a 1d-quantity of shape (n_telescopes)
        showermodel: optional, e.g. GaussianShowermodel
            Model description of shower properties
        """
        self.tel_positions = tel_positions
        self.tel_pix_coords_altaz = tel_pix_coords_altaz
        self.tel_solid_angles = tel_solid_angles
        self.tel_mirror_area = tel_mirror_area
        self.showermodel = showermodel

    def generate_images(self):
        """Predicts images for telescopes."""
        imgs = {}
        for tel_id, vec_oc in self._vec_oc.items():
            area = self.tel_mirror_area[tel_id]
            solid_angle = self.tel_solid_angles[tel_id]
            pix_coords_altaz = self.tel_pix_coords_altaz[tel_id]
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
            vec[tel_id] = self.showermodel.barycenter + SkyCoord(
                *position, unit=u.m, frame=GroundFrame()
            ).transform_to(EastingNorthingFrame()).cartesian.xyz.to_value(u.m)
        return vec

    def _vec_los(self, pix_altaz):
        """Calculates unit vector for each pixel pointing/altaz coord along the line of sight.

        Parameters
        ----------
        pix_altaz : u.Quantity[Angle]
            Pointing/AltAz of the pixel along the line of sight as a 1d-quantity of shape (n_pixels)
        """
        vec = altaz_to_righthanded_cartesian(alt=pix_altaz.alt, az=-pix_altaz.az)
        return vec

    def _telescope_axis(self, tel_pointing):
        """Calculates the unit vector of the telescope axis.

        Parameters
        ----------
        tel_pointing : u.Quantity[Angle]
            Pointing of the telescope in AltAz
        """
        vec = altaz_to_righthanded_cartesian(alt=tel_pointing.alt, az=-tel_pointing.az)
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
                np.einsum("ni,i->n", vec_los, self.showermodel.vec_shower_axis)
                / (norm(vec_los, axis=1) * norm(self.showermodel.vec_shower_axis)),
                a_min=-1,
                a_max=1,
            )
        )

        theta = np.arccos(
            np.clip(
                np.einsum("ni,i->n", vec_los, vec_pointing)
                / (norm(vec_los, axis=1) * norm(vec_pointing)),
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
        # print(self.showermodel.photon_integral(vec_oc, vec_los, epsilon).unit)
        return photons
