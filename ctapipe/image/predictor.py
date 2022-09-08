import numpy as np
from numpy.linalg import norm
from astropy.coordinates import spherical_to_cartesian
from astropy.utils.decorators import lazyproperty
import astropy.units as u


class Predictor:
    def __init__(
        self,
        tel_positions,
        tel_pix_coords_altaz,
        tel_solid_angles,
        tel_mirror_area,
        showermodel,
    ):
        """Creates images of a given showermodel for a set of telescopes given some telescope parameters.

        Parameters
        ----------
        tel_positions : u.Quantity[length]
            Position of the telescopes in the frame of the array
        tel_pix_coords_altaz : u.Quantity[Angle]
            Pointing of the pixels for each telescope in AltAz
        tel_solid_angles : u.Quantity[Angle**2]
            Solid angles of the pixels for each telescope
        tel_mirror_area : u.Quantity[length**2]
            Mirror area of each telescope
        showermodel :
            Showermodel to generate images from
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
            Solid angle for each pixel
        vec_oc : u.Quantity[length]
            Vector from optical center of telescope to barycenter of shower
        pix_coords_altaz : u.Quantity[Angle]
            Pointing of the pixels in AltAz
        vec_pointing : u.Quantity[length]
            Unit vector of the telescope axis/pointing
        """
        pix_content = self._photons(
            area, solid_angle, vec_oc, pix_coords_altaz, vec_pointing
        )
        return pix_content

    @lazyproperty
    def _vec_oc(self):
        """Calculates vector of optical center of each telescope to the barycenter of the shower."""
        vec = {}
        for tel_id, position in self.tel_positions.items():
            vec[tel_id] = self.showermodel.barycenter - position
        return vec

    def _vec_los(self, pix_altaz):
        """Calculates unit vector for a pixel pointing/altaz coord along the line of sight.

        Parameters
        ----------
        pix_altaz : u.Quantity[Angle]
            Pointing/AltAz of the pixel along the line of sight
        """
        x, y, z = spherical_to_cartesian(1, lat=pix_altaz.alt, lon=pix_altaz.az)
        vec = np.stack((x, y, z), -1)
        return vec * u.m

    def _telescope_axis(self, tel_pointing):
        """Calculates the unit vector of the telescope axis.

        Parameters
        ----------
        tel_pointing : u.Quantity[Angle]
            Pointing of the telescope in AltAz
        """
        return (
            np.stack(
                (spherical_to_cartesian(1, lat=tel_pointing.alt, lon=tel_pointing.az)),
                -1,
            )
            * u.m
        )

    def _photons(self, area, solid_angle, vec_oc, pix_coords_altaz, vec_pointing):
        """Calculates the photons contained in a pixel of interest.
        See https://arxiv.org/pdf/astro-ph/0601373.pdf Equation (1).

        Parameters
        ----------
        area : u.Quantity[length**2]
            Area of the mirror
        solid_angle : u.Quantity[Angle**2]
            Solid angle of the pixel.
        vec_oc : u.Quantity[lenght]
            Vector from optical center of the telescope to barycenter of the shower
        pix_coords_altaz : u.Quantity[Angle]
            Pointing of the pixel in AltAz
        vec_pointing : u.Quantity[lenght]
            Unit vector along the telescope axis
        """
        vec_los = self._vec_los(pix_coords_altaz)
        epsilon = np.arccos(
            np.einsum("ni,i->n", vec_los, self.showermodel.vec_shower_axis)
            / (norm(vec_los, axis=1) * norm(self.showermodel.vec_shower_axis))
        ).to_value(u.rad)

        theta = np.arccos(
            np.einsum("ni,i->n", vec_los, vec_pointing)
            / (norm(vec_los, axis=1) * norm(vec_pointing))
        )

        return (
            area
            * solid_angle
            * self.showermodel.emission_probability(epsilon)
            * np.cos(theta)
            * self.showermodel.photon_integral(vec_oc, vec_los, epsilon)
        )
