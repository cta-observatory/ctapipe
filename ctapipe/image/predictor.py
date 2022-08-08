import numpy as np
from numpy.linalg import norm
from astropy.coordinates import spherical_to_cartesian
from astropy.utils.decorators import lazyproperty
import astropy.units as u


class Predictor:
    def __init__(
        self,
        tel_positions,
        tel_pointings,
        tel_solid_angles,
        tel_mirror_area,
        showermodel,
    ):
        """Creates images of a given showermodel for a set of telescopes given some telescope parameters.

        Parameters
        ----------
        tel_positions : u.Quantity[length]
            Position of the telescopes in the frame of the array
        tel_pointings : u.Quantity[Angle]
            Pointing of the pixels for each telescope in AltAz
        tel_solid_angles : u.Quantity[Angle**2]
            Solid angles of the pixels for each telescope
        tel_mirror_area : u.Quantity[length**2]
            Mirror area of each telescope
        showermodel :
            Showermodel to generate images from
        """
        self.tel_positions = tel_positions
        self.tel_pointings = tel_pointings
        self.tel_solid_angles = tel_solid_angles
        self.tel_mirror_area = tel_mirror_area
        self.showermodel = showermodel

    def generate_images(self):
        """Predicts images for telescopes."""
        imgs = {}
        for tel_id, vec_oc in self._vec_oc.items():
            area = self.tel_mirror_area[tel_id]
            solid_angle = self.tel_solid_angles[tel_id]
            pointing = self.tel_pointings[tel_id]
            vec_axis = self._telescope_axis(pointing[0])
            imgs[tel_id] = self._generate_img(
                area, solid_angle, vec_oc, pointing, vec_axis
            )

        return imgs

    def _generate_img(self, area, solid_angle, vec_oc, pointing, vec_axis):
        """Generates one image of the shower for a given telescope.

        Parameters
        ----------
        area : u.Quantity[length**2]
            Area of the telescope mirror
        solid_angle : u.Quantity[Angle**2]
            Solid angle for each pixel
        vec_oc : u.Quantity[length]
            Vector from optical center of telescope to barycenter of shower
        pointing : u.Quantity[Angle]
            Pointing of the pixels in AltAz
        vec_axis : u.Quantity[length]
            Unit vector of the telescope axis
        """
        pix_content = []
        for pix_id, point in enumerate(pointing):
            pix_content.append(
                self._photons(area, solid_angle[pix_id], vec_oc, point, vec_axis).value
            )

        return np.array(pix_content)

    @lazyproperty
    def _vec_oc(self):
        """Calculates vector of optical center of each telescope to the barycenter of the shower."""
        vec = {}
        for tel_id, position in self.tel_positions.items():
            vec[tel_id] = self.showermodel.barycenter - position
        return vec

    def _vec_los(self, pix_pointing):
        """Calculates unit vector for a pixel pointing along the line of sight.

        Parameters
        ----------
        pix_pointing : u.Quantity[Angle]
            Pointing of the pixel along the line of sight
        """
        x, y, z = spherical_to_cartesian(1, lat=pix_pointing.alt, lon=pix_pointing.az)
        vec = np.stack((x, y, z), -1)
        return vec

    def _telescope_axis(self, tel_pointing):
        """Calculates the unit vector of the telescope axis.

        Parameters
        ----------
        tel_pointing : u.Quantity[Angle]
            Pointing of the telescope in AltAz
        """
        return np.stack(
            (spherical_to_cartesian(1, lat=tel_pointing.alt, lon=tel_pointing.az)), -1
        )

    def _photons(self, area, solid_angle, vec_oc, pointing, vec_axis):
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
        pointing : u.Quantity[Angle]
            Pointing of the pixel in AltAz
        vec_axis : u.Quantity[lenght]
            Unit vector along the telescope axis
        """

        vec_los = self._vec_los(pointing)
        epsilon = np.arccos(
            np.dot(vec_los, self.showermodel.vec_shower_axis)
            / (norm(vec_los) * norm(self.showermodel.vec_shower_axis))
        ).to_value(u.rad)

        theta = np.arccos(np.dot(vec_los, vec_axis) / (norm(vec_los) * norm(vec_axis)))

        return (
            area
            * solid_angle
            * self.showermodel.emission_probability(epsilon)
            * np.cos(theta)
            * self.showermodel.photon_integral(vec_oc, vec_los, epsilon)
        )
