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
        self.tel_positions = tel_positions
        self.tel_pointings = tel_pointings
        self.tel_solid_angles = tel_solid_angles
        self.tel_mirror_area = tel_mirror_area
        self.showermodel = showermodel

    def generate_images(self):
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
            vec[tel_id] = self.showermodel._barycenter - position
        return vec

    def _vec_los(self, pix_pointing):
        """Calculates vector along line of sight for each pixel."""
        x, y, z = spherical_to_cartesian(1, lat=pix_pointing.alt, lon=pix_pointing.az)
        vec = np.stack((x, y, z), -1)
        return vec

    def _telescope_axis(self, tel_pointing):
        return np.stack(
            (spherical_to_cartesian(1, lat=tel_pointing.alt, lon=tel_pointing.az)), -1
        )

    def _photons(self, area, solid_angle, vec_oc, pointing, vec_axis):
        """Calculate number of photons in pixel. See https://arxiv.org/pdf/astro-ph/0601373.pdf Equation (1).

        Parameters
        ----------
        theta : u.Quantity[angle]
            Angle between pixel and telecope axis
        epsilon: u.Quantity[angle]
            Angle between pointing direction of pixel and shower axis
        """
        vec_los = self._vec_los(pointing)
        epsilon = np.arccos(
            np.dot(vec_los, self.showermodel._vec_shower_axis)
            / (norm(vec_los) * norm(self.showermodel._vec_shower_axis))
        ).to_value(u.rad)

        theta = np.arccos(np.dot(vec_los, vec_axis) / (norm(vec_los) * norm(vec_axis)))

        return (
            area
            * solid_angle
            * self.showermodel.emission_probability(epsilon)
            * np.cos(theta)
            * self.showermodel.photon_integral(vec_oc, vec_los, epsilon)
        )
