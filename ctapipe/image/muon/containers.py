import numpy as np
from numpy import nan
import astropy.units as u
from ...core import Container, Field


class MuonRingParameter(Container):
    ring_center_x = Field(
        nan * u.deg, "center (x) of the fitted muon ring", unit=u.deg
    )
    ring_center_y = Field(nan * u.deg, "center (y) of the fitted muon ring", unit=u.deg)
    ring_radius = Field(nan * u.deg, "radius of the fitted muon ring", unit=u.deg)
    ring_center_phi = Field(
        nan * u.deg, "Angle of ring center within camera plane", unit=u.deg
    )
    ring_center_distance = Field(
        nan * u.deg, "Distance of ring center from camera center", unit=u.deg
    )
    ring_chi2_fit = Field(nan, "chisquare of the muon ring fit", unit=u.deg)
    ring_cov_matrix = Field(
        np.full((3, 3), nan), "covariance matrix of the muon ring fit"
    )
    ring_containment = Field(nan, "containment of the ring inside the camera")


class MuonIntensityParameter(Container):
    ring_completeness = Field(nan, "fraction of ring present")
    ring_pix_completeness = Field(nan, "fraction of pixels present in the ring")
    ring_num_pixel = Field(-1, "number of pixels in the ring image")
    ring_size = Field(nan, "size of the ring in pe")
    off_ring_size = Field(nan, "image size outside of ring in pe")
    ring_width = Field(nan, "width of the muon ring in degrees")
    ring_time_width = Field(nan, "duration of the ring image sequence")
    impact_parameter = Field(
        nan, "distance of muon impact position from center of mirror"
    )
    impact_parameter_chi2 = Field(nan, "impact parameter chi squared")
    intensity_cov_matrix = Field(nan, "covariance matrix of intensity")
    impact_parameter_pos_x = Field(nan, "impact parameter x position")
    impact_parameter_pos_y = Field(nan, "impact parameter y position")
    cog_x = Field(nan, "Center of Gravity x")
    cog_y = Field(nan, "Center of Gravity y")
    prediction = Field(None, "image prediction")
    mask = Field(None, "image pixel mask")
    optical_efficiency_muon = Field(nan, "optical efficiency muon")
