import numpy as np


class PointingCorrection:
    """
    Base class for pointing correction schemes
    """

    def get_camera_trans_matrix(self):
        """
        Get dummy transformation matrix, returns diagonal unit array

        Returns
        -------
        matrix: affine translate matrix for camera corrections
        """
        return np.matrix(np.diagflat(np.ones(3)))


class HESSStylePointingCorrection(PointingCorrection):
    """
    Class for taking HESS style pointing corrections and producing the transformation
    matrix required for transforming in the nominal system
    """
    def __init__(self, x_trans, y_trans, rotation, scale):
        """

        Parameters
        ----------
        x_trans: float
            Translation in x-dimension
        y_trans: float
            Translation in y-dimension
        rotation: float
            Image rotation
        scale: float
            Scaling of focal length
        """
        self.x_trans, self.y_trans = x_trans, y_trans
        self.rotation = rotation
        self.scale = scale

    def get_camera_trans_matrix(self):
        """
        Get transformation matrix in the telescope frame, applying translation rotation
        and scaling as in the HESS software

        Returns
        -------
        matrix: affine translate matrix for camera corrections
        """
        c, s = np.cos(self.rotation), np.sin(self.rotation)
        print(c, s, self.scale)
        matrix = np.matrix([[c * self.scale, -s * self.scale,
                             self.x_trans * c - self.y_trans * s],
                            [s* self.scale, c * self.scale,
                             self.x_trans * s + self.y_trans * c],
                            [0, 0, 1]])

        return matrix
