"""
Visualization routines using MatPlotLib
"""

from matplotlib import pyplot as plt
from matplotlib.collections import RegularPolyCollection
import numpy as np

__all__ = ['CameraDisplay']


class CameraDisplay(object):

    """ 
    Camera Display using MatPlotLib 

    Parameters
    ----------
    geometry : `~ctapipe.io.CameraGeometry`
        Definition of the Camera/Image
    axis : `matplotlib.axes._subplots.AxesSubplot`
        a MatPlotLib Axis object to plot on, or None to create a new one
    """

    def __init__(self, geometry, axes=None, title="Camera"):
        self.axes = axes if axes is not None else plt.gca()
        self.geom = geometry
        self.polys = None

        # initialize the plot and generate the pixels as a
        # PolyCollection

        xx, yy, rr = (self.geom.pix_x.value, self.geom.pix_y.value,
                      self.geom.pix_r.data)
        offsets = list(zip(xx, yy))

        if self.geom.pix_type == 'hexagonal':
            self.polys = RegularPolyCollection(numsides=6,
                                               rotation=np.radians(90),
                                               offsets=offsets,
                                               sizes=self._radius_to_size(rr),
                                               transOffset=self.axes.transData)
            self.polys.set_facecolor('black')
            self.polys.set_linestyle('none')
            self.axes.add_collection(self.polys, autolim=True)
        else:
            raise ValueError(
                "Unimplemented pixel type: {}", self.geom.pix_type)

        self.axes.set_title(title)
        self.axes.set_xlabel("X position ({})".format(self.geom.pix_x.unit))
        self.axes.set_ylabel("Y position ({})".format(self.geom.pix_y.unit))
        self.axes.autoscale_view()

    def _radius_to_size(self, radii):
        """compute radius in screen coordinates and returns the size in
        points^2, needed for the size parameter of
        RegularPolyCollection
        """
        rr = radii.ravel()
        center = np.zeros([len(rr), 2])
        offset = np.column_stack([center[:, 0], rr])

        offset_pix = (self.axes.transData.transform(center)
                      - self.axes.transData.transform(offset))
        rad_pix = offset_pix[:, 1]
        return np.pi * rad_pix ** 2

    def draw_image(self, image, scale=None):
        """
        Change the image displayed on the Camera. 


        Parameters
        ----------
        image: array_like
            array of values corresponding to the pixels in the CameraGeometry.
        scale: None or (float,float)
            if None, autoscale to min/max value of image. Otherwise
            pass the (min,max values)
        """

        if scale is None:
            self.polys.set_facecolors(image / np.max(image))
        else:
            # TODO: implement me
            pass

        self.axes.update()


    
