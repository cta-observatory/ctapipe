from matplotlib import pyplot as plt
import numpy as np
from ctapipe.visualization import ArrayDisplay
import astropy.units as u
from ctapipe.coordinates import TiltedGroundFrame, GroundFrame

class ArrayPlotter:
    """
    Simple plotter for drawing array level items
    """

    def __init__(self, instrument, telescopes=None, system=None):
        """

        Parameters
        ----------
        telescopes
        """

        self.instrument = instrument

        if telescopes is None:
            self.telescopes = instrument.telescope_ids
        else:
            self.telescopes = telescopes

        tel_x = [self.instrument.tel_pos[i][0].to(u.m).value for i in self.telescopes]
        tel_y = [self.instrument.tel_pos[i][1].to(u.m).value for i in self.telescopes]
        tel_z = [self.instrument.tel_pos[i][2].to(u.m).value for i in self.telescopes]

        if system is not None:
            ground = GroundFrame(x=np.asarray(tel_x)*u.m, y=np.asarray(tel_y)*u.m, z=np.asarray(tel_z)*u.m)
            new_sys = ground.transform_to(system)
            self.tel_x = new_sys.x
            self.tel_y = new_sys.y
        else:
            self.tel_x = tel_x*u.m
            self.tel_y = tel_y*u.m

    def draw_array(self, range=((-1000,1000),(-1000,1000)), hillas=None, background=None):

        plt.close()

        if background is not None:
            plt.imshow(background)

        array = ArrayDisplay(telx=np.asarray(self.tel_x), tely=np.asarray(self.tel_y),
                             mirrorarea=np.ones(len(self.tel_y))*100)

        array.axes.set_xlim(range[0])
        array.axes.set_ylim(range[1])

        plt.tight_layout()
        plt.show()