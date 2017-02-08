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

        self.array = ArrayDisplay(telx=np.asarray(self.tel_x), tely=np.asarray(self.tel_y),
                             mirrorarea=np.ones(len(self.tel_y))*100)

        self.hillas = None

    def overlay_hillas(self, hillas):
        count = 0
        for tel in self.telescopes:
            if tel in hillas:
                self.array.overlay_moments(hillas[tel], (self.tel_x[count], self.tel_y[count]))
            count += 1
        self.hillas = hillas

    def background_image(self, background, range):

        plt.imshow(background, extent=[range[0][0], range[0][1], range[1][0], range[1][1]], cmap="viridis")

        # Annoyingly we need to redraw everything
        self.array = ArrayDisplay(telx=np.asarray(self.tel_x), tely=np.asarray(self.tel_y),
                             mirrorarea=np.ones(len(self.tel_y))*100)

        if self.hillas is not None:
            self.overlay_hillas(self.hillas)

    def draw_array(self, range=((-2000,2000),(-2000,2000))):

        self.array.axes.set_xlim(range[0])
        self.array.axes.set_ylim(range[1])

        plt.tight_layout()
        plt.show()