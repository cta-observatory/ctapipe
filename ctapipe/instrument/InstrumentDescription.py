import matplotlib.pyplot as plt

from ctapipe.instrument.telescope import TelescopeDescription as TD
from ctapipe.instrument.telescope.camera import CameraDescription as CD
from ctapipe.instrument.telescope.optics import OpticsDescription as OD

__all__ = ['Optics','Camera','Telescope']

class Optics:
    
    """`Optics` is a class that provides and gets all the information about
    the optics of a specific telescope."""

    def __init__(self,mir_area,mir_number,opt_foclen):
        """
        Parameters
        ----------
        self: type
            description
        mir_area: float with unit
            area of the mirrors of the telescope
        mir_number: int number
            number of mirrors of the telescope
        opt_foclen: float with unit
            optical focus length of the optical system of the telescope
        """
        self.mir_area = mir_area
        self.mir_number = mir_number
        self.opt_foclen = opt_foclen

    @classmethod
    def initialize(cls,filename,tel_id,item):
        """
        Load all the information about the optics of a given telescope with
        ID `tel_id` from an open file with name `filename`.
        
        Parameters
        ----------
        filename: string
            name of the file
        tel_id: int
            ID of the telescope whose optics information should be loaded
        item: of various type depending on the file extension
            return value of the opening/loading process of the file
        """
        mir_area, mir_number,opt_foclen = OD.initialize(filename,tel_id,item)
        opt = cls(mir_area,mir_number,opt_foclen)
        return opt


class Camera:
    
    """`Camera` is a class that provides and gets all the information about
    the camera of a specific telescope."""

    def __init__(self,cam_class,cam_fov,pix_id,pix_posX,pix_posY,pix_posZ,
                 pix_area,pix_type,pix_neighbors):
        """
        Parameters
        ----------
        self: type
            description
        cam_class: string
            camera class of the telescope
        cam_fov: float
            field of view (FOV) of the camera
        pix_id: array (int)
            pixel ids of the camera of the telescope
        pix_posX: array with units
            position of each pixel (x-coordinate)
        pix_posY: array with units
            position of each pixel (y-coordinate)
        pix_posZ: array with units
            position of each pixel (z-coordinate)
        pix_area: array with units
            area of each pixel
        pix_type: string
            name of the pixel type (e.g. hexagonal)
        pix_neighbors: ndarray (int)
            nD-array with pixel IDs of neighboring
            pixels of the pixels (n=number of pixels)
        """
        self.cam_class = cam_class
        self.cam_fov = cam_fov
        self.pix_id = pix_id
        self.pix_posX = pix_posX
        self.pix_posY = pix_posY
        self.pix_posZ = pix_posZ
        self.pix_area = pix_area
        self.pix_type = pix_type
        self.pix_neighbors = pix_neighbors

    @classmethod
    def initialize(cls,filename,tel_id,item):
        """
        Load all the information about the camera of a given telescope with
        ID `tel_id` from an open file with name `filename`.
        
        Parameters
        ----------
        filename: string
            name of the file
        tel_id: int
            ID of the telescope whose optics information should be loaded
        item: of various type depending on the file extension
            return value of the opening/loading process of the file
        """
        (cam_class,cam_fov,pix_id,pix_posX,pix_posY,pix_posZ,pix_area,pix_type,
         pix_neighbors) = CD.initialize(filename,tel_id,item)
        cam = cls(cam_class,cam_fov,pix_id,pix_posX,pix_posY,pix_posZ,pix_area,
                pix_type,pix_neighbors)
        return cam


class Pixel:
    
     """`Pixel` is a class that provides and gets all the information about
    a specific pixel of a specific camera."""

     
    
class Telescope(Optics,Camera):
    
    """`Telescope` is a class that provides and gets all the information about
    all telescopes available in a run. It inherits the methods and variables of
    the classes `Optics` and `Camera`."""

    def __init__(self,tel_num,tel_id,tel_posX,tel_posY,tel_posZ):
        """
        Parameters
        ----------
        self: type
            description
        tel_num: int number
            number of telescopes available in the given run
        tel_id: array(int)
            telescope id numbers
        tel_posX: array with units
            position of each telescope (x-coordinate)
        tel_posY: array with units
            position of each telescope (y-coordinate)
        tel_posZ: array with units
            position of each telescope (z-coordinate)
        """
        self.tel_id = tel_id
        self.tel_num = tel_num
        self.tel_posX = tel_posX
        self.tel_posY = tel_posY
        self.tel_posZ = tel_posZ

    @classmethod
    def initialize(cls,filename,item):
        """
        Load all the information about the telescope and its components
        (= parameters of the inherited classes) from an open file with
        name `filename`.

        Parameters
        ----------
        filename: string
            name of the file
        item: of various type depending on the file extension
            return value of the opening/loading process of the file
        """
        tel_id, tel_num,tel_posX,tel_posY,tel_posZ = TD.initialize(filename,
                                                                   item)
        tel = cls(tel_num,tel_id,tel_posX,tel_posY,tel_posZ)

        opt = []
        cam = []
        for i in range(len(tel_id)):
            opt.append(Optics.initialize(filename,tel_id[i],item))
            cam.append(Camera.initialize(filename,tel_id[i],item))
        
        return tel,opt,cam

'''
class Subarray:
    #What should be in here?

    def __init__(self):
        self.telescope = Telescope()
    
    def plotSubArray(self):
        ad = ArrayDisplay(ld.telescope_posX, ld.telescope_posY, ld.mirror_area)
        for i in range(len(ld.telescope_id)):
            name = "CT%i" % ld.telescope_id[i]
            plt.text(ld.telescope_posX[i],ld.telescope_posY[i],name,fontsize=8)
        ad.axes.set_xlim(-1000, 1000)
        ad.axes.set_ylim(-1000, 1000)
        plt.show()
'''
