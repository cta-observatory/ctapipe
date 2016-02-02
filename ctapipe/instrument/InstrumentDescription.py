import numpy as np

from ctapipe.instrument import TelescopeDescription as TD
from ctapipe.instrument import CameraDescription as CD
from ctapipe.instrument import OpticsDescription as OD
from ctapipe.instrument import util_functions as uf
from astropy import units as u

__all__ = ['Optics','Camera','Telescope']

class Optics:
    
    """`Optics` is a class that provides and gets all the information about
    the optics of a specific telescope."""

    def __init__(self,mir_class,mir_area,mir_number,prim_mirpar,prim_refrad,
                 prim_diameter,prim_hole_diam,sec_mirpar,sec_refrad,sec_diameter,
                 sec_hole_diam,mir_reflection,opt_foclen,foc_surfparam,
                 foc_surf_refrad,tel_trans):
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
        ...
        """
        self.mir_class = mir_class
        self.mir_area = mir_area
        self.mir_number = mir_number
        self.prim_mirpar = prim_mirpar
        self.prim_refrad = prim_refrad
        self.prim_diameter = prim_diameter
        self.prim_hole_diam = prim_hole_diam
        self.sec_mirpar = sec_mirpar
        self.sec_refrad = sec_refrad
        self.sec_diameter = sec_diameter
        self.sec_hole_diam = sec_hole_diam
        self.mir_reflection = mir_reflection
        self.opt_foclen = opt_foclen
        self.foc_surfparam = foc_surfparam
        self.foc_surf_refrad = foc_surf_refrad
        self.tel_trans = tel_trans

    @classmethod
    def read_file(cls,filename='fake_data',tel_id=1,attribute='closed'):
        """
        Load all the information about the optics of a given telescope with
        ID `tel_id` from an open file with name `filename`.
        
        Parameters
        ----------
        filename: string
            name of the file, if no file name is given, faked data is produced
        tel_id: int
            ID of the telescope whose optics information should be loaded
        attribute: if file is closed, the attribute 'close' is given, else
            the astropy table with the whole data read from the file is given
        """
        ext = uf.get_file_type(filename)

        if attribute == 'closed':
            load = getattr(uf,"load_%s" % ext)
            instr_table = load(filename)
        else:
            instr_table = attribute
    
        (mir_class,mir_area,mir_number,prim_mirpar,prim_refrad,prim_diameter,
         prim_hole_diam,sec_mirpar,sec_refrad,sec_diameter,sec_hole_diam,
         mir_reflection,opt_foclen,foc_surfparam,foc_surf_refrad,
         tel_trans) = OD.get_data(instr_table,tel_id)
        opt = cls(mir_class,mir_area,mir_number,prim_mirpar,prim_refrad,
               prim_diameter,prim_hole_diam,sec_mirpar,sec_refrad,sec_diameter,
               sec_hole_diam,mir_reflection,opt_foclen,foc_surfparam,
               foc_surf_refrad,tel_trans)
        
        return opt,instr_table


class Camera:
    
    """`Camera` is a class that provides and gets all the information about
    the camera of a specific telescope."""

    def __init__(self,cam_class,cam_fov,pix_id,pix_posX,pix_posY,
                 pix_area,pix_type,pix_neighbors,fadc_pulsshape):
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
        self.pix_area = pix_area
        self.pix_type = pix_type
        self.pix_neighbors = pix_neighbors
        self.fadc_pulsshape = fadc_pulsshape

    @classmethod
    def read_file(cls,filename='fake_data',tel_id=1,attribute='closed'):
        """
        Load all the information about the camera of a given telescope with
        ID `tel_id` from an open file with name `filename`.
        
        Parameters
        ----------
        filename: string
            name of the file, if no file name is given, faked data is produced
        tel_id: int
            ID of the telescope whose optics information should be loaded
        attribute: if file is closed, the attribute 'close' is given, else
            the astropy table with the whole data read from the file is given
        """
        ext = uf.get_file_type(filename)

        if attribute == 'closed':
            load = getattr(uf,"load_%s" % ext)
            instr_table = load(filename)
        else:
            instr_table = attribute
        
        (cam_class,cam_fov,pix_id,pix_posX,pix_posY,pix_area,pix_type,
         pix_neighbors,fadc_pulsshape) = CD.get_data(instr_table,tel_id)
        cam = cls(cam_class,cam_fov,pix_id,pix_posX,pix_posY,pix_area,
                pix_type,pix_neighbors,fadc_pulsshape)
        
        return cam,instr_table
    
    @staticmethod
    def rotate(cls,angle):
        """
        rotates the camera coordinates about the center of the camera by
        specified angle. Modifies the CameraGeometry in-place (so
        after this is called, the pix_x and pix_y arrays are
        rotated).
        
        Note:
        -----
        This is intended only to correct simulated data that are
        rotated by a fixed angle.  For the more general case of
        correction for camera pointing errors (rotations,
        translations, skews, etc), you should use a true coordinate
        transformation defined in `ctapipe.coordinates`.      
        
        Parameters
        ----------
        cls: give the name of the class whose pixel positions should be rotated
        angle: value convertable to an `astropy.coordinates.Angle`
            rotation angle with unit (e.g. 12 * u.deg), or "12d" 
        """
        cls.pix_posX, cls.pix_posY = CD.rotate(cls.pix_posX,cls.pix_posY,angle)

    
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
    def read_file(cls,filename='fake_data',attribute='closed'):
        """
        Load all the information about the telescope and its components
        (= parameters of the inherited classes) from an open file with
        name `filename`.

        Parameters
        ----------
        filename: string
            name of the file, if no file name is given, faked data is produced
        attribute: if file is closed, the attribute 'close' is given, else
            the astropy table with the whole data read from the file is given
        """
        
        ext = uf.get_file_type(filename)

        if attribute == 'closed':
            load = getattr(uf,"load_%s" % ext)
            instr_table = load(filename)
        else:
            instr_table = attribute
        
        tel_id, tel_num,tel_posX,tel_posY,tel_posZ = TD.get_data(instr_table)
        tel = cls(tel_num,tel_id,tel_posX,tel_posY,tel_posZ)

        opt = []
        cam = []
        for i in range(len(tel_id)):
            opt.append(Optics.read_file(filename,tel_id[i],instr_table)[0])
            cam.append(Camera.read_file(filename,tel_id[i],instr_table)[0])
        
        return tel,opt,cam,instr_table

class Atmosphere:
    """Atmosphere is a class that provides data about the atmosphere. This
    data is stored in different files which are read by member functions"""
    def __init__(self,rho,thickness,ext_coeff):
        self.rho = rho
        self.thickness = thickness
        self.ext_coeff = ext_coeff
    
    def load_profile(filename):
        """
        Load atmosphere profile from file
        
        Parameter
        ---------
        filename: string
            name of file
        --------
        """
        altitude,rho,thickness,n_minus_1 = np.loadtxt(filename,unpack=True,
                                                      delimeter=' ')
        altitude = altitude*u.km
        rho = rho*u.g*u.cm**-3
        thickness = thickness*u.g*u.cm**-2
        return altitude,rho,thickness
    
    def load_extinction_coeff(filename):
        """
        Load atmosphere extinction profile from file
        
        Parameter
        ---------
        filename: string
            name of file
        --------
        """
        
        # still work to do

