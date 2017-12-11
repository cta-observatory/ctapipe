"""
Module for handling the camera specific information (such as pixel positions),
which differ between CHEC-S, CHEC-M and single module files.
"""
import numpy as np
from astropy import log
from os.path import dirname, realpath, join
from target_io import T_SAMPLES_PER_WAVEFORM_BLOCK as N_BLOCKSAMPLES
from target_calib import MappingModule, MappingCHEC


class Borg:
    _shared_state = {}

    def __init__(self):
        """
        A class used for inheritence to allow "global" configuration classes.
        """
        self.__dict__ = self._shared_state


class Config(Borg):
    def __init__(self, camera_id=None):
        """
        Class for handling the camera specific information (such as pixel
        positions), which differ between CHEC-S, CHEC-M and single module
        files.

        The correct configuration can be set globally by importing this class
        and initializing it as the desired `camera_id`.

        >>> from targetpipe.io.camera import Config
        >>> Config('checm')

        If a particular script needs a specific configuration other than the
        default ('checs'), then the above two lines should be the first
        lines in the script.

        Thanks to the `Borg` inheritence, this class is global, and any new
        initialization will use the pre-existing configuration, unless a new
        configuration is chosen.

        Parameters
        ----------
        camera_id
        """
        Borg.__init__(self)
        if not self.__dict__:
            self._id = None

            self.dir = dirname(realpath(__file__))

            self._cameraname = None

            self.n_pix = None
            self.optical_foclen = None
            self.pixel_pos = None
            self.refshape = None
            self.refstep = None
            self.time_slice = None
            self.dead_pixels = None

            self.n_rows = None
            self.n_columns = None
            self.n_blocksamples = None
            self.n_blocks = None
            self.n_cells = None
            self.skip_sample = None
            self.skip_end_sample = None
            self.skip_event = None
            self.skip_end_event = None

            self.options = dict(
                checm=self._case_checm,
                checm_single=self._case_checm_single,
                checs=self._case_checs,
                checs_single=self._case_checs_single
            )

            if not camera_id:
                self.id = 'checs'
                # self.id = 'checm'

        if camera_id:
            self.id = camera_id

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, val):
        if not self.id == val:
            self.switch_camera(val)

    def switch_camera(self, camera_id):
        """
        Change the current camera configuration to the one specified.

        Parameters
        ----------
        camera_id : str
            name of the camera configuration to use,
            check `self.options` for valid ids
        """
        log.info("Loading camera config: {}".format(camera_id))
        self._id = camera_id
        try:
            self.options[camera_id]()
        except KeyError:
            log.error("No camera with id: {}".format(camera_id))
            raise

        self.n_blocksamples = N_BLOCKSAMPLES
        self.n_blocks = self.n_rows * self.n_columns
        self.n_cells = self.n_rows * self.n_columns * self.n_blocksamples

    def switch_to_single_module(self):
        """
        Switch the current camera configuration to a single module version.
        """
        self.id = self._cameraname + '_single'

    def _case_checm(self):
        """
        Configuration for the CHEC-M camera

        The CHEC-M camera configuration reads a numpy file to obtain the
        correct pixel coordinates.
        """
        self._cameraname = 'checm'
        self.n_pix = 2048
        self.dead_pixels = [96, 276, 1906, 1910, 1916]
        self.optical_foclen = 2.283
        pixel_pos_path = join(self.dir, 'checm_pixel_pos.npy')
        self.pixel_pos = np.load(pixel_pos_path)[:, :self.n_pix]
        ref_pulse_path = join(self.dir, 'checm_reference_pulse.npz')
        ref_pulse_file = np.load(ref_pulse_path)
        self.refshape = ref_pulse_file['refshape']
        self.refstep = ref_pulse_file['refstep']
        self.time_slice = ref_pulse_file['time_slice']
        self.n_rows = 8
        self.n_columns = 64
        self.skip_sample = 32
        self.skip_end_sample = 0
        self.skip_event = 2
        self.skip_end_event = 1

    def _case_checm_single(self):
        """
        Configuration for a single CHEC-M TARGET module
        """
        self._case_checm()
        self.n_pix = 64
        self.dead_pixels = []

    def _case_checs(self):
        """
        Configuration for the CHEC-S camera

        The CHEC-S camera configuration uses TargetCalib to obtain the correct
        pixel coordinates
        """
        self._cameraname = 'checs'
        self.n_pix = 2048
        self.dead_pixels = []
        self.optical_foclen = 2.283
        m = MappingCHEC()
        self.pixel_pos = np.vstack([m.GetXPixVector(), m.GetYPixVector()])
        self.refshape = np.zeros(10)  # TODO: Get correct values for CHEC-S
        self.refstep = 0  # TODO: Get correct values for CHEC-S
        self.time_slice = 0  # TODO: Get correct values for CHEC-S
        self.n_rows = 8
        self.n_columns = 16
        self.skip_sample = 0
        self.skip_end_sample = 0
        self.skip_event = 2
        self.skip_end_event = 1

    def _case_checs_single(self):
        """
        Configuration for a single CHEC-S TARGET module
        """
        self._case_checs()
        m = MappingModule()
        self.pixel_pos = np.vstack([m.GetXPixVector(), m.GetYPixVector()])
        self.n_pix = 64
        self.dead_pixels = []
