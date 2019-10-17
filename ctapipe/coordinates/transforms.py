from astropy.coordinates import SkyCoord, AltAz
from ..coordinates import CameraFrame
import astropy.units as u



def event_pos_in_camera(event, tel_id, horizon_frame=AltAz()):
    """
    Return the position of the source in the camera frame

    Parameters
    ----------
    event: `ctapipe.io.containers.DataContainer`
    tel_id: int - id of the telescope
    horizon_frame: `astropy.coordinates.builtin_frames.altaz.AltAz`

    Returns
    -------
    (x, y) (float, float): position in the camera
    """

    focal = event.inst.subarray.tel[tel_id].optics.equivalent_focal_length

    telescope_pointing = SkyCoord(alt=event.mc.tel[tel_id].altitude_raw * u.rad,
                                  az=event.mc.tel[tel_id].azimuth_raw * u.rad,
                                  frame=horizon_frame)

    event_direction = SkyCoord(alt=event.mc.alt,
                               az=event.mc.az,
                               frame=horizon_frame)

    camera_frame = CameraFrame(focal_length=focal,
                               telescope_pointing=telescope_pointing)

    camera_pos = event_direction.transform_to(camera_frame)
    return camera_pos.x, camera_pos.y