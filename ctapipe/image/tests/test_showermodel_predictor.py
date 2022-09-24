import astropy.units as u
from astropy.coordinates import AltAz, SkyCoord, cartesian_to_spherical

from ctapipe.coordinates import CameraFrame, TelescopeFrame


def test_predictor(example_subarray):
    from ctapipe.image import GaussianShowermodel, ShowermodelPredictor

    # This is a shower straight from above
    total_photons = 1000
    x = 0 * u.meter
    y = 0 * u.meter
    azimuth = 0 * u.deg
    altitude = 90 * u.deg
    first_interaction = 10000 * u.meter
    width = 20 * u.meter
    length = 3000 * u.meter

    model = GaussianShowermodel(
        total_photons=total_photons,
        x=x,
        y=y,
        azimuth=azimuth,
        altitude=altitude,
        first_interaction=first_interaction,
        width=width,
        length=length,
    )

    lsts = example_subarray.select_subarray([1, 2, 3, 4])

    tel_pix_coords_altaz = {}
    tel_solid_angles = {}
    tel_mirror_area = {}
    for tel_id, tel in lsts.tel.items():
        geometry = tel.camera.geometry
        pix_x = geometry.pix_x
        pix_y = geometry.pix_y
        focal_length = tel.optics.equivalent_focal_length

        pointing = model.barycenter - lsts.positions[tel_id]
        pointing = cartesian_to_spherical(*pointing)
        altaz = AltAz(az=pointing[2], alt=pointing[1])
        camera_frame = CameraFrame(focal_length=focal_length, telescope_pointing=altaz)

        cam_coords = SkyCoord(x=pix_x, y=pix_y, frame=camera_frame)

        cam_altaz = cam_coords.transform_to(AltAz())
        tel_pix_coords_altaz[tel_id] = cam_altaz

        tel_solid_angles[tel_id] = geometry.transform_to(TelescopeFrame()).pix_area

        tel_mirror_area[tel_id] = tel.optics.mirror_area

    pred = ShowermodelPredictor(
        tel_positions=lsts.positions,
        tel_pix_coords_altaz=tel_pix_coords_altaz,
        tel_solid_angles=tel_solid_angles,
        tel_mirror_area=tel_mirror_area,
        showermodel=model,
    )

    imgs = pred.generate_images()

    for i in [1, 2, 3, 4]:
        geometry = lsts.tel[i].camera.geometry

        assert imgs[i].shape == geometry.pix_x.shape
