def test_hash():
    from ctapipe.instrument.telescope import TelescopeDescription
    from ctapipe.instrument.optics import OpticsDescription
    from ctapipe.instrument.camera import CameraGeometry

    types = ['LST', 'MST', 'SST']
    names = ['LST', 'MST', 'SST-1M']
    cameras = ['LSTCam', 'FlashCam', 'DigiCam']

    telescopes = []
    for name, type, camera in zip(names, types, cameras):
        for i in range(3):

            telescopes.append(TelescopeDescription(
                name=name,
                type=type,
                optics=OpticsDescription.from_name(name),
                camera=CameraGeometry.from_name(camera)
            ))

    assert len(telescopes) == 9
    assert len(set(telescopes)) == 3
