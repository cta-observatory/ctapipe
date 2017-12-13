import pytest
pytest.importorskip("target_calib")

from ctapipe.io.unofficial.targetio import camera


def test_checm():
    config = camera.Config('checm')
    assert(config.id == 'checm')
    assert(config.n_pix == 2048)
    assert(config.n_columns == 64)
    assert(round(config.pixel_pos[0, 1], 3) == -0.066)
    assert(round(config.pixel_pos[1, 1], 3) == 0.114)


def test_checm_single():
    config = camera.Config('checm_single')
    assert(config.id == 'checm_single')
    assert(config.n_pix == 64)
    assert(config.n_columns == 64)
    assert(round(config.pixel_pos[0, 1], 3) == -0.066)
    assert(round(config.pixel_pos[1, 1], 3) == 0.114)


def test_checs():
    config = camera.Config('checs')
    assert(config.id == 'checs')
    assert(config.n_pix == 2048)
    assert(config.n_columns == 16)
    assert(round(config.pixel_pos[0, 1], 3) == 0.136)
    assert(round(config.pixel_pos[1, 1], 3) == 0.057)


def test_checs_single():
    config = camera.Config('checs_single')
    assert(config.id == 'checs_single')
    assert(config.n_pix == 64)
    assert(config.n_columns == 16)
    assert(round(config.pixel_pos[0, 1], 3) == 0.022)
    assert(round(config.pixel_pos[1, 1], 3) == 0.003)


def test_switching_config():
    config = camera.Config('checs')
    config.id = 'checm'
    assert(config.id == 'checm')
    assert(config.n_pix == 2048)
    assert(config.n_columns == 64)
    assert(round(config.pixel_pos[0, 1], 3) == -0.066)
    assert(round(config.pixel_pos[1, 1], 3) == 0.114)


def test_borg_functionality():
    config1 = camera.Config('checs')
    camera.Config('checm')
    assert(config1.id == 'checm')

    config1 = camera.Config('checm')
    config2 = camera.Config()
    assert (config2.id == 'checm')

    config2.id = 'checs_single'
    assert (config1.id == 'checs_single')


def test_reset():
    config = camera.Config.reset()
    assert(config.id == config.default)


# Reset camera config
camera.Config.reset()
