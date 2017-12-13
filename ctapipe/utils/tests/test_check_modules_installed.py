from ctapipe.utils import check_modules_installed


def test_check_modules_installed():
    assert(not check_modules_installed(["unlikely_module_name"]))
    assert(not check_modules_installed(["unlikely_module_name", 'numpy']))
    assert(check_modules_installed(["numpy"]))
