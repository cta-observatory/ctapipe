import pytest


@pytest.mark.xfail(reason="IERS_URL_B unreachable", strict=False)
@pytest.mark.vizier
def test_export_astropy_cache(tmp_path):
    from ctapipe.tools.store_astropy_cache import main

    cache = tmp_path / "astropy_cache"

    main(["--directory", str(cache), "--force"])
    assert cache.is_dir()
