
def test_dunder_all():
    from ctapipe.image import reducers
    # test existence
    reducers.__all__  # there is no entry in __all__ at the moment
