from ctapipe.io.files import get_file_type


def test_get_file_type():

    test_filenames = {'test.fits.gz': 'fits',
                      'test.fits': 'fits',
                      'test.fits.bz2': 'fits',
                      'test.fit': 'fits',
                      'test_file.eventio.gz': 'eventio',
                      'test_file.eventio': 'eventio',
                      'more.complex.fileame.txt.gz': 'txt'}

    for filename, filetype in test_filenames.items():
        assert get_file_type(filename) == filetype
