from pathlib import Path
from ctapipe.utils.datasets import get_datasets_path
from ..files import get_file_type
from ..files import InputFile


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


def test_inputfile():
    dataset = get_datasets_path("gamma_test.simtel.gz")
    file = InputFile(dataset, 'hessio')
    datasets_path = Path(get_datasets_path(""))
    assert file.input_path == datasets_path.joinpath("gamma_test.simtel.gz").as_posix()
    assert file.directory == datasets_path.as_posix()
    assert file.extension == ".gz"
    assert file.filename == "gamma_test.simtel"
    assert file.origin == "hessio"
    source = file.read()
    event = next(source)
    assert event.dl0.tels_with_data == {38, 47}