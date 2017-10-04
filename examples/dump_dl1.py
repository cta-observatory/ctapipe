from ctapipe.io.dl1writer import DL1Writer
from ctapipe.calib import CameraCalibrator
from ctapipe.io.hessio import hessio_event_source
from ctapipe.utils import get_dataset
from tqdm import tqdm

if __name__ == '__main__':

    filename = get_dataset('gamma_test_large.simtel.gz')
    source = hessio_event_source(filename)
    calib = CameraCalibrator(None, None)
    writer = DL1Writer(None, outfile='dl1_dump.h5')

    for event in tqdm(source):
        calib.calibrate(event)
        writer.write(event)