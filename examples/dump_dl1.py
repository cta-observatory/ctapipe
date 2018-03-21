from tqdm import tqdm

from ctapipe.calib import CameraCalibrator
from ctapipe.io.dl1writer import DL1Writer
from ctapipe.io import event_source
from ctapipe.utils import get_dataset

if __name__ == '__main__':

    filename = get_dataset('gamma_test_large.simtel.gz')
    source = event_source(filename)
    calib = CameraCalibrator(None, None)
    writer = DL1Writer(None, outfile='dl1_dump.h5')

    for event in tqdm(source):
        calib.calibrate(event)
        writer.write(event)
