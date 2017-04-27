import numpy as np
from ctapipe.reco.ImPACT import ImPACTFitter


def test_brightest_mean():
    ImPACT = ImPACTFitter(root_dir="", fit_xmax=True)

    image = np.array([1,1,1,1])
    pixel_x = np.array([0,0,0,0])
    pixel_y = np.array([0,0,0,0])
    pixel_area = np.array([0,0,0,0])

    ImPACT.set_event_properties({1:image}, {1:pixel_x}, {1:pixel_y}, {1:pixel_area}, {1:0}, {1:0}, {1:0}, {1:0})
    ImPACT.get_brightest_mean(num_pix=3)
