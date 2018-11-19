import numpy as np
import pytest
from ctapipe.utils import get_dataset_path

from ctapipe.io.nectarcameventsource import NectarCAMEventSource
from ctapipe.calib.camera.flatfield import FlasherFlatFieldCalculator


def test_FlasherFlatFieldCalculator():
    
    example_file_path = get_dataset_path("NectarCAM.Run0890.10events.fits.fz") 
    inputfile_reader = NectarCAMEventSource(
        input_url=example_file_path,
        max_events=10
    )

    ff_calculator = FlasherFlatFieldCalculator()
    ff_calculator.max_events=3

    for i,event in enumerate(inputfile_reader):
        for tel_id in event.r0.tels_with_data:
    
            
            ff_data = ff_calculator.calculate_relative_gain(event.r0.tel[tel_id])
            
            if ff_calculator.count == ff_calculator.max_events:              
                assert(ff_data)

       
     
