"""
Extract flat field coefficients from flasher data files.
"""

import os

import numpy as np
from tqdm import tqdm
from traitlets import Dict, List, Int, Unicode


from ctapipe.core import Provenance
from ctapipe.io import HDF5TableWriter
from ctapipe.core import Tool
from ctapipe.io import EventSourceFactory
from ctapipe.calib.camera.flatfield import FlatFieldFactory
from ctapipe.io.containers import MonDataContainer


class FlatFieldGenerator(Tool):
    name = "FlatFieldGenerator"
    description = "Generate a HDF5 file with flat field coefficients" 
                   
    
    output_file = Unicode('flatfield.hdf5',
                          help='Name of the output flat field file ' \
                               'file').tag(config=True)

    aliases = Dict(dict(input_file='EventSourceFactory.input_url',
                        max_events='EventSourceFactory.max_events',
                        allowed_tels= 'EventSourceFactory.allowed_tels',
                        generator='FlatFieldFactory.product',
                        max_time_range_s='FlatFieldFactory.max_time_range_s',
                        ff_events='FlatFieldFactory.max_events',
                        n_channels='FlatFieldFactory.n_channels'                  
                        ))

    classes = List([EventSourceFactory,
                    FlatFieldFactory,
                    MonDataContainer,
                    HDF5TableWriter
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eventsource = None
        self.flatfield = None
        self.container = None
        self.writer = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)
        
        # open an extractor per camera
        self.eventsource = EventSourceFactory.produce(**kwargs)
        self.flatfield = FlatFieldFactory.produce(**kwargs)
        self.container = MonDataContainer()
        self.writer = HDF5TableWriter(
            filename=self.output_file, group_name='flatfield', overwrite=True
        )
        

    def start(self):
        desc = "Flat field coefficient calculator"
        
        for count, event in enumerate(self.eventsource):
           
            for tel_id in event.r0.tels_with_data:
                # initialize the flat filed  containers
                self.container.flatfield.tels_with_data.append(tel_id)
                #SSself.container.flatfield.tel[tel_id].tel_id = tel_id
            
                ff_data = self.flatfield.calculate_relative_gain(event.r0.tel[tel_id])
                
                if ff_data:   
                    self.container.flatfield.tel[tel_id] = ff_data
                    table_name='tel_'+str(tel_id)
                                 
                    self.log.info("write event in table: /flatfield/%s", table_name)
                    self.writer.write(table_name,ff_data)      
                    
                
                
    

    def finish(self):
 
        Provenance().add_output_file(self.output_file,
                                     role='mon.tel.flatfield')
        self.writer.close()


def main():
    exe = FlatFieldGenerator()
    exe.run()


if __name__ == '__main__':
    main()
