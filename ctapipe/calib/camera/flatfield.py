"""
Factory for the estimation of the flat field coefficients
"""
from abc import abstractmethod
import numpy as np

from ctapipe.core import Component, Factory
from ctapipe.core.traits import Int
from ctapipe.io.containers import FlatFieldCameraContainer

__all__ = [
    'FlatFieldCalculator',
    'FlasherFlatFieldCalculator',
    'FlatFieldFactory'
]


class FlatFieldCalculator(Component):
    """
    Parent class for the flat field calculators. Fills the MON.flatfield container.

    """
    max_time_range_s = Int(60, help='Define the maximum time interval per'
           ' coefficient flat-filed calculation').tag(config=True)
    max_events = Int(1000, help='Define the maximum number of events per '
           ' coefficient flat-filed calculation').tag(config=True)
    n_channels = Int(2, help='Define the number of channels to be '
                    'treated ').tag(config=True)


    def __init__(self, config=None, tool=None, **kwargs):
        """
        Parent class for the flat field calculators. Fills the MON.flatfield container.

        Parameters
        ----------
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        tool : ctapipe.core.Tool
            Tool executable that is calling this component.
            Passes the correct logger to the component.
            Set to None if no Tool to pass.

        kwargs

        """
        super().__init__(config=config, parent=tool, **kwargs)
        self.container = FlatFieldCameraContainer()
        
     
        
    @abstractmethod
    def calculate_relative_gain(self,event):
        """
        Parameters
        ----------
        event
        
        """
       
        
        
class FlasherFlatFieldCalculator(FlatFieldCalculator):
    """
    Class for calculating flat field coefficients witht the
    flasher data. Fills the MON.flatfield container.
    """

    def __init__(self, config=None, tool=None,  **kwargs):
        """
        Parent class for the flat field calculators. Fills the MON.flatfield container.

        Parameters
        ----------
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        tool : ctapipe.core.Tool
            Tool executable that is calling this component.
            Passes the correct logger to the component.
            Set to None if no Tool to pass.
        eventsource : ctapipe.io.eventsource.EventSource
            EventSource that is being used to read the events. The EventSource
            contains information (such as metadata or inst) which indicates
            the appropriate R1Calibrator to use.
        kwargs

        """
        super().__init__(config=config, tool=tool, **kwargs)
                      
        self.log.info("Used events statistics : %d", self.max_events)
        self.count = 0

         
                        
            
    def calculate_relative_gain(self, event):
        """
        calculate the relative flat filed coefficients

        Parameters
        ----------
        event : specific camera  

        """
        # initialize the np array at each cycle
        
        if (self.count == 0):
            self.time_start = event.trigger_time
            
            self.event_median = np.zeros((self.max_events, self.n_channels))
            
            n_pix = event.waveform.shape[1]
            self.trace_integral = np.zeros((self.max_events, self.n_channels, n_pix))

        
                     
        trace = event.waveform[:,:,:] 
           
            
        # here if necessary subtract  pedestals
        # e.g. with calc_pedestals_from_traces()
        # ...
                
        # extract the signal (for the moment very rough  integral):  x(i,j)
        # for the moment no check on the values
        self.trace_integral[self.count,:,:] = trace[:,:,:].sum(axis=2)
                             
        # extract the median on all the camera per event: <x>(i)
        self.event_median[self.count,:] = \
            np.median(self.trace_integral[self.count,:,:], axis=1)  
        time = event.trigger_time 
        
        # increment the internal counter
        self.count = self.count+1
            
        # check if to create a calibration event
        if ((time - self.time_start) > self.max_time_range_s or self.count == self.max_events):
        
                
            # extract for each pixel : x(i,j)/<x>(i) = g(i,j) 
            rel_gain_event = self.trace_integral/self.event_median[:,:, np.newaxis]
            
            # extract the median, mean and std over all the events <g>j and
            # fill the container and return it
            self.container.mean_time_s = (time - self.time_start)/2
            self.container.range_time_s = [self.time_start, time]       
            self.container.n_events = self.count + 1
            self.container.relative_gain_median =  np.median(rel_gain_event, axis=0)
            self.container.relative_gain_mean =  np.mean(rel_gain_event, axis=0)                  
            self.container.relative_gain_rms =  np.std(rel_gain_event, axis=0)
            
            # re-initialize the event count
            self.count = 0
            
            return self.container
         
        return None 
            
               


class FlatFieldFactory(Factory):
    """
    Factory to obtain flat-field coefficients
    """
    base = FlatFieldCalculator
    default = 'FlasherFlatFieldCalculator'
    custom_product_help = ('')

 