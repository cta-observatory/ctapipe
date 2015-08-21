from ctapipe.utils.datasets import get_path
from ctapipe.io.hessio import hessio_event_source
from ctapipe.configuration.core import Configuration, ConfigurationException
import threading


def HessioReader():
    print("--- HessioReader init ---")
    conf = Configuration()
    conf.read("./pipeline.ini", impl=Configuration.INI)
    raw_data = conf.get('source', section='HESSIO_READER')
    source = hessio_event_source(get_path(raw_data), max_events=10)
       
    counter = 0
    for event in source:
        print("HessioReader get next event")
        counter+=1
        print("--< Start Event",counter,">--")#,end="\r")
        yield event
    print("\n--- Done ---")
        