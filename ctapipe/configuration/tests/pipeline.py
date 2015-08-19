from ctapipe.configuration.core import Configuration, ConfigurationException
from importlib import import_module
from ctapipe.utils.datasets import get_path

import HessioReader as hr
#p, m = name.rsplit('.', 1)
#mod = import_module(p)
#met = getattr(mod, m)



if __name__ == "__main__":
    """ 
    r = hr.HessioReader()
    r.init()
    for event in r.do_it():
        print(event)
    """ 

    conf = Configuration()
    conf.read("./pipeline.ini", impl=Configuration.INI)
    
    
    pipeline = conf.get_section('PIPELINE')
    
    # import producer
    producer = pipeline['PRODUCER'][Configuration.VALUE_INDEX]
    
    producer_import = conf.get('import',section=producer)
    print("producer_import",producer_import)
    
    MyClass = getattr(import_module(producer_import), 'HessioReader')
    instance = MyClass()
    instance.init()
    
    for event in  instance.do_it(): 
        print(event.dl0.tels_with_data)
    
    
    
    #consumer = pipeline['CONSUMER'][Configuration.VALUE_INDEX]