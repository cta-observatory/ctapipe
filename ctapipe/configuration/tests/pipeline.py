from ctapipe.configuration.core import Configuration, ConfigurationException
from importlib import import_module
from ctapipe.utils.datasets import get_path

__all__=['dynamic_class_from_module']

import HessioReader as hr
#p, m = name.rsplit('.', 1)
#mod = import_module(p)
#met = getattr(mod, m)

def dynamic_class_from_module(section_name,configuration=None):
    module = conf.get('module',section=section_name)
    class_name = conf.get('class', section=section_name)
    _class = getattr(import_module(module), class_name)
    if configuration == None: instance = _class()
    else: instance = _class(configuration)
         
    return instance


if __name__ == "__main__":

    conf = Configuration()
    conf.read("./pipeline.ini", impl=Configuration.INI)
    
    
    # import producer
    producer_section_name = conf.get('PRODUCER',section='PIPELINE')
    producer = dynamic_class_from_module(producer_section_name,conf)
    producer.init()
    prev_section_name = producer_section_name
    prev = producer

    #init all stager and consumer
    while True:
        current_section_name =  conf.get('next',section=prev_section_name)
        if current_section_name != None:
           current = dynamic_class_from_module(current_section_name,conf)
           prev.__dict__["next_instance"]=current
           current.init()
           prev_section_name = current_section_name 
           prev = current

        else: break
        
    producer.do_it()
    producer.finish()
    

        


"""
FINISH
"""
    