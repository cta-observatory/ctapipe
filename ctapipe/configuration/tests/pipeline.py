from ctapipe.configuration.core import Configuration, ConfigurationException
from importlib import import_module
from ctapipe.utils.datasets import get_path

import HessioReader as hr
#p, m = name.rsplit('.', 1)
#mod = import_module(p)
#met = getattr(mod, m)

def dynamic_class_from_module(section_name):
    module = conf.get('module',section=section_name)
    print('module',module)
    class_name = conf.get('class', section=section_name)
    print('class_name',class_name)
    _class = getattr(import_module(module), class_name)
    print('class_',_class)
    instance = _class()
    return instance


if __name__ == "__main__":

    conf = Configuration()
    conf.read("./pipeline.ini", impl=Configuration.INI)
    
    
    # import producer
    producer_section_name = conf.get('PRODUCER',section='PIPELINE')
    producer = dynamic_class_from_module(producer_section_name)
    producer.init()

    # import stagers
    stager_list = conf.get_list('STAGES', section='PIPELINE')
    stager = dynamic_class_from_module(stager_list[0])
    stager.init()

    
    for result in  producer.do_it(): 
        stager.do_it(result)
    