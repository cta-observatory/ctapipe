import pipeline_zmq 
from ctapipe.configuration.core import Configuration, ConfigurationException
from sys import stderr


if __name__ == '__main__':
    
    conf = Configuration()    
    conf.add_argument("-s" , "--seq", dest="sequential", required=False,  action='store_true',
                      help='Pipeline runs in parallel mode by default. Use -s or (--seq) to run it in sequential mode')
    res = conf.parse_args()
    
    filename = "./pipeline_lapp.ini"
    if len(conf.read(filename, impl=Configuration.INI)) != 1:
        print("Could not read", filename, file=stderr)
    else:
        # create  and init pipeline 
        conf.list()
        pipeline = pipeline_zmq.Pipeline(conf)
        if pipeline.init() == False:
            print("Could not initialise pipeline",file=stderr)
        else:
            pipeline.run()
            # finish pipeline
    
    

    