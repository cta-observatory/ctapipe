import pipeline
from ctapipe.configuration.core import Configuration, ConfigurationException
from sys import stderr

if __name__ == '__main__':
    
    conf = Configuration()    
    conf.add_argument("-s" , "--seq", dest="sequential", required=False,  action='store_true',
                      help='Pipeline runs in parallel mode by default. Use -s or (--seq) to run it in sequential mode')
    res = conf.parse_args()
    
    
    conf.read("./pipeline.ini", impl=Configuration.INI)
    
    
    # create  and init pipeline 
    pipeline = pipeline.Pipeline(conf)
    if pipeline.init() == False:
        print("Could not initialise pipeline",file=stderr)
    else:
        
        
        # run pipeline
        if ( conf.get('sequential') == True):
            pipeline.run_sequential()
        else:
            pipeline.run_parallel()
            
        # finish pipeline
        pipeline.finish()
    
    

    