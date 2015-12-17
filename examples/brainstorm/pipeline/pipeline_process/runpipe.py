import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from zmqpipe import pipeline_zmq 
from ctapipe.configuration.core import Configuration, ConfigurationException
from sys import stderr


if __name__ == '__main__':
    
	conf = Configuration()    
	conf.add_argument("-c", "--conf" , dest="pipe_config_filename", required=False, default='pipeline.ini',
	help='configuration file containing pipeline configuration')

	conf.add_argument("-s" , "--seq", dest="sequential", required=False,
	action='store_true', help='Pipeline runs in parallel mode by default. Use -s or (--seq) to run it in sequential mode')

	conf.add_argument("-v" , "--verbose", dest="verbose", required=False, 
	action='store_true', help='enable verbose mode')
	res = conf.parse_args()

	if len(conf.read(conf.pipe_config_filename, impl=Configuration.INI)) != 1:
		print("Could not read", conf.pipe_config_filename, file=stderr)
	else:
		# create  and init pipeline 
		if conf.verbose: conf.list()
		pipeline = pipeline_zmq.Pipeline(conf)
		if pipeline.init() == False:
			print("Could not initialise pipeline",file=stderr)
		else:
			pipeline.run()
    
    

    
