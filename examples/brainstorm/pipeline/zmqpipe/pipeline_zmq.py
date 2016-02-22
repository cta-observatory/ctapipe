"""Simple but robust implementation of generator/coroutine-based
pipelines in Python. The pipelines may be run either sequentially
(single-threaded) or in parallel (one thread per pipeline stage).

This implementation supports pipeline bubbles (indications that the
processing for a certain item should abort). To use them, yield the
BUBBLE constant from any stage coroutine except the last.

In the parallel case, the implementation transparently handles thread
shutdown when the processing is complete and when a stage raises an
exception.
"""
from ctapipe.utils.datasets import get_path
from ctapipe.configuration.core import Configuration, ConfigurationException
from .producer_zmq import ProducerZmq
from .stager_zmq import StagerZmq
from .consumer_zmq import ConsumerZmq
from ctapipe.core import Container
from sys import exit,stderr



__all__=  ['Pipeline', 'PipelineError']


class PipeStep():
	def __init__(self,section_name, prev_step=None, name=None, port_in=None, port_out=None):
		self.section_name = section_name
		self.port_in = port_in
		self.port_out = port_out
		self.prev_step = prev_step
		
	def  __repr__(self): 
		return "Name[ "+str(self.section_name) + " ], previous step[ " + str(self.prev_step.section_name) + " ], port in[ " + str(self.port_in)  + " ], port out [ " + str(self.port_out) +" ]" 
    
        
        
        
class PipelineError(Exception):
	def __init__(self, msg):
		"""An indication that an exception occurred in the pipeline. The
		object is passed through the pipeline to shut down all threads
		before it is raised again in the main thread.
		"""
		self.msg = msg

class Pipeline():
    
	PRODUCER = 'PRODUCER'
	STAGER   = 'STAGER'
	CONSUMER = 'CONSUMER'
	PORT     = 'ports'
	ROLE     = 'role'

	def __init__(self,configuration):
		"""Represents a staged pattern of stage. Each stage in the pipeline
		is a coroutine that receives messages from the previous stage and
		yields messages to be sent to the next stage.
		
		Parameters
		----------
		configuration : Configuration object, required
			Pipeline asks to this configuration instance  to 
			create producers, stagers and consumers instances
			according to information in configuration
		"""
		self.conf = configuration
		self.ports=None
		self.producers_zmq = list()
		self.stagers_zmq = list()
		self.consumers_zmq = list()
		
		self.producer_steps = None
		self.stager_steps = None
		self.consumers_steps = None
		
	def init(self):
		"""
		Create producers, stagers and consumers instance according to configuration 
		Returns:
		--------
		bool: True if pipeline is correctly setup and all producer,stager and consumer initialised
				Otherwise False
		"""
		# Verify configuration instance
		if self.conf == None:
			print("Could not initialise a pipeline without configuration", file=stderr)
			return False
		
		# verify ports for ZMQ
		self.ports = self.conf.get_list('ports', 'PIPELINE')
		if self.conf.verbose:print('Pipeline:init self.ports', self.ports)
		if  self.ports == None:  
			print("Could not initialise a pipeline without any port available in configuration", file=stderr)
			return False
		
		# Gererate steps(producers, stagers and consumers) from configuration
		self._generate_steps()
		
		# Configure steps' port out
		if self._configure_port_out(self.producer_steps, self.stager_steps) == False:
			print("No enough available ports for ZMQ")
			return False
		
		# Configure steps' port in
		self._configure_port_in(self.stager_steps, self.consumer_steps)
		
		# import and init producers
		for producer_step in self.producer_steps:
			producer_zmq = self.instantiation( producer_step.section_name,self.PRODUCER,port_out = producer_step.port_out)
			if producer_zmq.init() == False : return False
			self.producers_zmq.append(producer_zmq)
			
		# import and init stagers
		for stager_step in self.stager_steps:
			stager_zmq = self.instantiation(stager_step.section_name ,self.STAGER,
											port_in = stager_step.port_in, port_out = stager_step.port_out)
			if stager_zmq.init() == False : return False
			self.stagers_zmq.append(stager_zmq)
			
		# import and init consumers
		for consumer_step in self.consumer_steps:
			consumer_zmq = self.instantiation( consumer_step.section_name, self.CONSUMER,port_in = consumer_step.port_in)
			if consumer_zmq.init() == False : return False
			self.consumers_zmq.append(consumer_zmq)
			
		self.print()
		return True


	def _generate_steps(self):
		self.producer_steps = self.get_pipe_steps(self.PRODUCER)
		self.stager_steps = self.get_pipe_steps(self.STAGER)
		self.consumer_steps = self.get_pipe_steps(self.CONSUMER)
		
		# Now that all steps exists, set previous step
		for step in self.consumer_steps + self.stager_steps:
			prev_section_name = self.get_prev_step_section_name(step.section_name)
			if prev_section_name != None:
				prev_step = self.get_step_by_section_name(prev_section_name)
				step.prev_step = prev_step
		return True
				
	def _configure_port_out(self, producer_steps, stager_steps):
		"""
		Configure port_out from pipeline's ports list for producers and stagers
		
		returns:
		--------
		True if every port is configured
		False if no more ports are available
		"""
		for producer_step in producer_steps:
			producer_step.port_out = self._get_next_available_port()
			if producer_step.port_out == None : return False
			
		for stager_step in stager_steps:
			stager_step.port_out =  self._get_next_available_port()
			if stager_step.port_out == None : return False
			
		return True
				
	def _configure_port_in(self,stager_steps,consumer_steps):
		"""
		Configure port_in from pipeline's ports list for stagers and consumers
		"""
		for stager_step in stager_steps:
			stager_step.port_in = self.get_prev_step_port_out(stager_step.section_name) 
		
		for consumer_step in consumer_steps:
			consumer_step.port_in = self.get_prev_step_port_out(consumer_step.section_name) 
		
	def _get_next_available_port(self):
		"""
		return an available port from  pipeline's ports list 
		"""
		if self.ports != None and len(self.ports) > 0:
			port = self.ports.pop(0)
			return port
		return None
		

	def instantiation(self,section_name, stage_type, port_in = None, port_out=None):
		if section_name == None : raise PipelineError("Cannot create instance of "+ section_name)

		obj = self.conf.dynamic_class_from_module(section_name,pass_configuration=True)
		obj.section_name = section_name
		if obj  == None: raise PipelineError("Cannot create instance of "+section_name)

		if stage_type == self.STAGER:
			thread = StagerZmq(obj,port_in,port_out)
		elif stage_type == self.PRODUCER:
			thread = ProducerZmq(obj,port_out)
		elif stage_type == self.CONSUMER:
			thread = ConsumerZmq(obj,port_in)
		
		else: raise PipelineError("Cannot create instance of",section_name,". Type",stage_type, "does not exist." )
		
		return thread

	def get_pipe_steps(self,role):
		"""
		Returns:
		--------
		List of section name filter by specific role (PRODUCER, STAGER, CONSUMER)
		"""
		result = list()
		for section in self.conf.get_section_list():
			if self.conf.has_key( self.ROLE, section) == True:
				if self.conf.get(self.ROLE, section ) == role:
					
					if role == self.PRODUCER:
						step = PipeStep(section)
						
					elif role == self.STAGER:
						step = PipeStep(section)
						
					elif role == self.CONSUMER:
						step = PipeStep(section)
						
					result.append(step)
		return result



		
	def get_prev_step_section_name(self,section):
		"""
		Parameters:
		-----------
		section_name : str
			section name of a  pipeline step
		
		Returns:
		section_name of previons step
		"""
		if self.conf.has_key( "prev", section) == True:
				return  self.conf.get("prev", section )
		return None
		
				
	def get_prev_step_port_out(self,section):
		"""
		Parameters:
		-----------
		section_name : str
			section name of a  pipeline step
		
		Returns:
		port_out of previons step
		"""
		prev_section = self.get_prev_step_section_name(section)
		if  prev_section != None: 
			if self.producer_steps != None: 
				for producer_step in self.producer_steps:
					if producer_step.section_name == prev_section:
						return producer_step.port_out
					
			if self.stager_steps != None: 
				for stager_step in self.stager_steps:
					if stager_step.section_name == prev_section:
						return stager_step.port_out
		return None
				

		
	def print(self):
		chaine = list() 
		for consumer in self.consumer_steps: 
			chaine.append(" /* \tconsumer " + consumer.section_name + " (port_in:" + str(consumer.port_in)  + ", port_out:" + str(consumer.port_out) + ")\t*/")
			prev = consumer.prev_step
			while prev != None:
				chaine.append(" /* \t" + str(prev.section_name) +" (port_in:" + str(prev.port_in)  + ", port_out:" + str(prev.port_out) +   " \t*/")
				prev = prev.prev_step
				
		chaine.reverse()
		print("\n\n ----------------- Pipeline configuration --------------- ")
		print(" /*\t\t\t\t\t*/")
		for  item  in chaine[:-1]:
			print (item)
			print(" /*\t\t|\t\t\t\t\t*/")
			print(" /*\t\t|\t\t\t\t\t*/")
			print(" /*\t\tV\t\t\t\t\t*/")
		print(chaine[-1])
		print(" /*\t\t\t\t\t\t\t*/")
		print(" -------------- End Pipeline configuration --------------- \n\n ")
		
		
	def get_step_by_section_name(self,section_name): 
		
		for step in self.producer_steps + self.stager_steps + self.consumer_steps:
			if step.section_name == section_name: return step
		
		return  None
			
	def run(self):
		
		for prod in self.producers_zmq:
			prod.start()
			
		for stage in self.stagers_zmq:
			stage.start()
			
		for cons in self.consumers_zmq:
			cons.start()
			
		# Wait that all producers end of run method
		for prod in self.producers_zmq:
			prod.join()
			
		# Execute finish cleanly
		for prod in self.producers_zmq:
			prod.finish()
		
		for stage in self.stagers_zmq:
			stage.join()
			
		for stage in self.stagers_zmq:
			stage.finish()
		
		for cons in self.consumers_zmq:
			cons.join()
			cons.finish()
        
        
        
                