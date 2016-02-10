from ctapipe.configuration.core import Configuration, ConfigurationException
from time import sleep
import subprocess
import os
  

class ConsumerProcess:
	def __init__(self,configuration=None):
		self.file = None
		self.configuration = configuration
		self.executable = None
		self.out_extension = None
		self.output_dir = None
		self.options = None

		
	def init(self):
		self.executable =  self.configuration.get('executable', section=self.section_name)
		self.output_dir = self.configuration.get('output_dir', section=self.section_name)
		self.out_extension = self.configuration.get('out_extension', section=self.section_name)
		self.options = self.configuration.get('options', section=self.section_name)
		if self.output_dir == "" :  self.output_dir = None
		
		if not self.output_dir == None : 
			if not os.path.exists(self.output_dir):
				try:
					os.mkdir(self.output_dir)
				except OSError:
					print("ConsumerProcess: could not create output_dir", self.output_dir)
					return False
		return True
	
	def run(self,input_file):
		print('ConsumerProcess', input_file)
		if  self.output_dir != None:
			output_file =  self.output_dir + '/'+input_file.split('/')[-1] 
			if self.out_extension != None:
				output_file = output_file.rsplit('.',2)[0]
				output_file+= "." + self.out_extension
			cmd = [self.executable , "-i " , input_file  , "-o" , output_file , self.options ]
		else:
			cmd =[self.executable , "-i " , input_file , self.options ]
		
		print("ConsumerProcess cmd", cmd)
		ps = subprocess.Popen(cmd,shell=True,stderr=subprocess.STDOUT)
		ps.communicate()[0]
		print("ConsumerProcess ps.wait() done")
	
	def finish(self):
		print("--- ConsumerProcess finish ---")
