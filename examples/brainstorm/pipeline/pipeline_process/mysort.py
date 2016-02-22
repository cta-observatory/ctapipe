from ctapipe.configuration.core import Configuration, ConfigurationException
from time import sleep
import subprocess
import os

SECTION_NAME='EXE3'    

class MySort:
	def __init__(self,configuration=None):
		self.file = None
		self.configuration = configuration
		self.output_dir = None
		
	def init(self):
		self.output_dir = self.configuration.get('output_dir', section=SECTION_NAME)
		
		if self.output_dir == None: 
			print("MySort :output_dir is not defined")
			return False
		if not os.path.exists(self.output_dir):
			try:
				os.mkdir(self.output_dir)
			except OSError:
				print("MySort: could not create output_dir", self.output_dir)
				return False
		return True
	
	def run(self,input_file):
		print('MySort', input_file)
		output_file =  self.output_dir + '/'+input_file.split('/')[-1] 
		cmd = 'sort ' + input_file + ' > '+  output_file
		proc = subprocess.Popen(cmd,shell=True, stderr=subprocess.STDOUT)
		proc.wait()
	
	def finish(self):
		print("--- MySort finish ---")
