from ctapipe.utils.datasets import get_path
from ctapipe.io.hessio import hessio_event_source
from ctapipe.configuration.core import Configuration, ConfigurationException
import threading
from sys import stderr
import os
import subprocess


SECTION_NAME = 'SHADOK'
class Shadok():   
	def __init__(self,configuration=None):
		self.configuration = configuration
		self.source_dir = None
		self.exe = None
		
	def init(self):
		print("--- Shadok init ---")
		if self.configuration == None:
			self.configuration = Configuration()
		self.source_dir = self.configuration.get('source_dir', section=SECTION_NAME)
		self.exe = self.configuration.get('executable', section=SECTION_NAME)
		return True
		
	def run(self):
		for input_file in os.listdir(self.source_dir):
			if input_file.find('simtel.gz') != -1: 
				output = input_file.split('.')[0]+'.prun'
				print("--- Shadok start for file", input_file, "---")
				cmd = [self.exe,'-i',self.source_dir+"/"+input_file,'-o',self.source_dir+"/"+output,'-l 5']
				print('SHADOK cmd', cmd)
				proc = subprocess.Popen(cmd)
				proc.wait()
				print('SHADOK return',proc.returncode)
				print('SHADOK yield',  output)
				yield output

		
		
		
	def finish(self):
		print ( "--- Shadok finish ---")
		pass

        