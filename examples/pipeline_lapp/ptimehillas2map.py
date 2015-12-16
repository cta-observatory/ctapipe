from ctapipe.configuration.core import Configuration, ConfigurationException
from time import sleep
import subprocess

SECTION_NAME = 'TIME2MAP'

class PTimeHillas2Map:
	def __init__(self,configuration=None):
		self.file = None
		self.configuration = configuration
		self.exe = None
		
	def init(self):
		self.exe = self.configuration.get('executable', section=SECTION_NAME)
		return True
	
	def run(self,input_file):
		output = input_file.split('.')[0]+'.png'
		cmd = [self.exe,'-i',input_file,'-o',output]  
		proc = subprocess.Popen(cmd)
		proc.wait()
	
	def finish(self):
		print("--- PTimeHillas2Map finish ---")
