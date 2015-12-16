from ctapipe.configuration.core import Configuration, ConfigurationException
from time import sleep
import subprocess

SECTION_NAME = 'HILLAS2TIME'

class PTableHillas2PTimeHillas:
	def __init__(self,configuration=None):
		self.file = None
		self.configuration = configuration
		self.exe = None
		
	
	def init(self):
		self.exe = self.configuration.get('executable', section=SECTION_NAME)
		print('-----------------------------------PTableHillas2PTimeHillas init')
		return True

	
	def run(self,input_file):
		print("--- PTableHillas2PTimeHillas start for file", input_file, "---")
		output = input_file.split('.')[0]+'.ptimehillas'
		cmd = [self.exe,'-i',input_file,'-o',output]  
		proc = subprocess.Popen(cmd)
		proc.wait()
		print("--- PTableHillas2PTimeHillas res ---" , proc.returncode)
		return output 

	def finish(self):
		print("--- PTableHillas2PTimeHillas finish ---")
