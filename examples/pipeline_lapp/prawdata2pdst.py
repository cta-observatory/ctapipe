from time import sleep
import threading
import subprocess
import os

    
SECTION_NAME = 'PRAW2PDST'

class PRawData2Pdst():
	def __init__(self,configuration=None):
		self.configuration = configuration
		self.exe = None
		self.source_dir = None
		
	def init(self):
		self.exe = self.configuration.get('executable', section=SECTION_NAME)
		self.source_dir = self.configuration.get('source_dir', section=SECTION_NAME)
		return True

	def run(self):
		for input_file in os.listdir(self.source_dir):
			sleep(2)
			if input_file.find('prun') != -1: 
				output = self.source_dir+"/"+input_file.split('.')[0]+'.ptabhillas'
				cmd = [self.exe,'-i',self.source_dir+"/"+input_file,'-o',output]
				proc = subprocess.Popen(cmd)
				proc.wait()
				yield output

	def finish(self):
		print("--- PRawData2Pdst finish ---")