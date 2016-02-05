from time import sleep
import threading
import subprocess
import os


SECTION_NAME='EXE1'    

class MyCp():
	def __init__(self,configuration=None):
		self.configuration = configuration
		self.source_dir = None
		self.out_extension = None
		self.output_dir = None
		self.section_name = None
		
	def init(self):
		self.source_dir = self.configuration.get('source_dir', section=SECTION_NAME)
		self.output_dir = self.configuration.get('output_dir', section=SECTION_NAME)
		self.out_extension = self.configuration.get('out_extension', section=SECTION_NAME)
		if self.source_dir == None or self.output_dir == None or self.out_extension == None: 
			print("MyCp :configuration error ")
			print('source_dir:',self.source_dir, 'output_dir:', self.output_dir, 'out_extension:',self.out_extension)
			return False
		
		if not os.path.exists(self.output_dir):
			try:
				os.mkdir(self.output_dir)
			except OSError:
				print("MyCp: could not create output_dir", self.output_dir)
				return False
		return True

	def run(self):
		for input_file in os.listdir(self.source_dir):
			print('Mycp start ', input_file)
			output_file = self.output_dir+"/"+input_file+self.out_extension
			cmd = ['cp',self.source_dir+"/"+input_file,output_file]
			proc = subprocess.Popen(cmd)
			proc.wait()
			yield output_file



	def finish(self):
		print("--- MyCp finish ---")