from time import sleep
import threading
import subprocess
import os


SECTION_NAME='EXE2'    

class UpperCase():
	def __init__(self,configuration=None):
		self.configuration = configuration
		self.out_extension = None
		self.output_dir = None
		
	def init(self):
		self.output_dir = self.configuration.get('output_dir', section=SECTION_NAME)
		self.out_extension = self.configuration.get('out_extension', section=SECTION_NAME)
		if self.output_dir == None or self.out_extension == None: 
			print("UpperCase :configuration error ")
			print('output_dir:', self.output_dir, 'out_extension:',self.out_extension)
			return False
		
		
		if not os.path.exists(self.output_dir):
			try:
				os.mkdir(self.output_dir)
			except OSError:
				print("UpperCase: could not create output_dir", self.output_dir)
				return False
		return True

	def run(self,input_file):
		print('UpperCase', input_file)
		output_file =  self.output_dir + '/'+input_file.split('/')[-1] + '.' + self.out_extension
		cmd = 'cat '+ input_file + ' | tr [a-z] [A-Z] > ' + output_file
		ps = subprocess.Popen(cmd,shell=True,stderr=subprocess.STDOUT)
		ps.communicate()[0]
		return  output_file



	def finish(self):
		print("--- UpperCase finish ---")