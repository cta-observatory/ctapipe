from time import sleep
import threading
import subprocess
import os


class StagerProcess():
	def __init__(self,configuration=None):
		self.section_name = None
		self.configuration = configuration
		self.executable = None
		self.out_extension = None
		self.output_dir = None
		self.options = None
		
	def init(self):
		self.executable =  self.configuration.get('executable', section=self.section_name)
		self.options = self.configuration.get('options', section=self.section_name)
		self.output_dir = self.configuration.get('output_dir', section=self.section_name)
		self.out_extension = self.configuration.get('out_extension', section=self.section_name)
		if self.output_dir == None or self.out_extension == None: 
			print("StagerProcess :configuration error ")
			print('output_dir:', self.output_dir, 'out_extension:',self.out_extension)
			return False
		
		
		if not os.path.exists(self.output_dir):
			try:
				os.mkdir(self.output_dir)
			except OSError:
				print("StagerProcess: could not create output_dir", self.output_dir)
				return False
		return True

	def run(self,input_file):
		print('StagerProcess', input_file)
		if  self.output_dir != None:
			output_file =  self.output_dir + '/'+input_file.split('/')[-1] 
			if self.out_extension != None:
				output_file = output_file.rsplit('.',2)[0]
				output_file+= "." + self.out_extension
			cmd = self.executable + " -i " + input_file  + " -o " + output_file + " " +  self.options 
		else:
			cmd = self.executable + " -i " + input_file + " " +  self.options 
		ps = subprocess.Popen(cmd,shell=True,stderr=subprocess.STDOUT)
		ps.communicate()[0]
		return  output_file



	def finish(self):
		print("--- StagerProcess finish ---")