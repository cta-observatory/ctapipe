from time import sleep
import threading
import subprocess
import os


class ProducerProcess():
	def __init__(self,configuration=None):
		self.configuration = configuration
		self.source_dir = None
		self.output_dir = None
		self.executable = None
		self.out_extension = None
		self.options = None
		
		
	def init(self):
		self.source_dir = self.configuration.get('source_dir', section=self.section_name)
		self.output_dir = self.configuration.get('output_dir', section=self.section_name)
		self.executable =  self.configuration.get('executable', section=self.section_name)
		self.out_extension = self.configuration.get('out_extension', section=self.section_name)
		self.options = self.configuration.get('options', section=self.section_name)
		if self.source_dir == None or self.output_dir == None : 
			print("ProducerProcess :configuration error ")
			print('source_dir:',self.source_dir, 'output_dir:', self.output_dir,)
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
			print('ProducerProcess start ', input_file)
			# remove path before name
			output_file = self.output_dir + "/" + input_file.split('/')[-1]
			
			if self.out_extension != None:
				# remove old_extension
				output_file = output_file.rsplit('.',2)[0]
				output_file+= "." + self.out_extension
			cmd = [self.executable,"-i",self.source_dir+"/"+input_file,"-o" , output_file]
			print("Producer command: ", cmd)
			proc = subprocess.Popen(cmd)
			proc.wait()
			yield output_file



	def finish(self):
		print("--- ProducerProcess finish ---")