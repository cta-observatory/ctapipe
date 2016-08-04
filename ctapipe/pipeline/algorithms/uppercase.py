from time import sleep
import threading
import subprocess
import os
from ctapipe.core import Component
from traitlets import Unicode



class UpperCase(Component):
	output_dir = Unicode('/tmp/test/out2', help='directory receving produced data').tag(
		config=True, allow_none=False)
	out_extension = Unicode('uc', help='directory receving produced data').tag(
		config=True, allow_none=False)

	def init(self):
		if self.output_dir == None or self.out_extension == None:
			self.log.error("UpperCase :configuration error ")
			self.log.error('output_dir: {} out_extension:'.format(self.output_dir,self.out_extension))
			return False

		if not os.path.exists(self.output_dir):
			try:
				os.makedirs(self.output_dir)
			except OSError:
				self.log.error('UpperCase: could not create output_dir {}'.format(self.output_dir))
				return False
		return True

	def run(self,input_file):
		self.log.info('UpperCase{}'.format(input_file))
		output_file =  self.output_dir + '/'+input_file.split('/')[-1] + '.' + self.out_extension
		cmd = 'cat '+ input_file + ' | tr [a-z] [A-Z] > ' + output_file
		ps = subprocess.Popen(cmd,shell=True,stderr=subprocess.STDOUT)
		ps.communicate()[0]
		return  output_file



	def finish(self):
		print("--- UpperCase finish ---")
