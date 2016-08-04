from time import sleep
import threading
import subprocess
import os
from ctapipe.core import Component
from traitlets import Unicode


class MyCp(Component):
	output_dir = Unicode('/tmp/test/in', help='directory receving produced data').tag(
	config=True, allow_none=False)
	source_dir = Unicode('/tmp/test/out', help='directory contianing data files').tag(
			config=True, allow_none=False)
	out_extension = Unicode('type1', help='directory receving produced data').tag(
	config=True, allow_none=False)

	def init(self):
		if self.source_dir == None or self.output_dir == None or self.out_extension == None:
			self.log.error("MyCp :configuration error ")
			self.log.error('source_dir: {} output_dir: {} out_extension: {}'
			.format(self.source_dir, self.output_dir,self.out_extension))
			return False

		if not os.path.exists(self.output_dir):
			try:
				os.makedirs(self.output_dir)
			except OSError as e:
				self.log.error(
					"{} : could not create output directory {}: {}".format(self.section_name,  self.output_dir, e))
				return False
		return True

	def run(self):
		for input_file in os.listdir(self.source_dir):
			self.log.info('Mycp start {}'.format(input_file))
			output_file = self.output_dir+"/"+input_file+self.out_extension
			cmd = ['cp',self.source_dir+"/"+input_file,output_file]
			proc = subprocess.Popen(cmd)
			proc.wait()
			yield output_file



	def finish(self):
		self.log.info("--- MyCp finish ---")
