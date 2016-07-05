from time import sleep
import threading
import subprocess
import os
from ctapipe.configuration.core import Configuration, ConfigurationException



class ListFiles():
	def __init__(self,configuration=None):
		self.configuration=configuration

	def init(self):
		self.source_dir = self.configuration.get('source_dir', section=self.section_name)

	def run(self):
		for input_file in os.listdir(self.source_dir):
			yield self.source_dir+"/"+input_file

	def finish(self):
		print('---', self.section_name, 'finish ---')



if __name__ == '__main__':
	conf = Configuration('pipiline_test.ini')
	foo = ListFiles(conf)
