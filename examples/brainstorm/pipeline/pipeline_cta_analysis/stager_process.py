from time import sleep
import threading
import subprocess
import os
from base_process import BaseProcess


class StagerProcess(BaseProcess):
	def __init__(self,configuration=None):
		BaseProcess.__init__(self,configuration)
	
	def run(self,input_file):
		cmd,output_file = super().build_command(input_file)
		proc = subprocess.Popen(cmd)
		proc.wait()
		return  output_file

	def finish(self):
		print('--- ', self.section_name, ' finish ---')