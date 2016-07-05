from time import sleep, clock
import threading
import subprocess
import os
from base_process import BaseProcess
import random


class FakeProcess_2():
	def __init__(self,configuration=None):
		BaseProcess.__init__(self,configuration)

	def run(self,input_file):
		return  clock()

	def finish(self):
		print('--- ', self.section_name, ' finish ---')

	def init(self):
		return True
