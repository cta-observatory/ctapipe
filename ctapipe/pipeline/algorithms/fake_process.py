from time import sleep, clock
import os
import random


class FakeProcess():
	def __init__(self,configuration=None):
		pass

	def run(self,_input):
		sleep(random.random())
		return  int(_input+1)

	def finish(self):
		pass

	def init(self):
		return True
