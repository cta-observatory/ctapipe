from time import sleep, clock
import os
import random


class FakeProcess2():
	def __init__(self,configuration=None):
		pass

	def run(self,_input):
		print("FakeProcess2 ",int(_input+1))
		return  int(_input+1)

	def finish(self):
		pass

	def init(self):
		return True
