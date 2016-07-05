from time import sleep, clock
import os
import random


class FakeProducer():
	def __init__(self,configuration=None):
		print("FakeProducer configuration",configuration)
		pass

	def run(self):
		for i in range(10):
			yield i*10

	def finish(self):
		pass

	def init(self):
		return True
