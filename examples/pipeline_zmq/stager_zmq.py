# coding: utf8
from time import sleep
import zmq
from threading import Thread, Lock
    
class StagerZmq(Thread):
	def __init__(self,coroutine,port_in, port_out):
		super(StagerZmq, self).__init__()
		self.receiver = None
		self.sender = None
		self.context = None
		self.coroutine = coroutine
		self.port_in = "tcp://localhost:" + str(port_in)
		self.port_out ="tcp://*:" + str(port_out)

		
	def init(self):

		self.context = zmq.Context()
		# Socket to receive messages on
		self.receiver = self.context.socket(zmq.PULL)
		self.receiver.setsockopt(zmq.RCVHWM, 10)
		self.receiver.connect(self.port_in)
		# Socket to send messages to
		self.sender = self.context.socket(zmq.PUSH)
		self.sender.setsockopt(zmq.SNDHWM, 10)
		self.sender.bind(self.port_out)
		
		print("-------------------- Stager init")
		if self.coroutine  == None: return False
		if self.coroutine.init() == None: return None
		print("-------------------- Stager init return True")
		
		return True

			
	def run(self):
		while (True):
			print("-------------------- Stager wait for new entry")
			result = self.receiver.recv_pyobj()
			print("-------------------- Stager receive a new entry")
			if isinstance(result,str):
				print("-------------------- Stager receive ["+result+"]")
				if result == "STOP": 
					self.sender.send_pyobj("STOP")
					break
			result = self.coroutine.run(result)
			self.sender.send_pyobj(result)
				


	def finish(self):
		self.coroutine.finish()