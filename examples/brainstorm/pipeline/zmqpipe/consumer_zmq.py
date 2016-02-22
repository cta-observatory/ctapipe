# coding: utf8


import zmq
from sys import stderr
from threading import Thread, Lock

class ConsumerZmq(Thread):
		def __init__(self,coroutine,port):
			super(ConsumerZmq, self).__init__()
			self.file = None
			self.receiver = None
			self.coroutine = coroutine
			self.port = "tcp://localhost:" + port
			
		
		def init(self):
			if self.coroutine  == None: return False
			if self.coroutine.init() == False: return False
			context = zmq.Context()
			# Socket to receive messages on
			self.receiver = context.socket(zmq.PULL)
			self.receiver.setsockopt(zmq.RCVHWM, 10)
			self.receiver.connect(self.port)
			return True

		def run(self):
			# Wait for start of batch
			while (True):
				result = self.receiver.recv_pyobj()
				if ( result == "STOP" ) : 
					break
				self.coroutine.run(result)
		
		
		def finish(self):
			self.coroutine.finish()
