from copy import deepcopy
from threading import Thread, Lock
import zmq


class ProducerZmq(Thread):
   
	def __init__(self,coroutine,port):
		super(ProducerZmq, self).__init__()
		self.sender = None
		self.context =None
		self.coroutine = coroutine
		self.port = "tcp://*:"+ port
		
	def init(self):
		if self.coroutine  == None: return False
		if self.coroutine.init() == False: return False
		self.context = zmq.Context()
		# Socket to send messages on
		self.sender = self.context.socket(zmq.PUSH)
		self.sender.setsockopt(zmq.SNDHWM, 10)
		self.sender.bind(self.port)
		return True
		
	def run(self):
		generator = self.coroutine.run()
		for result in generator:
			self.sender.send_pyobj(result)
		self.sender.send_pyobj("STOP")
		
	def finish(self):
		self.coroutine.finish()
        
        