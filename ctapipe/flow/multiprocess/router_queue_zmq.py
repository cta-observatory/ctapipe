from multiprocessing import Process
from multiprocessing import Value
from zmq import POLLIN
from zmq import PUB
from zmq import ROUTER
from zmq import Poller
from zmq import Context
from zmq.error import ZMQError
from pickle import dumps
from pickle import loads
from ctapipe.core import Component


class RouterQueue(Process, Component):

    """`RouterQueue` class represents a router between pipeline steps, and it
    manages queue for prev step.
    It is derived from Process class.
    RouterQueue class is connected to one or more input steps and to one
    or more output steps thanks to zmq.ROUTER sockets.
    If inputs arrive quickers than output are sent (because next stage have not
    enough time to compute) these inputs are queued in RouterQueue.
    RouterQueue send output the next steps in LRU(last recently used) pattern.
    If a queue limit is reach for a step, RouterQueue replies "FULL"
    without taking new job in account, so the prev step have to send it again
    later.
    """

    def __init__(
            self, connections=None, gui_address=None):
        """
        Parameters
        ----------
        connections: dict {'STEP_NANE' : (STEP port in, STEP port out,
                                            STEP queue lengh max )}
            Port in, port out  for socket for each next steps.
            Max queue lengh (-1 menans no maximum)
        gui_address : str
            GUI port for ZMQ 'hostname': + 'port'
        """
        Process.__init__(self)
        Component.__init__(self, parent=None)
        self.gui_address = gui_address
        # list of available stages which receive next job
        # This list allow to use next_stage in a LRU (Last recently used)
        # pattern
        self.next_available_stages = dict()
        # queue jobs to be send to next_stage
        self.queue_jobs = dict()
        self.router_sockets = dict()
        self.dealer_sockets = dict()
        self.queue_limit = dict()
        self.connections = connections or {}
        self.done = False
        self._stop = Value('i', 0)
        self._total_queue_size = Value('i', 0)

    def run(self):
        """
        Method representing the process’s activity.
        It sends a job present in its queue (FIFO) to an available stager
        (if exist). Then it polls its sockets (in and out).
        When received new input from input socket, it appends contains to
        its queue.
        When received a signal from its output socket, it append sender
        (a pipeline step) to availble stagers list
        """
        if self.init_connections():
            nb_job_remains = 0
            while not self.stop or nb_job_remains > 0:
                for name in self.connections:
                    queue = self.queue_jobs[name]
                    next_available = self.next_available_stages[name]
                    if queue and next_available:
                        # get that oldest job and remove it form list
                        job = self.queue_jobs[name].pop(0)
                        if self.gui_address:
                            self.update_gui(name)
                        # Get the next_stage for new job, and remove it from
                        # available list
                        next_stage = self.next_available_stages[name].pop(0)
                        # send new job
                        self.dealer_sockets[name].send_multipart(
                            [next_stage, b"", dumps(job)])
                    # check if new socket message arrive. Or skip after timeout
                    # (100 s)
                sockets = dict(self.poller.poll(100))
                # Test if message arrived from next_stages
                for n, socket_dealer in self.dealer_sockets.items():
                    if (socket_dealer in sockets and
                            sockets[socket_dealer] == POLLIN):
                        request = socket_dealer.recv_multipart()
                        # Get next_stage identity(to responde) and message
                        next_stage, _, _ = request[:3]
                        # add stager to next_available_stages
                        self.next_available_stages[n].append(next_stage)
                # Test if message arrived from prev_stage (stage or producer)
                for n, socket_router in self.router_sockets.items():
                    if (socket_router in sockets and
                            sockets[socket_router] == POLLIN):
                        # Get next prev_stage request
                        address, empty, request = socket_router.recv_multipart()
                        # store it to job queue
                        queue = self.queue_jobs[n]
                        if (len(queue) > self.queue_limit[n]
                                and self.queue_limit[n] != -1):
                            socket_router.send_multipart([address, b"", b"FULL"])
                        else:
                            queue.append(loads(request))
                            if self.gui_address:
                                self.update_gui(n)
                            socket_router.send_multipart([address, b"", b"OK"])
                nb_job_remains = 0
                for n, queue in self.queue_jobs.items():
                    nb_job_remains += len(queue)
                self._total_queue_size.value = nb_job_remains
            for socket in self.router_sockets.values():
                socket.close()
            for socket in self.dealer_sockets.values():
                socket.close()
        self.done = True

    def isQueueEmpty(self, stage_name=None):
        """
        Get status of steps' queue
        Parameters
        ----------
        stage_name: str
            router_name corresponding to stager name or consumer name
            If stage_name=None, it take in account all queues
        Returns
        -------
        True is corresponding queue is empty, otherwise False
        If stage_name = None it returns True if sum of all queues is 0
        """
        if stage_name:
            val = stage_name.find("$$process_number$$")
            if val != -1:
                name_to_search = stage_name[0:val]
            else:
                name_to_search = stage_name
            for name, queue in self.queue_jobs.items():
                if name.find("_router"):
                    pos = name.find("_router")
                    name = name[:pos]
                if name == name_to_search:
                    if not queue:
                        return True
            return False
        else:
            for queue in self.queue_jobs.values():
                if queue:
                    return False
            return True

    def init_connections(self):
        """
        Initialise zmq sockets, poller and queues.
        Because this class is s Process, This method must be call in the run
         method to be hold by the correct process.
        """
        # Prepare our context and sockets
        context = Context()
        # Socket to talk to prev_stages
        for name, connections in self.connections.items():
            self.queue_limit[name] = connections[2]
            sock_router = context.socket(ROUTER)
            try:
                sock_router.bind('tcp://*:' + connections[0])
            except ZMQError as e:
                self.log.error('{} : tcp://localhost:{}'
                               .format(e, connections[0]))
                return False
            self.router_sockets[name] = sock_router
            # Socket to talk to next_stages
            sock_dealer = context.socket(ROUTER)
            try:
                sock_dealer.bind("tcp://*:" + connections[1])
            except ZMQError as e:
                self.log.error('{} : tcp://localhost:{}'
                               .format(e, connections[1]))
                return False
            self.dealer_sockets[name] = sock_dealer
            self.next_available_stages[name] = list()
            self.queue_jobs[name] = list()
        # Use a ZMQ Pool to get multichannel message
        self.poller = Poller()
        # Register dealer socket to next_stage
        for n, dealer in self.dealer_sockets.items():
            self.poller.register(dealer, POLLIN)
        for n, router in self.router_sockets.items():
            self.poller.register(router, POLLIN)
        # Register router socket to prev_stages or producer
        self.socket_pub = context.socket(PUB)
        if self.gui_address is not None:
            try:
                self.socket_pub.connect("tcp://" + self.gui_address)
            except ZMQError as e:
                self.log.error("".format(e, self.gui_address))
                return False
        # This flag stop this current process
        return True

    def update_gui(self, name):
        """
        send status to GUI
        name: str
            step name
        """
        msg = [name, str(len(self.queue_jobs[name]))]
        self.socket_pub.send_multipart(
            [b'GUI_ROUTER_CHANGE', dumps(msg)])

    @property
    def stop(self):
        return self._stop.value

    @stop.setter
    def stop(self, value):
        self._stop.value = value

    @property
    def total_queue_size(self):
        return self._total_queue_size.value
