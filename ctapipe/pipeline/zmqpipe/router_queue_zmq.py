import time
import threading
import zmq
import pickle
from ctapipe.core import Component


class RouterQueue(threading.Thread, Component):

    """`RouterQueue` class represents a router between pipeline steps, and it
    manages queue for prev step.
    It is derived from Thread class.
    RouterQueue class is connected to one or more input steps and to one
    or more output steps thanks to zmq.ROUTER sockets.
    If inputs arrive quickers than output are sent (because next stage have not
    enough time to compute) these inputs are queued in RouterQueue.
    RouterQueue send output the next steps in LRU(last recently used) pattern.
    """

    def __init__(
            self, sock_router_port, socket_dealer_port,
            step_names=dict(), gui_address=None):
        """
        Parameters
        ----------
        sock_router_port: str
            Port number for input router socket url
        socket_dealer_port: str
            Port number for ouput router socket url
        """
        threading.Thread.__init__(self)
        self.gui_address = gui_address
        # list of available stages which receive next job
        # This list allow to use next_stage in a LRU (Last recently used)
        # pattern
        self.next_available_stages = dict()
        # queue jobs to be send to next_stage
        self.queue_jobs = dict()
        self.router_sockets = dict()
        self.dealer_sockets = dict()
        self.stop = False
        self.names = step_names
        self.sock_router_port = sock_router_port
        self.socket_dealer_port = socket_dealer_port

    def init(self):
        # Prepare our context and sockets
        context = zmq.Context.instance()
        # Socket to talk to prev_stages
        for name in self.names:
            sock_router = context.socket(zmq.ROUTER)
            try:
                sock_router.bind('inproc://' + self.sock_router_port[name])
            except zmq.error.ZMQError as e:
                self.log.error('{} : inproc://{}'
                               .format(e, self.sock_router_port[name]))
                return False
            self.router_sockets[name] = sock_router
            # Socket to talk to next_stages
            sock_dealer = context.socket(zmq.ROUTER)
            try:
                sock_dealer.bind("inproc://" + self.socket_dealer_port[name])
            except zmq.error.ZMQError as e:
                self.log.error('{} : inproc://{}'
                               .format(e, self.sock_router_port[name]))
                return False

            self.dealer_sockets[name] = sock_dealer

            self.next_available_stages[name] = list()
            self.queue_jobs[name] = list()

        # Use a ZMQ Pool to get multichannel message
        self.poller = zmq.Poller()
        # Register dealer socket to next_stage
        for n, dealer in self.dealer_sockets.items():

            self.poller.register(dealer, zmq.POLLIN)
        for n, router in self.router_sockets.items():
            self.poller.register(router, zmq.POLLIN)

        # Register router socket to prev_stages or producer
        self.socket_pub = context.socket(zmq.PUB)
        if self.gui_address is not None:
            try:
                self.socket_pub.connect("tcp://" + self.gui_address)
            except zmq.error.ZMQError as e:
                self.log.error("".format(e, self.gui_address))
                return False
        # This flag stop this current thread
        return True

    def run(self):
        """
        Method representing the thread’s activity.
        It sends a job present in its queue (FIFO) to an available stager
        (if exist). Then it polls its sockets (in and out).
        When received new input from input socket, it appends contains to
        its queue.
        When received a signal from its output socket, it append sender
        (a pipeline step) to availble stagers list
        """
        nb_job_remains = 0

        while not self.stop or nb_job_remains > 0:
            for name in self.names:
                queue = self.queue_jobs[name]

                # queue,next_available in
                # zip(self.queue_jobs,self.next_available_stages):
                next_available = self.next_available_stages[name]
                if queue and next_available:
                    # get that oldest job and remove it form list
                    job = self.queue_jobs[name].pop(0)
                    self.update_gui(name)
                    # Get the next_stage for new job, and remove it from
                    # available list
                    next_stage = self.next_available_stages[name].pop(0)
                    # send new job
                    self.dealer_sockets[name].send_multipart(
                        [next_stage, b"", pickle.dumps(job)])
                # check if new socket message arrive. Or skip after timeout
                # (100 s)
            sockets = dict(self.poller.poll(100))

            # Test if message arrived from next_stages
            for n, socket_dealer in self.dealer_sockets.items():
                if (socket_dealer in sockets and
                        sockets[socket_dealer] == zmq.POLLIN):

                    request = socket_dealer.recv_multipart()
                    # Get next_stage identity(to responde) and message
                    next_stage, _, message = request[:3]

                    cmd = pickle.loads(message)
                    # add stager to next_available_stages
                    self.next_available_stages[n].append(next_stage)

            # Test if message arrived from prev_stage (stage or producer)
            for n, socket_router in self.router_sockets.items():
                if (socket_router in sockets and
                        sockets[socket_router] == zmq.POLLIN):
                    # Get next prev_stage request
                    address, empty, request = socket_router.recv_multipart()
                    # store it to job queue
                    queue = self.queue_jobs[n]
                    queue.append(pickle.loads(request))
                    self.update_gui(n)
                    socket_router.send_multipart([address, b"", b"OK"])
            nb_job_remains = 0
            for n, queue in self.queue_jobs.items():
                nb_job_remains += len(queue)
        for socket in self.router_sockets.values():
            socket.close()
        for socket in self.dealer_sockets.values():
            socket.close()

    def isQueueEmpty(self, stage_name):
        """
        Parameters
        ----------
        stage_name: str
            router_name corresponding to stager name or consumer name
        Returns
        -------
        True is corresponding queue is empty, otherwise False
        """
        val = stage_name.find("$$thread_number$$")
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

    def finish(self):
        """
        set stop flag to True to stop Thread activity
        """
        self.stop = True

    def update_gui(self, name):
        msg = [name, str(len(self.queue_jobs[name]))]
        self.socket_pub.send_multipart(
            [b'GUI_ROUTER_CHANGE', pickle.dumps(msg)])
