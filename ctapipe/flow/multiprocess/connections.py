from time import sleep
import zmq


class Connections():
    """
    implements ZMQ connections between process for PRODUCER and STAGER and CONSUMER
    """

    def __init__(self, main_connection_name, connections=None):
        """
        Parameters
        ----------
        connections : dict
        main_connection_name : str
            Default next step name. Used to send data when destination is not provided
        """
        self.connections = connections or {}
        self.sockets = dict()
        self.context = zmq.Context()
        self.main_out_socket = None
        self.main_connection_name = main_connection_name


    def close_connections(self):
        """
        Close all zmq socket connections
        """
        for sock in self.sockets.values():
            sock.close()

    def get_destination_msg_from_result(self, result):
        """
        If result is a tuple, check if last tuple elem is a valid next step name.
        If yes, return a destination defined to  the last tuple elem and send
        result without the destination
        If no return None as destination

        Parameters
        ----------
        result : any type
            value to send. If type(result) is tuple, it can contain next step name)
        Returns
        --------
        tulpe conaining two elements: msg and destination

        """
        if isinstance(result, tuple):
            # look is last tuple elem is a valid next step
            if result[-1] in self.connections.keys():
                destination = result[-1]
                if len(result[:-1]) == 1:
                    msg = result[:-1][0]
                else:
                    msg = result[:-1]
                return msg, destination
            else:
                return result, None
        else:
            return result, None

    def send_msg(self, msg, destination_step_name=None):
        """
        Send a message thanks to ZMQ

        Parameters
        ----------
        msg: a Pickle.dump message
        destination_step_name: str
            msg will be send to corresponding step
        """
        send = False
        if not destination_step_name:
            socket = self.main_out_socket
        else:
            socket = self.sockets[destination_step_name]
        while not send:
            socket.send_pyobj(msg)
            request = socket.recv()
            if request == b'OK':
                send = True
            else:
                sleep(0.1)

    def init_connections(self):
        """
        Initialise zmq sockets.
        Because this class is s Process, This method must be call in the run
         method to be hold by the correct process.
        """
        for name, connection in self.connections.items():
            self.sockets[name] = self.context.socket(zmq.REQ)
            try:
                self.sockets[name].connect('tcp://localhost:' + connection)
                if self.main_connection_name == name:
                    self.main_out_socket = self.sockets[name]
            except zmq.error.ZMQError as e:
                print(' {} : tcp://localhost:{}'
                      .format(e, connection))
                return False
        return True
