# Licensed under a 3-clause BSD style license - see LICENSE.rst
'''a parallelization system. It executes ctapipe algorithms in a multithread
environment.
It is based on ZeroMQ library (http://zeromq.org) to pass messages between
threads. ZMQ library allows to stay away from class concurrency mechanisms
like mutexes, critical sections semaphores, while being thread safe.
User defined steps thanks to Python classes.
Passing data between steps is managed by the router.
If a step is executed by several threads, the router uses LRU pattern
(least recently used ) to choose the step that will receive next data.
The router also manage Queue for each step.
'''
from ctapipe.utils import dynamic_class_from_module
from ctapipe.core import Tool
from threading import Thread
from ctapipe.pipeline.zmqpipe.producer_zmq import ProducerZmq
from ctapipe.pipeline.zmqpipe.stager_zmq import StagerZmq
from ctapipe.pipeline.zmqpipe.consumer_zmq import ConsumerZMQ
from ctapipe.pipeline.zmqpipe.router_queue_zmq import RouterQueue
import sys
import os
import zmq
import time
import pickle
from ctapipe.core import Tool
from traitlets import (Integer, Float, List, Dict, Unicode)

__all__ = ['Pipeline', 'PipelineError']


class PipeStep():

    '''
PipeStep reprensents a Pipeline step. One or several threads can be attach
    to this step.
Parameters
----------
    name : str
            pipeline configuration name
    port_in : str
            port number to connect prev Router
    port_out : str
            port number to connect next Router
nb_thread: int
    Number of thread to instantiate for this step
Note: The firewall must be configure to accept input/output on theses port
'''

    def __init__(self, name,
                 #prev_step=None,
                 next_steps_name=list(),
                 port_in=None,
                 port_out=None, nb_thread=1):
        self.name = name
        self.port_in = port_in
        self.port_out = port_out
        #self.prev_step = prev_step
        self.next_steps_name = next_steps_name
        self.threads = list()
        self.nb_thread = nb_thread

    def __repr__(self):
        '''standard representation
        '''
        return ('Name[ ' + str(self.name)
                + '], next_steps_name[' + str(self.next_steps_name)
                + '], port in[ ' + str(self.port_in)
                + '], port out [ ' + str(self.port_out) + ' ]')


class PipelineError(Exception):

    def __init__(self, msg):
        '''Mentions that an exception occurred in the pipeline.
        '''
        self.msg = msg


class GUIStepInfo():

    '''
    This class is used to send step or router information to GUI to inform
    about changes.
    We can not used directly thread class (i.e. ProducerZmq) nor PipeStep
    (because it contains a thread as member) because pickle cannot dumps class
    containing thread.
    Parameters
    ----------
    step_zmq : thread(ProducerZmq or StagerZmq or ConsumerZMQ or RouterQueue)
            copy thread informations to this structure
    '''
    PRODUCER = 'PRODUCER'
    STAGER = 'STAGER'
    CONSUMER = 'CONSUMER'
    ROUTER = 'ROUTER'

    def __init__(self, step_zmq, name=None):
        self.running = False
        self.nb_job_done = 0
        if step_zmq is not None:
            if name is None:
                self.name = step_zmq.name
            else:
                self.name = name
            if isinstance(step_zmq, ProducerZmq):
                self.type = self.PRODUCER
                self.nb_job_done = step_zmq.nb_job_done
            if isinstance(step_zmq, StagerZmq):
                self.type = self.STAGER
                self.running = step_zmq.running
                self.nb_job_done = step_zmq.nb_job_done
            if isinstance(step_zmq, ConsumerZMQ):
                self.nb_job_done = step_zmq.nb_job_done
                self.type = self.CONSUMER
            if isinstance(step_zmq, RouterQueue):
                self.type = self.ROUTER
        else:
            self.name = 'ND'
            self.type = 'ND'
        self.queue_size = 0

    def __repr__(self):
        '''standard representation
        '''
        return (self.name + ', ' + self.type + ', ' + str(self.running) +
                ', ' + str(self.queue_size))


class Pipeline(Tool):

    '''
    Represents a staged pattern of stage. Each stage in the pipeline
    is one or several threads containing a coroutine that receives messages
    from the previous stage and	yields messages to be sent to the next stage
    thanks to RouterQueue instances	'''

    description = 'run stages in multithread pipeline'
    gui_address = Unicode('localhost:5565', help='GUI adress and port').tag(
        config=True, allow_none=True)
    producer_conf = Dict(
        help='producer description: name , module, class',
                                            allow_none=False).tag(config=True)
    stagers_conf = List(
        help='stagers list description in a set order:',
         allow_none=False).tag(config=True)
    consumer_conf = Dict(
        default_value={'name': 'CONSUMER', 'class': 'Producer',
                       'module': 'producer',  'prev': 'STAGE1'},
        help='producer description: name , module, class',
                allow_none=False).tag(config=True)
    aliases = Dict({'gui_address': 'Pipeline.gui_address'})
    examples = ('protm%> ctapipe-pipeline \
    --config=examples/brainstorm/pipeline/pipeline_py/example.json')
    # TO DO: register steps class for configuration
    # classes = List()

    PRODUCER = 'PRODUCER'
    STAGER = 'STAGER'
    CONSUMER = 'CONSUMER'
    ROUTER = 'ROUTER'
    producer = None
    consumer = None
    stagers = list()
    router = None
    producer_step = None
    stager_steps = None
    consumer_step = None
    step_threads = list()
    router_thread = None
    context = zmq.Context().instance()
    socket_pub = context.socket(zmq.PUB)
    levels_for_gui = list()

    def setup(self):
        if self.init() == False:
            self.log.error('Could not initialise pipeline')
            sys.exit()

    def init(self):
        '''
        Create producers, stagers and consumers instance according to
         configuration
        Returns:
        --------
        bool : True if pipeline is correctly setup and all producer,stager
         and consumer initialised Otherwise False
        '''
        # Verify configuration instance
        if not os.path.isfile(self.config_file):
            self.log.error('Could not open pipeline config_file {}'
                           .format(self.config_file))
            return False

        # Get port for GUI
        if self.gui_address is not None:
            try:
                self.socket_pub.connect('tcp://' + self.gui_address)
            except zmq.error.ZMQError as e:
                self.log.info(str(e) + 'tcp://' + self.gui_address)
                return False
        # Gererate steps(producers, stagers and consumers) from configuration
        if self.generate_steps() == False:
            self.log.error("Error during steps generation")
            return False

        self.configure_ports()
        # import and init producers

        conf = self.producer_conf
        try:
            producer_zmq = self.instantiation(
                self.producer_step.name, self.PRODUCER,
                port_out=self.producer_step.port_out, config=conf)
        except PipelineError as e:
            self.log.error(e)
            return False
        if producer_zmq.init() == False:
            self.log.error('producer_zmq init failed')
            return False
        self.producer = producer_zmq

        # ROUTER
        sock_router_ports = dict()
        socket_dealer_ports = dict()
        router_names = dict()

        # import and init consumers
        # each consumer need a router to connect it to prev stage
        name = self.consumer_step.name + '_' + 'router'
        #router_names[name] = [self.consumer_step.port_in,self.consumer_step.name]
        router_names[name] = [self.consumer_step.name+'_in',self.consumer_step.name+'_out']
        #sock_router_ports[name] = consumer_step.port_in
        #socket_dealer_ports[name] = router_port_out
        conf = self.consumer_conf
        try:
            consumer_zmq = self.instantiation(self.consumer_step.name,
                                      self.CONSUMER,
                                      port_in=self.consumer_step.port_in,
                                      config=conf)
        except PipelineError as e:
            self.log.error(e)
            return False
        if consumer_zmq.init() == False:
            self.log.error('consumer_zmq init failed')
            return False
        self.consumer = consumer_zmq

        # import and init stagers
        for stager_step in self.stager_steps:
            # each stage need a router to connect it to prev stage
            name = stager_step.name + '_' + 'router'
            router_names[name] = [stager_step.name+'_in',stager_step.name+'_out']
            #sock_router_ports[name] = stager_step.port_in
            #socket_dealer_ports[name] = router_port_out

            for i in range(stager_step.nb_thread):
                conf = self.get_step_conf(stager_step.name)
                try:
                    stager_zmq = self.instantiation(
                        stager_step.name ,
                        self.STAGER,
                        thread_name = stager_step.name
                            +'$$thread_number$$'
                            + str(i),
                        port_in=stager_step.port_in, port_out=stager_step.port_out,
                        config=conf)
                except PipelineError as e:
                    self.log.error(e)
                    return False
                if stager_zmq.init() == False:
                    self.log.error('stager_zmq init failed')
                    return False
                self.stagers.append(stager_zmq)
                stager_step.threads.append(stager_zmq)
        router = RouterQueue(connexions=router_names,
                             gui_address=self.gui_address)
        if router.init() == False:
            return False
        self.router=router
        # Define order in which step have to be stop
        self.def_thread_order()
        # self.log.info pipeline configuration
        self.display_conf()
        return True

    def generate_steps(self):
        ''' Generate pipeline steps from configuration'''
        self.producer_step = self.get_pipe_steps(self.PRODUCER)
        self.stager_steps = self.get_pipe_steps(self.STAGER)
        self.consumer_step = self.get_pipe_steps(self.CONSUMER)
        if not self.producer_step:
            self.log.error("No producer in configuration")
            return False
        if not self.stager_steps:
            self.log.error("No stager inb configuration")
            return False
        if not self.consumer_step:
            self.log.error("No consumer inb configuration")
            return False

        # Now that all steps exists, set previous step
        """
        for step in self.consumer_step + self.stager_steps:
            prev_name = self.get_prev_step_name(step.name)
            if prev_name is not None:
                prev_step = self.get_step_by_name(prev_name)
                step.prev_step = prev_step
            else:
                return False
        """
        return True

    def configure_ports(self):

        self.producer_step.port_out = self.producer_step.next_steps_name[0]+'_in'
        self.log.debug('---> configure_ports producer {}, port_out {}'
            .format(self.producer_step.name,self.producer_step.port_out))
        for next_step in self.producer_step.next_steps_name:
            step = self.get_step_by_name(next_step)
            if step:
                step.port_in = step.name+'_out'
                self.log.debug('---> configure_ports step {}, port_in {}'.format(step.name,step.port_in))

        for stage in self.stager_steps:
            stage.port_out = stage.next_steps_name[0]+'_in'
            self.log.debug('---> configure_ports step {}, port_out {}'.format(stage.name,stage.port_out))
            for next_step in stage.next_steps_name:
                step = self.get_step_by_name(next_step)
                if step:
                    step.port_in = stage.next_steps_name[0]+'_out'
                    self.log.debug('---> configure_ports step {}, port_in {}'.format(step.name,step.port_in))


    def get_step_by_name(self,name):
        for step in (self.stager_steps
        + [self.consumer_step,self.producer_step]):
            if step.name == name:
                return step
        return None
    """
    def configure_port_out(self, producer_steps, stager_steps):
        '''
        Configure port_out from pipeline's ports list for producers and stagers
        returns:
        --------
        True if every ports is configured
        False if no more ports are available
        Parameters
        ----------
        producer_steps : list of producer step
        stager_steps   : list of stager step
        '''
        for producer_step in producer_steps:
            producer_step.port_out = producer_step.name

            if producer_step.port_out is None:
                return False
        for stager_step in stager_steps:
            stager_step.port_out = stager_step.name
            if stager_step.port_out is None:
                return False
        return True
    """
    """
    def configure_port_in(self, stager_steps, consumer_steps):
        '''
        Configure port_in from pipeline's ports list for stagers and consumers
        Parameters
        ----------
        consumer_steps : list of consumer step
        stager_steps   : list of stager step
        '''

        for stager_step in stager_steps:
            stager_step.port_in = self.get_prev_step_port_out(
                stager_step.name)

        for consumer_step in consumer_steps:
            consumer_step.port_in = self.get_prev_step_port_out(
                consumer_step.name)
        """

    def instantiation(
            self, name, stage_type, thread_name=None, port_in=None,
            port_out=None, config=None):
        '''
        Instantiate on Pytohn object from name found in configuration
        Parameters
        ----------
        name : str
                stage name
        stage_type	: str
        port_in : str
                step port in
        port_out: str
                step port out
        '''
        stage = self.get_step_conf(name)
        module = stage['module']
        class_name = stage['class']
        obj = dynamic_class_from_module(class_name, module, self)

        if obj is None:
            raise PipelineError('Cannot create instance of ' + name)
        obj.name = name
        if stage_type == self.STAGER:
            thread = StagerZmq(
                obj, port_in, port_out, thread_name, gui_address=self.gui_address)
        elif stage_type == self.PRODUCER:
            thread = ProducerZmq(
                obj,port_out, name, gui_address=self.gui_address)
        elif stage_type == self.CONSUMER:
            thread = ConsumerZMQ(
                obj,port_in,
                name, parent=self,
                gui_address=self.gui_address)
        else:
            raise PipelineError(
                'Cannot create instance of', name, '. Type',
                 stage_type, 'does not exist.')
        # set coroutine socket to it's stager or producer socket .
        return thread

    def get_pipe_steps(self, role):
        '''
        Create a list of pipeline step corresponding to configuration and role
        Parameters
        ----------
        role: str
                filter with role for step to be add in result list
                Accepted values: self.PRODUCER - self.STAGER  - self.CONSUMER
        Returns:
        --------
        PRODUCER,CONSUMER: a section name filter by specific role (PRODUCER,CONSUMER)
        STAGER: List of section name filter by specific role

        '''

        # Create producer step
        try:
            if role == self.PRODUCER:
                prod_step = PipeStep(self.producer_conf['name'])
                prod_step.type = self.PRODUCER
                prod_step.next_steps_name = self.producer_conf['next_steps'].split(',')
                return prod_step
            elif role == self.STAGER:
                # Create stagers steps
                result = list()
                for stage_conf in self.stagers_conf:
                    nb_thread = int(stage_conf['nb_thread'])
                    next_steps_name = stage_conf['next_steps'].split(',')
                    stage_step = PipeStep(
                        stage_conf['name'],
                        next_steps_name=next_steps_name,nb_thread=nb_thread)
                    stage_step.type = self.STAGER
                    result.append(stage_step)
                return result
            elif role == self.CONSUMER:
                # Create consumer step
                cons_step = PipeStep(self.consumer_conf['name'])
                cons_step.type = self.CONSUMER
                return  cons_step
            return result
        except KeyError as e:
            return None

    """
    def get_prev_step_name(self, section):
        '''
        Parameters:
        -----------
        name : str
                section name of a  pipeline step
        Returns:
        --------
        name of previons step
        '''
        # If section correspond to consumer name, returl last stage
        if self.consumer_conf['name'] == section:
            return self.stagers_conf[-1]['name']
        indice = self.get_stager_indice(section)
        if indice != -1:
            if indice == 0:  # This is the first dtage, so prev is Producer
                return self.producer_conf['name']
            return self.stagers_conf[indice - 1]['name']
        return None
    """
    """
    def get_prev_step_port_out(self, section):
        '''
        return port out of prev stage
        Parameters:
        -----------
        name : str
                section name of a  pipeline step
        Returns:
        port_out of previons step
        '''
        prev_section = self.get_prev_step_name(section)
        if prev_section is not None:
            if self.producer_steps is not None:
                for producer_step in self.producer_steps:
                    if producer_step.name == prev_section:
                        return producer_step.port_out
            if self.stager_steps is not None:
                for stager_step in self.stager_steps:
                    if stager_step.name == prev_section:
                        return stager_step.port_out
        return None
    """
    def def_thread_order(self):
        ''' Define order in which step have to be stop.
        Fill self.step_threads
        '''
        """
        for consumer in self.consumer_step:
            self.router_thread = self.router_queues[0]
            prev = consumer.prev_step
            while prev is not None:
                stages = list()
                for t in prev.threads:
                    self.step_threads.append(t)
                    stages.append(GUIStepInfo(t))
                prev = prev.prev_step
        """
        self.router_thread = self.router

        next_steps_name =  self.producer_step.next_steps_name
        for next_step in  next_steps_name:
            for step_name in next_steps_name:
                next_step = self.get_step_by_name(step_name)
                for t in next_step.threads:
                    self.step_threads.append(t)
                    next_steps_name = next_step.next_steps_name
                    while next_steps_name:
                        for step_name in next_steps_name:
                            next_step = self.get_step_by_name(step_name)
                            for t in next_step.threads:
                                self.step_threads.append(t)
                        next_steps_name = next_step.next_steps_name

        self.log.debug("def_thread_order-> self.step_threads {}".format(self.step_threads))



    def def_step_for_gui(self):
        ''' Create a list (self.levels_for_gui) containing GUIStepInfo instances
         representing pipeline configuration and Threads activity
        Fill self.step_threads
        Returns: Actual time
        '''
        self.levels_for_gui = list()
        self.levels_for_gui.append([GUIStepInfo(self.consumer)])
        self.levels_for_gui.append(
            [GUIStepInfo(self.router_thread, name=self.consumer_step.name +
             '_router')])
        """
        prev = self.consumer_step.prev_step
        while prev is not None:
            stages = list()
            for t in prev.threads:
                stages.append(GUIStepInfo(t))
            if stages:
                self.levels_for_gui.append(stages)
                self.levels_for_gui.append(
                    [GUIStepInfo(self.router_thread, name=prev.name +
                     '_router')])
            prev = prev.prev_step
        """
        next_steps_name = self.producer_step.next_steps_name
        while next_steps_name:
            for next_step_name in next_steps_name:
                next_step = self.get_step_by_name(next_step_name)
                stages = list()
                for t in next_step.threads:
                    stages.append(GUIStepInfo(t))
                if stages:
                    self.levels_for_gui.append(stages)
                    self.levels_for_gui.append(
                        [GUIStepInfo(self.router_thread, name=next_step.name +
                         '_router')])

            next_steps_name = next_step.next_steps_name
        self.levels_for_gui.append([GUIStepInfo(self.producer)])
        self.levels_for_gui = list(reversed(self.levels_for_gui))
        return time.clock()

    def display_conf(self):
        ''' self.log.info pipeline configuration
        '''
        chaine = list()
        chaine.append('    \t\t' + self.producer_step.name)
        next_steps_name = self.producer_step.next_steps_name
        while next_steps_name:
            for step_name in next_steps_name:
                chaine.append('    \t\t' + str(step_name))
                next_step = self.get_step_by_name(step_name)
                next_steps_name = next_step.next_steps_name

        self.log.info(' ------------- Pipeline configuration ----------- ')
        self.log.info(' \t\t\t\t\t\t')
        for item in chaine[:-1]:
            self.log.info(item)
            self.log.info(' \t\t\t  |')
            self.log.info(' \t\t\t  |')
            self.log.info(' \t\t\t  V')
        self.log.info(chaine[-1])
        self.log.info(' \t\t\t\t\t\t')
        self.log.info(
            ' ---------- End Pipeline configuration ----------- \n\n ')


    def get_step_by_name(self, name):
        ''' Find a PipeStep in self.producer_step or  self.stager_steps or
        self.consumer_step
        Return: PipeStep if found, otherwise None
        '''
        for step in (self.stager_steps+[self.producer_step,self.consumer_step]):
            if step.name == name:
                return step
        return None

    def start(self):
        ''' Start all pipeline threads.
        Regularly inform GUI of pipeline configuration in case of a new GUI
        instance was lunch
        Stop all thread in set order
        '''
        # send pipeline cofiguration to an optinal GUI instance
        conf_time = self.def_step_for_gui()
        self.socket_pub.send_multipart(
            [b'GUI_GRAPH', pickle.dumps([conf_time, self.levels_for_gui])])
        # Start all Threads
        self.consumer.start()

        self.router.start()
        for stage in self.stagers:
            stage.start()
        self.producer.start()
        # Wait that all producers end of run method
        self.wait_and_send_levels(self.producer, conf_time)
        # Now send stop to thread and wait they join(when their queue will be
        # empty)
        #for worker in reversed(self.step_threads):
        for worker in self.step_threads:
            if worker is not None:
                while not self.router_thread.isQueueEmpty(worker.name):
                    self.socket_pub.send_multipart(
                        [b'GUI_GRAPH', pickle.dumps([conf_time,
                         self.levels_for_gui])])
                    time.sleep(1)
                self.wait_and_send_levels(worker, conf_time)
        self.wait_and_send_levels(self.router_thread, conf_time)
        self.wait_and_send_levels(self.consumer, conf_time)
        self.socket_pub.close()
        self.context.destroy()
        # self.context.term()

    def finish(self):
        self.log.info('===== Pipeline END ======')

    def wait_and_send_levels(self, thread_to_wait, conf_time):
        '''
        Wait for a thread to join and regularly send pipeline state to GUI
        Parameters:
        -----------
        thread_to_wait : thread
                thread to join
        conf_time : str
                represents time at which configuration has been built
        '''
        thread_to_wait.finish()
        self.def_step_for_gui()
        while True:
            thread_to_wait.join(timeout=1.0)
            self.socket_pub.send_multipart(
                [b'GUI_GRAPH', pickle.dumps([conf_time, self.levels_for_gui])])
            if not thread_to_wait.is_alive():
                break

    def get_step_conf(self, name):
        '''
        Search step by its name in self.stage_conf list,
        self.producer_conf and self.consumer_conf
        Parameters:
        -----------
        name : str
                stage name
        Returns:
        --------
        Step name matching instance, or None is not found
        '''
        if self.producer_conf['name'] == name:
            return self.producer_conf
        if self.consumer_conf['name'] == name:
            return self.consumer_conf
        for step in self.stagers_conf:
            if step['name'] == name:
                return step
        return None

    def get_stager_indice(self, name):
        '''
        Search step by its name in self.stage_conf list
        Parameters:
        -----------
        name : str
                stage name
        Returns:
        --------
        indice in list, -1 if not found
        '''
        for index, step in enumerate(self.stagers_conf):
            if step['name'] == name:
                return index
        return -1


def main():
    tool = Pipeline()
    tool.run()

if __name__ == 'main':
    main()
