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
    section_name : str
            pipeline configuration section_name
    prev_step : PipeStep
            previous step in pipeline
    port_in : str
            port number to connect prev Router
    port_out : str
            port number to connect next Router
nb_thread: int
    Number of thread to instantiate for this step
Note: The firewall must be configure to accept input/output on theses port
'''

    def __init__(self, section_name, prev_step=None, port_in=None,
                 port_out=None, nb_thread=1):
        self.section_name = section_name
        self.port_in = port_in
        self.port_out = port_out
        self.prev_step = prev_step
        self.threads = list()
        self.nb_thread = nb_thread

    def __repr__(self):
        '''standard representation
        '''
        if self.prev_step is not None:
            return ('Name[ ' + str(self.section_name) + ' ], previous step[ '
                    + str(self.prev_step.section_name) + ' ], port in[ '
                    + str(self.port_in) + ' ], port out [ '
                    + str(self.port_out) + ' ]')
        return ('WARNING prev_step is None Name[ ' + str(self.section_name)
                + ' ], previous step[ ' + ' port in[ ' + str(self.port_in)
                + ' ], port out [ ' + str(self.port_out) + ' ]')


class PipelineError(Exception):

    def __init__(self, msg):
        '''Mentions that an exception occurred in the pipeline.
        '''
        self.msg = msg


class StepInfo():

    '''
    This class is used to send step or router  information to GUI to inform
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
    router_queues = list()
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
        # Configure steps' port out
        if self.configure_port_out(self.producer_steps,
                                   self.stager_steps) == False:
            self.log.info('No enough available ports for ZMQ')
            return False

        # Configure steps' port in
        self.configure_port_in(self.stager_steps, self.consumer_step)

        # import and init producers
        for producer_step in self.producer_steps:
            conf = self.producer_conf
            try:
                producer_zmq = self.instantiation(
                    producer_step.section_name, self.PRODUCER,
                    port_out=producer_step.port_out, config=conf)
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
        for consumer_step in self.consumer_step:
            # each consumer need a router to connect it to prev stage
            router_port_out = consumer_step.section_name + '_router'
            name = consumer_step.section_name + '_' + 'router'
            router_names[name] = name
            sock_router_ports[name] = consumer_step.port_in
            socket_dealer_ports[name] = router_port_out
            conf = self.consumer_conf
            try:
                consumer_zmq = self.instantiation(consumer_step.section_name,
                                                  self.CONSUMER,
                                                  port_in=router_port_out,
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
            router_port_out = stager_step.section_name + '_router'
            name = stager_step.section_name + '_' + 'router'
            router_names[name] = name
            sock_router_ports[name] = stager_step.port_in
            socket_dealer_ports[name] = router_port_out

            for i in range(stager_step.nb_thread):
                conf = self.get_step_conf(stager_step.section_name)
                try:
                    stager_zmq = self.instantiation(
                        stager_step.section_name, self.STAGER,
                        port_in=router_port_out, port_out=stager_step.port_out,
                        name=stager_step.section_name +
                            '$$thread_number$$' + str(i),
                        config=conf)
                except PipelineError as e:
                    self.log.error(e)
                    return False
                if stager_zmq.init() == False:
                    self.log.error('stager_zmq init failed')
                    return False
                self.stagers.append(stager_zmq)
                stager_step.threads.append(stager_zmq)
        router = RouterQueue(sock_router_ports,
                             socket_dealer_ports,
                             step_names=router_names,
                             gui_address=self.gui_address)
        if router.init() == False:
            return False
        self.router_queues.append(router)
        # Define order in which step have to be stop
        self.def_thread_order()
        # self.log.info pipeline configuration
        self.display_conf()
        return True

    def generate_steps(self):
        ''' Generate pipeline steps from configuration'''
        self.producer_steps = self.get_pipe_steps(self.PRODUCER)
        self.stager_steps = self.get_pipe_steps(self.STAGER)
        self.consumer_step = self.get_pipe_steps(self.CONSUMER)
        if not self.producer_steps:
            self.log.error("No producer in configuration")
            return False
        if not self.stager_steps:
            self.log.error("No stager inb configuration")
            return False
        if not self.consumer_step:
            self.log.error("No consumer inb configuration")
            return False

        # Now that all steps exists, set previous step
        for step in self.consumer_step + self.stager_steps:
            prev_section_name = self.get_prev_step_section_name(
                step.section_name)
            if prev_section_name is not None:
                prev_step = self.get_step_by_section_name(prev_section_name)
                step.prev_step = prev_step
            else:
                return False
        return True

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
            producer_step.port_out = producer_step.section_name

            if producer_step.port_out is None:
                return False
        for stager_step in stager_steps:
            stager_step.port_out = stager_step.section_name
            if stager_step.port_out is None:
                return False
        return True

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
                stager_step.section_name)
        for consumer_step in consumer_steps:
            consumer_step.port_in = self.get_prev_step_port_out(
                consumer_step.section_name)

    def instantiation(
            self, section_name, stage_type, port_in=None,
            port_out=None, config=None, name=''):
        '''
        Instantiate on Pytohn object from name found in configuration
        Parameters
        ----------
        section_name : str
                section name in configuration file
        stage_type	: str
        port_in : str
                step port in
        port_out: str
                step port out
        name : str
                stage name
        '''
        stage = self.get_step_conf(section_name)
        module = stage['module']
        class_name = stage['class']
        obj = dynamic_class_from_module(class_name, module, self)

        if obj is None:
            raise PipelineError('Cannot create instance of ' + section_name)
        obj.section_name = section_name
        if stage_type == self.STAGER:
            thread = StagerZmq(
                obj, port_in, port_out, name=name, gui_address=self.gui_address)
        elif stage_type == self.PRODUCER:
            thread = ProducerZmq(
                obj, port_out, 'producer', gui_address=self.gui_address)
        elif stage_type == self.CONSUMER:
            thread = ConsumerZMQ(
                obj, port_in,
                'consumer', parent=self,
                gui_address=self.gui_address)
        else:
            raise PipelineError(
                'Cannot create instance of', section_name, '. Type',
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
        List of section name filter by specific role (PRODUCER,STAGER,CONSUMER)
        '''
        result = list()
        # Create producer step
        try:
            if role == self.PRODUCER:
                prod_step = PipeStep(self.producer_conf['name'])
                prod_step.type = self.PRODUCER
                result.append(prod_step)
            elif role == self.STAGER:
                # Create stagers steps
                for stage_conf in self.stagers_conf:
                    nb_thread = int(stage_conf['nb_thread'])
                    stage_step = PipeStep(
                        stage_conf['name'], nb_thread=nb_thread)
                    stage_step.type = self.STAGER
                    result.append(stage_step)
            elif role == self.CONSUMER:
                # Create consumer step
                cons_step = PipeStep(self.consumer_conf['name'])
                cons_step.type = self.CONSUMER
                result.append(cons_step)
            return result
        except KeyError as e:
            return result

    def get_prev_step_section_name(self, section):
        '''
        Parameters:
        -----------
        section_name : str
                section name of a  pipeline step
        Returns:
        --------
        section_name of previons step
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

    def get_prev_step_port_out(self, section):
        '''
        return port out of prev stage
        Parameters:
        -----------
        section_name : str
                section name of a  pipeline step
        Returns:
        port_out of previons step
        '''
        prev_section = self.get_prev_step_section_name(section)
        if prev_section is not None:
            if self.producer_steps is not None:
                for producer_step in self.producer_steps:
                    if producer_step.section_name == prev_section:
                        return producer_step.port_out
            if self.stager_steps is not None:
                for stager_step in self.stager_steps:
                    if stager_step.section_name == prev_section:
                        return stager_step.port_out
        return None

    def def_thread_order(self):
        ''' Define order in which step have to be stop.
        Fill self.step_threads
        '''
        chaine = list()
        for consumer in self.consumer_step:
            self.router_thread = self.router_queues[0]
            prev = consumer.prev_step
            while prev is not None:
                stages = list()
                for t in prev.threads:
                    self.step_threads.append(t)
                    stages.append(StepInfo(t))
                prev = prev.prev_step

    def def_step_for_gui(self):
        ''' Create a list (self.levels_for_gui) containing StepInfo instances
         representing pipeline configuration and Threads activity
        Fill self.step_threads
        Returns: Actual time
        '''
        self.levels_for_gui = list()
        consumer = self.consumer_step[0]
        self.levels_for_gui.append([StepInfo(self.consumer)])
        self.levels_for_gui.append(
            [StepInfo(self.router_thread, name=consumer.section_name +
             '_router')])
        prev = consumer.prev_step
        while prev is not None:
            stages = list()
            for t in prev.threads:
                stages.append(StepInfo(t))
            if stages:
                self.levels_for_gui.append(stages)
                self.levels_for_gui.append(
                    [StepInfo(self.router_thread, name=prev.section_name +
                     '_router')])
            prev = prev.prev_step
        self.levels_for_gui.append([StepInfo(self.producer)])
        self.levels_for_gui = list(reversed(self.levels_for_gui))
        return time.clock()

    def display_conf(self):
        ''' self.log.info pipeline configuration
        '''
        chaine = list()
        for consumer in self.consumer_step:
            chaine.append('    \t\t' + consumer.section_name)
            prev = consumer.prev_step
            while prev != None:
                chaine.append('    \t\t' + str(prev.section_name))
                prev = prev.prev_step
        chaine.reverse()
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

    def get_step_by_section_name(self, section_name):
        ''' Find a PipeStep in self.producer_steps or  self.stager_steps or
        self.consumer_step
        Return: PipeStep if found, otherwise None
        '''
        for step in (self.producer_steps
            + self.stager_steps
            + self.consumer_step):
            if step.section_name == section_name:
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
        for router in self.router_queues:
            router.start()
        for stage in self.stagers:
            stage.start()
        self.producer.start()
        # Wait that all producers end of run method
        self.wait_and_send_levels(self.producer, conf_time)
        # Now send stop to thread and wait they join(when their queue will be
        # empty)
        for worker in reversed(self.step_threads):
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
