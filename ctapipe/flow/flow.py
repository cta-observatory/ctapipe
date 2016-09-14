import zmq
from sys import exit
from os import path
from time import time
from time import sleep
from pickle import dumps
from traitlets import Bool
from traitlets import List
from traitlets import Dict
from traitlets import Unicode
from ctapipe.flow.multiprocessus.producer_zmq import ProducerZmq
from ctapipe.flow.multiprocessus.stager_zmq import StagerZmq
from ctapipe.flow.multiprocessus.consumer_zmq import ConsumerZMQ
from ctapipe.flow.multiprocessus.router_queue_zmq import RouterQueue
from ctapipe.flow.sequential.producer_sequential import ProducerSequential
from ctapipe.flow.sequential.stager_sequential import StagerSequential
from ctapipe.flow.sequential.consumer_sequential import ConsumerSequential
from ctapipe.flow.stager_rep import StagerRep
from ctapipe.utils import dynamic_class_from_module
from ctapipe.utils.dynamic_class import DynamicClassError
from ctapipe.core import Tool

__all__ = ['Flow', 'FlowError']

class PipeStep():

    '''
PipeStep reprensents a Flow step. One or several processus can be attach
    to this step.
Parameters
----------
    name : str
            Flow based framework configuration name
    next_steps_name: list(str)
    port_in : str
            port number to connect prev Router
    connexions : dict {'str : 'str'}
            key: connexion name(step name) , value port name
    main_connexion_name: str
            First step in next_steps configuration
    nb_processus : int
            mumber of processus to instantiate for this step
    level : step level in Flow based framework. Producer is level 0.
            Used to start/stop processus in correct order
    queue_limit: int
            Maximum number of element the router can queue
'''
    def __init__(self, name,
                 next_steps_name=list(),
                 port_in=None,
                 main_connexion_name=None,
                 nb_processus=1, level=0,
                 queue_limit = 0):

        self.name = name
        self.port_in = port_in
        self.next_steps_name = next_steps_name
        self.nb_processus = nb_processus
        self.level = level
        self.connexions = dict()
        self.processus = list()
        self.main_connexion_name = main_connexion_name
        self.queue_limit = queue_limit
        self.order_defined = False
        self.coroutine = None

    def __repr__(self):
        '''standard representation
        '''
        return ('Name[ ' + str(self.name)
                + '], next_steps_name[' + str(self.next_steps_name)
                + '], port in[ ' + str(self.port_in)
                + '], main connexion name  [ ' + str(self.main_connexion_name) + ' ]'
                + '], port in[ ' + str(self.port_in)
                + '], nb processus[ ' + str(self.nb_processus)
                + '], level[ ' + str(self.level)
                + '], queue_limit[ ' + str(self.queue_limit) + ']')

class FlowError(Exception):
    def __init__(self, msg):
        '''Mentions that an exception occurred in the Flow based framework.
        '''
        self.msg = msg

class Flow(Tool):
    '''
    A Flow-based framework. It executes steps in a sequential or
    multiprocessus environment.
    User defined steps thanks to Python classes, and configuration in a json file
    The multiprocessus mode is based on ZeroMQ library (http://zeromq.org) to
    pass messages between processus. ZMQ library allows to stay away from class
    concurrency mechanisms like mutexes, critical sections semaphores,
    while being processus safe. Passing data between steps is managed by the router.
    If a step is executed by several processus, the router uses LRU pattern
    (least recently used ) to choose the step that will receive next data.
    The router also manage Queue for each step.
    '''
    description = 'run stages in multiprocessus Flow based framework'
    gui = Bool(False, help='send status to GUI').tag(
        config=True, allow_none=True)
    gui_address = Unicode('localhost:5565', help='GUI adress and port').tag(
        config=True, allow_none=True)
    mode = Unicode('sequential', help='Flow mode [sequential | multiprocessus]').tag(
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
    ports_list = list(range(5555,5600,1))
    zmq_ports = List(ports_list, help='ZMQ ports').tag(
        config=True, allow_none=True)
    aliases = Dict({'gui_address': 'Flow.gui_address',
                    'mode':'Flow.mode','gui': 'Flow.gui'})
    examples = ('prompt%> ctapipe-flow \
    --config=examples/brainstorm/flow/flow_py/example.json')

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
    step_processus = list()
    router_processus = None
    ports = dict()

    def setup(self):
        if self.init() == False:
            self.log.error('Could not initialise Flow based framework')
            exit()

    def init(self):
        '''
        Create producers, stagers and consumers instance according to
         configuration
        Returns:
        --------
        bool : True if Flow based framework is correctly setup and all producer,stager
         and consumer initialised Otherwise False
        '''
        # Verify configuration instance
        if not path.isfile(self.config_file):
            self.log.error('Could not open Flow based framework config_file {}'
                           .format(self.config_file))
            return False
        if not self.generate_steps():
            self.log.error("Error during steps generation")
            return False
        if self.gui :
            self.context = zmq.Context()
            self.socket_pub = self.context.socket(zmq.PUB)
            if not self.connect_gui():  return False
        if self.mode == 'sequential':
            return self.init_sequential()
        elif self.mode == 'multiprocessus':
            return self.init_multiprocessus()
        else:
            self.log.error("{} is not a valid mode for Flow based framework".format(self.mode))

    def init_multiprocessus(self):
        """
        Initialise Flow for multiprocessus mode
        Returns:
        ========
        True if every initialisation are correct
        Otherwise False
        """
        if not self.configure_ports() : return False
        if not self.configure_producer() : return False
        router_names =  self.add_consumer_to_router()
        if not self.configure_consumer(): return False
        if not self.configure_stagers(router_names) : return False
        gui_address = None
        if self.gui:
            gui_address = self.gui_address
        self.router = RouterQueue(connexions=router_names,
                             gui_address=gui_address)
        for step in self.stager_steps:
            for t in step.processus:
                self.step_processus.append(t)
        self.display_conf()
        return True

    def init_sequential(self):
        """
        Initialise Flow for sequential mode
        Returns:
        ========
        True if every initialisation are correct
        Otherwise False
        """
        self.configure_ports()
        self.sequential_instances = dict()
        # set coroutines
        #producer
        conf = self.get_step_conf(self.producer_step.name)
        module = conf['module']
        class_name = conf['class']
        try:
            coroutine = dynamic_class_from_module(class_name, module, self)
        except DynamicClassError as e:
            self.log.error('{}'.format(e))
            return False

        self.producer = ProducerSequential(coroutine, name=self.producer_step.name,
                                  connexions=self.producer_step.connexions,
                                  main_connexion_name = self.producer_step.main_connexion_name)
        self.producer.init()
        self.producer_step.processus.append(self.producer)
        self.sequential_instances[self.producer_step.name] = self.producer
        #stages
        for step in (self.stager_steps ):
            conf = self.get_step_conf(step.name)
            module = conf['module']
            class_name = conf['class']
            try:
                coroutine = dynamic_class_from_module(class_name, module, self)
            except DynamicClassError as e:
                self.log.error('{}'.format(e))
                return False

            stage = StagerSequential(coroutine,name = step.name, connexions=step.connexions,
                                     main_connexion_name=step.main_connexion_name)
            step.processus.append(stage)
            self.sequential_instances[step.name] = stage
            self.stagers.append(stage)
            stage.init()
        #consumer
        conf = self.get_step_conf(self.consumer_step.name)
        module = conf['module']
        class_name = conf['class']
        try:
            coroutine = dynamic_class_from_module(class_name, module, self)
        except DynamicClassError as e:
            self.log.error('{}'.format(e))
            return False
        self.consumer = ConsumerSequential(coroutine, name =  conf['name'])
        self.consumer_step.processus.append(self.consumer)
        self.consumer.init()
        self.sequential_instances[self.consumer_step.name] = self.consumer
        self.display_conf()
        return True

    def configure_stagers(self,router_names):
        """ Creates Processus with users's coroutines for all stages
        Parameters:
        ===========
        router_names: List
            List to fill with routers name
        Returns:
        ========
        True if every instantialtion are correct
        Otherwise False
        """
        #STAGERS
        for stager_step in self.stager_steps:
            # each stage need a router to connect it to prev stages
            name = stager_step.name + '_' + 'router'
            router_names[name] = [self.ports[stager_step.name+'_in'],
                                  self.ports[stager_step.name+'_out'],
                                  stager_step.queue_limit]

            for i in range(stager_step.nb_processus):
                conf = self.get_step_conf(stager_step.name)
                try:
                    stager_zmq = self.instantiation(
                        stager_step.name ,
                        self.STAGER,
                        processus_name = stager_step.name
                            +'$$processus_number$$'
                            + str(i),
                        port_in=stager_step.port_in,
                        connexions = stager_step.connexions,
                        main_connexion_name = stager_step.main_connexion_name,
                        config=conf)
                except FlowError as e:
                    self.log.error(e)
                    return False
                self.stagers.append(stager_zmq)
                stager_step.processus.append(stager_zmq)
        return True


    def configure_consumer(self):
        """ Creates consumer Processus with users's coroutines
        Returns:
        ========
        True if every instantialtion are correct
        Otherwise False
        """
        try:
            consumer_zmq = self.instantiation(self.consumer_step.name,
                                      self.CONSUMER,
                                      port_in=self.consumer_step.port_in,
                                      config=self.consumer_conf)
        except FlowError as e:
            self.log.error(e)
            return False
        self.consumer = consumer_zmq
        return True

    def add_consumer_to_router(self):
        """ Create router_names dictionary and
        Add consumer router ports
        Returns:
        ========
        The new router_names dictionary
        """
        # ROUTER
        router_names = dict()
        # each stage need a router to connect it to prev stages
        name = self.consumer_step.name + '_' + 'router'
        router_names[name] = [self.ports[self.consumer_step.name+'_in'],
                              self.ports[self.consumer_step.name+'_out'],
                              self.consumer_step.queue_limit]
        return router_names

    def configure_producer(self):
        """ Creates producer Processus with users's coroutines
        Returns:
        ========
        True if every instantialtion are correct
        Otherwise False
        """
        #PRODUCER
        try:
            producer_zmq = self.instantiation(
                self.producer_step.name, self.PRODUCER,
                connexions = self.producer_step.connexions,
                main_connexion_name = self.producer_step.main_connexion_name,
                config= self.producer_conf)
        except FlowError as e:
            self.log.error(e)
            return False
        self.producer = producer_zmq
        return True

    def connect_gui(self):
        """ Connect ZMQ socket to send informations to GUI
        Returns:
        ========
        True if everything correct
        Otherwise False
        """
        # Get port for GUI
        if self.gui_address is not None:
            try:
                self.socket_pub.connect('tcp://' + self.gui_address)
            except zmq.error.ZMQError as e:
                self.log.info(str(e) + 'tcp://' + self.gui_address)
                return False
        return True


    def generate_steps(self):
        """ Generate Flow based framework steps from configuration
        Returns:
        ========
        True if everything correct
        Otherwise False
        """
        self.producer_step = self.get_pipe_steps(self.PRODUCER)
        self.stager_steps = self.get_pipe_steps(self.STAGER)
        self.consumer_step = self.get_pipe_steps(self.CONSUMER)
        if not self.producer_step:
            self.log.error("No producer in configuration")
            return False
        if not self.consumer_step:
            self.log.error("No consumer in configuration")
            return False
        return True

    def configure_ports(self):
        """
        Configures producer, stagers and consumer ZMQ ports
        Returns:
        ========
        True if everything correct
        Otherwise False
        """
        #configure connexions (zmq port) for producer (one per next step)
        try:
            for next_step_name in self.producer_step.next_steps_name:
                if not next_step_name+'_in' in self.ports:
                    self.ports[next_step_name+'_in'] = str(self.zmq_ports.pop())
                self.producer_step.connexions[next_step_name]=self.ports[next_step_name+'_in']
            self.producer_step.main_connexion_name = self.producer_step.next_steps_name[0]

            #configure port_in and connexions (zmq port)  for all stages (one per next step)
            for stage in self.stager_steps:
                if not stage.name+'_out' in self.ports:
                    self.ports[stage.name+'_out'] = str(self.zmq_ports.pop())
                stage.port_in = self.ports[stage.name+'_out']
                for next_step_name in stage.next_steps_name:
                    if not next_step_name+'_in' in self.ports:
                        self.ports[next_step_name+'_in'] = str(self.zmq_ports.pop())
                    stage.connexions[next_step_name]=self.ports[next_step_name+'_in']
                stage.main_connexion_name = stage.next_steps_name[0]

            #configure port-in  (zmq port) for consumer
            if not  self.consumer_step.name+'_out' in self.ports:
                self.ports[ self.consumer_step.name+'_out'] = str(self.zmq_ports.pop())
            self.consumer_step.port_in = self.ports[ self.consumer_step.name+'_out']
            return True
        except IndexError as e:
            self.log.error("Not enough ZMQ ports. Consider adding some port to configuration.")
        except Exception as e:
            self.log.error("Could not configure ZMQ ports. {}".format(e))
            return False

    def get_step_by_name(self, name):
        ''' Find a PipeStep in self.producer_step or  self.stager_steps or
        self.consumer_step
        Parameters:
        ===========
        name : str
            step name
        Return: PipeStep if found, otherwise None
        '''
        for step in (self.stager_steps+[self.producer_step,self.consumer_step]):
            if step.name == name:
                return step
        return None

    def instantiation(
            self, name, stage_type, processus_name=None,
            port_in=None, connexions=None, main_connexion_name=None, config=None):
        '''
        Instantiate on Python object from name found in configuration
        Parameters
        ----------
        name : str
                stage name
        stage_type	: str
        processus_name : str
        port_in : str
                step ZMQ port in
        connexions : dict
                key: StepName, value" connexion ZMQ ports
        main_connexion_name : str
            main ZMQ connection name. Connexion to use when user not precise
        '''
        stage = self.get_step_conf(name)
        module = stage['module']
        class_name = stage['class']
        obj = dynamic_class_from_module(class_name, module, self)
        if obj is None:
            raise FlowError('Cannot create instance of ' + name)
        obj.name = name
        if stage_type == self.STAGER:
            processus = StagerZmq(
                obj, port_in, processus_name,
                connexions=connexions,
                main_connexion_name = main_connexion_name)
        elif stage_type == self.PRODUCER:
            processus = ProducerZmq(
                obj, name, connexions=connexions,
                main_connexion_name = main_connexion_name)
        elif stage_type == self.CONSUMER:
            processus = ConsumerZMQ(
                obj,port_in,
                name)
        else:
            raise FlowError(
                'Cannot create instance of', name, '. Type',
                 stage_type, 'does not exist.')
        # set coroutine socket to it's stager or producer socket .
        return processus

    def get_pipe_steps(self, role):
        '''
        Create a list of Flow based framework steps from configuration and filter by role
        Parameters
        ----------
        role: str
                filter with role for step to be add in result list
                Accepted values: self.PRODUCER - self.STAGER  - self.CONSUMER
        Returns:
        --------
        PRODUCER,CONSUMER: a step name filter by specific role (PRODUCER,CONSUMER)
        STAGER: List of steps name filter by specific role
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
                    try:
                        nb_processus = int(stage_conf['nb_process'])
                    except Exception as e:
                        nb_processus = 1
                    next_steps_name = stage_conf['next_steps'].split(',')
                    try: queue_limit = stage_conf['queue_limit']
                    except Exception: queue_limit = -1
                    stage_step = PipeStep(  stage_conf['name'],
                        next_steps_name=next_steps_name,nb_processus=nb_processus,
                        queue_limit = queue_limit)
                    stage_step.type = self.STAGER
                    result.append(stage_step)
                return result
            elif role == self.CONSUMER:
                # Create consumer step
                try:  queue_limit = self.consumer_conf['queue_limit']
                except: queue_limit = -1
                cons_step = PipeStep(self.consumer_conf['name'],queue_limit = queue_limit)
                cons_step.type = self.CONSUMER
                return  cons_step
            return result
        except KeyError as e:
            return None

    def def_step_for_gui(self):
        ''' Create a list (levels_for_gui) containing all steps
        Returns: the created list and actual time
        '''
        levels_for_gui = list()

        levels_for_gui.append(StagerRep(self.producer_step.name,
                            self.producer_step.next_steps_name,
                            nb_job_done=self.producer.nb_job_done,
                            running=self.producer.running,
                            step_type=StagerRep.PRODUCER))
        for step in self.stager_steps:
            nb_job_done = 0
            running = 0
            if self.mode == 'sequential':
                running = step.processus[0].running
                nb_job_done = step.processus[0].nb_job_done
                levels_for_gui.append(StagerRep(step.name,step.next_steps_name,
                                      nb_job_done=nb_job_done,
                                      running=running,
                                      nb_processus = len(step.processus)))

            elif self.mode == 'multiprocessus':
                for processus in step.processus:
                    nb_job_done+=processus.nb_job_done
                    running += processus.running
                levels_for_gui.append(StagerRep(processus.name,step.next_steps_name,
                                      nb_job_done=nb_job_done,
                                      running=running,
                                      nb_processus = len(step.processus)))

        levels_for_gui.append(StagerRep(self.consumer_step.name,
                                nb_job_done=self.consumer.nb_job_done,
                                running=self.consumer.running,
                                step_type=StagerRep.CONSUMER))

        return (levels_for_gui,time())


    def display_conf(self):
        ''' Print steps and their next_steps
        '''
        self.log.info('')
        self.log.info('------------------ Flow configuration ------------------')
        for step in  ([self.producer_step ] + self.stager_steps
            + [self.consumer_step]):
            if self.mode == 'multiprocessus':
                self.log.info('step {} (nb processus {}) '.format(step.name,str(step.nb_processus)))
            else:
                self.log.info('step {}'.format(step.name))
            for next_step_name in step.next_steps_name:
                self.log.info('--> next {} '.format(next_step_name))
        self.log.info('------------------ End Flow configuration ------------------')
        self.log.info('')

    def display_statistics(self):
        """
        Log each StagerRep statistic
        """
        steps,_ = self.def_step_for_gui()
        for step in steps:
            self.log.info(step.get_statistics())

    def start(self):
        """ run the Flow based framework steps
        """
        if self.mode == 'multiprocessus':
            self.start_multiprocessus()
        elif self.mode == 'sequential':
            self.start_sequential()

    def start_sequential(self):
        """ run the Flow based framework in sequential mode
        """
        if self.gui :
            self.socket_pub.send_multipart(
            [b'MODE', dumps('sequential')])
        start_time = time()
        #self.producer.running = 0
        # Get producer instance's generator
        self.producer = self.sequential_instances[self.producer_step.name]
        #execute producer run coroutine
        prod_gen = self.producer.run()
        # only for gui
        if self.gui :
            self.producer.running = 1
            self.send_status_to_gui()
        #for each producer output
        for prod_result in prod_gen:
            if self.gui :
                self.producer.running = 0
                self.send_status_to_gui()
            # get next stage destination and input from producer output
            msg,destination = prod_result
            # run each steps until consumer return
            while msg != None:
                destination, msg=self.run_generator(destination,msg)
            if self.gui :
                self.producer.running = 1
                self.send_status_to_gui()
        if self.gui :
            self.consumer.running=0
            self.send_status_to_gui()
            # execute finish method for all steps
        for step in self.sequential_instances.values():
            step.finish()
        end_time = time()
        self.log.info('=== SEQUENTIAL MODE END ===')
        self.log.info('Compute time {} sec'.format(end_time - start_time))
        self.display_statistics()
        # send finish to GUI and close connexions
        if self.gui :
            self.socket_pub.send_multipart(
            [b'FINISH', dumps('finish')])
            self.socket_pub.close()
            self.context.destroy()
            self.context.term()

    def run_generator(self, destination ,msg):
        """ Get step for destination. Create a genetor from its run method.
        re-enter in run_generator until Generator send values
        Parameters:
        ===========
        destination: str
            Next step name
        msg: a Pickle dumped msg
        Returns:
        ========
        Next destination and msg
        """
        stage = self.sequential_instances[destination]
        stage.running = 1
        if self.gui : self.send_status_to_gui()
        stage_gen = stage.run(msg)
        stage.running = 0
        if stage_gen:
            for result in stage_gen:
                if result:
                    msg,destination = result
                    destination, msg=self.run_generator(destination,msg)
                else:
                    msg = destination = None
        else:
            msg = destination = None
        return  (msg,destination)


    def send_status_to_gui(self):
        """
        Update all StagerRep status and send them to GUI
        """
        self.socket_pub.send_multipart([b'MODE', dumps(self.mode)])
        levels_gui,conf_time = self.def_step_for_gui()
        self.socket_pub.send_multipart(
            [b'GUI_GRAPH', dumps([conf_time,
            levels_gui])])

    def start_multiprocessus(self):
        ''' Start all Flow based framework processus.
        Regularly inform GUI of Flow based framework configuration in case of a new GUI
        instance was lunch
        Stop all processus without loosing data
        '''
        # send Flow based framework cofiguration to an optinal GUI instance
        if self.gui :
            self.send_status_to_gui()
        start_time = time()
        # Start all processus
        self.consumer.start()
        self.router.start()
        for stage in self.stagers:
            stage.start()
        self.producer.start()
        # Wait producer end of run method
        self.wait_and_send_levels(self.producer)

        # Ensure that all queues are empty and all processus are waiting for
        # new data since more that a specific tine
        while not self.wait_all_stagers(1000): # 1000 ms
            if self.gui :
                self.send_status_to_gui()
            sleep(1)

        # Now send stop to stage processus and wait they join
        for worker in self.step_processus:
            self.wait_and_send_levels(worker)
        # Stop consumer and router processus
        self.wait_and_send_levels(self.consumer)
        self.wait_and_send_levels(self.router)
        if self.gui :
            self.send_status_to_gui()
        # Wait 1 s to be sure this message will be display
        end_time = time()
        self.log.info('=== MULTUPROCESSUS MODE END ===')
        self.log.info('Compute time {} sec'.format(end_time - start_time))
        self.display_statistics()

        sleep(1)
        if self.gui :
            self.socket_pub.send_multipart(
            [b'FINISH', dumps('finish')])
            self.socket_pub.close()
            self.context.destroy()
            self.context.term()


    def wait_all_stagers(self,mintime):
        """ Verify id all steps (stage + consumers) are finised their
        jobs and waiting
        Returns:
        ========
        True if all stages queue are empty and all Processus
        wait since mintime
        Otherwise False
        """
        if self.router.total_queue_size == 0 :
            for worker in self.step_processus:
                if worker.wait_since < mintime: # 5000ms
                    return False
            return True
        return False


    def finish(self):
        self.log.info('===== Flow END ======')

    def wait_and_send_levels(self, processus_to_wait):
        '''
        Wait for a processus to join and regularly send Flow based framework state to GUI
        in case of a GUI will connect later
        Parameters:
        -----------
        processus_to_wait : processus
                processus to join
        conf_time : str
                represents time at which configuration has been built
        '''
        processus_to_wait.stop = 1

        while True:
            processus_to_wait.join(timeout=.1)
            if self.gui :
                self.send_status_to_gui()
            if not processus_to_wait.is_alive():
                return

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
    tool = Flow()
    tool.run()

if __name__ == '__main__':
    main()
