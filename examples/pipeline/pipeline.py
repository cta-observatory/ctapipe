"""Simple but robust implementation of generator/coroutine-based
pipelines in Python. The pipelines may be run either sequentially
(single-threaded) or in parallel (one thread per pipeline stage).

This implementation supports pipeline bubbles (indications that the
processing for a certain item should abort). To use them, yield the
BUBBLE constant from any stage coroutine except the last.

In the parallel case, the implementation transparently handles thread
shutdown when the processing is complete and when a stage raises an
exception.
"""
from queue import Queue
from threading import Thread, Lock

from ctapipe.utils.datasets import get_path
from ctapipe.io.hessio import hessio_event_source
from ctapipe.configuration.core import Configuration, ConfigurationException
from ctapipe.core import Container
from sys import exit,stderr

BUBBLE = '__PIPELINE_BUBBLE__'
POISON = '__PIPELINE_POISON__'

DEFAULT_QUEUE_SIZE = 0

__all__=  ['Pipeline', 'PipelineError']


class PipelineError(Exception):
    def __init__(self, msg):
        """An indication that an exception occurred in the pipeline. The
        object is passed through the pipeline to shut down all threads
        before it is raised again in the main thread.
        """
        self.msg = msg

class Pipeline():
    
    def __init__(self,configuration):
        """Represents a staged pattern of stage. Each stage in the pipeline
        is a coroutine that receives messages from the previous stage and
        yields messages to be sent to the next stage.
        
        Parameters
        ----------
        configuration : Configuration object, required
            Pipeline asks to this configuration instance  to 
            create producers, stagers and consumers instances
            according to information in configuration
        """
        self.conf = configuration
        self.producer = None
        self.stager = None
        self.consuner = None
        self.stages = [self.produce(), self.stage(), self.consume()]
        
    def init(self):
        """
        Create producers, stagers and consumers instance according to configuration 
        Returns:
        --------
        bool: True if pipeline is correctly setup and all producer,stager and consumer initialised
              Otherwise False
        """
        # import and init producer
        producer_section_name = self.conf.get('PRODUCER',section='PIPELINE')
        if (producer_section_name == None):
            print("No PRODUCER section in configuration",file=stderr)
            return False
        self.producer = self.conf.dynamic_class_from_module(producer_section_name)
        if self.producer  == None: return False
        if self.producer.init() == False: return False
        
        # import and init stager
        stagers = self.conf.getNextStager(producer_section_name)
        if stagers != None and len(stagers)==0:
            print("no stager found in configuration.", file=stderr)
            return False
        else:
            stager_section_name =stagers[0]
            self.stager = self.conf.dynamic_class_from_module(stager_section_name) 
            if self.stager == None : return False
            self.stager.init()
            
            # import and init consumer
            consumers = self.conf.getNextStager(stager_section_name)
            if consumers != None and len(consumers)>0:
                consumer_section_name = consumers[0]
                print("consumer_section_name:", consumer_section_name)
                self.consumer = self.conf.dynamic_class_from_module(consumer_section_name)
                if self.consumer == None: return False
                self.consumer.init()
        
            else: stager_section_name = None
            return True
        
        
    def produce(self):
        """
        Generates output for next stages 
        """
        generator = self.producer.run()
        for output in generator:
            yield output
        
    def stage(self):
        """
        Receives input from prev stage and generates output for next stages 
        """
        input = yield
        while True:
            output = self.stager.run(input) 
            input = yield output

    def consume(self):
        """
        Receives input from prev stage and consums it
        """
        while True:
            input = yield
            self.consumer.run(input)
            
            
    def run_sequential(self):
        """Run the pipeline sequentially in the current thread. The
        stages are run one after the other.
        """
        # "Prime" the coroutines.
        for coro in self.stages[1:]:
            next(coro)
        
        # Begin the pipeline.
        for msg in self.stages[0]:
            for stage in self.stages[1:]:
                msg = stage.send(msg)
                if msg is BUBBLE:
                    # Don't continue to the next stage.
                    break
    
    def run_parallel(self, queue_size=DEFAULT_QUEUE_SIZE):
        """Run the pipeline in parallel using one thread per stage. The
        messages between the stages are stored in queues of the given
        size.
        """
        queues = [Queue(queue_size) for i in range(len(self.stages)-1)]
        threads = [_ProducerThread(self.stages[0], queues[0])]
        for i in range(1, len(self.stages)-1):
            threads.append(_StagerThread(
                self.stages[i], queues[i-1], queues[i]
            ))
        threads.append(_ConsumerThread(self.stages[-1], queues[-1]))
        
        # Start threads.
        for thread in threads:
            thread.start()
        
        # Wait for termination.
        try:
            for thread in threads:
                thread.join()
        except:
            # Shut down the pipeline by telling the first thread to
            # poison its channel.
            threads[0].abort()
            raise
        
        # Was there an exception?
        exc = threads[-1].exc
        if exc:
            raise exc

    def finish(self):
        """
        Execute finish method on all producers, stagers and consumers
        self.producer.finish()
        self.stager.finish()
        self.consumer.finish()
        
        
class _ProducerThread(Thread):
    """The thread running the first stage in a parallel pipeline setup.
    The coroutine should just be a generator.
    """
    def __init__(self, coro, out_queue):
        super(_ProducerThread, self).__init__()
        self.coro = coro
        self.out_queue = out_queue
        
        self.abort_lock = Lock()
        self.abort_flag = False
    
    def run(self):
        while True:
            # Time to abort?
            with self.abort_lock:
                if self.abort_flag:
                    break
            
            # Get the value from the generator.
            try:
                msg = next(self.coro)
            except StopIteration:
                break
            except (Exception, exc):
                self.out_queue.put(PipelineError(exc))
                return
            
            # Send it to the next stage.
            self.out_queue.put(msg)
            if msg is BUBBLE:
                continue
        
        # Generator finished; shut down the pipeline.
        self.out_queue.put(POISON)
    
    def abort(self):
        """Shut down the pipeline by canceling this thread and
        poisoning out_channel.
        """
        with self.abort_lock:
            self.abort_flag = True

class _StagerThread(Thread):
    """A thread running any stage in the pipeline except the first or
    last.
    """
    def __init__(self, coro, in_queue, out_queue):
        super(_StagerThread, self).__init__()
        self.coro = coro
        self.in_queue = in_queue
        self.out_queue = out_queue

    def run(self):
        # Prime the coroutine.
        next(self.coro)
        
        while True:
            # Get the message from the previous stage.
            msg = self.in_queue.get()

            if msg is POISON:
                break
            elif isinstance(msg, PipelineError):
                self.out_queue.put(msg)
                return
            
            # Invoke the current stage.
            try:    
                out = self.coro.send(msg)
            except (Exception, exc):    
                self.out_queue.put(PipelineError(exc))
                return
            
            # Send message to next stage.
            if out is BUBBLE:
                continue 
            
            self.out_queue.put(out)
        
        # Pipeline is shutting down normally.
        self.out_queue.put(POISON)

class _ConsumerThread(Thread):
    """A thread running the last stage in a pipeline. The coroutine
    should yield nothing.
    """
    def __init__(self, coro, in_queue):
        super(_ConsumerThread, self).__init__()
        self.coro = coro
        self.in_queue = in_queue

    def run(self):
        # Prime the coroutine.
        next(self.coro)

        while True:
            # Get the message from the previous stage.
            msg = self.in_queue.get()
            if msg is POISON:
                break
            elif isinstance(msg, PipelineError):
                self.exc = msg.exc
                return
            
            # Send to consumer.
            try:
                self.coro.send(msg)
            except (Exception, exc):
                self.exc = exc
                return
        
        # No exception raised in pipeline.
        self.exc = None
