.. _pipeline:

================
Pipeline
================

.. currentmodule:: ctapipe.pipeline

Introduction
============

`ctapipe.pipeline`
it is a parallelization system. It executes ctapipe processing modules in a multithread environment.

It is based on ZeroMQ library (http:queue//zeromq.org) for messages passing between threads.
ZMQ library allows to stay away from class concurrency mechanisms like mutexes,
critical sections semaphores, while being thread safe.

User implements steps in Python class. Passing data between steps is managed by the router thanks to Pickle serialization.
If a step is executed by several threads, the router uses LRU pattern (least recently used ) to
choose the step that will receive next data. The router also manage Queue for each step.

.. figure:: pipeline.png
    :scale: 70 %
    :alt: pipeline example

    ctapipe-guipipe application. It displays a complete pipeline instance containing:
    A producer step (SimTelArrayReader) that reads events in a SimTelArray MC file and sends them one by one to next step.
    A stager step (ShuntTelescope) that receives event and guides it to the next step according to it type (LST or other)
    A stager step (LSTDump) that receives LST telescope raw data, and sends a string with LST tag and raw data string representation.
    A stager step (OtherDump) that receives OTHER telescope raw data, and sends a string with OTHER tag and raw data string representation.
    This step runs on 2 thread (this is represented by 2 arrows)
    A consumer step (StringWriter) that receives strings and writes them to a file.


Getting Started
===============
ZMQ library installation
------------------------
   *prompt$> conda install pyzmq*

Pipeline configuration
======================
Pipeline configuration is read from a json configuration file, or command line arguments, thanks to traitlets config.

Mandatories configuration entries:
----------------------------------

- One producer_conf containing 1 step.

- One consumer_conf containing 1 step.

- One stagers_conf containing 1 to n step(s).

Mandatory configuration per step
--------------------------------
- name  : step name

- class : class name containing algorithm

- module: python module containing class (defined above)

- next_steps: list of next steps (you should use comma to separate items)

Optional entry per step
^^^^^^^^^^^^^^^^^^^^^^^
- nb_thread:  only available for stage, not for producer or consumer. Define how many thread will execute this stage
- queue_limit:  Define maximum number of message a router can queue for this step. Used it to limit memery consumption.

User option for step
^^^^^^^^^^^^^^^^^^^^
If step class derived form Component, you can add all required parameters for the step, and get them at execution time.
You have to add a new entry with step's class name.
! Do not use entries in stagers_conf, producer_conf or consumer_conf


json example
^^^^^^^^^^^^
.. code-block:: json

    {
      "version": 1,
      "Pipeline": {
          "producer_conf": { "name" : "SimTelArrayReader", "module": "ctapipe.pipeline.algorithms.simtelarray_reader",
               "class": "SimTelArrayReader","next_steps" : "ShuntTelescope"},
          "consumer_conf": { "name" : "StringWriter", "module": "ctapipe.pipeline.algorithms.string_writer",
                    "class": "StringWriter"},
          "stagers_conf" : [ {"name": "ShuntTelescope", "class": "ShuntTelescope",
                                          "module": "ctapipe.pipeline.algorithms.shunt_telescope",
                                          "next_steps" : "OtherDump,LSTDump" },
                            {"name": "OtherDump", "class": "OtherDump",
                                          "module": "ctapipe.pipeline.algorithms.other_dump",
                                          "next_steps" : "StringWriter", "nb_thread" : 2},
                            {"name": "LSTDump", "class": "LSTDump",
                                          "module": "ctapipe.pipeline.algorithms.lst_dump",
                                          "next_steps" : "StringWriter", "nb_thread" : 1, "queue_limit" : 10}
                          ]
      },
      "SimTelArrayReader": { "filename": "gamma_test.simtel.gz"},
      "StringWriter": { "filename": "/tmp/string_writter.txt"}
    }


Steps implementation
====================
Step is defined in a Python class. Each class defines 3 methods: init, run and finish
These 3 methods are executed by the pipeline.

Producer run method
-------------------
Producer class run method does not have any input parameter.
It must yield nothing if you want to get correct number of job salready done via the GUI.
Use self.send_msg mrthod to send data to next step.



.. code-block:: python

    >>> def run(self):
    >>>     for input_file in os.listdir(self.source_dir):
    >>>         self.send_msg(self.source_dir + "/" + input_file)
    >>>         yield

Stager run method
-----------------
Stager class run method takes one parameter (sent by the previous step).
Use self.send_msg mrthod to send data to next step.
Do not return anything (or it will be lose)


.. code-block:: python

    >>> def run(self,event):
    >>>     if event != None:
    >>>         self.send_msg(event.dl0.tels_with_data)


In case a step has to send several output for one input to the next step :

.. code-block:: python

    >>> def run(self,event):
    >>>    if event != None:
    >>>        tels = event.dl0.tels_with_data
    >>>        for tel in tels:
    >>>             self.send_msg(tel)

Consumer run method
-------------------
Consumer class run method takes one parameter and does not return anything

Send message with several next steps.
-------------------------------------
In case of producer or stage have got several next step (next_steps keyword in configuration),
you can choose the step that will receive data by passing its name as parameter of send_msg method

.. code-block:: python

    >>> def run(self,event):
    >>>    if event != None:
    >>>        tels = event.dl0.tels_with_data
    >>>        for tel in tels:
    >>>             if tel in lst_list:
    >>>                 self.send_msg(tel,'LST_CALIBRATION')
    >>>             elif tel in mst_list
    >>>                 self.send_msg(tel,'MST_CALIBRATION')

Running the pipeline
====================
   *prompt$> ctapipe-pipeline --config=mypipeconfig.json*
By default pipeline send its activity to a GUI  on tcp://localhost:5565.
But if the GUI is running on another system, you can use --Pipeline.gui_address
option to define another address.
Configure the firewall to allow access to that port for authorized computers.

Execution examples
------------------
    *prompt$> ctapipe-pipeline --config=examples/brainstorm/pipeline/pipeline_py/example.json*

Pipeline Graphical representation
=================================
A GUI can be launch to keep a close watch on pipeline execution.
This GUI can be launch on the same system than the pipeline or on a different one.
By default GUI is binded to port 5565. You can change it with --PipeGui.port option
    *prompt$> ctapipe-guipipe*

Optional packages for GUI
-------------------------

PyQt4 installation
^^^^^^^^^^^^^^^^^^
   *prompt$> conda install pyqt*

graphviz installation
^^^^^^^^^^^^^^^^^^^^^
*prompt$> conda install graphviz*

Examples
========

json configuration example
--------------------------
Refer to `json example`_.

.. _example:

Producer example
----------------
.. code-block:: python

    from ctapipe.utils.datasets import get_path
    from ctapipe.io.hessio import hessio_event_source
    import threading
    from ctapipe.core import Component
    from traitlets import Unicode

    class SimTelArrayReader(Component):

        """`SimTelArrayReader` class represents a Producer for pipeline.
            It opens simtelarray file and yiekld even in run method
        """
        filename = Unicode('gamma', help='simtel MC input file').tag(
            config=True, allow_none=True)
        source = None

        def init(self):
            self.log.info("--- SimTelArrayReader init {}---".format(self.filename))
            try:
                in_file = get_path(self.filename)
                self.source = hessio_event_source(in_file)
                self.log.info('{} successfully opened {}'.format(self.filename,self.source))
            except:
                self.log.error('could not open ' + in_file)
                return False
            return True

        def run(self):
            counter = 0
            for event in self.source:
                event.dl0.event_id = counter
                counter += 1
                # send new job to next step thanks to router
                self.send_msg(event)
                yield
            self.log.info("\n--- SimTelArrayReader Done ---")

        def finish(self):
            self.log.info("--- SimTelArrayReader finish ---")
            pass



Stager example
--------------
.. code-block:: python

    from ctapipe.utils.datasets import get_path
    import ctapipe.instrument.InstrumentDescription as ID
    from ctapipe.core import Component
    from traitlets import Unicode


    LST=1
    OTHER = 2
    class ShuntTelescope(Component):
        """ShuntTelescope class represents a Stage for pipeline.
            It shunts event based on telescope type
        """

        def init(self):
            self.log.info("--- ShuntTelescope init ---")
            self.telescope_types=dict()
            for index in range(50):
                self.telescope_types[index]=LST
            for index in range(50,200,1):
                self.telescope_types[index]=OTHER
            return True

        def run(self, event):
            triggered_telescopes = event.dl0.tels_with_data
            for telescope_id in triggered_telescopes:
            if self.telescope_types[telescope_id] == LST:
                self.send_msg(event.dl0.tel[telescope_id],'LSTDump')
            if self.telescope_types[telescope_id] == OTHER:
                self.send_msg(event.dl0.tel[telescope_id],'OtherDump')

        def finish(self):
            self.log.info("--- ShuntTelescope finish ---")
            pass


Consumer example
----------------
.. code-block:: python

    from ctapipe.configuration.core import Configuration, ConfigurationException
    from ctapipe.core import Component

    class StringWriter(Component):

            def init(self):
                filename = '/tmp/test.txt'
                self.file = open(filename, 'w')
                self.log.info("--- StringWriter init ---")
                return True

            def run(self, object):
                if (object != None):
                    self.file.write(str(object) + "\n")

            def finish(self):
                self.log.info("--- StringWriter finish ---")
                self.file.close()
