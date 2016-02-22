==========
Brainstorm
==========

This directory contains a bunch of demo scripts that I had used to
explore some concepts for the framework. They do not use the ctapipe
module directly.

**************
proposal_demos
**************
runnable code that is in the PDF proposal summary document

misc_demos
==========
random pieces of code (mostly working, but may need some support
files to be copied from the ctapipe_data directory)

notebooks
=========

random thoughts and tests in IPython notebook format (run `ipython notebook` to access)

pipeline
========
zmqpipe
-------

A multithreading pipeline implementation build on top of ZMQ

Each pipeline stage must be a Python Class with a least 3 methods( init, run and finish).

Each pipeline stage must be defined in a configuration file (see ctapipe.configuration and pipeline.ini example) with at least::

        -module :Python mosdule name
        -class  :Python class name within the module
        -role   :PRODUCER, STAGER or CONSUMER
        -prev   :previous stage iniside pipeline(Only for STAGER and CONSUMER) 

        ZMQ takes care of filling queue and passing next input to pipeline stager and consumer.

pipeline_py
^^^^^^^^^^^
pipeline example using only Python modules and hessio MC data

pipeline_process
^^^^^^^^^^^^^^^^
pipeline example using system executables instead of Python modules