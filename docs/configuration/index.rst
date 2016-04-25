.. _configuration:

================
Configuration
================

.. currentmodule:: ctapipe.configuration

Introduction
============

`ctapipe.configuration`
This module allows to get command-line options and configuation values from file in a same object instance.   
On the one hand, this module implements a basic configuration file parser language which provides a FITS stucture and a structure similar to what you would find on Microsoft Windows INI files.   
On the other hand, as It derives form argparse.ArgumentParser, this module makes it easy to write user-friendly command-line interfaces. The program defines what arguments it requires, and it will figure out how to parse those out of sys.argv. This module also automatically generates help and usage messages and issues errors when users give the program invalid arguments.

Getting Started
===============
* [For command-line options refer to argparse.ArgumentParser documentation](https://docs.python.org/3/library/argparse.html)  
.. code-block:: python

	>>> from ctapipe.configuration import Configuration
	>>> conf = Configuration()  
	>>> conf.add_argument("--name", dest="name", action='store')
	>>> conf.parse_args()
	>>> current_name = conf.get("name") == "CTA"


* [For configuation parser refer to configparser documentation](https://docs.python.org/3/library/configparser.html#module-ConfigParser)
.. code-block:: python

	>>> from ctapipe.configuration import Configuration
	>>> conf = Configuration()
	>>> conf.add("key", "value", section="section", comment="my comment")
	>>> conf.write('example.fits')
	>>> readed_conf = Configuration()
	>>> readed_conf.read('example.fits')
	>>> value = readed_conf.get("key", "section")



Reference/API
=============

.. automodapi:: ctapipe.configuration.core
