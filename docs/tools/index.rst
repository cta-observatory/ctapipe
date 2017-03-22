.. _tools:

============================
Command line tools (`tools`)
============================

.. currentmodule:: ctapipe.tools

Introduction
============

`ctapipe.tools` contains the command line tools that use the ctapipe
framework.  All commandline tools inherit from `ctapipe.core.Tool`
which provides a common user-interface experience, complete with
command-line and configuration file handling, logging, and provenance
features.  See `ctapipe.core` documentation for examples.


Getting Started
===============

You can find out what *ctapipe* tools are installed using the
`ctapipe-info` command-line tool:

.. code:: sh

   ctapipe-info --tools        # list all tools
   ctapipe-info --version      # show ctapipe version
   ctapipe-info --dependencies # list dependencies


To find out more about a specific tool, use it's ``--help`` command-line option


Common Tool Functionality
=========================

* High-level command-line options can be seen with ``--help``
* Low-level (Component) options can be seen with ``--help-all``
* You can either specify options on the command-line, or in a *config
  file* (so far in JSON or python format, see the documentation for
  `traitlets.config`). Specify this configuration file using the
  ``--config_file=FILENAME`` option.
* You can set the logging level of a tool using ``--log_level`` to a
  value or name in the list `[0, 10, 20, 30, 40, 50, 'DEBUG', 'INFO',
  'WARN', 'ERROR', 'CRITICAL']`.  By default, only warning messages are
  shown, but if you specify DEBUG you will get everything.


Developing a new Tool
=====================

To create a new command-line Tool, follow the following procedure:

1. make a subclass of `ctapipe.core.Tool`
2. define any user-configurable parameters of the tool
3. register any configurable `ctapipe.core.Component` classes
   that you will use with the Tool via its `classes` attribute
4. use the `aliases` attribute to promote any parameters from one of
   the sub-Components to a high-level a command-line parameter (they
   can still be specified using the advanced command-line command
   syntax or via a config file if you do not do this, but if the
   parameter is commonly changed by the user, having it be high-level
   is more convenient
5. overload the `setup(self)`, `start(self)`, and `finish(self)`
   methods to implement the functinatlity of the tool (these will be
   called in order when you call the `run()` method.
6. make sure you have a function called `main()` that creates and runs
   your tool in the source file.
7. add the file and `main()` function you made in step 6 to the
   top-level `setup.py` file under the section called
   `entry_points['console_scripts']`.  This will automatically create
   a executable called `ctapipe-*` where `*` is the name of the source
   file of your tool (e.g. `mytool.py` will become `ctapipe-mytool`.
8. this new tool will be installed in the bin directory next time you
   run `make develop` at the top-level of ctapipe, or if you
   re-install the ctapipe package in non-developer mode.

   

Reference/API
=============

.. automodapi:: ctapipe.tools
    :no-inheritance-diagram:
