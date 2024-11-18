***************
Code Guidelines
***************

Coding should follow the CTA coding guidelines from the **CTAO Software
Programming Standards** document :cite:p:`ctao-software-standards`.

Here, we list useful guidelines for the logical structure of code (see
also the style-guide for code style).  This guide is subject to change
as we further develop the framework, so it should be checked
regularly.


References for Good Coding Practices
====================================

* `Good Python Style <https://docs.python-guide.org/en/latest/writing/style/>`_
* `Best Practices in Scientific Computing (Presentation) <https://swcarpentry.github.io/slideshows/best-practices/index.html>`_
* `Best Practices for Scientific Computing (Paper) <https://arxiv.org/abs/1210.0530>`_


Checking for Logistic Errors
============================

Several static analysis packages exist to help look for common coding
errors, and these should be used frequently.

.. code-block:: sh

	% pip install hacking  # installs all checker tools

	% pyflakes file.py # checks for code errors
	% flake8 file.py   # checks style and code errors
	% flake8           # checks code in all subdirs


If you use *PyCharm* as an IDE, there is also a GUI function to find
and review all common code errors and style issues.


Unit-tests
==========

A *unit test* is a piece of code that tests a single functionality of
a library (e.g. a function, method, or class).

All code your write should have associated *unit tests* to ensure the
code works, gives reasonable results, handles error cases properly, and
to keep bugs at a minimum

Unit tests in ``ctapipe`` uses `pytest <https://docs.pytest.org>`_ .
Each module should put tests in a ``[module_name]/test`` subdirectory,
which can contain one or more files called ``test_[X]`` containing tests to run,
which are then automatically discovered.

To run the test suite, you can run ``make test`` from the top-level
ctapipe directory (which is just an alias to ``python -m pytest``).
You can also run tests in subdirectories to limit which ones are run.

Follow these basic guidelines:

1. There should be at least a unit test that *executes* all
   functions/classes/methods that you have written (minimally just
   runs them)
2. You should write tests that give simple inputs and check that the
   expected output is returned
3. Make sure to test edge and error cases for your functions
   (e.g. test what happens if an unexpected but still valid input is
   given)
4. Any time you fix a bug, it is good practice to add a unit test to
   make sure that bug does not appear again in the future (this is
   called regression testing)


Data Structures
===============

Python is very flexible with data structures: data can be in classes,
dictionaries, lists, tuples, and numpy ``NDArrays``.  Furthermore, the
structure of a class or dict is flexible: members can be added or
removed at runtime.  Therefore, we should be careful to follow some
basic guidelines:

* basic array-like data should be in an ``numpy.NDArray``, with a suitable
  ``dtype`` (fixed data type)

* for complex sets of arrays, each with a column name and unit, can be
  managed via an ``astropy.table.Table`` object (which also provides
  methods to read and write such a table to/from nearly any file
  format, including FITS and HDF5).
  Other packages like ``pandas`` or ``dask`` may be explored.

* ``ctapipe.core.Container`` should be used for any
  high-level data structures that you want to be able to write to
  disk (they are not necessary for simple function return values)


Logging and Debugging
=====================

* do not use the ``print()`` function to output text. Instead use the
  common logging failities of ``ctapipe``.  Log messages should be
  simple, and no not include the filename, function name, time, or any
  other metadata (which can be attached automatically by the logging
  system). See https://docs.python.org/3/howto/logging.html for more info

* Logging within a ``Tool`` or ``Component`` subclass: use the ``self.log`` logger
  instance

* logging in a library file that is not part of Tool or Component: define a
  logger at the top of the python file, and name it by using ``__name__`` as
  follows:


.. code-block:: python

	# at the top of your file:

	import logging
	logger = logging.getLogger(__name__)


Python logging works as follows:

.. code-block:: python

	logger.warning("this might be a problem")
	logger.info("basic status")
	logger.debug("debugging message")
	logger.error("a serious problem")
	logger.critical("this should never happen!")

And which messages print out and in what logging format can be defined at
run-time, along with filtering capabilities (e.g. only show log messages from
a particular file or class).

Some logging guidelines:

* you should **not** include the name of your function/class, line number, name
  of the file, or similar info in a log message. That information can be added
  automatically by the logger by changing the log format if needed (all log
  messages come with an attached ``LogRecord`` which contains all of the
  necessary metadata: name, level, pathname, filename, line number, message,
  arguments,exc_info (for exceptions), function name, stack info, process name, and
  optional user-defined fields.

* the log message should be human-readable and explain to a user not fully
  familiar with the code what is happening.

* if the message refers to a value, you can insert it into the message using
  format ``logger.debug("some message: {}".format(val)")`` or the log syntax
  ``logger.debug("some message: %d", val)``


Function or Method Input/Output
===============================

Functions and methods should *not modify input parameters*. Therefore
any data you pass in should be independent of what is output (do not
e.g. fill in a large data structure with several algorithms). The
reason for this is to allow for parallelization and flow-based
chaining of algorithms, which is impossible if one algorithm modifies
the input to another.


Unit Quantities
===============

When appropriate (e.g. in high-level algorithms APIs), use
``astropy.units`` for any quantity where the unit may be ambiguous or
where units need to be transformed.  Internally in a function, this is not necessary since the coder can ensure unit consistency, but for public APIs (function inputs, etc), units are useful.  You can even enforce a function to have particular unit inputs:

.. code-block:: python

   from astropy import units as u
   from astropy.units.decorators import quantity_input

   @quantity_input
   def my_function_that_should_enforce_units(width: u.m , length:u.m, angle:u.deg):
	   print(width.value, "is in meters") # no need for further conversion


With this decorator, the inputs will be automatically converted to the
given units, or an exception will be thrown if they cannot. So one can
call this like:

.. code-block:: python

   # works:
   my_function_that_should_enforce_units(width=12*u.cm,
								 length=16*u.m,
					 angle=1.0*u.rad)

   # throws exception
   my_function_that_should_enforce_units(width=12,   # no units, fails
								 length=16,
					 angle=1.0)
   # throws exception
   my_function_that_should_enforce_units(width=12*u.TeV, # bad conversion, fails
								 length=16*u.m,
					 angle=1.0*u.rad)

Note however, that this introduces some overhead as the units are
tested and converted for each function call. For functions that are
called frequently, it's best to enforce a unit earlier (e.g when the
parameters are defined), and assume it.


Writing Algorithms
==================

Note that if you write an algorithm, it may be used in many ways: in a
command-line tool used in a batch-based system, in a server that
processes events or data in real-time on-line, or in a variety of
other data processing systems (map-reduce, Spark, dask,
etc). Therefore the main request of ``ctapipe`` managers is that
algorithms should be written as simply as possible without depending
on any particular data flow mechanism. The following guidelines can
help when writing algorithms:

* Keep the design of algorithm code as simple as possible. Inputs and
  outputs should be simple values or arrays, avoiding complex structures
  if possible.

* Separate algorithms cleanly from the framework: Do not try to
  implement any *framework* features in your algorithm:
  - do not parse command-line or other options
  - do not make a way to choose a method from input parameters
  - do not write data streams to disk yourself (use framework
  features, or just ``print()`` until they are available) data flow
  between algorithms, etc).
  - If a framework feature is missing, request it via the issue
  tracker.

* If the algorithm needs no *state* to be stored between calls, use a
  simple function with explicit parameters as keyword arguments.

  .. code-block:: python

	def mangle_signal(signal, px, py, center_point=(0, 0), setpoint=2.0 * u.m):
        """
        Mangles an image

        Parameters:
        -----------
        signal : np.ndarray
            array of signal values for each point in space
        px,py  : np.ndarray
            arrays of x and y values of each signal value
        centerpoint : (x,y)
            center value in pixel coordinates
        setpoint : float quantity
            a parameter in meters
        """
	    ...


* if the algorithm must maintain some state information between calls
  (loaded tables, etc) or needs some common initialization parameters,
  a class may be used to facilitate this. Again, use keyword arguments.

  .. code-block:: python

    class SignalMangler:

        def __init__(self, px, py, lookup_table_filename):
            self.transform_table = Table.read(lookup_table_filename)
            self.px = px
            self.py = py

        def mangle(self, signal):
            ...

* if there are multiple implementations of the same generic algorithm,
  a *class hierarchy* should be use where the base class defines the
  common interface to all algorithm instances.


* Algorithms that need user-definable parameters (that end up in a
  config file or as command-line parameters), need to use
  :py:class:`ctapipe.core.Component` as a base class, and follow its guidelines
  (see related documentation)


* When writing example or integration test code for an algorithm,
  **keep it simple**: use a basic for loop to chain your algorithms
  together.
  An algorithm test (not unit test, but integration test) should look roughly like this:

  .. code-block:: python


    # these should become user-defined parameters:
    filename = "events.tar.gz"
    tel_id = 1

    # initialize any algorithms

    source = EventSource(filename)
    geom = source.subarray.tel[tel_id].camera.geometry
    ImageMangler = mangler(geom.pix_x, geom.pix_y, "transformtable.fits")

    # simple loop over events, calling each algorithm and directly
    # passing data

    for event in source:
        image = event.dl1.tel[tel_id].image
        mangled_image = mangler.mangle(image)
        image_parameters = parameterize_image(mangled_image)


* When your algorithm test code (as above) works well and you are
  happy with the results, you should convert your test code into a set of
  :py:class:`ctapipe.core.Component` or :py:class:`ctapipe.core.Tool`
  so that it is usable with the configuration system or becomes a
  command-line program released with ctapipe.
