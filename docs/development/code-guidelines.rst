Code Guidelines
===============

Coding should follow the CTA coding guidelines from the **CTA Code
Standards** document. 

Here, we list useful guidelines for the logical structure of code (see
also the style-guide for code style).  This guide is subject to change
as we further develop the framework, so it should be checked
regularly.

References for good coding practices
------------------------------------

* `Good Python Style <http://docs.python-guide.org/en/latest/writing/style/>`_
* `Best Practices in Scientific Computing (Presentation) <http://swcarpentry.github.io/slideshows/best-practices/index.html>`_
* `Best Practices for Scientific Computing (Paper) <http://arxiv.org/abs/1210.0530>`_
  

Data Structures
---------------

Python is very flexible with data structures: data can be in classes,
dictionaries, lists, tuples, and numpy `NDArrays`.  Furthermore, the
structure of a class or dict is flexible: members can be added or
removed at runtime.  Therefore, we should be careful to follow some
basic guidelines:

* basic array-like data should be in an `numpy.NDArray`, with a suitable
  `dtype` (fixed data type)

* for complex sets of arrays, each with a column name and unit, can be
  managed via an `astropy.table.Table` object (which also provides
  methods to read and write such a table to/from nearly any file
  format, including FITS and HDF5). Other packages like `pandas` or
  `dask` may be explored.

* `dict` s or classes can be used for more flexible containers for
  output parameters.

* `ctapipe.core.Container` should be used for any
  high-level data structures that you want to be able to write to
  disk (they are not necessary for simple function return values)

* Classes don't need to sub-class `object`, because we only support
  Python 3 and new-style classes are the default, i.e. subclassing
  `object` is superfluous.


Logging and debugging
---------------------
  
* do not use `print()` statements to output text. Instead use the
  common logging failities of `ctapipe`.  Log messages should be
  simple, and no not include the filename, function name, time, or any
  other metadata (which can be attached automatically by the logging
  system)

Function or method Input/Output
-------------------------------

Functions and methods should *not modify input parameters*. Therefore
any data you pass in should be independent of what is output (do not
e.g. fill in a large data structure with several algorithms). The
reason for this is to allow for parallelization and flow-based
chaining of algorithms, which is impossible if one algorithm modifies
the input to another.

Unit Quantities
---------------

When approprate (e.g. in high-level algorithms APIs), use
`astropy.units` for any quantity where the unit may be ambiguous or
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
   
Unit-tests
----------


Writing Algorithms
------------------

Note that if you write an algorithm, it may be used in many ways: in a
command-line tool used in a batch-based system, in a server that
processes events or data in real-time on-line, or in a variety of
other data processing systems (map-reduce, Spark, dask,
etc). Therefore the main request of `ctapipe` mangers is that
algorithms should be written as simply as possible without depending
on any particular data flow mechanism. The following guidelines can
help when writing algorithms:

* Keep the design of algorithm code as simple as possible. Inputs and
  outputs should be simple values or arrays, avoiding complex structures
  if possible.

* Separate algorithms cleanly from the framework: Do not try to
  implement any *framework* features in your algorithm:
  - do not parse command-line or other options
  - do not make a way to choose a method from an input parameters
  (there will be a common factory class for that in the framework for
  all algorothms that have multiple implementations)
  - do not write data streams to disk yourself (use framework
  features, or just `print()` until they are available) data flow
  between algorithms, etc).
  - If a framework feature is missing, request it via the issue
  tracker.

* If the algorithm needs no *state* to be stored between calls, use a
  simple function with explicit parameters as keyword arguments. 

  .. code-block:: python

     def mangle_signal(signal, px, py, centerpoint=(0,0), setpoint=2.0*u.m):
         """
	 Mangles an image
		  
	 Parameters:
	 -----------
	 signal : np.ndarray
	     array of signal values for each point in space
	 px,py  : np.ndarray
	     arrays of x and y valyes of each signal value
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
	    
* if there are multiple implemenations of the same generic algorithm,
  a *class hierarchy* should be use where the base class defines the
  common interface to all algorithm instances.


* Algorithms that need user-definable parameters (that end up in a
  config file or as command-line parameters), need to use
  `ctapipe.core.Component` as a base class, and follow its guidelines
  (see related documentation)


* When writing example or integration test code for an algorithm,
  **keep it simple**: use a basic for loop to chain your algorithms
  together. This example code will later be transformed by *framework
  experts* into a modular system that can be parallelized and chained,
  so don't do that yourself. Algorithm test (not unit test, but
  integration test) code should look roughtly like this:

  .. code-block:: python


     # these should become user-defined parameters:
     
     filename = "events.tar.gz"
     tel_id = 1

     # initialize any algorithms
     
     source = calibrated_event_source(filename)
     ImageMangler mangler(geom.pix_x, geom.pix_y, "transformtable.fits")
     Serializer serializer = ...

     # simple loop over events, calling each algorithm and directly
     #passing data
     
     for event in source:
  
         image = event.dl1.tel[tel_id].image
         mangled_image = mangler.mangle(image)
         image_parameters = parameterize_image(mangled_image)

         # here you may here pack your output values into a Container if 
         # they are not already in one. We assume here that mangled_image
         # and image_parameters are already Container subclasses
     
         serializer.write([mangled_image, image_parameters])

* When your algorithm test code (as above) works well and you are
  happy with the results, you can do two things:
  
  1. convert your test code into a `ctapipe.core.Tool` so that it
     becomes a command-line program released with ctapipe (with no
     modification to the data flow).  This should be done anyway, if
     it is useful, since the `Tool` you create can be refactored
     later.
  2. request to the framework experts to have each algorithm wrapped
     in a chainable flow framework to allow parallelization and other
     advanced features.  Note that the choice of flow-framework is
     under study, so leaving things simple as above lets multiple
     systems be tested.


