*****************
Support Libraries
*****************

In general, we will try to avoid re-implementing complex mathematical
algorhtms, and use instead a small set of well-tested, community
supported packages. All packages chosen have very large developer
bases, and a wide community of users, ensuring long-term support. 

The following are support libraries that are
recommended when developing CTA Pipeline algorithms:

.. contents::


Math/Stats
==========

A large variety of advanced math and statistcs functions can be found
in the `numpy <http://www.numpy.org>`_  (data structures and numerics
)and `scipy <http://www.scipy.org>`_ packages (advanced mathematics).

* NumPy user guide : http://docs.scipy.org/doc/numpy/user/  (more high-level)
* NumPy reference  : http://docs.scipy.org/doc/numpy/reference/
* SciPy user guide : http://docs.scipy.org/doc/scipy/reference/
* SciPy reference : http://docs.scipy.org/doc/scipy/reference/index.html
  
Specific functionality:

* `scipy.stats`: statistical models (pdfs, random sampling, etc) for
  discrete or continuous distributions
* `scipy.linalg`:  fast linear algebra operations
* `scipy.optimize`: fitting and minimization
* `scipy.integrate`: integration, including multivariate integration
* `scipy.signal`: signal processing
* `scipy.interpolate`: interpolation of multi-variate datasets
* `scipy.spatial`: spatial algorithms (clustering, nearest neighbors)
* `scipy.special`: special functions 

these functions are all based on `numpy.ndarray` data structures,
which provide c-like speeds. 

Multivariate Analysis and Machine Learning
==========================================

`SciKit-Learn <http://scikit-learn.org>`_, an extension of SciPy, provides
very friendly and well-documented interface to a wide variety of MVA
techniques, both for classification and regression.

* Decision Trees
* Support vector machines
* Random Forrests
* Perceptrons
* Clustering
* Dimensionality Reduction
* training/cross-checks
* etc.


Astronomical Calculations
=========================

`AstroPy <http://astropy.org>`_ is the accepted package for all
astronomical calculations. Specifically:

* `astropy.coordinates`: coordinate transforms and frames
* `astropy.time`: time transformations and frames
* `astropy.wcs`: projections
* `astropy.io`: low-level FITS access
* `astropy.units`: unit quantities with automatic propegation and
  cross-checks
* `astropy.table`: quick and easy reading of tables in nearly any
  format (FITS, ascii, HDF, VO, etc)
* `astropy.convolution`: convolution and filtering (built upon
  scipy.signal, but with more robust defaults) 

subpackages of Astropy that are not marked as "reasonable stable" or
"mature" should be avoided until their interfaces are solidified. The
list can be found on the astropy documentation page, under the list
*current status of subpackages* 
  

FITS Table Access
=================

FITS Tables can be read via `astropy.table`, or `astropy.io.fits`
(formerly called `pyfits`), however these implementations are not
intended for efficient access to very large files (As they access all
tables column-wise). In the case we want to load GBs or more of data
in a FITS table, the `fitsio` module should be used instead. It is a
simple wrapper for libCFITSIO, and supports efficient row-wise table
access.  It is not yet included in Anaconda's distribution, so must be
installed via pip

.. code-block:: bash

   pip install --user fitsio
