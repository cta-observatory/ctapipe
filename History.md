
0.5.2 / 2017-07-31
==================

  * add system data in ctapipe-info
  * don't import info in tools/init
  * add concept of entity role to provenance
  * change warning to debug message
  * turn default activity warning into a debug message (not really a problem)
  * test for failure of writing output file
  * better log format + tool finish message
  * pep8 reformatting
  * more verbose default log message
  * add accessor for activity outputs and inputs
  * register files loaded via "from_name" in provenance
  * make sure setup is called after provenance is initialized
  * test for failure of writing output file
  * better log format + tool finish message
  * pep8 reformatting
  * Fixed auto documentation errors
  * more verbose default log message
  * add accessor for activity outputs and inputs
  * register files loaded via "from_name" in provenance
  * make sure setup is called after provenance is initialized
  * continue with warning if output file exists (do not overwrite)
  * Fixed underline bug in docs
  * Fixed underline bug in docs
  * Fixed underline bug in docs
  * Fixed underline bug in docs
  * add telescope_description column to optics table for convenience
  * Fixed failing tests
  * Updating naming scheme for telescopes
  * fixed some bugs in example notebooks
  * explicitly catch ImportError
  * removed old muon code that somehow crept back in
  * behavior of constructor also changed in astropy 2
  * remove healpy from list of packages (not needed)
  * add missing call to super().__init__(), which broke PlanarRepresentation for astropy 2
  * import FrameAttribute as Attribute for cross-compaibility with astropy 2.x
  * fixed tests
  * minor changes
  * Updated likelihood expectation calculation to use safe version of likelihood
  * modified test to check for relative uncertainty (should be more stable)
  * added deepcopy in base init -- protect against empty X[] in fit
  * pipeline should work now
  * reverting shower_max to enable pull request
  * Fixed error when taking log of quantity
  * Fixed import problem in hillas intersection test
  * minor cleanup of bug
  * Inproved documentation for functions guessing the shower maximum given the reconstructed energy
  * Significantly improved documentation for the ImPACT code
  * solving migrad nan issue and change shower max to return depth (not height)
  * register outputs from ctapipe-dump-instrument with provenance
  * really basic example
  * fix bug in definition of mcheader
  * Use StandardScaler class
  * added table output for OpticsDescriptions (in SubarrayDescription)
  * Cope with data in multiple telescopes in an event
  * update reco/__init__.py so that ShowerMaxEstimator appears in docs
  * removed __all__ token
  * Added scale_features method to RegressorClassifierBase class
  * added test unit for event classification with sklearn MLPClassifier
  * Added simple tests for image axis intersection
  * Added Hillas algorithm 1 style event reconstruction
  * moved from simple lists to named tuples
  * Removed print statements
  * Added LM options
  * Updated scaling factors
  * Updated viewers to allow backgrounds to be added more easily, updated scaling factors on the Hillas parameters
  * Added low gain channel
  * Changed scaling values
  * Removed unused funtions and added optional fit priors
  * Added option to use LM minimiser
  * Fixed camera rotation problems and added functions to draw likelihood surfaces
  * Allowed selection of minimiser to be used for ImPACt
  * Changed references from GCT to GATE
  * fixed zenith angle calculation
  * Added config options of cuts
  * Removed fixed Xmax from fit
  * Added comments
  * Added comments
  * general checks
  * Fixed for use with DSTs
  * Added comments
  * Changed structure such that set parameters function is not need
  * Changed structure such that set parameters function is not need
  * Added tool for perfomring hillas based reconstruction
  * Fixed energy reconstruction and added ouput into the correct format
  * Created tool for reco using Hillas parameters
  * Added MVA based energy reconstruction algorithm
  * Fixed incorrect calculation of impact distance when getting the training parameters
  * Added Tool to output Hillas parameters into FITS files
  * Code cleanup and addtion of command line options
  * Added cut to prevent 0 width events
  * Switched to use of FITS templates and fixed use of depth tables
  * Store system as class member
  * Fixed incorrect inversion of axes
  * renamed to work correctly with new ctapipe version
  * Fixed mis labelled axes
  * Added ImPACT reconstruction tool
  * Added fix for pixel neighbour list
  * Added Xmax reconstruction
  * modified for use with arbitrary numpy axis
  * Fixed error in assignment of telescope coordinates in core reconstruction
  * Converted the reconstruction code to a class, which inherits from the higher level reconstruction classes. Added calculation of the uncertatinties on the position and core. Changed output to the common shower output.
  * Removed print statement
  * Removed print
  * converted to use dictionaries as input
  * reverted changes
  * added fix to numpy array size
  * Added simple script for testing energy reconstruction methods
  * Update Hillas intersection code to use correct rotation angle
  * added ground projection info
  * Added function for tilted reconstruction
  * Added simple reconstruction tests
  * Changed to use camera object
  * Added Ground reconstruction function
  * Added Ground reconstruction function
  * Added instrument info class
  * Added simple Hillas reco code
  * Cleaned up imports
  * Created example script for simple Hillas Reco
  * Added Hillas reco example

v0.5.1 / 2017-07-05
===================

  * remove unused import
  * get rid of hard-coded telid in camera plotter test
  * use SubarrayDescription in array plotter test
  * use SubarrayDescription in HillasReconstructor
  * get rid of instrument.get_camera_types (replaced by SubarrayDescription)
  * remove geometry cache from EventViewer
  * remove geom_dict from CameraPlotter (replace with new instrument class)
  * update docs for instrument classes
  * get git tags in case they are not there
  * fix spelling error in directory name
  * update examples to use SubarrayDescription
  * added better docs for instrument module
  * fix bug in filling mirror_area and num_mirror_tiles
  * improve SubarrayDescription and add tests
  * add example for instrument info
  * added demo notebook
  * moved obsolete examples away
  * update calibration_pipeline example to use SubarrayDescription
  * add __all__ to __init__
  * add more docstring info
  * remove reference to old InstrumentDescription
  * removed some unmaintained code and renamed old InstrumentDescription to mc_config
  * added `select_subarray` to get sub-subarrays from a SubarrayDescription
  * reformatting
  * add missing name option to subarray
  * fix some formatting problems and extra imports
  * fix doc error
  * implement ArrayDescription.to_table() and .info()
  * add docs
  * added SubarrayDescription and use in event.inst
  * start to replace the `inst` container with TelescopeDescription
  * fix tests
  * updated docstrings
  * updated docstrings
  * fixed optics lookup table
  * added better __str__ and __repr__
  * added more docstring details for OpticsDescription and TelescopeDescription
  * import optics and camera classes to instrument level
  * add preliminary OpticsDescription and TelescopeDescription
  * dump basic optics desc to file in dump_instrument
  * fix option in pep8 config
  * remove config from .lanscape.yml that is duplicated in setup.cfg
  * update pep8 options
  * also set max-line-length in setup.cfg pep8 section
  * ignore log-format interpolation warning for now
  * added some pep8 exclusions
  * Fix docs and examples
  * Fix repr of Container
  * Remove more merge artifacts
  * Fix merge artifacts
  * Added function for parameterisation of image timing
  * Use fields instead of attributes for Container
  * Rename Item to Field, fix meta attribute
  * TableReader didn't correctly copy the headers to the container
  * added spaces around some more operators
  * Revert "Improve containers to use __slots__"
  * code now Landscape compliant
  * added landscape.io settings file
  * add code health badge and reformat
  * Remove unnecessary if and add test
  * fixed imports and unused vars
  * fixed unused imports and __all__
  * fix a bunch of flake8 warnings
  * remove unused variable
  * allow binary operator on newline rule
  * pep8 formatting
  * removed strange comment
  * autopep8 formatting
  * autopep8 to remove style warnings
  * remove commented-out code
  * removed redundant goal/goal → 1
  * removed incorrect return in multiprocess/connections.py __init__
  * memory issues? reduced number of draws
  * sensitivity now uses reconstructed energy instead of MC energy
  * update flake8 options
  * improved getting-started for developers
  * Catch ImportError instead of ModuleNotFoundError for python 3.5
  * Fix items() and __getstate__ of Container
  * Fix typo
  * Fix typo
  * Fix docstring
  * Reimplement setitem
  * Fix pickling and reset
  * Add missing Item
  * Make container use __slots__ through metaclass
  * docs must build without warning for build to pass
  * fix bad docstrings and reformat
  * fixed bad docstrings
  * cleaned up docs and added missing __all__
  * fixed intersphinx inventory warnings
  * fix wrong exception type in read_release_version
  * add option to disable progressbar
  * pep8
  * removed outdated example
  * clean up flake8 options
  * fix tests accordingly
  * renamed some N to n
  * added seed at beginning of function
  * implemented comments
  * loosened an assertion a bit; fixed some doc-strings
  * one more test for data
  * more updates and added a test specific for sensitivity
  * more work on test and event-drawing methods
  * started fixing some tests
  * small overhaul
  * cleanup of test plots
  * moved from log to linear binning in energy -- modified event weights; looks okayish now
  * loads of stuff
  * removed function call in test file
  * added sensitivity_energy_bin_edges argument to have a dedicated binning for sensitivity
  * fixed random draw from histogram
  * added energy event sampler -- WiP, needs testing
  * so, now...
  * added time stamp generation to sensitivity module
  * some renames, doc-fixes and default value tweaks
  * renamed function argument
  * fixed that last one...
  * minor tweak in reverting table order
  * inverted count-logic of `cut` and added `keep` instead added capability to add cut functions in bulk through a dict
  * minor tweak
  * brought test_Sensitivity.py up to speed
  * improved summing over events in on- and off-region
  * optimisation, bug-fixing, document-updating
  * refactored to be more flexible towards particle classes
  * minor changes and typos in docs
  * added format to efficiency column
  * modified docs
  * added module for PS sensitivity
  * switched arguments in set_cut function
  * fixed test assertion
  * updated documentation
  * switched from dict to OrderedDict
  * added cutflow class and unit test
