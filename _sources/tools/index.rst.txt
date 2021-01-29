.. _command_line_tools:

Command-Line Tools
==================

You can get a list of all available command-line tools by typing

.. code-block:: sh
   
    ctapipe-info --tools


Data Processing Tools:
----------------------

* `ctapipe-stage1`: input R0, R1, or DL0 data and output DL1 data in HDF5 DL1 format
* `ctapipe-merge`: merge DL1 (and other) data files into a single file
* `ctapipe-reconstruct-muons`: detect and parameterize muons (deprecated, to be merged with stage1 tool)

Other Tools:
------------
* `ctapipe-dump-instrument`: writes instrumental info from any supported event input file, and writes them out as FITS files for external use.

