.. _command_line_tools:

Command-Line Tools
==================

You can get a list of all available command-line tools by typing

.. code-block:: sh

    ctapipe-info --tools


Data Processing Tools:
----------------------

* ``ctapipe-process``: input R0, R1, or DL0 data and output DL1/DL2 data in HDF5 format
* ``ctapipe-merge``: merge DL1-DL2 data files into a single file
* ``ctapipe-reconstruct-muons``: detect and parameterize muons (deprecated, to be merged with process tool)

Other Tools:
------------

* ``ctapipe-dump-instrument``: writes instrumental info from any supported event input file, and writes them out as FITS files for external use.
