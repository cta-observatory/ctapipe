.. _analysis_steps:

**************
Analysis Steps
**************

The analysis of CTAO data in ``ctapipe`` follows a staged approach, moving from raw
data to high-level reconstructed parameters.

Data Levels
===========

- **R1**: Calibrated waveforms (photo-electrons).
- **DL0**: Reduced data (e.g. after volume reduction).
- **DL1**: Integrated camera images and timing information (image parameters).
- **DL2**: Reconstructed event parameters (energy, direction, particle type).

Regarding the data format for each level, please refer to the :ref:`data_format` section of the user guide.

Processing Pipeline
===================

1. **Calibration**: Converts raw ADC counts to photo-electrons.
2. **Image Extraction**: Integrates waveforms to produce 2D camera images (DL1a).
3. **Image Cleaning**: Removes noise pixels, leaving only the Cherenkov shower signal.
4. **Parameterization**: Calculates Hillas and timing parameters from the cleaned image (DL1b).
5. **Reconstruction**: 
    - **Direction**: Uses multiple telescope views (stereo) to find the source position.
    - **Energy**: Estimates the primary particle's energy.
    - **Classification**: Distinguishes between gamma-rays and hadronic background.

.. toctree::
   :maxdepth: 1

   image_parameters
