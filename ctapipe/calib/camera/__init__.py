# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Camera calibration module.

--------------
calibrators.py
--------------
This module selects the correct calibrator based on the
origin of the file. For example, it will use the calibrator in mc.py if the
file’s origin is hessio. It then calibrates every telescope in that event,
and stores the pe_charge in the new dl1 container of the event (along with
some useful information such as the integration window used). Now the returned
event container contains the original dl0 and mc information
(actually a reference to the original) and the new dl1 container.

This module also contains a generator that calibrates the events as you loop
through this “calibrated_source”. This is useful when you wish to calibrate
many or all the events, as it stores the geometries into a dict, so they do
not need to be recalculated, speeding up the process. These functions will
also only calculate the geometry if the integrator you have specified needs it.

When calibration is to be added for a specific real-camera source, the correct
addition to the switch in this module should be made.

-----
mc.py
-----
This module handles the common calibration steps required for MC
files, irregardless of which integration technique is to be used. There
should be a version of this format of file for each real-camera calibration,
receiving the same input and returning the same output as this file does, and
performing similar operations as required (such as pedestal subtraction).

--------------
integrators.py
--------------
This module contains the integration techniques, which are common across the
cameras. The only input into this module are the waveforms and camera
geometries, ensuring its compatibility with the calibrators for any camera.
This module avoids duplication of integration code, and allows the sharing of
better methods. It also makes investigations in the best charge extraction
techniques a lot easier.
"""
