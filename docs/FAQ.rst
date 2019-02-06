==========================
Frequently Asked Questions
==========================

----------------
Technical Issues
----------------

Missing standard library when compiling on macOS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you get an error when running `make develop` (or `python setup.py
develop`) that says `fatal error: 'iostream' file not found`, this is likely
because latest versions of macOS put the standard C++ libraries inside a
package, rather than in /usr/lib (where older versions of setuptools expect
them to be apparently). If this error occurs, you can just run the following
command to re-create the /usr/lib links:

.. code-block:: console

    sudo installer -pkg /Library/Developer/CommandLineTools/Packages/macOS_SDK_headers_for_macOS_10.14.pkg -target /


Hard crash when loading a SimTelArray file?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Note*: as of ctapipe 0.6.1, the new `SimTelEventSource` (based on `pyeventio`)
is used by default instead of `HESSIOEventSource` (based on
`pyhessio`/`libhessio`).  This should prevent any crashes as it does not use
the `libhessio` library. If you are still using an older version, or have
explicitly enabled the `HESSIOEventSource` rather than the new
`SimTelEventSource`, see the following:

Sometimes when reading a simtel file, the code crashes (not even an
out-of-memory error is given). Loading a simtel file right now needs a
lot of ram. However, a second issue with SimTelArray files is that we
have compiled `libhessio` to allow very large arrays (>500 telescopes,
with high-resolution cameras like the SCTCam), and this sometimes
causes memory allocation to hit your OS's stack limit.  THis issue
will go away when the final array layout is fixed (at which point we
will no longer support reading older simtelarray files, however).  In
the mean time, you can increase your stack memory limit by using the
commmand `ulimit -s <SIZE IN MB>` for a single terminal
session. Increasing it above the default should help stop these random
crashes.

