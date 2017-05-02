Frequently Asked Questions
==========================

Hard crash when loading a SimTelArray file?
-----------------------------------------------------------------------

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
