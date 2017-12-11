"""
This module contains the modules and files required for reading data for the
TargetIOFileReader.

checm_pixel_pos.npy and checm_reference_pulse.npz are numpy files which are
currently used to load instrument information that is normally included as
part of the sim_telarray file. In the future this information would be
included in the instrument database. However, as the instument database does
not exist yet, these file are used as a work-around.
"""