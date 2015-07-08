from ctapipe.io import provenance

"""note that provenance is a singleton, and contains a global variable:

    _actcollect = ActivityCollection()

that contains the provenance information, and the helper functions
operate on that variable. . Thus, any module just has to import
provenance, and call the functions like `start_activity` or
`add_input`, and their info will be added to the global provenance
info that is written at the end of the program.

- version info is taken from the module.__version__ attribute

- where do actors get added? (could be automatically via username of
  person running tool)



- Method is the description of possible inputs and configuration (like
 a schema)

- Activity is the realization of the Method (e.g. launched on a file),
  with a start and stop time.

- ConfigBuilder: should build a MethodCollection? or ActivityColleciton?

  * MethodCollection would imply list of Components/Tools, but without
    parameter values

  * ActivityCollection would be the same with the values for parameters

"""


@method
def anothermethod(data, calfile, threshold=6):
    provenance.add_input(calfile)
    # do method
    return 0


# activity: all method parameters are fixed


if __name__ == '__main__':

    with provenance.activity(name="reconstruction", output="prov.fits"):

        # inputs (entities) can be added at any time (even in sub-modules)
        # with copy=True, any provenance info in the file is read in
        # (e.g. previous Activities).
        provenance.add_input(lfn="/data/stuff/somedatafile.fits", copy=True)
        provenance.add_input(lfn="/data/stuff/anotherfile.fits")

        # also, config parameters can be registered, grouped by some
        # identifier like the algorithm name (this could be done
        # automatically in each algorithm).
        config = {'x': 3, 'var': 'test', 'cut': (-1, 3)}
        provenance.add_activity("myalogorithm", params=config, version=1)
        provenance.add_activity(anothermethod)

        # do stuff... (TOOL CODE HERE)

    # when the activity context-manager ends, the status is recorded
    # (any exceptions will be caught and the status set).
    #
    # The output file is written, containing an ActivityCollection
    # with this activity, and all previous ones that were copied from
    # the input file (the "reconstruction" activity is appended to the
    # end of the Collection). May need to allow writing prov data to
    # multiple output files, if there are more than one
