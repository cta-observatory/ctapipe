"""
Creating command-line Tools
===========================

"""

from time import sleep

from ctapipe.core import Component, TelescopeComponent, Tool
from ctapipe.core.traits import (
    FloatTelescopeParameter,
    Integer,
    Path,
    TraitError,
    observe,
)
from ctapipe.instrument import SubarrayDescription
from ctapipe.utils import get_dataset_path

######################################################################
GAMMA_FILE = get_dataset_path("gamma_prod5.simtel.zst")


######################################################################
# see https://github.com/ipython/traitlets/blob/master/examples/myapp.py
#


######################################################################
# Setup:
# ------
#
# Create a few ``Component``\ s that we will use later in a ``Tool``:
#


class MyComponent(Component):
    """A Component that does stuff"""

    value = Integer(default_value=-1, help="Value to use").tag(config=True)

    def do_thing(self):
        self.log.debug("Did thing")


# in order to have 2 of the same components at once
class SecondaryMyComponent(MyComponent):
    """A second component"""


class AdvancedComponent(Component):
    """An advanced technique"""

    value1 = Integer(default_value=-1, help="Value to use").tag(config=True)
    infile = Path(
        help="input file name",
        exists=None,  # set to True to require existing, False for requiring non-existing
        directory_ok=False,
    ).tag(config=True)
    outfile = Path(help="output file name", exists=False, directory_ok=False).tag(
        config=True
    )

    def __init__(self, config=None, parent=None, **kwargs):
        super().__init__(config=config, parent=parent, **kwargs)
        # components can have sub components, but these must have
        # then parent=self as argument and be assigned as member
        # so the full config can be received later
        self.subcompent = MyComponent(parent=self)

    @observe("outfile")
    def on_outfile_changed(self, change):
        self.log.warning("Outfile was changed to '{}'".format(change))


class TelescopeWiseComponent(TelescopeComponent):
    """a component that contains parameters that are per-telescope configurable"""

    param = FloatTelescopeParameter(
        help="Something configurable with telescope patterns", default_value=5.0
    ).tag(config=True)


######################################################################
MyComponent()

######################################################################
AdvancedComponent(infile="test.foo", outfile="out.foo")


######################################################################
# ``TelescopeComponents`` need to have a subarray given to them in order
# to work (since they need one to turn a ``TelescopeParameter`` into a
# concrete list of values for each telescope. Here we will give a dummy
# one:
#


subarray = SubarrayDescription.read(GAMMA_FILE)
subarray.info()

######################################################################
TelescopeWiseComponent(subarray=subarray)


######################################################################
# This TelescopeParameters can then be set using a list of patterns like:
#
#
#    component.param = [("type", "LST*",3.0),("type", "MST*", 2.0),(id, 25, 4.0)]
#
# These get translated into per-telescope-id values once the subarray is
# registered. After that one access the per-telescope id values via:
#
#
#    component.param.tel[tel_id]
#


######################################################################
# Now create an executable Tool that contains the Components
# ----------------------------------------------------------
#
# Note that all the components we wish to be configured via the tool must
# be added to the ``classes`` attribute.
#


class MyTool(Tool):
    name = "mytool"
    description = "do some things and stuff"
    aliases = dict(
        infile="AdvancedComponent.infile",
        outfile="AdvancedComponent.outfile",
        iterations="MyTool.iterations",
    )

    # Which classes are registered for configuration
    classes = [
        MyComponent,
        AdvancedComponent,
        SecondaryMyComponent,
        TelescopeWiseComponent,
    ]

    # local configuration parameters
    iterations = Integer(5, help="Number of times to run", allow_none=False).tag(
        config=True
    )

    def setup(self):
        self.comp = MyComponent(parent=self)
        self.comp2 = SecondaryMyComponent(parent=self)
        self.comp3 = TelescopeWiseComponent(parent=self, subarray=subarray)
        self.advanced = AdvancedComponent(parent=self)

    def start(self):
        self.log.info("Performing {} iterations...".format(self.iterations))
        for ii in range(self.iterations):
            self.log.info("ITERATION {}".format(ii))
            self.comp.do_thing()
            self.comp2.do_thing()
            sleep(0.1)

    def finish(self):
        self.log.warning("Shutting down.")


######################################################################
# Get Help info
# -------------
#
# The following allows you to print the help info within a Jupyter
# notebook, but this same information would be displayed if the user
# types:
#
# ::
#
#      mytool --help
#

tool = MyTool()
tool

######################################################################
tool.print_help()


######################################################################
# The following is equivalent to the user typing ``mytool --help-all``
#

tool.print_help(classes=True)


######################################################################
# Run the tool
# ------------
#
# here we pass in argv since it is a Notebook, but if argv is not
# specified it’s read from ``sys.argv``, so the following is the same as
# running:
#
#
#    mytool --log_level=INFO --infile gamma_test.simtel.gz --iterations=3
#
# As Tools are intended to be executables, they are raising
# ``SystemExit`` on exit. Here, we use them to demonstrate how it would
# work, so we catch the ``SystemExit``.
#

try:
    tool.run(argv=["--infile", str(GAMMA_FILE), "--outfile", "out.csv"])
except SystemExit as e:
    assert e.code == 0, f"Tool returned with error status {e}"

tool.log_format = "%(asctime)s : %(levelname)s [%(name)s %(funcName)s] %(message)s"


######################################################################
try:
    tool.run(
        argv=[
            "--log-level",
            "INFO",
            "--infile",
            str(GAMMA_FILE),
            "--outfile",
            "out.csv",
            "--iterations",
            "3",
        ]
    )
except SystemExit as e:
    assert e.code == 0, f"Tool returned with error status {e}"


######################################################################
# here we change the log-level to DEBUG:
#

try:
    tool.run(
        argv=[
            "--log-level",
            "DEBUG",
            "--infile",
            str(GAMMA_FILE),
            "--outfile",
            "out.csv",
        ]
    )
except SystemExit as e:
    assert e.code == 0, f"Tool returned with error status {e}"


######################################################################
# you can also set parameters directly in the class, rather than using the
# argument/configfile parser. This is useful if you are calling the Tool
# from a script rather than the command-line
#

tool.iterations = 1
tool.log_level = 0

try:
    tool.run(["--infile", str(GAMMA_FILE), "--outfile", "out.csv"])
except SystemExit as e:
    assert e.code == 0, f"Tool returned with error status {e}"


######################################################################
# see what happens when a value is set that is not of the correct type:
#

try:
    tool.iterations = "badval"
except TraitError as E:
    print("bad value:", E)
except SystemExit as e:
    assert e.code == 0, f"Tool returned with error status {e}"


######################################################################
# Example of what happens when you change a parameter that is being
# “observed” in a class. It’s handler is called:
#

tool.advanced.outfile = "Another.txt"


######################################################################
# we see that the handler for ``outfile`` was called, and it receive a
# change dict that shows the old and new values.
#


######################################################################
# create a tool using a config file:
#

tool2 = MyTool()

######################################################################
try:
    tool2.run(argv=["--config", "config.json"])
except SystemExit as e:
    assert e.code == 0, f"Tool returned with error status {e}"

######################################################################
print(tool2.advanced.infile)

######################################################################
print(tool2.config)

######################################################################
tool2.is_setup

######################################################################
tool3 = MyTool()

######################################################################
tool3.is_setup

######################################################################
tool3.initialize(argv=[])

######################################################################
tool3.is_setup

######################################################################
tool3

######################################################################
tool.setup()
tool

######################################################################
tool.comp2


######################################################################
# Getting the configuration of an instance
# ----------------------------------------
#

tool.get_current_config()

######################################################################
tool.iterations = 12
tool.get_current_config()


######################################################################
# Writing a Sample Config File
# ----------------------------
#

print(tool.generate_config_file())
