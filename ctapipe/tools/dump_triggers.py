"""
dump a table of the event times and trigger patterns from a
simtelarray input file.
"""

import numpy as np
from astropy import units as u
from astropy.table import Table

from ctapipe.io import EventSource
from ctapipe.core import Provenance, ToolConfigurationError
from ctapipe.core.traits import Unicode, Dict, Bool, Path
from ..core import Tool

MAX_TELS = 1000


class DumpTriggersTool(Tool):
    description = Unicode(__doc__)
    name = "ctapipe-dump-triggers"

    # =============================================
    # configuration parameters:
    # =============================================

    infile = Path(exists=True, directory_ok=False, help="input simtelarray file").tag(
        config=True
    )

    outfile = Path(
        default_value="triggers.fits",
        directory_ok=False,
        help="output filename (*.fits, *.h5)",
    ).tag(config=True)

    overwrite = Bool(False, help="overwrite existing output file").tag(config=True)

    # =============================================
    # map low-level options to high-level command-line options
    # =============================================

    aliases = Dict(
        {"infile": "DumpTriggersTool.infile", "outfile": "DumpTriggersTool.outfile"}
    )

    flags = Dict(
        {
            "overwrite": (
                {"DumpTriggersTool": {"overwrite": True}},
                "Enable overwriting of output file",
            )
        }
    )

    examples = (
        "ctapipe-dump-triggers --infile gamma.simtel.gz "
        "--outfile trig.fits --overwrite"
        "\n\n"
        "If you want to see more output, use --log_level=DEBUG"
    )

    # =============================================
    # The methods of the Tool (initialize, start, finish):
    # =============================================

    def add_event_to_table(self, event):
        """
        add the current hessio event to a row in the `self.events` table
        """
        time = event.trigger.time

        if self._prev_time is None:
            self._prev_time = time

        if self._current_starttime is None:
            self._current_starttime = time

        relative_time = time - self._current_starttime
        delta_t = time - self._prev_time
        self._prev_time = time

        # build the trigger pattern as a fixed-length array
        # (better for storage in FITS format)
        # trigtels = event.get_telescope_with_data_list()
        trigtels = event.dl0.tel.keys()
        self._current_trigpattern[:] = 0  # zero the trigger pattern
        self._current_trigpattern[list(trigtels)] = 1  # set the triggered tels
        # to 1

        # insert the row into the table
        self.events.add_row(
            (
                event.index.event_id,
                relative_time.sec,
                delta_t.sec,
                len(trigtels),
                self._current_trigpattern,
            )
        )

    def setup(self):
        """ setup function, called before `start()` """

        if self.infile == "":
            raise ToolConfigurationError("No 'infile' parameter was specified. ")

        self.events = Table(
            names=["EVENT_ID", "T_REL", "DELTA_T", "N_TRIG", "TRIGGERED_TELS"],
            dtype=[np.int64, np.float64, np.float64, np.int32, np.uint8],
        )

        self.events["TRIGGERED_TELS"].shape = (0, MAX_TELS)
        self.events["T_REL"].unit = u.s
        self.events["T_REL"].description = "Time relative to first event"
        self.events["DELTA_T"].unit = u.s
        self.events.meta["INPUT"] = self.infile

        self._current_trigpattern = np.zeros(MAX_TELS)
        self._current_starttime = None
        self._prev_time = None

    def start(self):
        """ main event loop """
        with EventSource(self.infile) as source:
            for event in source:
                self.add_event_to_table(event)

    def finish(self):
        """
        finish up and write out results (called automatically after
        `start()`)
        """
        # write out the final table
        try:
            if ".fits" in self.outfile.suffixes:
                self.events.write(self.outfile, overwrite=self.overwrite)
            elif self.outfile.suffix in (".hdf5", ".h5", ".hdf"):
                self.events.write(
                    self.outfile, path="/events", overwrite=self.overwrite
                )
            else:
                self.events.write(self.outfile)

            Provenance().add_output_file(self.outfile)
        except IOError as err:
            self.log.warning("Couldn't write output (%s)", err)

        self.log.info("\n %s", self.events)


def main():
    tool = DumpTriggersTool()
    tool.run()
