"""
Detect and extract muon ring parameters, and write the muon ring and
intensity parameters to an output table.

The resulting output can be read e.g. using for example
`pandas.read_hdf(filename, 'muons/LSTCam')`
"""

import warnings
from collections import defaultdict

from tqdm import tqdm

from ctapipe.calib import CameraCalibrator
from ctapipe.core import Provenance
from ctapipe.core import Tool, ToolConfigurationError
from ctapipe.core import traits as t
from ctapipe.image.muon.muon_diagnostic_plots import plot_muon_event
from ctapipe.image.muon.muon_reco_functions import analyze_muon_event
from ctapipe.io import EventSource, event_source
from ctapipe.io import HDF5TableWriter

warnings.filterwarnings("ignore")  # Supresses iminuit warnings


def _exclude_some_columns(subarray, writer):
    """ a hack to exclude some columns of all output tables here we exclude
    the prediction and mask quantities, since they are arrays and thus not
    readable by pandas.  Also, prediction currently is a variable-length
    quantity (need to change it to be fixed-length), so it cannot be written
    to a fixed-length table.
    """
    all_camids = {str(x.camera) for x in subarray.tel.values()}
    for cam in all_camids:
        writer.exclude(cam, 'prediction')
        writer.exclude(cam, 'mask')


class MuonDisplayerTool(Tool):
    name = 'ctapipe-reconstruct-muons'
    description = t.Unicode(__doc__)

    events = t.Unicode("",
                       help="input event data file").tag(config=True)

    outfile = t.Unicode("muons.hdf5", help='HDF5 output file name').tag(
        config=True)

    display = t.Bool(
        help='display the camera events', default=False
    ).tag(config=True)

    classes = t.List([
        CameraCalibrator, EventSource
    ])

    aliases = t.Dict({
        'input': 'MuonDisplayerTool.events',
        'outfile': 'MuonDisplayerTool.outfile',
        'display': 'MuonDisplayerTool.display',
        'max_events': 'EventSource.max_events',
        'allowed_tels': 'EventSource.allowed_tels',
    })

    def setup(self):
        if self.events == '':
            raise ToolConfigurationError("please specify --input <events file>")
        self.log.debug("input: %s", self.events)
        self.source = event_source(self.events)
        self.calib = CameraCalibrator(
            config=self.config, parent=self, eventsource=self.source
        )
        self.writer = HDF5TableWriter(self.outfile, "muons")

    def start(self):

        numev = 0
        self.num_muons_found = defaultdict(int)

        for event in tqdm(self.source, desc='detecting muons'):

            self.calib.calibrate(event)
            muon_evt = analyze_muon_event(event)

            if numev == 0:
                _exclude_some_columns(event.inst.subarray, self.writer)

            numev += 1

            if not muon_evt['MuonIntensityParams']:
                # No telescopes  contained a good muon
                continue
            else:
                if self.display:
                    plot_muon_event(event, muon_evt)

                for tel_id in muon_evt['TelIds']:
                    idx = muon_evt['TelIds'].index(tel_id)
                    intens_params = muon_evt['MuonIntensityParams'][idx]

                    if intens_params is not None:
                        ring_params = muon_evt['MuonRingParams'][idx]
                        cam_id = str(event.inst.subarray.tel[tel_id].camera)
                        self.num_muons_found[cam_id] += 1
                        self.log.debug("INTENSITY: %s", intens_params)
                        self.log.debug("RING: %s", ring_params)
                        self.writer.write(table_name=cam_id,
                                          containers=[intens_params,
                                                      ring_params])

                self.log.info(
                    "Event Number: %d, found %s muons",
                    numev, dict(self.num_muons_found)
                )

    def finish(self):
        Provenance().add_output_file(self.outfile,
                                     role='dl1.tel.evt.muon')
        self.writer.close()


def main():
    tool = MuonDisplayerTool()
    tool.run()
