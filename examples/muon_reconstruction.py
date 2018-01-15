"""
Example to load raw data (hessio format), calibrate and reconstruct muon
ring parameters, and write some parameters to an output table
"""

import warnings
from astropy.table import Table
from ctapipe.calib import CameraCalibrator
from ctapipe.core import Tool
from ctapipe.core import traits as t
from ctapipe.image.muon.muon_diagnostic_plots import plot_muon_event
from ctapipe.image.muon.muon_reco_functions import analyze_muon_event
from ctapipe.io.hessio import hessio_event_source
from ctapipe.utils import get_dataset

warnings.filterwarnings("ignore")  # Supresses iminuit warnings


def print_muon(event, printer=print):
    for tid in event['TelIds']:
        idx = event['TelIds'].index(tid)
        if event['MuonIntensityParams'][idx]:
            printer("MUON: Run ID {} Event ID {} \
                    Impact Parameter {} Ring Width {} Optical Efficiency {}".format(
                event['MuonRingParams'][idx].run_id,
                event['MuonRingParams'][idx].event_id,
                event['MuonIntensityParams'][idx].impact_parameter,
                event['MuonIntensityParams'][idx].ring_width,
                event['MuonIntensityParams'][idx].optical_efficiency_muon)
            )
    pass


class MuonDisplayerTool(Tool):
    name = 'ctapipe-display-muons'
    description = t.Unicode(__doc__)

    infile = t.Unicode(
        help='input file name',
        default=get_dataset('gamma_test_large.simtel.gz')
    ).tag(config=True)

    outfile = t.Unicode(help='output file name',
                        default=None).tag(config=True)

    display = t.Bool(
        help='display the camera events', default=False
    ).tag(config=True)

    classes = t.List([CameraCalibrator, ])

    aliases = t.Dict({'infile': 'MuonDisplayerTool.infile',
                      'outfile': 'MuonDisplayerTool.outfile',
                      'display': 'MuonDisplayerTool.display'
                      })


    def setup(self):
        self.calib = CameraCalibrator(config=self.config, tool=self)

    def start(self):

        output_parameters = {'MuonEff': [],
                             'ImpactP': [],
                             'RingWidth': []}

        numev = 0
        num_muons_found = 0

        for event in hessio_event_source(self.infile):
            self.log.info("Event Number: %d, found %d muons across all events",
                          numev, num_muons_found)

            self.calib.calibrate(event)
            muon_evt = analyze_muon_event(event)

            numev += 1

            if not muon_evt['MuonIntensityParams']:  # No telescopes contained a good muon
                continue
            else:
                if self.display:
                    plot_muon_event(event, muon_evt)

                for tid in muon_evt['TelIds']:
                    idx = muon_evt['TelIds'].index(tid)
                    if muon_evt['MuonIntensityParams'][idx] is not None:
                        self.log.info("** Muon params: %s",
                                      muon_evt['MuonIntensityParams'][idx])

                        output_parameters['MuonEff'].append(
                            muon_evt['MuonIntensityParams'][idx].optical_efficiency_muon
                        )
                        output_parameters['ImpactP'].append(
                            muon_evt['MuonIntensityParams'][idx].impact_parameter.value
                        )
                        output_parameters['RingWidth'].append(
                            muon_evt['MuonIntensityParams'][idx].ring_width.value
                        )
                        print_muon(muon_evt, printer=self.log.info)
                        num_muons_found += 1



        t = Table(output_parameters)
        t['ImpactP'].unit = 'm'
        t['RingWidth'].unit = 'deg'
        if self.outfile:
            t.write(self.outfile)

if __name__ == '__main__':
    tool = MuonDisplayerTool()
    tool.run()
