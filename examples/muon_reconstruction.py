"""
Example to load raw data (hessio format), calibrate and reconstruct muon
ring parameters
"""



import argparse
import os

from astropy import log
from astropy.table import Table

from ctapipe.calib import CameraCalibrator
from ctapipe.image.muon.muon_diagnostic_plots import plot_muon_efficiency, \
    plot_muon_event
from ctapipe.image.muon.muon_reco_functions import analyze_muon_event
from ctapipe.io.hessio import hessio_event_source
from ctapipe.utils import get_dataset
from ctapipe.core import Tool
from ctapipe.core import traits as t



def print_muon(event):
    for tid in event['TelIds']:
        idx = event['TelIds'].index(tid)
        if event['MuonIntensityParams'][idx]:
            print("MUON:", event['MuonRingParams'][idx].run_id, event['MuonRingParams'][idx].event_id,
                  event['MuonIntensityParams'][idx].impact_parameter, event['MuonIntensityParams'][idx].ring_width, "mu_eff=",
                  event['MuonIntensityParams'][idx].optical_efficiency_muon)
    pass


class MuonDisplayerTool(Tool):
    name = 'ctapipe-display-muons'
    description = t.Unicode(__doc__)

    infile = t.Unicode(
        help='input file name',
        default=get_dataset('gamma_test_large.simtel.gz')
    ).tag(config=True)

    outfile = t.Unicode(help='output file name',
                        default=None)).tag(config=True)

    display = t.Bool(
        help='display the camera events', default=False
    ).tag(config=True)

    classes = t.List([CameraCalibrator,])

    aliases = t.Dict({'infile': 'MuonDisplayerTool.infile',
                      'outfile': 'MuonDisplayerTool.outfile',
                      'display' : 'MuonDisplayerTool.display'
                      })


    def setup(self):
        self.calib = CameraCalibrator(config=self.config, tool=self)

    def start(self):

        muoneff = []
        impactp = []
        ringwidth = []
        plot_dict = {'MuonEff': muoneff, 'ImpactP': impactp,
                     'RingWidth': ringwidth}

        numev = 0

        source = hessio_event_source(self.infile)

        for event in source:
            self.log.info("Event Number: %d", numev)

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
                    if not muon_evt['MuonIntensityParams'][idx]:
                        continue

                    self.log.info("** Muon params: %s", muon_evt[idx])

                    plot_dict['MuonEff'].append(
                        muon_evt['MuonIntensityParams'][idx].optical_efficiency_muon
                    )
                    plot_dict['ImpactP'].append(
                        muon_evt['MuonIntensityParams'][idx].impact_parameter.value
                    )
                    plot_dict['RingWidth'].append(
                        muon_evt['MuonIntensityParams'][idx].ring_width.value
                    )

                    print_muon(muon_evt)


        t = Table(plot_dict)
        t['ImpactP'].unit = 'm'
        t['RingWidth'].unit = 'deg'
        if self.outfile:
            t.write(self.outfile)

if __name__ == '__main__':
    tool = MuonDisplayerTool()
    tool.run()
