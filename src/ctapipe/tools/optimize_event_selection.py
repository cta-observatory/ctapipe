"""Tool to generate selections for IRFs production"""
import astropy.units as u
from astropy.table import vstack

from ..core import Provenance, Tool, traits
from ..core.traits import Float, Integer, Unicode
from ..irf import (
    PYIRF_SPECTRA,
    EventPreProcessor,
    EventsLoader,
    FovOffsetBinning,
    GridOptimizer,
    OutputEnergyBinning,
    Spectra,
    ThetaCutsCalculator,
)


class IrfEventSelector(Tool):
    name = "ctapipe-optimize-event-selection"
    description = "Tool to create optimized cuts for IRF generation"

    gamma_file = traits.Path(
        default_value=None, directory_ok=False, help="Gamma input filename and path"
    ).tag(config=True)
    gamma_sim_spectrum = traits.UseEnum(
        Spectra,
        default_value=Spectra.CRAB_HEGRA,
        help="Name of the pyrif spectra used for the simulated gamma spectrum",
    ).tag(config=True)
    proton_file = traits.Path(
        default_value=None, directory_ok=False, help="Proton input filename and path"
    ).tag(config=True)
    proton_sim_spectrum = traits.UseEnum(
        Spectra,
        default_value=Spectra.IRFDOC_PROTON_SPECTRUM,
        help="Name of the pyrif spectra used for the simulated proton spectrum",
    ).tag(config=True)
    electron_file = traits.Path(
        default_value=None, directory_ok=False, help="Electron input filename and path"
    ).tag(config=True)
    electron_sim_spectrum = traits.UseEnum(
        Spectra,
        default_value=Spectra.IRFDOC_ELECTRON_SPECTRUM,
        help="Name of the pyrif spectra used for the simulated electron spectrum",
    ).tag(config=True)

    chunk_size = Integer(
        default_value=100000,
        allow_none=True,
        help="How many subarray events to load at once when preselecting events.",
    ).tag(config=True)

    output_path = traits.Path(
        default_value="./Selection_Cuts.fits",
        allow_none=False,
        directory_ok=False,
        help="Output file storing optimization result",
    ).tag(config=True)

    obs_time = Float(default_value=50.0, help="Observation time").tag(config=True)
    obs_time_unit = Unicode(
        default_value="hour",
        help="Unit used to specify observation time as an astropy unit string.",
    ).tag(config=True)

    alpha = Float(
        default_value=0.2, help="Ratio between size of on and off regions"
    ).tag(config=True)

    aliases = {
        "gamma-file": "IrfEventSelector.gamma_file",
        "proton-file": "IrfEventSelector.proton_file",
        "electron-file": "IrfEventSelector.electron_file",
        "output": "IrfEventSelector.output_path",
        "chunk_size": "IrfEventSelector.chunk_size",
    }

    classes = [GridOptimizer, FovOffsetBinning, OutputEnergyBinning, EventPreProcessor]

    def setup(self):
        self.go = GridOptimizer(parent=self)
        self.theta = ThetaCutsCalculator(parent=self)
        self.e_bins = OutputEnergyBinning(parent=self)
        self.bins = FovOffsetBinning(parent=self)
        self.epp = EventPreProcessor(parent=self)

        self.reco_energy_bins = self.e_bins.reco_energy_bins()
        self.true_energy_bins = self.e_bins.true_energy_bins()

        self.fov_offset_bins = self.bins.fov_offset_bins()

        self.particles = [
            EventsLoader(
                self.epp,
                "gammas",
                self.gamma_file,
                PYIRF_SPECTRA[self.gamma_sim_spectrum],
            ),
            EventsLoader(
                self.epp,
                "protons",
                self.proton_file,
                PYIRF_SPECTRA[self.proton_sim_spectrum],
            ),
            EventsLoader(
                self.epp,
                "electrons",
                self.electron_file,
                PYIRF_SPECTRA[self.electron_sim_spectrum],
            ),
        ]

    def start(self):
        # TODO: this event loading code seems to be largely repeated between all the tools,
        # try to refactor to a common solution

        reduced_events = dict()
        for sel in self.particles:
            evs, cnt, meta = sel.load_preselected_events(
                self.chunk_size, self.obs_time * u.Unit(self.obs_time_unit), self.bins
            )
            reduced_events[sel.kind] = evs
            reduced_events[f"{sel.kind}_count"] = cnt
            if sel.kind == "gammas":
                self.sim_info = meta["sim_info"]
                self.gamma_spectrum = meta["spectrum"]

        self.log.debug(
            "Loaded %d gammas, %d protons, %d electrons"
            % (
                reduced_events["gammas_count"],
                reduced_events["protons_count"],
                reduced_events["electrons_count"],
            )
        )
        self.log.debug(
            "Keeping %d gammas, %d protons, %d electrons"
            % (
                len(reduced_events["gammas"]),
                len(reduced_events["protons"]),
                len(reduced_events["electrons"]),
            )
        )
        self.signal_events = reduced_events["gammas"]
        self.background_events = vstack(
            [reduced_events["protons"], reduced_events["electrons"]]
        )

        self.log.info(
            "Optimizing cuts using %d signal and %d background events"
            % (len(self.signal_events), len(self.background_events)),
        )
        result, ope_sens = self.go.optimize_gh_cut(
            self.signal_events,
            self.background_events,
            self.alpha,
            self.bins.fov_offset_min * u.deg,
            self.bins.fov_offset_max * u.deg,
            self.theta,
            self.epp,
        )

        self.log.info("Writing results to %s" % self.output_path)
        Provenance().add_output_file(self.output_path, role="Optimization Result")
        result.write(self.output_path, self.overwrite)


def main():
    tool = IrfEventSelector()
    tool.run()


if __name__ == "main":
    main()
