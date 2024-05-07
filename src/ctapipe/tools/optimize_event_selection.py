"""Tool to generate selections for IRFs production"""
import astropy.units as u
from astropy.table import vstack

from ..core import Provenance, Tool, traits
from ..core.traits import AstroQuantity, Bool, Float, Integer, flag
from ..irf import (
    SPECTRA,
    CutOptimizerBase,
    EventsLoader,
    FovOffsetBinning,
    Spectra,
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
        help="Name of the pyirf spectra used for the simulated gamma spectrum",
    ).tag(config=True)

    proton_file = traits.Path(
        default_value=None, directory_ok=False, help="Proton input filename and path"
    ).tag(config=True)

    proton_sim_spectrum = traits.UseEnum(
        Spectra,
        default_value=Spectra.IRFDOC_PROTON_SPECTRUM,
        help="Name of the pyirf spectra used for the simulated proton spectrum",
    ).tag(config=True)

    electron_file = traits.Path(
        default_value=None, directory_ok=False, help="Electron input filename and path"
    ).tag(config=True)

    electron_sim_spectrum = traits.UseEnum(
        Spectra,
        default_value=Spectra.IRFDOC_ELECTRON_SPECTRUM,
        help="Name of the pyirf spectra used for the simulated electron spectrum",
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

    obs_time = AstroQuantity(
        default_value=50.0 * u.hour,
        physical_type=u.physical.time,
        help="Observation time in the form ``<value> <unit>``",
    ).tag(config=True)

    alpha = Float(
        default_value=0.2, help="Ratio between size of on and off regions."
    ).tag(config=True)

    optimization_algorithm = traits.ComponentName(
        CutOptimizerBase,
        default_value="GridOptimizer",
        help="The cut optimization algorithm to be used.",
    ).tag(config=True)

    full_enclosure = Bool(
        False,
        help="Compute only the G/H separation cut needed for full enclosure IRF.",
    ).tag(config=True)

    aliases = {
        "gamma-file": "IrfEventSelector.gamma_file",
        "proton-file": "IrfEventSelector.proton_file",
        "electron-file": "IrfEventSelector.electron_file",
        "output": "IrfEventSelector.output_path",
        "chunk_size": "IrfEventSelector.chunk_size",
    }

    flags = {
        **flag(
            "full-enclosure",
            "IrfEventSelector.full_enclosure",
            "Compute only the G/H separation cut.",
            "Compute the G/H separation cut and the theta cut.",
        )
    }

    classes = [CutOptimizerBase, FovOffsetBinning, EventsLoader]

    def setup(self):
        self.optimizer = CutOptimizerBase.from_name(
            self.optimization_algorithm, parent=self
        )
        self.bins = FovOffsetBinning(parent=self)

        self.particles = [
            EventsLoader(
                parent=self,
                kind="gammas",
                file=self.gamma_file,
                target_spectrum=SPECTRA[self.gamma_sim_spectrum],
            ),
            EventsLoader(
                parent=self,
                kind="protons",
                file=self.proton_file,
                target_spectrum=SPECTRA[self.proton_sim_spectrum],
            ),
            EventsLoader(
                parent=self,
                kind="electrons",
                file=self.electron_file,
                target_spectrum=SPECTRA[self.electron_sim_spectrum],
            ),
        ]

    def start(self):
        # TODO: this event loading code seems to be largely repeated between all the tools,
        # try to refactor to a common solution

        reduced_events = dict()
        for sel in self.particles:
            evs, cnt, meta = sel.load_preselected_events(
                self.chunk_size, self.obs_time, self.bins
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
        result = self.optimizer.optimize_cuts(
            signal=self.signal_events,
            background=self.background_events,
            alpha=self.alpha,
            min_fov_radius=self.bins.fov_offset_min,
            max_fov_radius=self.bins.fov_offset_max,
            precuts=self.particles[0].epp,  # identical precuts for all particle types
            clf_prefix=self.particles[0].epp.gammaness_classifier,
            point_like=not self.full_enclosure,
        )
        self.result = result

    def finish(self):
        self.log.info("Writing results to %s" % self.output_path)
        Provenance().add_output_file(self.output_path, role="Optimization Result")
        self.result.write(self.output_path, self.overwrite)


def main():
    tool = IrfEventSelector()
    tool.run()


if __name__ == "main":
    main()
