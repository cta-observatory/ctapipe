"""Tool to generate selections for IRFs production"""
import astropy.units as u
from astropy.table import vstack

from ..core import Provenance, Tool, traits
from ..core.traits import AstroQuantity, Bool, Float, Integer, flag
from ..irf import (
    SPECTRA,
    EventsLoader,
    FovOffsetBinning,
    GridOptimizer,
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

    point_like = Bool(
        False,
        help=(
            "Optimize both G/H separation cut and theta cut"
            " for computing point-like IRFs"
        ),
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
            "point-like",
            "IrfEventSelector.point_like",
            "Optimize both G/H separation cut and theta cut.",
            "Optimize G/H separation cut without prior theta cut.",
        )
    }

    classes = [GridOptimizer, ThetaCutsCalculator, FovOffsetBinning, EventsLoader]

    def setup(self):
        self.go = GridOptimizer(parent=self)
        self.theta = ThetaCutsCalculator(parent=self)
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
        result, ope_sens = self.go.optimize_gh_cut(
            self.signal_events,
            self.background_events,
            self.alpha,
            self.bins.fov_offset_min * u.deg,
            self.bins.fov_offset_max * u.deg,
            self.theta,
            self.particles[0].epp,  # precuts are the same for all particle types
            self.point_like,
        )

        self.log.info("Writing results to %s" % self.output_path)
        if not self.point_like:
            self.log.info("Writing dummy theta cut to %s" % self.output_path)
        Provenance().add_output_file(self.output_path, role="Optimization Result")
        result.write(self.output_path, self.overwrite)


def main():
    tool = IrfEventSelector()
    tool.run()


if __name__ == "main":
    main()
