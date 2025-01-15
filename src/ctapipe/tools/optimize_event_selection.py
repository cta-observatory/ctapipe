"""Tool to generate selections for IRFs production"""

import astropy.units as u
from astropy.table import vstack

from ..core import Provenance, Tool, traits
from ..core.traits import AstroQuantity, Integer, classes_with_traits
from ..irf import EventLoader, Spectra
from ..irf.optimize import CutOptimizerBase

__all__ = ["EventSelectionOptimizer"]


class EventSelectionOptimizer(Tool):
    "Tool to create optimized cuts for IRF generation"

    name = "ctapipe-optimize-event-selection"
    description = __doc__
    examples = """
    ctapipe-optimize-event-selection \\
        --gamma-file gamma.dl2.h5 \\
        --proton-file proton.dl2.h5 \\
        --electron-file electron.dl2.h5 \\
        --output cuts.fits
    """

    gamma_file = traits.Path(
        default_value=None, directory_ok=False, help="Gamma input filename and path"
    ).tag(config=True)

    gamma_target_spectrum = traits.UseEnum(
        Spectra,
        default_value=Spectra.CRAB_HEGRA,
        help="Name of the spectrum used for weights of gamma events.",
    ).tag(config=True)

    proton_file = traits.Path(
        default_value=None,
        directory_ok=False,
        allow_none=True,
        help=(
            "Proton input filename and path. "
            "Not needed, if ``optimization_algorithm = 'PercentileCuts'``."
        ),
    ).tag(config=True)

    proton_target_spectrum = traits.UseEnum(
        Spectra,
        default_value=Spectra.IRFDOC_PROTON_SPECTRUM,
        help="Name of the spectrum used for weights of proton events.",
    ).tag(config=True)

    electron_file = traits.Path(
        default_value=None,
        directory_ok=False,
        allow_none=True,
        help=(
            "Electron input filename and path. "
            "Not needed, if ``optimization_algorithm = 'PercentileCuts'``."
        ),
    ).tag(config=True)

    electron_target_spectrum = traits.UseEnum(
        Spectra,
        default_value=Spectra.IRFDOC_ELECTRON_SPECTRUM,
        help="Name of the spectrum used for weights of electron events.",
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
        default_value=u.Quantity(50, u.hour),
        physical_type=u.physical.time,
        help=(
            "Observation time in the form ``<value> <unit>``."
            " This is used for flux normalization when calculating sensitivities."
        ),
    ).tag(config=True)

    optimization_algorithm = traits.ComponentName(
        CutOptimizerBase,
        default_value="PointSourceSensitivityOptimizer",
        help="The cut optimization algorithm to be used.",
    ).tag(config=True)

    aliases = {
        "gamma-file": "EventSelectionOptimizer.gamma_file",
        "proton-file": "EventSelectionOptimizer.proton_file",
        "electron-file": "EventSelectionOptimizer.electron_file",
        "output": "EventSelectionOptimizer.output_path",
        "chunk_size": "EventSelectionOptimizer.chunk_size",
    }

    classes = [EventLoader] + classes_with_traits(CutOptimizerBase)

    def setup(self):
        """
        Initialize components from config.
        """
        self.optimizer = CutOptimizerBase.from_name(
            self.optimization_algorithm, parent=self
        )
        self.particles = [
            EventLoader(
                parent=self,
                kind="gammas",
                file=self.gamma_file,
                target_spectrum=self.gamma_target_spectrum,
            )
        ]
        if self.optimization_algorithm != "PercentileCuts":
            if not self.proton_file or (
                self.proton_file and not self.proton_file.exists()
            ):
                raise ValueError(
                    "Need a proton file for cut optimization "
                    f"using {self.optimization_algorithm}."
                )

            self.particles.append(
                EventLoader(
                    parent=self,
                    kind="protons",
                    file=self.proton_file,
                    target_spectrum=self.proton_target_spectrum,
                )
            )
            if self.electron_file and self.electron_file.exists():
                self.particles.append(
                    EventLoader(
                        parent=self,
                        kind="electrons",
                        file=self.electron_file,
                        target_spectrum=self.electron_target_spectrum,
                    )
                )
            else:
                self.log.warning("Optimizing cuts without electron file.")

    def start(self):
        """
        Load events and optimize g/h (and theta) cuts.
        """
        reduced_events = dict()
        for sel in self.particles:
            evs, cnt, meta = sel.load_preselected_events(self.chunk_size, self.obs_time)
            if self.optimization_algorithm == "PointSourceSensitivityOptimizer":
                evs = sel.make_event_weights(
                    evs,
                    meta["spectrum"],
                    (
                        self.optimizer.min_bkg_fov_offset,
                        self.optimizer.max_bkg_fov_offset,
                    ),
                )

            reduced_events[sel.kind] = evs
            reduced_events[f"{sel.kind}_count"] = cnt
            if sel.kind == "gammas":
                self.sim_info = meta["sim_info"]
                self.gamma_spectrum = meta["spectrum"]

        self.signal_events = reduced_events["gammas"]

        if self.optimization_algorithm == "PercentileCuts":
            self.log.debug("Loaded %d gammas" % reduced_events["gammas_count"])
            self.log.debug("Keeping %d gammas" % len(reduced_events["gammas"]))
            self.log.info("Optimizing cuts using %d signal" % len(self.signal_events))
        else:
            if "electrons" not in reduced_events.keys():
                reduced_events["electrons"] = []
                reduced_events["electrons_count"] = 0
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
            self.background_events = vstack(
                [reduced_events["protons"], reduced_events["electrons"]]
            )
            self.log.info(
                "Optimizing cuts using %d signal and %d background events"
                % (len(self.signal_events), len(self.background_events)),
            )

        result = self.optimizer(
            signal=self.signal_events,
            background=self.background_events
            if self.optimization_algorithm != "PercentileCuts"
            else None,
            precuts=self.particles[0].epp,  # identical precuts for all particle types
            clf_prefix=self.particles[0].epp.gammaness_classifier,
        )
        self.result = result

    def finish(self):
        """
        Write optimized cuts to the output file.
        """
        self.log.info("Writing results to %s" % self.output_path)
        Provenance().add_output_file(self.output_path, role="Optimization Result")
        self.result.write(self.output_path, self.overwrite)


def main():
    tool = EventSelectionOptimizer()
    tool.run()


if __name__ == "main":
    main()
