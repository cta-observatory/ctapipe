"""Tool to generate selections for IRFs production"""

import astropy.units as u
from astropy.table import vstack

from ..core import Provenance, Tool, traits
from ..core.traits import AstroQuantity, Bool, Float, Integer, classes_with_traits, flag
from ..irf import EventLoader, Spectra
from ..irf.optimize import CutOptimizerBase


class IrfEventSelector(Tool):
    name = "ctapipe-optimize-event-selection"
    description = "Tool to create optimized cuts for IRF generation"

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
        help="Observation time in the form ``<value> <unit>``",
    ).tag(config=True)

    alpha = Float(
        default_value=0.2, help="Ratio between size of on and off regions."
    ).tag(config=True)

    optimization_algorithm = traits.ComponentName(
        CutOptimizerBase,
        default_value="PointSourceSensitivityOptimizer",
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

    classes = [EventLoader] + classes_with_traits(CutOptimizerBase)

    def setup(self):
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
            self.particles.append(
                EventLoader(
                    parent=self,
                    kind="protons",
                    file=self.proton_file,
                    target_spectrum=self.proton_target_spectrum,
                )
            )
            self.particles.append(
                EventLoader(
                    parent=self,
                    kind="electrons",
                    file=self.electron_file,
                    target_spectrum=self.electron_target_spectrum,
                )
            )

    def start(self):
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

        result = self.optimizer.optimize_cuts(
            signal=self.signal_events,
            background=self.background_events
            if self.optimization_algorithm != "PercentileCuts"
            else None,
            alpha=self.alpha,
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
