"""Tool to generate IRFs"""

import operator
from functools import partial

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.table import QTable, vstack
from pyirf.cuts import evaluate_binned_cut
from pyirf.io import create_rad_max_hdu

from ..core import Provenance, Tool, ToolConfigurationError, traits
from ..core.traits import AstroQuantity, Bool, Integer, classes_with_traits, flag
from ..irf import (
    EventLoader,
    EventPreProcessor,
    OptimizationResultStore,
    Spectra,
    check_bins_in_range,
)
from ..irf.benchmarks import (
    AngularResolutionMakerBase,
    EnergyBiasResolutionMakerBase,
    SensitivityMakerBase,
)
from ..irf.irfs import (
    BackgroundRateMakerBase,
    EffectiveAreaMakerBase,
    EnergyDispersionMakerBase,
    PsfMakerBase,
)


class IrfTool(Tool):
    name = "ctapipe-make-irf"
    description = "Tool to create IRF files in GADF format"

    do_background = Bool(
        True,
        help="Compute background rate using supplied files.",
    ).tag(config=True)

    range_check_error = Bool(
        False,
        help="Raise error if asking for IRFs outside range where cut optimisation is valid.",
    ).tag(config=True)

    cuts_file = traits.Path(
        default_value=None,
        directory_ok=False,
        help="Path to optimized cuts input file.",
    ).tag(config=True)

    gamma_file = traits.Path(
        default_value=None, directory_ok=False, help="Gamma input filename and path."
    ).tag(config=True)

    gamma_target_spectrum = traits.UseEnum(
        Spectra,
        default_value=Spectra.CRAB_HEGRA,
        help="Name of the spectrum used for weights of gamma events.",
    ).tag(config=True)

    proton_file = traits.Path(
        default_value=None,
        allow_none=True,
        directory_ok=False,
        help="Proton input filename and path.",
    ).tag(config=True)

    proton_target_spectrum = traits.UseEnum(
        Spectra,
        default_value=Spectra.IRFDOC_PROTON_SPECTRUM,
        help="Name of the spectrum used for weights of proton events.",
    ).tag(config=True)

    electron_file = traits.Path(
        default_value=None,
        allow_none=True,
        directory_ok=False,
        help="Electron input filename and path.",
    ).tag(config=True)

    electron_target_spectrum = traits.UseEnum(
        Spectra,
        default_value=Spectra.IRFDOC_ELECTRON_SPECTRUM,
        help="Name of the spectrum used for weights of electron events.",
    ).tag(config=True)

    chunk_size = Integer(
        default_value=100000,
        allow_none=True,
        help="How many subarray events to load at once while selecting.",
    ).tag(config=True)

    output_path = traits.Path(
        default_value="./IRF.fits.gz",
        allow_none=False,
        directory_ok=False,
        help="Output file",
    ).tag(config=True)

    benchmarks_output_path = traits.Path(
        default_value=None,
        allow_none=True,
        directory_ok=False,
        help="Optional second output file for benchmarks.",
    ).tag(config=True)

    obs_time = AstroQuantity(
        default_value=u.Quantity(50, u.hour),
        physical_type=u.physical.time,
        help=(
            "Observation time in the form ``<value> <unit>``."
            " This is used for flux normalization and estimating a background rate."
        ),
    ).tag(config=True)

    edisp_maker = traits.ComponentName(
        EnergyDispersionMakerBase,
        default_value="EnergyDispersion2dMaker",
        help="The parameterization of the energy dispersion to be used.",
    ).tag(config=True)

    aeff_maker = traits.ComponentName(
        EffectiveAreaMakerBase,
        default_value="EffectiveArea2dMaker",
        help="The parameterization of the effective area to be used.",
    ).tag(config=True)

    psf_maker = traits.ComponentName(
        PsfMakerBase,
        default_value="Psf3dMaker",
        help="The parameterization of the point spread function to be used.",
    ).tag(config=True)

    bkg_maker = traits.ComponentName(
        BackgroundRateMakerBase,
        default_value="BackgroundRate2dMaker",
        help="The parameterization of the background rate to be used.",
    ).tag(config=True)

    energy_bias_resolution_maker = traits.ComponentName(
        EnergyBiasResolutionMakerBase,
        default_value="EnergyBiasResolution2dMaker",
        help=(
            "The parameterization of the bias and resolution benchmark "
            "for the energy prediction."
        ),
    ).tag(config=True)

    angular_resolution_maker = traits.ComponentName(
        AngularResolutionMakerBase,
        default_value="AngularResolution2dMaker",
        help="The parameterization of the angular resolution benchmark.",
    ).tag(config=True)

    sensitivity_maker = traits.ComponentName(
        SensitivityMakerBase,
        default_value="Sensitivity2dMaker",
        help="The parameterization of the point source sensitivity benchmark.",
    ).tag(config=True)

    point_like = Bool(
        False,
        help=(
            "Compute a point-like IRF by applying a theta cut (``RAD_MAX``) "
            "which makes calculating a point spread function unnecessary."
        ),
    ).tag(config=True)

    aliases = {
        "cuts": "IrfTool.cuts_file",
        "gamma-file": "IrfTool.gamma_file",
        "proton-file": "IrfTool.proton_file",
        "electron-file": "IrfTool.electron_file",
        "output": "IrfTool.output_path",
        "benchmark-output": "IrfTool.benchmarks_output_path",
        "chunk_size": "IrfTool.chunk_size",
    }

    flags = {
        **flag(
            "do-background",
            "IrfTool.do_background",
            "Compute background rate.",
            "Do not compute background rate.",
        ),
        **flag(
            "point-like",
            "IrfTool.point_like",
            "Compute a point-like IRF.",
            "Compute a full-enclosure IRF.",
        ),
    }

    classes = (
        [
            EventLoader,
        ]
        + classes_with_traits(BackgroundRateMakerBase)
        + classes_with_traits(EffectiveAreaMakerBase)
        + classes_with_traits(EnergyDispersionMakerBase)
        + classes_with_traits(PsfMakerBase)
        + classes_with_traits(AngularResolutionMakerBase)
        + classes_with_traits(EnergyBiasResolutionMakerBase)
        + classes_with_traits(SensitivityMakerBase)
    )

    def setup(self):
        self.opt_result = OptimizationResultStore().read(self.cuts_file)

        if self.point_like and self.opt_result.theta_cuts is None:
            raise ToolConfigurationError(
                "Computing a point-like IRF requires an (optimized) theta cut."
            )

        check_e_bins = partial(
            check_bins_in_range,
            valid_range=self.opt_result.valid_energy,
            raise_error=self.range_check_error,
        )
        self.particles = [
            EventLoader(
                parent=self,
                kind="gammas",
                file=self.gamma_file,
                target_spectrum=self.gamma_target_spectrum,
            ),
        ]
        if self.do_background:
            if not self.proton_file:
                raise RuntimeError(
                    "At least a proton file required when specifying `do_background`."
                )

            self.particles.append(
                EventLoader(
                    parent=self,
                    kind="protons",
                    file=self.proton_file,
                    target_spectrum=self.proton_target_spectrum,
                )
            )
            if self.electron_file:
                self.particles.append(
                    EventLoader(
                        parent=self,
                        kind="electrons",
                        file=self.electron_file,
                        target_spectrum=self.electron_target_spectrum,
                    )
                )

            self.bkg = BackgroundRateMakerBase.from_name(self.bkg_maker, parent=self)
            check_e_bins(
                bins=self.bkg.reco_energy_bins, source="background reco energy"
            )

        self.edisp = EnergyDispersionMakerBase.from_name(self.edisp_maker, parent=self)
        self.aeff = EffectiveAreaMakerBase.from_name(self.aeff_maker, parent=self)

        if not self.point_like:
            self.psf = PsfMakerBase.from_name(self.psf_maker, parent=self)

        if self.benchmarks_output_path is not None:
            self.angular_resolution = AngularResolutionMakerBase.from_name(
                self.angular_resolution_maker, parent=self
            )
            if not self.angular_resolution.use_true_energy:
                check_e_bins(
                    bins=self.angular_resolution.reco_energy_bins,
                    source="Angular resolution energy",
                )

            self.bias_resolution = EnergyBiasResolutionMakerBase.from_name(
                self.energy_bias_resolution_maker, parent=self
            )
            self.sensitivity = SensitivityMakerBase.from_name(
                self.sensitivity_maker, parent=self
            )
            check_e_bins(
                bins=self.sensitivity.reco_energy_bins, source="Sensitivity reco energy"
            )

    def calculate_selections(self, reduced_events: dict) -> dict:
        """
        Add the selection columns to the signal and optionally background tables.

        Parameters
        ----------
        reduced_events: dict
            dict containing the signal (``"gammas"``) and optionally background
            tables (``"protons"``, ``"electrons"``)

        Returns
        -------
        dict
            ``reduced_events`` with selection columns added.
        """
        reduced_events["gammas"]["selected_gh"] = evaluate_binned_cut(
            reduced_events["gammas"]["gh_score"],
            reduced_events["gammas"]["reco_energy"],
            self.opt_result.gh_cuts,
            operator.ge,
        )
        if self.point_like:
            reduced_events["gammas"]["selected_theta"] = evaluate_binned_cut(
                reduced_events["gammas"]["theta"],
                reduced_events["gammas"]["reco_energy"],
                self.opt_result.theta_cuts,
                operator.le,
            )
            reduced_events["gammas"]["selected"] = (
                reduced_events["gammas"]["selected_theta"]
                & reduced_events["gammas"]["selected_gh"]
            )
        else:
            reduced_events["gammas"]["selected"] = reduced_events["gammas"][
                "selected_gh"
            ]

        if self.do_background:
            bkgs = ("protons", "electrons") if self.electron_file else ("protons")
            for bg_type in bkgs:
                reduced_events[bg_type]["selected_gh"] = evaluate_binned_cut(
                    reduced_events[bg_type]["gh_score"],
                    reduced_events[bg_type]["reco_energy"],
                    self.opt_result.gh_cuts,
                    operator.ge,
                )

        if self.do_background:
            self.log.info(
                "Keeping %d signal, %d proton events, and %d electron events"
                % (
                    np.count_nonzero(reduced_events["gammas"]["selected"]),
                    np.count_nonzero(reduced_events["protons"]["selected_gh"]),
                    np.count_nonzero(reduced_events["electrons"]["selected_gh"]),
                )
            )
        else:
            self.log.info(
                "Keeping %d signal events"
                % (np.count_nonzero(reduced_events["gammas"]["selected"]))
            )
        return reduced_events

    def _make_signal_irf_hdus(self, hdus, sim_info):
        hdus.append(
            self.aeff.make_aeff_hdu(
                events=self.signal_events[self.signal_events["selected"]],
                point_like=self.point_like,
                signal_is_point_like=self.signal_is_point_like,
                sim_info=sim_info,
            )
        )
        hdus.append(
            self.edisp.make_edisp_hdu(
                events=self.signal_events[self.signal_events["selected"]],
                point_like=self.point_like,
            )
        )
        if not self.point_like:
            hdus.append(
                self.psf.make_psf_hdu(
                    events=self.signal_events[self.signal_events["selected"]]
                )
            )
        else:
            # TODO: Support fov binning
            self.log.debug(
                "Currently multiple fov bins is not supported for RAD_MAX. "
                "Using `fov_offset_bins = [valid_offset.min, valid_offset.max]`."
            )
            hdus.append(
                create_rad_max_hdu(
                    rad_max=self.opt_result.theta_cuts["cut"].reshape(-1, 1),
                    reco_energy_bins=np.append(
                        self.opt_result.theta_cuts["low"],
                        self.opt_result.theta_cuts["high"][-1],
                    ),
                    fov_offset_bins=u.Quantity(
                        [
                            self.opt_result.valid_offset.min.to_value(u.deg),
                            self.opt_result.valid_offset.max.to_value(u.deg),
                        ],
                        u.deg,
                    ),
                )
            )
        return hdus

    def _make_benchmark_hdus(self, hdus):
        hdus.append(
            self.bias_resolution.make_bias_resolution_hdu(
                events=self.signal_events[self.signal_events["selected"]],
            )
        )
        hdus.append(
            self.angular_resolution.make_angular_resolution_hdu(
                events=self.signal_events[self.signal_events["selected_gh"]],
            )
        )
        if self.do_background:
            if not self.point_like:
                # Create a dummy theta cut since `pyirf.sensitivity.estimate_background`
                # needs a theta cut atm.
                self.log.info(
                    "Using all signal events with `theta < fov_offset_max` "
                    "to compute the sensitivity."
                )
                theta_cuts = QTable()
                theta_cuts["center"] = 0.5 * (
                    self.sensitivity.reco_energy_bins[:-1]
                    + self.sensitivity.reco_energy_bins[1:]
                )
                theta_cuts["cut"] = self.sensitivity.fov_offset_max
            else:
                theta_cuts = self.opt_result.theta_cuts

            hdus.append(
                self.sensitivity.make_sensitivity_hdu(
                    signal_events=self.signal_events[self.signal_events["selected"]],
                    background_events=self.background_events[
                        self.background_events["selected_gh"]
                    ],
                    theta_cut=theta_cuts,
                    gamma_spectrum=self.gamma_target_spectrum,
                )
            )
        return hdus

    def start(self):
        reduced_events = dict()
        for sel in self.particles:
            if sel.epp.quality_criteria != self.opt_result.precuts.quality_criteria:
                self.log.warning(
                    "Precuts are different from precuts used for calculating "
                    "g/h / theta cuts. Provided precuts:\n%s. "
                    "\nUsing the same precuts as g/h / theta cuts:\n%s. "
                    % (
                        sel.epp.to_table(functions=True)["criteria", "func"],
                        self.opt_result.precuts.to_table(functions=True)[
                            "criteria", "func"
                        ],
                    )
                )
                sel.epp = EventPreProcessor(
                    parent=sel,
                    quality_criteria=self.opt_result.precuts.quality_criteria,
                )

            if sel.epp.gammaness_classifier != self.opt_result.gh_cuts.meta["CLFNAME"]:
                raise RuntimeError(
                    "G/H cuts are only valid for gammaness scores predicted by "
                    "the same classifier model. Requested model: %s. "
                    "Model used for g/h cuts: %s."
                    % (
                        sel.epp.gammaness_classifier,
                        self.opt_result.gh_cuts.meta["CLFNAME"],
                    )
                )

            self.log.debug("%s Precuts: %s" % (sel.kind, sel.epp.quality_criteria))
            evs, cnt, meta = sel.load_preselected_events(self.chunk_size, self.obs_time)
            # Only calculate event weights if background or sensitivity should be calculated.
            if self.do_background:
                # Sensitivity is only calculated, if do_background is true
                # and benchmarks_output_path is given.
                if self.benchmarks_output_path is not None:
                    evs = sel.make_event_weights(
                        evs, meta["spectrum"], self.sensitivity.fov_offset_bins
                    )
                # If only background should be calculated,
                # only calculate weights for protons and electrons.
                elif sel.kind in ("protons", "electrons"):
                    evs = sel.make_event_weights(evs, meta["spectrum"])

            reduced_events[sel.kind] = evs
            reduced_events[f"{sel.kind}_count"] = cnt
            reduced_events[f"{sel.kind}_meta"] = meta
            self.log.debug(
                "Loaded %d %s events" % (reduced_events[f"{sel.kind}_count"], sel.kind)
            )
            if sel.kind == "gammas":
                self.signal_is_point_like = (
                    meta["sim_info"].viewcone_max - meta["sim_info"].viewcone_min
                ).value == 0

        if self.signal_is_point_like:
            errormessage = """The gamma input file contains point-like simulations.
                Therefore, the IRF can only be calculated at a single point
                in the FoV, but `fov_offset_n_bins > 1`."""

            if self.edisp.fov_offset_n_bins > 1 or self.aeff.fov_offset_n_bins > 1:
                raise ToolConfigurationError(errormessage)

            if not self.point_like and self.psf.fov_offset_n_bins > 1:
                raise ToolConfigurationError(errormessage)

            if self.do_background and self.bkg.fov_offset_n_bins > 1:
                raise ToolConfigurationError(errormessage)

            if self.benchmarks_output_path is not None and (
                self.angular_resolution.fov_offset_n_bins > 1
                or self.bias_resolution.fov_offset_n_bins > 1
                or self.sensitivity.fov_offset_n_bins > 1
            ):
                raise ToolConfigurationError(errormessage)

        reduced_events = self.calculate_selections(reduced_events)

        self.signal_events = reduced_events["gammas"]
        if self.do_background:
            if self.electron_file:
                self.background_events = vstack(
                    [reduced_events["protons"], reduced_events["electrons"]]
                )
            else:
                self.background_events = reduced_events["protons"]

        hdus = [fits.PrimaryHDU()]
        hdus = self._make_signal_irf_hdus(
            hdus, reduced_events["gammas_meta"]["sim_info"]
        )
        if self.do_background:
            hdus.append(
                self.bkg.make_bkg_hdu(
                    self.background_events[self.background_events["selected_gh"]],
                    self.obs_time,
                )
            )
            if "protons" in reduced_events.keys():
                hdus.append(
                    self.aeff.make_aeff_hdu(
                        events=reduced_events["protons"][
                            reduced_events["protons"]["selected_gh"]
                        ],
                        point_like=self.point_like,
                        signal_is_point_like=False,
                        sim_info=reduced_events["protons_meta"]["sim_info"],
                        extname="EFFECTIVE AREA PROTONS",
                    )
                )
            if "electrons" in reduced_events.keys():
                hdus.append(
                    self.aeff.make_aeff_hdu(
                        events=reduced_events["electrons"][
                            reduced_events["electrons"]["selected_gh"]
                        ],
                        point_like=self.point_like,
                        signal_is_point_like=False,
                        sim_info=reduced_events["electrons_meta"]["sim_info"],
                        extname="EFFECTIVE AREA ELECTRONS",
                    )
                )
        self.hdus = hdus

        if self.benchmarks_output_path is not None:
            b_hdus = [fits.PrimaryHDU()]
            b_hdus = self._make_benchmark_hdus(b_hdus)
            self.b_hdus = b_hdus

    def finish(self):
        self.log.info("Writing outputfile '%s'" % self.output_path)
        fits.HDUList(self.hdus).writeto(
            self.output_path,
            overwrite=self.overwrite,
        )
        Provenance().add_output_file(self.output_path, role="IRF")
        if self.benchmarks_output_path is not None:
            self.log.info(
                "Writing benchmark file to '%s'" % self.benchmarks_output_path
            )
            fits.HDUList(self.b_hdus).writeto(
                self.benchmarks_output_path,
                overwrite=self.overwrite,
            )
            Provenance().add_output_file(self.benchmarks_output_path, role="Benchmark")


def main():
    tool = IrfTool()
    tool.run()


if __name__ == "main":
    main()
