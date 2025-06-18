"""Tool to generate IRFs"""

from importlib.util import find_spec

if find_spec("pyirf") is None:
    from ..exceptions import OptionalDependencyMissing

    raise OptionalDependencyMissing("pyirf") from None

import operator
from functools import partial

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.table import vstack
from pyirf.cuts import evaluate_binned_cut
from pyirf.io import create_rad_max_hdu

from ..core import Provenance, Tool, ToolConfigurationError, traits
from ..core.traits import AstroQuantity, Bool, Integer, classes_with_traits, flag
from ..io.dl2_tables_preprocessing import (
    DL2EventLoader,
    DL2EventQualityQuery,
)
from ..irf import (
    OptimizationResult,
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
    PSFMakerBase,
)

__all__ = ["IrfTool"]


class IrfTool(Tool):
    "Tool to create IRF files in GADF format"

    name = "ctapipe-compute-irf"
    description = __doc__
    examples = """
    ctapipe-compute-irf \\
        --cuts cuts.fits \\
        --gamma-file gamma.dl2.h5 \\
        --proton-file proton.dl2.h5 \\
        --electron-file electron.dl2.h5 \\
        --output irf.fits.gz \\
        --benchmark-output benchmarks.fits.gz
    """

    do_background = Bool(
        True,
        help="Compute background rate using supplied files.",
    ).tag(config=True)

    range_check_error = Bool(
        False,
        help="Raise error if asking for IRFs outside range where cut optimisation is valid.",
    ).tag(config=True)

    cuts_file = traits.Path(
        exists=True,
        directory_ok=False,
        help="Path to optimized cuts input file.",
    ).tag(config=True)

    gamma_file = traits.Path(
        exists=True, directory_ok=False, help="Gamma input filename and path."
    ).tag(config=True)

    gamma_target_spectrum = traits.UseEnum(
        Spectra,
        default_value=Spectra.CRAB_HEGRA,
        help="Name of the spectrum used for weights of gamma events.",
    ).tag(config=True)

    proton_file = traits.Path(
        exists=True,
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
        exists=True,
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

    energy_dispersion_maker_name = traits.ComponentName(
        EnergyDispersionMakerBase,
        default_value="EnergyDispersion2dMaker",
        help="The parameterization of the energy dispersion to be used.",
    ).tag(config=True)

    effective_area_maker_name = traits.ComponentName(
        EffectiveAreaMakerBase,
        default_value="EffectiveArea2dMaker",
        help="The parameterization of the effective area to be used.",
    ).tag(config=True)

    psf_maker_name = traits.ComponentName(
        PSFMakerBase,
        default_value="PSF3DMaker",
        help="The parameterization of the point spread function to be used.",
    ).tag(config=True)

    background_maker_name = traits.ComponentName(
        BackgroundRateMakerBase,
        default_value="BackgroundRate2dMaker",
        help="The parameterization of the background rate to be used.",
    ).tag(config=True)

    energy_bias_resolution_maker_name = traits.ComponentName(
        EnergyBiasResolutionMakerBase,
        default_value="EnergyBiasResolution2dMaker",
        help=(
            "The parameterization of the bias and resolution benchmark "
            "for the energy prediction."
        ),
    ).tag(config=True)

    angular_resolution_maker_name = traits.ComponentName(
        AngularResolutionMakerBase,
        default_value="AngularResolution2dMaker",
        help="The parameterization of the angular resolution benchmark.",
    ).tag(config=True)

    sensitivity_maker_name = traits.ComponentName(
        SensitivityMakerBase,
        default_value="Sensitivity2dMaker",
        help="The parameterization of the point source sensitivity benchmark.",
    ).tag(config=True)

    spatial_selection_applied = Bool(
        False,
        help=(
            "Compute an IRF after applying a direction cut (``SpatialSelection=RAD_MAX``) "
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
            "spatial-selection-applied",
            "IrfTool.spatial_selection_applied",
            "Compute an IRF after applying a direction cut (``SpatialSelection=RAD_MAX``).",
            "Compute an IRF without any direction cut (``SpatialSelection=None``).",
        ),
    }

    classes = (
        [
            DL2EventLoader,
        ]
        + classes_with_traits(BackgroundRateMakerBase)
        + classes_with_traits(EffectiveAreaMakerBase)
        + classes_with_traits(EnergyDispersionMakerBase)
        + classes_with_traits(PSFMakerBase)
        + classes_with_traits(AngularResolutionMakerBase)
        + classes_with_traits(EnergyBiasResolutionMakerBase)
        + classes_with_traits(SensitivityMakerBase)
    )

    def _check_config(self):
        if self.gamma_file is None:
            self.log.critical(
                "Setting gamma_file is required (via --gamma-file or a config file)."
            )
            self.exit(1)

        if self.cuts_file is None:
            self.log.critical(
                "Setting cuts_file is required (via --cuts or a config file)."
            )
            self.exit(1)

        if self.output_path is None:
            self.log.critical(
                "Setting output_path is required (via --output or a config file)."
            )
            self.exit(1)

    def setup(self):
        """
        Initialize components from config and load g/h (and theta) cuts.
        """
        self._check_config()

        self.opt_result = OptimizationResult.read(self.cuts_file)

        if (
            self.spatial_selection_applied
            and self.opt_result.spatial_selection_table is None
        ):
            raise ToolConfigurationError(
                f"{self.cuts_file} does not contain any direction cut, "
                "but --spatial-selection-applied was given."
            )

        check_e_bins = partial(
            check_bins_in_range,
            valid_range=self.opt_result.valid_energy,
            raise_error=self.range_check_error,
        )
        self.event_loaders = {
            "gammas": DL2EventLoader(
                parent=self,
                file=self.gamma_file,
                target_spectrum=self.gamma_target_spectrum,
            ),
        }
        if self.do_background:
            if not self.proton_file or (
                self.proton_file and not self.proton_file.exists()
            ):
                raise ValueError(
                    "At least a proton file required when specifying `do_background`."
                )

            self.event_loaders["protons"] = DL2EventLoader(
                parent=self,
                file=self.proton_file,
                target_spectrum=self.proton_target_spectrum,
            )
            if self.electron_file and self.electron_file.exists():
                self.event_loaders["electrons"] = DL2EventLoader(
                    parent=self,
                    file=self.electron_file,
                    target_spectrum=self.electron_target_spectrum,
                )
            else:
                self.log.warning("Estimating background without electron file.")

            self.background_maker = BackgroundRateMakerBase.from_name(
                self.background_maker_name, parent=self
            )
            check_e_bins(
                bins=self.background_maker.reco_energy_bins,
                source="background reco energy",
            )

        self.energy_dispersion_maker = EnergyDispersionMakerBase.from_name(
            self.energy_dispersion_maker_name, parent=self
        )
        self.effective_area_maker = EffectiveAreaMakerBase.from_name(
            self.effective_area_maker_name, parent=self
        )
        self.psf_maker = PSFMakerBase.from_name(self.psf_maker_name, parent=self)

        if self.benchmarks_output_path is not None:
            self.angular_resolution_maker = AngularResolutionMakerBase.from_name(
                self.angular_resolution_maker_name, parent=self
            )
            if self.angular_resolution_maker.use_reco_energy:
                check_e_bins(
                    bins=self.angular_resolution_maker.reco_energy_bins,
                    source="Angular resolution energy",
                )

            self.bias_resolution_maker = EnergyBiasResolutionMakerBase.from_name(
                self.energy_bias_resolution_maker_name, parent=self
            )
            self.sensitivity_maker = SensitivityMakerBase.from_name(
                self.sensitivity_maker_name, parent=self
            )
            check_e_bins(
                bins=self.sensitivity_maker.reco_energy_bins,
                source="Sensitivity reco energy",
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
        reduced_events["gammas"]["selected"] = reduced_events["gammas"]["selected_gh"]
        if self.spatial_selection_applied:
            reduced_events["gammas"]["selected_theta"] = evaluate_binned_cut(
                reduced_events["gammas"]["theta"],
                reduced_events["gammas"]["reco_energy"],
                self.opt_result.spatial_selection_table,
                operator.le,
            )
            reduced_events["gammas"]["selected"] &= reduced_events["gammas"][
                "selected_theta"
            ]

        if self.opt_result.multiplicity_cuts is not None:
            reduced_events["gammas"]["selected_multiplicity"] = evaluate_binned_cut(
                reduced_events["gammas"]["multiplicity"],
                reduced_events["gammas"]["reco_energy"],
                self.opt_result.multiplicity_cuts,
                operator.ge,
            )
            reduced_events["gammas"]["selected"] &= reduced_events["gammas"][
                "selected_multiplicity"
            ]

        if self.do_background:
            backgrounds = (
                ["protons", "electrons"] if self.electron_file else ["protons"]
            )
            n_sel = {"protons": 0, "electrons": 0}
            for bkg_type in backgrounds:
                reduced_events[bkg_type]["selected_gh"] = evaluate_binned_cut(
                    reduced_events[bkg_type]["gh_score"],
                    reduced_events[bkg_type]["reco_energy"],
                    self.opt_result.gh_cuts,
                    operator.ge,
                )
                reduced_events[bkg_type]["selected"] = reduced_events[bkg_type][
                    "selected_gh"
                ]
                if self.opt_result.multiplicity_cuts is not None:
                    reduced_events[bkg_type][
                        "selected_multiplicity"
                    ] = evaluate_binned_cut(
                        reduced_events[bkg_type]["multiplicity"],
                        reduced_events[bkg_type]["reco_energy"],
                        self.opt_result.multiplicity_cuts,
                        operator.ge,
                    )
                    reduced_events[bkg_type]["selected"] &= reduced_events[bkg_type][
                        "selected_multiplicity"
                    ]

                n_sel[bkg_type] = np.count_nonzero(reduced_events[bkg_type]["selected"])

            self.log.info(
                "Keeping %d signal, %d proton events, and %d electron events"
                % (
                    np.count_nonzero(reduced_events["gammas"]["selected"]),
                    n_sel["protons"],
                    n_sel["electrons"],
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
            self.effective_area_maker(
                events=self.signal_events[self.signal_events["selected"]],
                spatial_selection_applied=self.spatial_selection_applied,
                signal_is_point_like=self.signal_is_point_like,
                sim_info=sim_info,
            )
        )
        hdus.append(
            self.energy_dispersion_maker(
                events=self.signal_events[self.signal_events["selected"]],
                spatial_selection_applied=self.spatial_selection_applied,
            )
        )
        hdus.append(
            self.psf_maker(events=self.signal_events[self.signal_events["selected_gh"]])
        )
        if self.spatial_selection_applied:
            # TODO: Support fov binning
            self.log.debug(
                "Currently multiple fov bins is not supported for RAD_MAX. "
                "Using `fov_offset_bins = [valid_offset.min, valid_offset.max]`."
            )
            hdus.append(
                create_rad_max_hdu(
                    rad_max=self.opt_result.spatial_selection_table["cut"].reshape(
                        -1, 1
                    ),
                    reco_energy_bins=np.append(
                        self.opt_result.spatial_selection_table["low"],
                        self.opt_result.spatial_selection_table["high"][-1],
                    ),
                    fov_offset_bins=u.Quantity(
                        [
                            self.opt_result.valid_offset.min,
                            self.opt_result.valid_offset.max,
                        ]
                    ).reshape(-1),
                )
            )
        return hdus

    def _make_benchmark_hdus(self, hdus):
        hdus.append(
            self.bias_resolution_maker(
                events=self.signal_events[self.signal_events["selected"]],
            )
        )
        hdus.append(
            self.angular_resolution_maker(
                events=self.signal_events[self.signal_events["selected_gh"]],
            )
        )
        if self.do_background:
            if self.opt_result.spatial_selection_table is None:
                raise ValueError(
                    "Calculating the point-source sensitivity requires "
                    f"theta cuts, but {self.cuts_file} does not contain any."
                )

            hdus.append(
                self.sensitivity_maker(
                    signal_events=self.signal_events[self.signal_events["selected"]],
                    background_events=self.background_events[
                        self.background_events["selected_gh"]
                    ],
                    spatial_selection_table=self.opt_result.spatial_selection_table,
                    gamma_spectrum=self.gamma_target_spectrum,
                )
            )
        return hdus

    def start(self):
        """
        Load events and calculate the irf (and the benchmarks).
        """
        reduced_events = dict()
        for particle_type, loader in self.event_loaders.items():
            if loader.epp.gammaness_classifier != self.opt_result.clf_prefix:
                raise RuntimeError(
                    "G/H cuts are only valid for gammaness scores predicted by "
                    "the same classifier model. Requested model: %s. "
                    "Model used for g/h cuts: %s."
                    % (
                        loader.epp.gammaness_classifier,
                        self.opt_result.clf_prefix,
                    )
                )

            if (
                loader.epp.quality_query.quality_criteria
                != self.opt_result.quality_query.quality_criteria
            ):
                self.log.warning(
                    "Quality criteria are different from quality criteria used for "
                    "calculating g/h / theta cuts. Provided quality criteria:\n%s. "
                    "\nUsing the same quality criteria as g/h / theta cuts:\n%s. "
                    % (
                        loader.epp.quality_query.to_table(functions=True)[
                            "criteria", "func"
                        ],
                        self.opt_result.quality_query.to_table(functions=True)[
                            "criteria", "func"
                        ],
                    )
                )
                loader.epp.quality_query = DL2EventQualityQuery(
                    parent=loader,
                    quality_criteria=self.opt_result.quality_query.quality_criteria,
                )

            self.log.debug(
                "%s Quality criteria: %s"
                % (particle_type, loader.epp.quality_query.quality_criteria)
            )
            events, count, meta = loader.load_preselected_events(
                self.chunk_size, self.obs_time
            )
            # Only calculate event weights if background or sensitivity should be calculated.
            if self.do_background:
                # Sensitivity is only calculated, if do_background is true
                # and benchmarks_output_path is given.
                if self.benchmarks_output_path is not None:
                    events = loader.make_event_weights(
                        events,
                        meta["spectrum"],
                        particle_type,
                        self.sensitivity_maker.fov_offset_bins,
                    )
                # If only background should be calculated,
                # only calculate weights for protons and electrons.
                elif particle_type in ("protons", "electrons"):
                    events = loader.make_event_weights(
                        events, meta["spectrum"], particle_type
                    )

            reduced_events[particle_type] = events
            reduced_events[f"{particle_type}_count"] = count
            reduced_events[f"{particle_type}_meta"] = meta
            self.log.debug(
                "Loaded %d %s events"
                % (reduced_events[f"{particle_type}_count"], particle_type)
            )
            if particle_type == "gammas":
                self.signal_is_point_like = (
                    meta["sim_info"].viewcone_max - meta["sim_info"].viewcone_min
                ).value == 0

        if self.signal_is_point_like:
            errormessage = """The gamma input file contains point-like simulations.
                Therefore, the IRF can only be calculated at a single point
                in the FoV, but `fov_offset_n_bins > 1`."""

            if (
                self.energy_dispersion_maker.fov_offset_n_bins > 1
                or self.effective_area_maker.fov_offset_n_bins > 1
            ):
                raise ToolConfigurationError(errormessage)

            if (
                not self.spatial_selection_applied
                and self.psf_maker.fov_offset_n_bins > 1
            ):
                raise ToolConfigurationError(errormessage)

            if self.do_background and self.background_maker.fov_offset_n_bins > 1:
                raise ToolConfigurationError(errormessage)

            if self.benchmarks_output_path is not None and (
                self.angular_resolution_maker.fov_offset_n_bins > 1
                or self.bias_resolution_maker.fov_offset_n_bins > 1
                or self.sensitivity_maker.fov_offset_n_bins > 1
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
                self.background_maker(
                    self.background_events[self.background_events["selected_gh"]],
                    self.obs_time,
                )
            )
            if "protons" in reduced_events.keys():
                hdus.append(
                    self.effective_area_maker(
                        events=reduced_events["protons"][
                            reduced_events["protons"]["selected_gh"]
                        ],
                        spatial_selection_applied=self.spatial_selection_applied,
                        signal_is_point_like=False,
                        sim_info=reduced_events["protons_meta"]["sim_info"],
                        extname="EFFECTIVE AREA PROTONS",
                    )
                )
            if "electrons" in reduced_events.keys():
                hdus.append(
                    self.effective_area_maker(
                        events=reduced_events["electrons"][
                            reduced_events["electrons"]["selected_gh"]
                        ],
                        spatial_selection_applied=self.spatial_selection_applied,
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
        """
        Write the irf (and the benchmarks) to the (respective) output file(s).
        """
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
