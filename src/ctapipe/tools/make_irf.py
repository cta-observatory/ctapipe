"""Tool to generate IRFs"""
import operator

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.table import QTable, vstack
from pyirf.benchmarks import angular_resolution, energy_bias_resolution
from pyirf.binning import create_histogram_table
from pyirf.cuts import evaluate_binned_cut
from pyirf.io import create_rad_max_hdu
from pyirf.sensitivity import calculate_sensitivity, estimate_background

from ..core import Provenance, Tool, ToolConfigurationError, traits
from ..core.traits import Bool, Float, Integer, Unicode, flag
from ..irf import (
    PYIRF_SPECTRA,
    Background2dIrf,
    Background3dIrf,
    EffectiveAreaIrf,
    EnergyMigrationIrf,
    EventsLoader,
    FovOffsetBinning,
    OptimizationResultStore,
    OutputEnergyBinning,
    PsfIrf,
    Spectra,
    ThetaCutsCalculator,
    check_bins_in_range,
)


class IrfTool(Tool):
    name = "ctapipe-make-irf"
    description = "Tool to create IRF files in GAD format"

    do_background = Bool(
        True,
        help="Compute background rate IRF using supplied files",
    ).tag(config=True)
    do_benchmarks = Bool(
        False,
        help="Produce IRF related benchmarks",
    ).tag(config=True)

    cuts_file = traits.Path(
        default_value=None, directory_ok=False, help="Path to optimized cuts input file"
    ).tag(config=True)

    gamma_file = traits.Path(
        default_value=None, directory_ok=False, help="Gamma input filename and path"
    ).tag(config=True)
    gamma_target_spectrum = traits.UseEnum(
        Spectra,
        default_value=Spectra.CRAB_HEGRA,
        help="Name of the pyrif spectra used for the simulated gamma spectrum",
    ).tag(config=True)
    proton_file = traits.Path(
        default_value=None,
        allow_none=True,
        directory_ok=False,
        help="Proton input filename and path",
    ).tag(config=True)
    proton_target_spectrum = traits.UseEnum(
        Spectra,
        default_value=Spectra.IRFDOC_PROTON_SPECTRUM,
        help="Name of the pyrif spectra used for the simulated proton spectrum",
    ).tag(config=True)
    electron_file = traits.Path(
        default_value=None,
        allow_none=True,
        directory_ok=False,
        help="Electron input filename and path",
    ).tag(config=True)
    electron_target_spectrum = traits.UseEnum(
        Spectra,
        default_value=Spectra.IRFDOC_ELECTRON_SPECTRUM,
        help="Name of the pyrif spectra used for the simulated electron spectrum",
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

    obs_time = Float(default_value=50.0, help="Observation time").tag(config=True)
    obs_time_unit = Unicode(
        default_value="hour",
        help="Unit used to specify observation time as an astropy unit string.",
    ).tag(config=True)

    alpha = Float(
        default_value=0.2, help="Ratio between size of on and off regions."
    ).tag(config=True)

    full_enclosure = Bool(
        False,
        help=(
            "Compute a full enclosure IRF by not applying a theta cut and only use"
            " the G/H separation cut."
        ),
    ).tag(config=True)

    aliases = {
        "cuts": "IrfTool.cuts_file",
        "gamma-file": "IrfTool.gamma_file",
        "proton-file": "IrfTool.proton_file",
        "electron-file": "IrfTool.electron_file",
        "output": "IrfTool.output_path",
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
            "do-benchmarks",
            "IrfTool.do_benchmarks",
            "Produce IRF related benchmarks.",
            "Do not produce IRF related benchmarks.",
        ),
        **flag(
            "full-enclosure",
            "IrfTool.full_enclosure",
            "Compute a full-enclosure IRF.",
            "Compute a point-like IRF.",
        ),
    }

    classes = [
        ThetaCutsCalculator,
        EventsLoader,
        Background2dIrf,
        Background3dIrf,
        EffectiveAreaIrf,
        EnergyMigrationIrf,
        FovOffsetBinning,
        OutputEnergyBinning,
        PsfIrf,
    ]

    def setup(self):
        self.theta = ThetaCutsCalculator(parent=self)
        self.e_bins = OutputEnergyBinning(parent=self)
        self.bins = FovOffsetBinning(parent=self)

        self.opt_result = OptimizationResultStore().read(self.cuts_file)

        self.reco_energy_bins = self.e_bins.reco_energy_bins()
        self.true_energy_bins = self.e_bins.true_energy_bins()
        self.fov_offset_bins = self.bins.fov_offset_bins()

        check_bins_in_range(self.reco_energy_bins, self.opt_result.valid_energy)
        check_bins_in_range(self.fov_offset_bins, self.opt_result.valid_offset)

        if (
            not self.full_enclosure
            and "n_events" not in self.opt_result.theta_cuts.colnames
        ):
            raise ToolConfigurationError(
                "Computing a point-like IRF requires an (optimized) theta cut."
            )

        self.particles = [
            EventsLoader(
                parent=self,
                kind="gammas",
                file=self.gamma_file,
                target_spectrum=PYIRF_SPECTRA[self.gamma_target_spectrum],
            ),
        ]
        if self.do_background and self.proton_file:
            self.particles.append(
                EventsLoader(
                    parent=self,
                    kind="protons",
                    file=self.proton_file,
                    target_spectrum=PYIRF_SPECTRA[self.proton_target_spectrum],
                )
            )
        if self.do_background and self.electron_file:
            self.particles.append(
                EventsLoader(
                    parent=self,
                    kind="electrons",
                    file=self.electron_file,
                    target_spectrum=PYIRF_SPECTRA[self.electron_target_spectrum],
                )
            )
        if self.do_background and len(self.particles) == 1:
            raise RuntimeError(
                "At least one electron or proton file required when speficying `do_background`."
            )

        if self.do_background:
            self.bkg = Background2dIrf(
                parent=self,
                valid_offset=self.opt_result.valid_offset,
            )
            self.bkg3 = Background3dIrf(
                parent=self,
                valid_offset=self.opt_result.valid_offset,
            )

        self.mig_matrix = EnergyMigrationIrf(
            parent=self,
        )
        if self.do_benchmarks:
            self.b_output = self.output_path.with_name(
                self.output_path.name.replace(".fits", "-benchmark.fits")
            )

    def calculate_selections(self):
        """Add the selection columns to the signal and optionally background tables"""
        self.signal_events["selected_gh"] = evaluate_binned_cut(
            self.signal_events["gh_score"],
            self.signal_events["reco_energy"],
            self.opt_result.gh_cuts,
            operator.ge,
        )
        if not self.full_enclosure:
            self.theta_cuts_opt = self.theta.calculate_theta_cuts(
                self.signal_events[self.signal_events["selected_gh"]]["theta"],
                self.signal_events[self.signal_events["selected_gh"]]["reco_energy"],
                self.reco_energy_bins,
            )
            self.signal_events["selected_theta"] = evaluate_binned_cut(
                self.signal_events["theta"],
                self.signal_events["reco_energy"],
                self.theta_cuts_opt,
                operator.le,
            )
            self.signal_events["selected"] = (
                self.signal_events["selected_theta"] & self.signal_events["selected_gh"]
            )
        else:
            # Re-"calculate" the dummy theta cut because of potentially different reco energy binning
            self.theta_cuts_opt = QTable()
            self.theta_cuts_opt["low"] = self.reco_energy_bins[:-1]
            self.theta_cuts_opt["center"] = 0.5 * (
                self.reco_energy_bins[:-1] + self.reco_energy_bins[1:]
            )
            self.theta_cuts_opt["high"] = self.reco_energy_bins[1:]
            self.theta_cuts_opt["cut"] = self.opt_result.valid_offset.max

            self.signal_events["selected"] = self.signal_events["selected_gh"]

        if self.do_background:
            self.background_events["selected_gh"] = evaluate_binned_cut(
                self.background_events["gh_score"],
                self.background_events["reco_energy"],
                self.opt_result.gh_cuts,
                operator.ge,
            )
            if not self.full_enclosure:
                self.background_events["selected_theta"] = evaluate_binned_cut(
                    self.background_events["theta"],
                    self.background_events["reco_energy"],
                    self.theta_cuts_opt,
                    operator.le,
                )
                self.background_events["selected"] = (
                    self.background_events["selected_theta"]
                    & self.background_events["selected_gh"]
                )
            else:
                self.background_events["selected"] = self.background_events[
                    "selected_gh"
                ]

        # TODO: maybe rework the above so we can give the number per
        # species instead of the total background
        if self.do_background:
            self.log.debug(
                "Keeping %d signal, %d background events"
                % (
                    sum(self.signal_events["selected"]),
                    sum(self.background_events["selected"]),
                )
            )
        else:
            self.log.debug(
                "Keeping %d signal events" % (sum(self.signal_events["selected"]))
            )

    def _stack_background(self, reduced_events):
        bkgs = []
        if self.proton_file:
            bkgs.append("protons")
        if self.electron_file:
            bkgs.append("electrons")
        if len(bkgs) == 2:
            background = vstack(
                [reduced_events["protons"], reduced_events["electrons"]]
            )
        else:
            background = reduced_events[bkgs[0]]
        return background

    def _make_signal_irf_hdus(self, hdus):
        hdus.append(
            self.aeff.make_effective_area_hdu(
                signal_events=self.signal_events[self.signal_events["selected"]],
                fov_offset_bins=self.fov_offset_bins,
                point_like=not self.full_enclosure,
                signal_is_point_like=self.signal_is_point_like,
            )
        )
        hdus.append(
            self.mig_matrix.make_energy_dispersion_hdu(
                signal_events=self.signal_events[self.signal_events["selected"]],
                fov_offset_bins=self.fov_offset_bins,
                point_like=not self.full_enclosure,
            )
        )
        hdus.append(
            self.psf.make_psf_table_hdu(
                signal_events=self.signal_events[self.signal_events["selected"]],
                fov_offset_bins=self.fov_offset_bins,
            )
        )
        hdus.append(
            create_rad_max_hdu(
                self.theta_cuts_opt["cut"].reshape(-1, 1),
                self.reco_energy_bins,
                self.fov_offset_bins,
            )
        )
        return hdus

    def _make_benchmark_hdus(self, hdus):
        bias_resolution = energy_bias_resolution(
            self.signal_events[self.signal_events["selected"]],
            self.true_energy_bins,
            bias_function=np.mean,
            energy_type="true",
        )
        hdus.append(fits.BinTableHDU(bias_resolution, name="ENERGY_BIAS_RESOLUTION"))

        # Here we use reconstructed energy instead of true energy for the sake of
        # current pipelines comparisons
        ang_res = angular_resolution(
            self.signal_events[self.signal_events["selected_gh"]],
            self.reco_energy_bins,
            energy_type="reco",
        )
        hdus.append(fits.BinTableHDU(ang_res, name="ANGULAR_RESOLUTION"))

        if self.do_background:
            signal_hist = create_histogram_table(
                self.signal_events[self.signal_events["selected"]],
                bins=self.reco_energy_bins,
            )
            background_hist = estimate_background(
                self.background_events[self.background_events["selected_gh"]],
                reco_energy_bins=self.reco_energy_bins,
                theta_cuts=self.theta_cuts_opt,
                alpha=self.alpha,
                fov_offset_min=self.fov_offset_bins[0],
                fov_offset_max=self.fov_offset_bins[-1],
            )
            sensitivity = calculate_sensitivity(
                signal_hist, background_hist, alpha=self.alpha
            )
            gamma_spectrum = PYIRF_SPECTRA[self.gamma_target_spectrum]
            # scale relative sensitivity by Crab flux to get the flux sensitivity
            sensitivity["flux_sensitivity"] = sensitivity[
                "relative_sensitivity"
            ] * gamma_spectrum(sensitivity["reco_energy_center"])

            hdus.append(fits.BinTableHDU(sensitivity, name="SENSITIVITY"))

        return hdus

    def start(self):

        reduced_events = dict()
        for sel in self.particles:
            # TODO: not very elegant to pass them this way, refactor later
            sel.epp.quality_criteria = self.opt_result.precuts.quality_criteria
            self.log.debug("%s Precuts: %s" % (sel.kind, sel.epp.quality_criteria))
            evs, cnt, meta = sel.load_preselected_events(
                self.chunk_size,
                self.obs_time * u.Unit(self.obs_time_unit),
                self.fov_offset_bins,
            )
            reduced_events[sel.kind] = evs
            reduced_events[f"{sel.kind}_count"] = cnt
            self.log.debug(
                "Loaded %d %s events" % (reduced_events[f"{sel.kind}_count"], sel.kind)
            )
            if sel.kind == "gammas":
                self.aeff = EffectiveAreaIrf(parent=self, sim_info=meta["sim_info"])
                self.signal_is_point_like = (
                    meta["sim_info"].viewcone_max - meta["sim_info"].viewcone_min
                ).value == 0

        if self.signal_is_point_like:
            self.log.info(
                "The gamma input file contains point-like simulations."
                " Therefore, the IRF is only calculated at a single point in the FoV."
                " Changing `fov_offset_n_bins` to 1."
            )
            self.fov_offset_bins.fov_offset_n_bins = 1

        self.signal_events = reduced_events["gammas"]
        if self.do_background:
            self.background_events = self._stack_background(reduced_events)

        self.calculate_selections()

        self.log.debug("True Energy bins: %s" % str(self.true_energy_bins.value))
        self.log.debug("Reco Energy bins: %s" % str(self.reco_energy_bins.value))
        self.log.debug("FoV offset bins: %s" % str(self.fov_offset_bins))

        self.psf = PsfIrf(
            parent=self,
            valid_offset=self.opt_result.valid_offset,
        )
        hdus = [fits.PrimaryHDU()]
        hdus = self._make_signal_irf_hdus(hdus)
        if self.do_background:
            hdus.append(
                self.bkg.make_bkg2d_table_hdu(
                    self.background_events, self.obs_time * u.Unit(self.obs_time_unit)
                )
            )
            hdus.append(
                self.bkg3.make_bkg3d_table_hdu(
                    self.background_events, self.obs_time * u.Unit(self.obs_time_unit)
                )
            )
        self.hdus = hdus

        if self.do_benchmarks:
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
        if self.do_benchmarks:
            self.log.info("Writing benchmark file to '%s'" % self.b_output)
            fits.HDUList(self.b_hdus).writeto(
                self.b_output,
                overwrite=self.overwrite,
            )
            Provenance().add_output_file(self.b_output, role="Benchmark")


def main():
    tool = IrfTool()
    tool.run()


if __name__ == "main":
    main()
