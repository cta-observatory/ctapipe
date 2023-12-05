"""Tool to generate IRFs"""
import operator

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.table import vstack
from pyirf.benchmarks import angular_resolution, energy_bias_resolution
from pyirf.binning import create_histogram_table
from pyirf.cuts import evaluate_binned_cut
from pyirf.io import create_background_2d_hdu, create_rad_max_hdu
from pyirf.irf import background_2d
from pyirf.sensitivity import calculate_sensitivity, estimate_background

from ..core import Provenance, Tool, traits
from ..core.traits import Bool, Float, Integer, Unicode
from ..irf import (
    PYIRF_SPECTRA,
    EffectiveAreaIrf,
    EnergyMigrationIrf,
    EventPreProcessor,
    EventsLoader,
    FovOffsetBinning,
    OptimisationResultStore,
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
        default_value=None, directory_ok=False, help="Path to optimised cuts input file"
    ).tag(config=True)

    gamma_file = traits.Path(
        default_value=None, directory_ok=False, help="Gamma input filename and path"
    ).tag(config=True)
    gamma_sim_spectrum = traits.UseEnum(
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
    proton_sim_spectrum = traits.UseEnum(
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
    electron_sim_spectrum = traits.UseEnum(
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

    overwrite = Bool(
        False,
        help="Overwrite the output file if it exists",
    ).tag(config=True)

    obs_time = Float(default_value=50.0, help="Observation time").tag(config=True)
    obs_time_unit = Unicode(
        default_value="hour",
        help="Unit used to specify observation time as an astropy unit string.",
    ).tag(config=True)

    alpha = Float(
        default_value=0.2, help="Ratio between size of on and off regions"
    ).tag(config=True)

    classes = [
        OutputEnergyBinning,
        FovOffsetBinning,
        EventPreProcessor,
        PsfIrf,
        EnergyMigrationIrf,
        EffectiveAreaIrf,
    ]

    def calculate_selections(self):
        """Add the selection columns to the signal and optionally background tables"""
        self.signal_events["selected_gh"] = evaluate_binned_cut(
            self.signal_events["gh_score"],
            self.signal_events["reco_energy"],
            self.opt_result.gh_cuts,
            operator.ge,
        )
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

        if self.do_background:
            self.background_events["selected_gh"] = evaluate_binned_cut(
                self.background_events["gh_score"],
                self.background_events["reco_energy"],
                self.opt_result.gh_cuts,
                operator.ge,
            )
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

        # TODO: maybe rework the above so we can give the number per
        # species instead of the total background
        self.log.debug(
            "Keeping %d signal, %d backgrond events"
            % (
                sum(self.signal_events["selected"]),
                sum(self.background_events["selected"]),
            )
        )

    def setup(self):
        self.theta = ThetaCutsCalculator(parent=self)
        self.e_bins = OutputEnergyBinning(parent=self)
        self.bins = FovOffsetBinning(parent=self)

        self.opt_result = OptimisationResultStore().read(self.cuts_file)
        self.epp = EventPreProcessor(parent=self)
        # TODO: not very elegant to pass them this way, refactor later
        self.epp.quality_criteria = self.opt_result.precuts.quality_criteria
        self.reco_energy_bins = self.e_bins.reco_energy_bins()
        self.true_energy_bins = self.e_bins.true_energy_bins()
        self.fov_offset_bins = self.bins.fov_offset_bins()

        check_bins_in_range(self.reco_energy_bins, self.opt_result.valid_energy)
        check_bins_in_range(self.fov_offset_bins, self.opt_result.valid_offset)

        self.particles = [
            EventsLoader(
                self.epp,
                "gammas",
                self.gamma_file,
                PYIRF_SPECTRA[self.gamma_sim_spectrum],
            ),
        ]
        if self.do_background and self.proton_file:
            self.particles.append(
                EventsLoader(
                    self.epp,
                    "protons",
                    self.proton_file,
                    PYIRF_SPECTRA[self.proton_sim_spectrum],
                )
            )
        if self.do_background and self.electron_file:
            self.particles.append(
                EventsLoader(
                    self.epp,
                    "electrons",
                    self.electron_file,
                    PYIRF_SPECTRA[self.electron_sim_spectrum],
                )
            )
        if self.do_background and len(self.particles) == 1:
            raise RuntimeError(
                "At least one electron or proton file required when speficying `do_background`."
            )

        self.aeff = None

        self.psf = PsfIrf(
            parent=self,
            energy_bins=self.true_energy_bins,
            valid_offset=self.opt_result.valid_offset,
        )
        self.mig_matrix = EnergyMigrationIrf(
            parent=self,
            energy_bins=self.true_energy_bins,
        )
        if self.do_benchmarks:
            self.b_hdus = None
            self.b_output = self.output_path.with_name(
                self.output_path.name.replace(".fits", "-benchmark.fits")
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
            )
        )
        hdus.append(
            self.mig_matrix.make_energy_dispersion_hdu(
                signal_events=self.signal_events[self.signal_events["selected"]],
                fov_offset_bins=self.fov_offset_bins,
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
                self.true_energy_bins,
                self.fov_offset_bins,
            )
        )
        return hdus

    def _make_background_hdu(self):
        sel = self.background_events["selected_gh"]
        self.log.debug("%d background events selected" % sel.sum())
        self.log.debug("%f obs time" % self.obs_time)

        background_rate = background_2d(
            self.background_events[sel],
            self.reco_energy_bins,
            fov_offset_bins=self.fov_offset_bins,
            t_obs=self.obs_time * u.Unit(self.obs_time_unit),
        )
        return create_background_2d_hdu(
            background_rate,
            self.reco_energy_bins,
            fov_offset_bins=self.fov_offset_bins,
        )

    def _make_benchmark_hdus(self, hdus):
        # Here we use reconstructed energy instead of true energy for the sake of
        # current pipelines comparisons
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
                fov_offset_min=self.fov_offset_bins["offset_min"],
                fov_offset_max=self.fov_offset_bins["offset_max"],
            )
            sensitivity = calculate_sensitivity(
                signal_hist, background_hist, alpha=self.alpha
            )

            # scale relative sensitivity by Crab flux to get the flux sensitivity
            sensitivity["flux_sensitivity"] = sensitivity[
                "relative_sensitivity"
            ] * self.gamma_spectrum(sensitivity["reco_energy_center"])

            hdus.append(fits.BinTableHDU(sensitivity, name="SENSITIVITY"))

        return hdus

    def start(self):
        # TODO: this event loading code seems to be largely repeated between both
        # tools, try to refactor to a common solution
        reduced_events = dict()
        for sel in self.particles:
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
                self.gamma_spectrum = meta["spectrum"]

        self.signal_events = reduced_events["gammas"]
        if self.do_background:
            self.background_events = self._stack_background(reduced_events)

        self.calculate_selections()

        self.log.debug("True Energy bins: %s" % str(self.true_energy_bins.value))
        self.log.debug("FoV offset bins: %s" % str(self.fov_offset_bins))

        hdus = [fits.PrimaryHDU()]
        hdus = self._make_signal_irf_hdus(hdus)
        if self.do_background:
            hdus.append(self._make_background_hdu())
        self.hdus = hdus

        if self.do_benchmarks:
            b_hdus = [fits.PrimaryHDU()]
            b_hdus = self._make_benchmark_hdus(b_hdus)
            self.b_hdus = self.b_hdus

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
