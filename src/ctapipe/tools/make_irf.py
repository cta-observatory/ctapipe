"""Tool to generate IRFs"""
import operator

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.table import vstack
from pyirf.benchmarks import angular_resolution, energy_bias_resolution
from pyirf.binning import create_histogram_table
from pyirf.cuts import evaluate_binned_cut
from pyirf.io import (
    create_aeff2d_hdu,
    create_background_2d_hdu,
    create_energy_dispersion_hdu,
    create_psf_table_hdu,
    create_rad_max_hdu,
)
from pyirf.irf import (
    background_2d,
    effective_area_per_energy_and_fov,
    energy_dispersion,
    psf_table,
)
from pyirf.sensitivity import calculate_sensitivity, estimate_background

from ..core import Provenance, Tool, traits
from ..core.traits import Bool, Float, Integer, Unicode
from ..irf import (
    PYIRF_SPECTRA,
    DataBinning,
    EventPreProcessor,
    EventSelector,
    GridOptimizer,
    OutputEnergyBinning,
    Spectra,
    ThetaCutsCalculator,
)


class IrfTool(Tool):
    name = "ctapipe-make-irfs"
    description = "Tool to create IRF files in GAD format"

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
        help="How many subarray events to load at once for making predictions.",
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
    # ON_radius = Float(default_value=1.0, help="Radius of ON region in degrees").tag(
    #    config=True
    # )
    max_bg_radius = Float(
        default_value=3.0, help="Radius used to calculate background rate in degrees"
    ).tag(config=True)

    classes = [GridOptimizer, DataBinning, OutputEnergyBinning, EventPreProcessor]

    def setup(self):
        self.go = GridOptimizer(parent=self)
        self.theta = ThetaCutsCalculator(parent=self)
        self.e_bins = OutputEnergyBinning(parent=self)
        self.bins = DataBinning(parent=self)
        epp = EventPreProcessor(parent=self)

        self.reco_energy_bins = self.e_bins.reco_energy_bins()
        self.true_energy_bins = self.e_bins.true_energy_bins()
        self.energy_migration_bins = self.e_bins.energy_migration_bins()

        self.source_offset_bins = self.bins.source_offset_bins()
        self.fov_offset_bins = self.bins.fov_offset_bins()

        self.particles = [
            EventSelector(
                epp, "gammas", self.gamma_file, PYIRF_SPECTRA[self.gamma_sim_spectrum]
            ),
            EventSelector(
                epp,
                "protons",
                self.proton_file,
                PYIRF_SPECTRA[self.proton_sim_spectrum],
                epp,
            ),
            EventSelector(
                epp,
                "electroms",
                self.electron_file,
                PYIRF_SPECTRA[self.electron_sim_spectrum],
            ),
        ]

    def start(self):
        reduced_events = dict()
        for sel in self.particles:
            evs, cnt = sel.load_preselected_events(self.chunk_size)
            reduced_events[sel.kind] = evs
            reduced_events[f"{sel.kind}_count"] = cnt

        self.log.debug(
            "Loaded %d gammas, %d protons, %d electrons"
            % (
                reduced_events["gamma_count"],
                reduced_events["proton_count"],
                reduced_events["electron_count"],
            )
        )
        self.log.debug(
            "Keeping %d gammas, %d protons, %d electrons"
            % (
                len(reduced_events["gamma"]),
                len(reduced_events["proton"]),
                len(reduced_events["electron"]),
            )
        )
        # select_fov = (
        #     reduced_events["gamma"]["true_source_fov_offset"]
        #     <= self.bins.fov_offset_max * u.deg
        # )
        # TODO: verify that this fov cut on only gamma is ok
        self.signal_events = reduced_events["gamma"]  # [select_fov]
        self.background_events = vstack(
            [reduced_events["proton"], reduced_events["electron"]]
        )

        self.load_preselected_events()
        self.log.info(
            "Optimising cuts using %d signal and %d background events"
            % (len(self.signal_events), len(self.background_events)),
        )
        self.gh_cuts, self.theta_cuts_opt, self.sens2 = self.go.optimise_gh_cut(
            self.signal_events,
            self.background_events,
            self.alpha,
            self.bins.fov_offset_min,
            self.bins.fov_offset_max,
            self.theta,
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
        self.background_events["selected_theta"] = evaluate_binned_cut(
            self.background_events["theta"],
            self.background_events["reco_energy"],
            self.theta_cuts_opt,
            operator.le,
        )
        # calculate sensitivity
        signal_hist = create_histogram_table(
            self.signal_events[self.signal_events["selected"]],
            bins=self.reco_energy_bins,
        )

        background_hist = estimate_background(
            self.background_events[self.background_events["selected_gh"]],
            reco_energy_bins=self.reco_energy_bins,
            theta_cuts=self.theta_cuts_opt,
            alpha=self.alpha,
            fov_offset_min=self.bins.fov_offset_min * u.deg,
            fov_offset_max=self.bins.fov_offset_max * u.deg,
        )
        self.sensitivity = calculate_sensitivity(
            signal_hist, background_hist, alpha=self.alpha
        )

        # scale relative sensitivity by Crab flux to get the flux sensitivity
        for s in (self.sens2, self.sensitivity):
            s["flux_sensitivity"] = s["relative_sensitivity"] * self.spectrum(
                s["reco_energy_center"]
            )

    def finish(self):
        masks = {
            "": self.signal_events["selected"],
            "_NO_CUTS": slice(None),
            "_ONLY_GH": self.signal_events["selected_gh"],
            "_ONLY_THETA": self.signal_events["selected_theta"],
        }
        hdus = [
            fits.PrimaryHDU(),
            fits.BinTableHDU(self.sensitivity, name="SENSITIVITY"),
            fits.BinTableHDU(self.sens2, name="SENSITIVITY_STEP_2"),
            fits.BinTableHDU(self.theta_cuts_opt, name="THETA_CUTS_OPT"),
            fits.BinTableHDU(self.gh_cuts, name="GH_CUTS"),
        ]

        self.log.debug("True Energy bins: %s" % str(self.true_energy_bins.value))
        self.log.debug("FoV offset bins: %s" % str(self.fov_offset_bins.value))
        for label, mask in masks.items():
            effective_area = effective_area_per_energy_and_fov(
                self.signal_events[mask],
                self.sim_info,
                true_energy_bins=self.true_energy_bins,
                fov_offset_bins=self.fov_offset_bins,
            )
            hdus.append(
                create_aeff2d_hdu(
                    effective_area[..., np.newaxis],  # +1 dimension for FOV offset
                    self.true_energy_bins,
                    self.fov_offset_bins,
                    extname="EFFECTIVE AREA" + label,
                )
            )
            edisp = energy_dispersion(
                self.signal_events[mask],
                true_energy_bins=self.true_energy_bins,
                fov_offset_bins=self.fov_offset_bins,
                migration_bins=self.energy_migration_bins,
            )
            hdus.append(
                create_energy_dispersion_hdu(
                    edisp,
                    true_energy_bins=self.true_energy_bins,
                    migration_bins=self.energy_migration_bins,
                    fov_offset_bins=self.fov_offset_bins,
                    extname="ENERGY_DISPERSION" + label,
                )
            )
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

        sel = self.background_events["selected_gh"]
        self.log.debug("%d background events selected" % sel.sum())
        self.log.debug("%f obs time" % self.obs_time)
        background_rate = background_2d(
            self.background_events[sel],
            self.reco_energy_bins,
            fov_offset_bins=self.fov_offset_bins,
            t_obs=self.obs_time * u.Unit(self.obs_time_unit),
        )
        hdus.append(
            create_background_2d_hdu(
                background_rate,
                self.reco_energy_bins,
                fov_offset_bins=self.fov_offset_bins,
            )
        )

        psf = psf_table(
            self.signal_events[self.signal_events["selected_gh"]],
            self.true_energy_bins,
            fov_offset_bins=self.fov_offset_bins,
            source_offset_bins=self.source_offset_bins,
        )
        hdus.append(
            create_psf_table_hdu(
                psf,
                self.true_energy_bins,
                self.source_offset_bins,
                self.fov_offset_bins,
            )
        )

        hdus.append(
            create_rad_max_hdu(
                self.theta_cuts_opt["cut"].reshape(-1, 1),
                self.true_energy_bins,
                self.fov_offset_bins,
            )
        )

        self.log.info("Writing outputfile '%s'" % self.output_path)
        fits.HDUList(hdus).writeto(
            self.output_path,
            overwrite=self.overwrite,
        )
        Provenance().add_output_file(self.output_path, role="IRF")


def main():
    tool = IrfTool()
    tool.run()


if __name__ == "main":
    main()
