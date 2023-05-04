"""Tool to generate IRFs"""
import operator

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.table import vstack
from pyirf.benchmarks import angular_resolution, energy_bias_resolution
from pyirf.binning import (
    add_overflow_bins,
    create_bins_per_decade,
    create_histogram_table,
)
from pyirf.cut_optimization import optimize_gh_cut
from pyirf.cuts import calculate_percentile_cut, evaluate_binned_cut
from pyirf.io import (
    create_aeff2d_hdu,
    create_background_2d_hdu,
    create_energy_dispersion_hdu,
    create_psf_table_hdu,
    create_rad_max_hdu,
)
from pyirf.irf import (
    background_2d,
    effective_area_per_energy,
    energy_dispersion,
    psf_table,
)
from pyirf.sensitivity import calculate_sensitivity, estimate_background
from pyirf.simulations import SimulatedEventsInfo
from pyirf.spectral import (
    CRAB_HEGRA,
    IRFDOC_ELECTRON_SPECTRUM,
    IRFDOC_PROTON_SPECTRUM,
    PowerLaw,
    calculate_event_weights,
)
from pyirf.utils import calculate_source_fov_offset, calculate_theta

from ..core import Provenance, Tool
from ..io import TableLoader
from ..irf import DataBinning, ToolConfig, EventPreProcessor

PYIRF_SPECTRA = {
    "CRAB_HEGRA": CRAB_HEGRA,
    "IRFDOC_ELECTRON_SPECTRUM": IRFDOC_ELECTRON_SPECTRUM,
    "IRFDOC_PROTON_SPECTRUM": IRFDOC_PROTON_SPECTRUM,
}


class IrfTool(Tool):
    name = "ctapipe-make-irfs"
    description = "Tool to create IRF files in GAD format"

    classes = [ DataBinning, ToolConfig,EventPreProcessor]

    def make_derived_columns(self,kind, events, spectrum, target_spectrum):
        events["pointing_az"] = 0 * u.deg
        events["pointing_alt"] = 70 * u.deg

        events["theta"] = calculate_theta(
            events,
            assumed_source_az=events["true_az"],
            assumed_source_alt=events["true_alt"],
        )

        events["true_source_fov_offset"] = calculate_source_fov_offset(
            events, prefix="true"
        )
        events["reco_source_fov_offset"] = calculate_source_fov_offset(
            events, prefix="reco"
        )
        # Gamma source is assumed to be pointlike 
        if kind == "gamma":
            spectrum = spectrum.integrate_cone(0*u.deg,self.tc.ON_radius*u.deg)
        events["weight"] = calculate_event_weights(
            events["true_energy"],
            target_spectrum=target_spectrum,
            simulated_spectrum=spectrum,
        )

        return events

    def get_sim_info_and_spectrum(self, loader):
        sim = loader.read_simulation_configuration()

        # These sims better have the same viewcone!
        assert( sim["max_viewcone_radius"].std() == 0)
        sim_info = SimulatedEventsInfo(
            n_showers=sum(sim["n_showers"] * sim["shower_reuse"]),
            energy_min=sim["energy_range_min"].quantity[0],
            energy_max=sim["energy_range_max"].quantity[0],
            max_impact=sim["max_scatter_range"].quantity[0],
            spectral_index=sim["spectral_index"][0],
            viewcone=sim["max_viewcone_radius"].quantity[0],
        )

        return sim_info, PowerLaw.from_simulation(
            sim_info, obstime=self.tc.obs_time * u.Unit(self.tc.obs_time_unit)
        )

    def setup(self):
        self.tc = ToolConfig(parent=self)
        self.bins = DataBinning(parent=self)
        self.eps = EventPreProcessor(parent=self)

        opts = dict(load_dl2=True, load_simulated=True, load_dl1_parameters=False)
        reduced_events = dict()
        for kind, file, target_spectrum in [
            ("gamma", self.tc.gamma_file, PYIRF_SPECTRA[self.tc.gamma_sim_spectrum]),
            ("proton", self.tc.proton_file, PYIRF_SPECTRA[self.tc.proton_sim_spectrum]),
            ("electron", self.tc.electron_file, PYIRF_SPECTRA[self.tc.electron_sim_spectrum]),
        ]:
            with TableLoader(file, **opts) as load:
                Provenance().add_input_file(file)
                table = self.eps._make_empty_table()
                sim_info, spectrum = self.get_sim_info_and_spectrum(load)
                if kind == "gamma":
                    self.sim_info = sim_info
                    self.spectrum = spectrum
                for start, stop, events in load.read_subarray_events_chunked(
                    self.tc.chunk_size
                ):
                    selected = self.eps._preselect_events(events)
                    selected = self.make_derived_columns(
                        kind,
                        selected, spectrum, target_spectrum
                    )
                    table = vstack([table, selected])

                reduced_events[kind] = table

        select_ON = reduced_events["gamma"]["theta"] <= self.tc.ON_radius*u.deg
        self.signal = reduced_events["gamma"][select_ON]
        self.background = vstack([reduced_events["proton"], reduced_events["electron"]])

        self.theta_bins = add_overflow_bins(
            create_bins_per_decade(
                self.sim_info.energy_min, self.sim_info.energy_max, 50
            )
        )

        self.reco_energy_bins = self.bins.reco_energy_bins()
        self.true_energy_bins = self.bins.true_energy_bins()
        self.source_offset_bins = self.bins.source_offset_bins()
        self.fov_offset_bins = self.bins.fov_offset_bins()
        self.bkg_fov_offset_bins = self.bins.bkg_fov_offset_bins()
        self.energy_migration_bins = self.bins.energy_migration_bins()

    def start(self):
        INITIAL_GH_CUT = np.quantile(
            self.signal["gh_score"], (1 - self.tc.initial_gh_cut_efficency)
        )
        self.log.info(
            f"Using fixed G/H cut of {INITIAL_GH_CUT} to calculate theta cuts"
        )

        mask_theta_cuts = self.signal["gh_score"] >= INITIAL_GH_CUT

        theta_cuts = calculate_percentile_cut(
            self.signal["theta"][mask_theta_cuts],
            self.signal["reco_energy"][mask_theta_cuts],
            bins=self.theta_bins,
            min_value=self.bins.theta_min_angle * u.deg,
            max_value=self.bins.theta_max_angle * u.deg,
            fill_value=self.bins.theta_fill_value * u.deg,
            min_events=self.bins.theta_min_counts,
            percentile=68,
        )

        self.log.info("Optimizing G/H separation cut for best sensitivity")
        gh_cut_efficiencies = np.arange(
            self.tc.gh_cut_efficiency_step,
            self.tc.max_gh_cut_efficiency + self.tc.gh_cut_efficiency_step / 2,
            self.tc.gh_cut_efficiency_step,
        )

        sens2, self.gh_cuts = optimize_gh_cut(
            self.signal,
            self.background,
            reco_energy_bins=self.reco_energy_bins,
            gh_cut_efficiencies=gh_cut_efficiencies,
            op=operator.ge,
            theta_cuts=theta_cuts,
            alpha=self.tc.alpha,
            fov_offset_max=self.tc.max_bg_radius * u.deg,
        )

        # now that we have the optimized gh cuts, we recalculate the theta
        # cut as 68 percent containment on the events surviving these cuts.
        self.log.info("Recalculating theta cut for optimized GH Cuts")
        for tab in (self.signal, self.background):
            tab["selected_gh"] = evaluate_binned_cut(
                tab["gh_score"], tab["reco_energy"], self.gh_cuts, operator.ge
            )

        self.theta_cuts_opt = calculate_percentile_cut(
            self.signal[self.signal["selected_gh"]]["theta"],
            self.signal[self.signal["selected_gh"]]["reco_energy"],
            self.theta_bins,
            percentile=68,
            min_value=self.bins.theta_min_angle * u.deg,
            max_value=self.bins.theta_max_angle * u.deg,
            fill_value=self.bins.theta_fill_value * u.deg,
            min_events=self.bins.theta_min_counts,
        )
        self.signal["selected_theta"] = evaluate_binned_cut(
            self.signal["theta"],
            self.signal["reco_energy"],
            self.theta_cuts_opt,
            operator.le,
        )
        self.signal["selected"] = (
            self.signal["selected_theta"] & self.signal["selected_gh"]
        )

        # calculate sensitivity
        signal_hist = create_histogram_table(
            self.signal[self.signal["selected"]], bins=self.reco_energy_bins
        )

        background_hist = estimate_background(
            self.background[self.background["selected_gh"]],
            reco_energy_bins=self.reco_energy_bins,
            theta_cuts=self.theta_cuts_opt,
            alpha=self.tc.alpha,
            fov_offset_min=self.bins.fov_offset_min*u.deg,
            fov_offset_max=self.bins.fov_offset_max*u.deg,
        )
        self.sensitivity = calculate_sensitivity(
            signal_hist, background_hist, alpha=self.tc.alpha
        )

        # scale relative sensitivity by Crab flux to get the flux sensitivity
        for s in (sens2, self.sensitivity):
            s["flux_sensitivity"] = s["relative_sensitivity"] * self.spectrum(
                s["reco_energy_center"]
            )

    def finish(self):
        masks = {
            "": self.signal["selected"],
            "_NO_CUTS": slice(None),
            "_ONLY_GH": self.signal["selected_gh"],
            "_ONLY_THETA": self.signal["selected_theta"],
        }
        hdus = [
            fits.PrimaryHDU(),
            fits.BinTableHDU(self.sensitivity, name="SENSITIVITY"),
            #            fits.BinTableHDU(sensitivity_step_2, name="SENSITIVITY_STEP_2"),
            #            fits.BinTableHDU(self.theta_cuts, name="THETA_CUTS"),
            fits.BinTableHDU(self.theta_cuts_opt, name="THETA_CUTS_OPT"),
            fits.BinTableHDU(self.gh_cuts, name="GH_CUTS"),
        ]

        for label, mask in masks.items():
            effective_area = effective_area_per_energy(
                self.signal[mask],
                self.sim_info,
                true_energy_bins=self.true_energy_bins,
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
                self.signal[mask],
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
            self.signal[self.signal["selected"]],
            self.reco_energy_bins,
            energy_type="reco",
        )

        # Here we use reconstructed energy instead of true energy for the sake of
        # current pipelines comparisons
        ang_res = angular_resolution(
            self.signal[self.signal["selected_gh"]],
            self.reco_energy_bins,
            energy_type="reco",
        )

        psf = psf_table(
            self.signal[self.signal["selected_gh"]],
            self.true_energy_bins,
            fov_offset_bins=self.fov_offset_bins,
            source_offset_bins=self.source_offset_bins,
        )

        background_rate = background_2d(
            self.background[self.background["selected_gh"]],
            self.reco_energy_bins,
            fov_offset_bins=self.bkg_fov_offset_bins,
            t_obs=self.tc.obs_time * u.Unit(self.tc.obs_time_unit),
        )

        hdus.append(
            create_background_2d_hdu(
                background_rate,
                self.reco_energy_bins,
                fov_offset_bins=self.bkg_fov_offset_bins,
            )
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
                self.theta_cuts_opt["cut"][:, np.newaxis],
                self.theta_bins,
                self.fov_offset_bins,
            )
        )
        hdus.append(fits.BinTableHDU(ang_res, name="ANGULAR_RESOLUTION"))
        hdus.append(fits.BinTableHDU(bias_resolution, name="ENERGY_BIAS_RESOLUTION"))

        self.log.info("Writing outputfile '%s'" % self.tc.output_path)
        fits.HDUList(hdus).writeto(
            self.tc.output_path,
            overwrite=self.overwrite,
        )


def main():
    tool = IrfTool()
    tool.run()


if __name__ == "main":
    main()
