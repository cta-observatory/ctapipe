"""Tool to generate IRFs"""
import operator
from enum import Enum

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.table import vstack
from pyirf.benchmarks import angular_resolution, energy_bias_resolution
from pyirf.binning import create_histogram_table
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

from ..core import Provenance, Tool, traits
from ..core.traits import Bool, Float, Integer, Unicode
from ..io import TableLoader
from ..irf import CutOptimising, DataBinning, EnergyBinning, EventPreProcessor


class Spectra(Enum):
    CRAB_HEGRA = 1
    IRFDOC_ELECTRON_SPECTRUM = 2
    IRFDOC_PROTON_SPECTRUM = 3


PYIRF_SPECTRA = {
    Spectra.CRAB_HEGRA: CRAB_HEGRA,
    Spectra.IRFDOC_ELECTRON_SPECTRUM: IRFDOC_ELECTRON_SPECTRUM,
    Spectra.IRFDOC_PROTON_SPECTRUM: IRFDOC_PROTON_SPECTRUM,
}


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
    ON_radius = Float(default_value=1.0, help="Radius of ON region in degrees").tag(
        config=True
    )
    max_bg_radius = Float(
        default_value=3.0, help="Radius used to calculate background rate in degrees"
    ).tag(config=True)

    classes = [CutOptimising, DataBinning, EnergyBinning, EventPreProcessor]

    def make_derived_columns(self, kind, events, spectrum, target_spectrum, obs_conf):

        if obs_conf["subarray_pointing_lat"].std() < 1e-3:
            assert all(obs_conf["subarray_pointing_frame"] == 0)
            # Lets suppose 0 means ALTAZ
            events["pointing_alt"] = obs_conf["subarray_pointing_lat"][0] * u.deg
            events["pointing_az"] = obs_conf["subarray_pointing_lon"][0] * u.deg
        else:
            raise NotImplementedError(
                "No support for making irfs from varying pointings yet"
            )

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
            spectrum = spectrum.integrate_cone(0 * u.deg, self.ON_radius * u.deg)
        events["weight"] = calculate_event_weights(
            events["true_energy"],
            target_spectrum=target_spectrum,
            simulated_spectrum=spectrum,
        )

        return events

    def get_metadata(self, loader):
        obs = loader.read_observation_information()
        sim = loader.read_simulation_configuration()

        # These sims better have the same viewcone!
        assert sim["max_viewcone_radius"].std() == 0
        sim_info = SimulatedEventsInfo(
            n_showers=sum(sim["n_showers"] * sim["shower_reuse"]),
            energy_min=sim["energy_range_min"].quantity[0],
            energy_max=sim["energy_range_max"].quantity[0],
            max_impact=sim["max_scatter_range"].quantity[0],
            spectral_index=sim["spectral_index"][0],
            viewcone=sim["max_viewcone_radius"].quantity[0],
        )

        return (
            sim_info,
            PowerLaw.from_simulation(
                sim_info, obstime=self.obs_time * u.Unit(self.obs_time_unit)
            ),
            obs,
        )

    def load_preselected_events(self):
        opts = dict(load_dl2=True, load_simulated=True, load_dl1_parameters=False)
        reduced_events = dict()
        for kind, file, target_spectrum in [
            ("gamma", self.gamma_file, PYIRF_SPECTRA[self.gamma_sim_spectrum]),
            ("proton", self.proton_file, PYIRF_SPECTRA[self.proton_sim_spectrum]),
            (
                "electron",
                self.electron_file,
                PYIRF_SPECTRA[self.electron_sim_spectrum],
            ),
        ]:
            with TableLoader(file, **opts) as load:
                Provenance().add_input_file(file)
                header = self.epp.make_empty_table()
                sim_info, spectrum, obs_conf = self.get_metadata(load)
                if kind == "gamma":
                    self.sim_info = sim_info
                    self.spectrum = spectrum
                bits = [header]
                for start, stop, events in load.read_subarray_events_chunked(
                    self.chunk_size
                ):
                    selected = self.epp.normalise_column_names(events)
                    selected = selected[self.epp.get_table_mask(selected)]
                    selected = self.make_derived_columns(
                        kind, selected, spectrum, target_spectrum, obs_conf
                    )
                    bits.append(selected)

                table = vstack(bits, join_type="exact")
                reduced_events[kind] = table

        select_ON = reduced_events["gamma"]["theta"] <= self.ON_radius * u.deg
        self.signal = reduced_events["gamma"][select_ON]
        self.background = vstack([reduced_events["proton"], reduced_events["electron"]])

    def setup(self):
        self.co = CutOptimising(parent=self)
        self.e_bins = EnergyBinning(parent=self)
        self.bins = DataBinning(parent=self)
        self.epp = EventPreProcessor(parent=self)

        self.reco_energy_bins = self.e_bins.reco_energy_bins()
        self.true_energy_bins = self.e_bins.true_energy_bins()
        self.energy_migration_bins = self.e_bins.energy_migration_bins()

        self.source_offset_bins = self.bins.source_offset_bins()
        self.fov_offset_bins = self.bins.fov_offset_bins()
        self.bkg_fov_offset_bins = self.bins.bkg_fov_offset_bins()

    def start(self):
        self.load_preselected_events()
        self.gh_cuts, sens2 = self.co.optimise_gh_cut(
            self.signal,
            self.background,
            self.true_energy_bins,
            self.reco_energy_bins,
            self.bins,
            self.alpha,
            self.max_bg_radius,
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
            self.true_energy_bins,
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
            alpha=self.alpha,
            fov_offset_min=self.bins.fov_offset_min * u.deg,
            fov_offset_max=self.bins.fov_offset_max * u.deg,
        )
        self.sensitivity = calculate_sensitivity(
            signal_hist, background_hist, alpha=self.alpha
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
            self.log.debug(self.true_energy_bins)
            self.log.debug(self.fov_offset_bins)
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
        hdus.append(fits.BinTableHDU(bias_resolution, name="ENERGY_BIAS_RESOLUTION"))

        # Here we use reconstructed energy instead of true energy for the sake of
        # current pipelines comparisons
        ang_res = angular_resolution(
            self.signal[self.signal["selected_gh"]],
            self.reco_energy_bins,
            energy_type="reco",
        )
        hdus.append(fits.BinTableHDU(ang_res, name="ANGULAR_RESOLUTION"))

        background_rate = background_2d(
            self.background[self.background["selected_gh"]],
            self.reco_energy_bins,
            fov_offset_bins=self.bkg_fov_offset_bins,
            t_obs=self.obs_time * u.Unit(self.obs_time_unit),
        )
        hdus.append(
            create_background_2d_hdu(
                background_rate,
                self.reco_energy_bins,
                fov_offset_bins=self.bkg_fov_offset_bins,
            )
        )

        psf = psf_table(
            self.signal[self.signal["selected_gh"]],
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


def main():
    tool = IrfTool()
    tool.run()


if __name__ == "main":
    main()
