"""Tool to generate IRFs"""
import operator

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.table import vstack
from pyirf.cuts import evaluate_binned_cut
from pyirf.io import (
    create_aeff2d_hdu,
    create_energy_dispersion_hdu,
    create_psf_table_hdu,
    create_rad_max_hdu,
)
from pyirf.irf import effective_area_per_energy_and_fov, energy_dispersion, psf_table

from ..core import Provenance, Tool, traits
from ..core.traits import Bool, Float, Integer, Unicode
from ..irf import (
    PYIRF_SPECTRA,
    EventPreProcessor,
    EventsLoader,
    GridOptimizer,
    OptimisationResultSaver,
    OutputEnergyBinning,
    SourceOffsetBinning,
    Spectra,
    ThetaCutsCalculator,
)


class IrfTool(Tool):
    name = "ctapipe-make-irf"
    description = "Tool to create IRF files in GAD format"

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
        GridOptimizer,
        SourceOffsetBinning,
        OutputEnergyBinning,
        EventPreProcessor,
    ]

    def calculate_selections(self):
        self.signal_events["selected_gh"] = evaluate_binned_cut(
            self.signal_events["gh_score"],
            self.signal_events["reco_energy"],
            self.opt_result.gh_cuts,
            operator.ge,
        )
        self.background_events["selected_gh"] = evaluate_binned_cut(
            self.background_events["gh_score"],
            self.background_events["reco_energy"],
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
        self.background_events["selected_theta"] = evaluate_binned_cut(
            self.background_events["theta"],
            self.background_events["reco_energy"],
            self.theta_cuts_opt,
            operator.le,
        )
        self.signal_events["selected"] = (
            self.signal_events["selected_theta"] & self.signal_events["selected_gh"]
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

    def _check_bins_in_range(self, bins, range):
        low = bins >= range.min
        hig = bins <= range.max

        if not all(low & hig):
            raise ValueError(f"Valid range is {range.min} to {range.max}, got {bins}")

    def setup(self):
        self.theta = ThetaCutsCalculator(parent=self)
        self.e_bins = OutputEnergyBinning(parent=self)
        self.bins = SourceOffsetBinning(parent=self)

        self.opt_result = OptimisationResultSaver().read(self.cuts_file)
        # TODO: not very elegant to pass them this way, refactor later
        self.epp = EventPreProcessor(parent=self)
        self.epp.quality_criteria = self.opt_result.precuts.quality_criteria
        self.reco_energy_bins = self.e_bins.reco_energy_bins()
        self.true_energy_bins = self.e_bins.true_energy_bins()
        self.energy_migration_bins = self.e_bins.energy_migration_bins()
        self.source_offset_bins = self.bins.source_offset_bins()

        self._check_bins_in_range(self.reco_energy_bins, self.opt_result.valid_energy)
        self._check_bins_in_range(self.source_offset_bins, self.opt_result.valid_offset)
        self.fov_offset_bins = self.opt_result.valid_offset.bins
        self.particles = [
            EventsLoader(
                self.epp,
                "gammas",
                self.gamma_file,
                PYIRF_SPECTRA[self.gamma_sim_spectrum],
            ),
            EventsLoader(
                self.epp,
                "protons",
                self.proton_file,
                PYIRF_SPECTRA[self.proton_sim_spectrum],
            ),
            EventsLoader(
                self.epp,
                "electrons",
                self.electron_file,
                PYIRF_SPECTRA[self.electron_sim_spectrum],
            ),
        ]

    def start(self):
        # TODO: this event loading code seems to be largely repeated between all the tools,
        # try to refactor to a common solution
        reduced_events = dict()
        for sel in self.particles:
            evs, cnt, meta = sel.load_preselected_events(
                self.chunk_size,
                self.obs_time * u.Unit(self.obs_time_unit),
                self.fov_offset_bins,
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

        self.signal_events = reduced_events["gammas"]
        self.background_events = vstack(
            [reduced_events["protons"], reduced_events["electrons"]]
        )

        self.calculate_selections()

        hdus = [
            fits.PrimaryHDU(),
        ]
        self.log.debug("True Energy bins: %s" % str(self.true_energy_bins.value))
        self.log.debug("FoV offset bins: %s" % str(self.fov_offset_bins))
        effective_area = effective_area_per_energy_and_fov(
            self.signal_events[self.signal_events["selected"]],
            self.sim_info,
            true_energy_bins=self.true_energy_bins,
            fov_offset_bins=self.fov_offset_bins,
        )
        hdus.append(
            create_aeff2d_hdu(
                effective_area[..., np.newaxis],  # +1 dimension for FOV offset
                self.true_energy_bins,
                self.fov_offset_bins,
                extname="EFFECTIVE AREA",
            )
        )
        edisp = energy_dispersion(
            self.signal_events[self.signal_events["selected"]],
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
                extname="ENERGY_DISPERSION",
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
        self.hdus = hdus

    def finish(self):

        self.log.info("Writing outputfile '%s'" % self.output_path)
        fits.HDUList(self.hdus).writeto(
            self.output_path,
            overwrite=self.overwrite,
        )
        Provenance().add_output_file(self.output_path, role="IRF")


def main():
    tool = IrfTool()
    tool.run()


if __name__ == "main":
    main()
