"""
Estimate the level of NSB required to match MC simulations to observations.
"""

import pathlib

import numpy as np
import yaml
from scipy.stats import median_abs_deviation
from traitlets import Float, Int, List, Path, TraitError

from ctapipe.calib.camera.gainselection import GainChannel
from ctapipe.core import Provenance, Tool
from ctapipe.core.traits import ComponentName, classes_with_traits
from ctapipe.image.extractor import ImageExtractor, VarianceExtractor
from ctapipe.image.modifications import PedestalEventTypeFilter, WaveformModifier
from ctapipe.io import EventSource


class EstimateNSBlevelTool(Tool):
    """
    Tool to estimate the level of NSB required to match the diffuse
    NSB from a NSB-only simulation file to a target observation.

    The NSB-only file must be produced with the same telescope
    array configuration and simulation settings to the MC files
    to add the NSB.

    This tool estimates the NSB as follows:
    1) Exclude faulty pixels (broken and outliers) from real observations
    from the median of the waveform variance per pixel.
    2) From median pixel charge variance of NSB-only file and median
    variance of (sky) pedestals, the approximate NSB level can be computed.
    3) Compute the 90% quantile for the real data (excluding outliers) and
    NSB-only MC 90% quantile for different NSB level.
    4) Fit the MC 90% quantile for different NSB level with a parabola and
    find the closer node to the real 90% quantile.
    """

    name = "ctapipe-estimate-nsb"
    description = __doc__

    # Required parameters:
    image_extractor_type = ComponentName(
        ImageExtractor,
        default_value="FixedWindowSum",
        help="Name of the ImageExtractor subclass to be used.",
    ).tag(config=True)

    nsb_file = Path(
        help="Path to a dedicated NSB-only file (requires r1 waveforms)",
    ).tag(config=True)

    shower_file = Path(
        help="Path to shower file to estimate the NSB level (requires dl0 "
        "for real data, r1 for simulations)",
    ).tag(config=True)

    # Optional parameters:
    output_path = Path(
        help="Output filename", default_value=pathlib.Path("nsb_params.yaml")
    ).tag(config=True)

    # Parameters to identify faulty pixel
    mod_zscore_th = Float(
        default_value=3.5,
        help="Modified Z-score threshold for identifying bright outlier pixels.",
    ).tag(config=True)

    min_npixels_valid = Int(
        default_value=1000,
        help="Minimum number of usable pixels (excluding outliers and broken pixels) needed to estimate the pedestal statistics",
    ).tag(config=True)

    min_nped_events = Int(
        default_value=5,
        help="Minimum number of pedestal events needed to estimate the pedestal statistics per pixel.",
    ).tag(config=True)

    # Other parameters
    nevents = Int(
        default_value=20,
        help="Number of noise realisations in WaveformModifier",
    ).tag(config=True)

    quantile_threshold = Float(
        default_value=0.9,
        help="The quantile threshold to compute the quantile of the pixel charge distribution",
    ).tag(config=True)

    n_grid_points = Int(
        default_value=3,
        help=(
            "Number of NSB realisations in the NSB grid below and above the obtained NSB through the "
            "variance method. It is not used if nsb_range is different than None"
        ),
    ).tag(config=True)

    nsb_range = List(
        trait=Int(),
        default_value=None,
        allow_none=True,
        help=(
            "Defines the range of the NSB levels (lower, upper) in the grid. "
            "None means that the range is estimated from the variance method."
        ),
    ).tag(config=True)

    aliases = {
        ("s", "shower-file"): "EstimateNSBlevelTool.shower_file",
        ("n", "nsb-file"): "EstimateNSBlevelTool.nsb_file",
    }

    classes = [
        ImageExtractor,
        WaveformModifier,
    ] + classes_with_traits(ImageExtractor)

    def setup(self):
        if self.shower_file is None:
            self.log.critical("Specifying 'shower_file' is required.")
            self.exit(1)
        if self.nsb_file is None:
            self.log.critical("Specifying 'nsb_file' is required.")
            self.exit(1)

        self.check_output(self.output_path)

        with EventSource(self.shower_file) as source:
            self.subarray = source.subarray

        self.image_extractor = ImageExtractor.from_name(
            self.image_extractor_type,
            subarray=self.subarray,
            parent=self,
        )

        # Verify that nsb_range only contains
        if self.nsb_range is not None and len(self.nsb_range) != 2:
            raise TraitError("`nsb_range` must contain two element: [lower, upper]")

        self.event_type_filter = PedestalEventTypeFilter(parent=self)

    def _extract_images_per_pixel_and_event(self, extractor):
        """
        Function to extract images (gain selected) per pixel and event for a certain extractor.

        Output dimensions of pixel_data: (n_events, n_pixels), broken_pixels (n_pixels)
        """

        pixel_data = {}
        broken_pixels = {}

        with EventSource(self.shower_file) as shower_source:
            for event in shower_source:
                if shower_source.is_simulation:
                    for tel_id in event.trigger.tels_with_trigger:
                        true_pe = event.simulation.tel[tel_id].true_image
                        nwf = event.r1.tel[tel_id].waveform

                        # select noise-only waveforms
                        nwf[:, true_pe > 0, :] = np.nan

                        # use high gain for pedestals
                        selected_gain_channel = np.broadcast_to(
                            GainChannel.HIGH,
                            shower_source.subarray.tel[tel_id].camera.geometry.n_pixels,
                        )
                        broken_pixels[tel_id] = event.monitoring.tel[
                            tel_id
                        ].camera.coefficients.outlier_mask[GainChannel.HIGH]
                        q_ev = self.image_extractor(
                            waveforms=nwf,
                            tel_id=tel_id,
                            selected_gain_channel=selected_gain_channel,
                            broken_pixels=broken_pixels[tel_id],
                        )

                        if tel_id in pixel_data:
                            pixel_data[tel_id].append(q_ev.image)
                        else:
                            pixel_data[tel_id] = [q_ev.image]

                else:
                    if not self.event_type_filter(event):
                        continue

                    for tel_id, dl0 in event.dl0.tel.items():
                        broken_pixels[tel_id] = event.monitoring.tel[
                            tel_id
                        ].camera.coefficients.outlier_mask
                        q_ev = extractor(
                            waveforms=dl0.waveform,
                            tel_id=tel_id,
                            selected_gain_channel=dl0.selected_gain_channel,
                            broken_pixels=broken_pixels[tel_id],
                        )

                        if tel_id in pixel_data:
                            pixel_data[tel_id].append(q_ev.image)
                        else:
                            pixel_data[tel_id] = [q_ev.image]

        return pixel_data, broken_pixels

    def calculate_pixel_charge_ped(self):

        self.ped_pixel_charge, self.bool_broken_pixels = (
            self._extract_images_per_pixel_and_event(self.image_extractor)
        )
        self.tel_id_list = list(self.ped_pixel_charge.keys())

    def identify_faulty_pixels(self):

        variance_extractor = VarianceExtractor(subarray=self.subarray)
        shower_pixel_variance, broken_pixels = self._extract_images_per_pixel_and_event(
            variance_extractor
        )

        self.faulty_pixels = {}
        self.pixel_var = {}
        for tel_id in np.sort(self.tel_id_list):
            # faulty pixels = bright pixels + broken pixels
            if broken_pixels[tel_id].ndim == 2:
                mask_broken = broken_pixels[tel_id][GainChannel.HIGH]
                self.pixel_var[tel_id] = np.nanmedian(
                    np.array(shower_pixel_variance[tel_id])[:, GainChannel.HIGH, :],
                    axis=0,
                )
            else:
                mask_broken = broken_pixels[tel_id]
                self.pixel_var[tel_id] = np.nanmedian(
                    np.array(shower_pixel_variance[tel_id]), axis=0
                )

            mad_pix = median_abs_deviation(self.pixel_var[tel_id])
            median_pix = np.median(self.pixel_var[tel_id])
            abs_mod_zscore = np.abs(
                (self.pixel_var[tel_id] - median_pix) / (1.486 * mad_pix)
            )

            # Obtain outliers pixels
            mask_outliers = abs_mod_zscore > self.mod_zscore_th

            self.faulty_pixels[tel_id] = mask_outliers | mask_broken

            n_valid = np.sum(~self.faulty_pixels[tel_id])
            if n_valid < self.min_npixels_valid:
                raise ValueError(
                    f"The number of faulty pixels (outliers and broken pixels) {n_valid} "
                    + f"for tel_id={tel_id} are too large to estimate pedestal statistics"
                )
            n_ped_events = np.shape(self.ped_pixel_charge.get(tel_id, []))[0]
            if n_ped_events < self.min_nped_events:
                raise ValueError(
                    f"The number of pedestal events ({n_ped_events}) for "
                    + f"tel_id={tel_id} is to small to obtain good estimates for pedestal statistics"
                )

    def calculate_pixel_charge_nsb_only_grid(self, nsb_levels, nevents):
        """
        Compute the charge NSB-only waveforms for nevents events
        for different NSB levels.
        The pixel charge is computed for each telescope, event, and NSB.
        """

        self.nsb_levels = nsb_levels
        self.nsb_pixel_charge = {}

        with EventSource(
            input_url=self.nsb_file, skip_calibration_events=False
        ) as source:
            for tel_id in self.tel_id_list:
                camera_geom = self.subarray.tel[tel_id].camera.geometry
                self.nsb_pixel_charge[tel_id] = np.zeros(
                    (len(self.nsb_levels), nevents, camera_geom.n_pixels)
                )

            for i_nsb, nsb in enumerate(self.nsb_levels):
                wf_mod = WaveformModifier(
                    subarray=source.subarray,
                    nsb_file=self.nsb_file,
                    nsb_level=nsb,
                    n_noise_realizations=nevents,
                )

                for tel_id in self.tel_id_list:
                    if tel_id not in wf_mod.total_noise:
                        raise KeyError(
                            f"File {self.nsb_file} does not contain tel_id={tel_id}"
                        )

                    camera_geom = source.subarray.tel[tel_id].camera.geometry
                    # Use only high gain
                    selected_gain_channel = np.broadcast_to(
                        GainChannel.HIGH, camera_geom.n_pixels
                    )
                    # Consider no broken pixels
                    bool_broken_pixels = np.zeros(camera_geom.n_pixels, dtype=bool)

                    for i_ev in range(nevents):
                        q_ped = self.image_extractor(
                            waveforms=wf_mod.total_noise[tel_id][i_ev],
                            tel_id=tel_id,
                            selected_gain_channel=selected_gain_channel,
                            broken_pixels=bool_broken_pixels,
                        )

                        self.nsb_pixel_charge[tel_id][i_nsb][i_ev] = q_ped.image

        return self.nsb_pixel_charge

    def nsb_from_variance(self, nsb_lvl1_pixel_charge):
        """
        Estimate the nsb from the ratio between variances of the pixel charge

        The obtained results may yield wrong estimated for biased extractors
        and small extractor windows.

        Parameters
        ----------
        nsb_lvl1_pixel_charge: dict with (n_nsb_levels, n_events, n_pixels)
            Pixel charges.
        """
        nsb_lvl = {}
        for tel_id in self.tel_id_list:
            # pixel charge variance per pixel
            ped_var_pix = np.nanvar(self.ped_pixel_charge[tel_id], axis=0)

            # obtain the pixel charge variance per pixel from the lowest nsb level
            ped_sim_var_pix = np.nanvar(nsb_lvl1_pixel_charge[tel_id], axis=(1))[0]

            # median pixel charge variance
            ped_var_pix_median = np.median(ped_var_pix[~self.faulty_pixels[tel_id]])
            ped_sim_var_pix_median = np.median(
                ped_sim_var_pix[~self.faulty_pixels[tel_id]]
            )

            nsb_lvl[tel_id] = ped_var_pix_median / ped_sim_var_pix_median

        nsb_min = np.floor(
            max(nsb_lvl[min(nsb_lvl, key=nsb_lvl.get)] - self.n_grid_points, 1)
        ).astype(int)
        nsb_max = np.ceil(
            nsb_lvl[max(nsb_lvl, key=nsb_lvl.get)] + self.n_grid_points + 1
        ).astype(int)
        nsb_grid = np.arange(nsb_min, nsb_max)

        return nsb_grid, nsb_lvl

    def quantile_curve_fit(self):
        """
        Compute the quantile at quantile_threshold. Fit a parabola to the quantile curve
        vs NSB and find the NSB level that matches the real data.
        """

        self.qsim_q = {}
        self.qped_q = {}
        for tel_id in self.tel_id_list:
            shape_ped = np.shape(self.nsb_pixel_charge[tel_id])
            all_nsb_pixel_charge = np.reshape(
                self.nsb_pixel_charge[tel_id],
                (shape_ped[0], shape_ped[1] * shape_ped[2]),
            )
            self.qsim_q[tel_id] = np.nanquantile(
                all_nsb_pixel_charge, self.quantile_threshold, axis=1
            )

            self.qped_q[tel_id] = np.nanquantile(
                np.array(self.ped_pixel_charge[tel_id])[
                    :, ~self.faulty_pixels[tel_id]
                ].flatten(),
                self.quantile_threshold,
                axis=0,
            )

        required_nsb_level = {}
        exact_nsb_level = {}
        self.models = {}

        for tel_id in self.tel_id_list:
            # Fits a parabola
            coeffs = np.polyfit(self.qsim_q[tel_id], self.nsb_levels, deg=2)
            model_fit = np.poly1d(coeffs)

            max_qsim_q = np.max(self.qsim_q[tel_id])
            min_qsim_q = np.min(self.qsim_q[tel_id])

            if self.qped_q[tel_id] > max_qsim_q or self.qped_q[tel_id] < min_qsim_q:
                self.log.info(
                    f"Warning: 90% quantile for pedestals ({self.qped_q[tel_id]:1.3f}) is outside the range"
                    + f" of the MC files [{min_qsim_q:1.3f}, {max_qsim_q:1.3f}]"
                    + f"for tel_id={tel_id}."
                )
                self.log.info(
                    "The NSB level is extrapolated from the fitted parabola between"
                    + f" NSB levels [{self.nsb_levels[0]}, {self.nsb_levels[-1]}]."
                )
                required_nsb_level[tel_id] = round(model_fit(self.qped_q[tel_id]))

            else:
                arg_min = np.argmin(
                    np.abs(model_fit(self.qped_q[tel_id]) - self.nsb_levels)
                )
                required_nsb_level[tel_id] = self.nsb_levels[arg_min]

            exact_nsb_level[tel_id] = model_fit(self.qped_q[tel_id])
            self.models[tel_id] = model_fit

        return required_nsb_level, exact_nsb_level

    def start(self):
        """
        Calculate the pixel charge statistics and the NSB level
        """
        # executed first to create tel_id_list
        self.calculate_pixel_charge_ped()

        self.identify_faulty_pixels()

        # estimate the NSB range from median variance
        if self.nsb_range is not None:
            nsb_grid = np.arange(self.nsb_range[0], self.nsb_range[1] + 1)
        else:
            # We only need the pixel-charge variance from the first NSB level
            nsb_lvl1_pixel_charge = self.calculate_pixel_charge_nsb_only_grid(
                nsb_levels=[1], nevents=self.nevents
            )
            nsb_grid, _ = self.nsb_from_variance(nsb_lvl1_pixel_charge)

        self.calculate_pixel_charge_nsb_only_grid(
            nsb_levels=nsb_grid, nevents=self.nevents
        )
        self.nsb_level, self.exact_nsb_level = self.quantile_curve_fit()

    def finish(self):
        """
        Write the parameters to the output file.
        """
        output_info = {
            "nsb-only-file": str(self.nsb_file),
            "shower-file": str(self.shower_file),
            "exact_nsb_level": {
                int(tel_id): float(val)
                for tel_id, val in sorted(self.exact_nsb_level.items())
            },
            "nsb_level": {
                int(tel_id): int(val) for tel_id, val in sorted(self.nsb_level.items())
            },
        }
        with open(self.output_path, "w") as yaml_file:
            yaml.dump(output_info, yaml_file, sort_keys=False)

        self.log.info(f"NSB parameters were written to {self.output_path}")

        Provenance().add_output_file(self.output_path, role="NSB parameters")

        self.log.info("Tool is shutting down")


def main():
    tool = EstimateNSBlevelTool()
    tool.run()


if __name__ == "__main__":
    main()
