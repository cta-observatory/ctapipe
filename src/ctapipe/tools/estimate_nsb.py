"""
Estimate the level of NSB required to match MC simulations to observations.
"""

import pathlib

import numpy as np
import yaml
from scipy.stats import median_abs_deviation, mode
from traitlets import Float, Int, List, Path, TraitError

from ctapipe.calib.camera.gainselection import GainChannel
from ctapipe.core import Provenance, Tool
from ctapipe.core.traits import ComponentName, classes_with_traits
from ctapipe.image.extractor import ImageExtractor, VarianceExtractor
from ctapipe.image.modifications import PedestalEventTypeFilter, WaveformModifier
from ctapipe.io import EventSource


class EstimateNSBlevelTool(Tool):
    """
    Tool to estimate the level of NSB required to match the diffuse NSB from an
    NSB-only simulation file (nsb_file) to a target observation (input_file).

    The NSB level is defined as the multiplicative factor (natural number)
    that NSB-only waveforms must be used to match the diffuse NSB of the target
    observation. For example, an NSB-only simulation produced with 25% of dark NSB,
    the required NSB level to match the dark NSB is 4.

    The NSB-only file must be produced with the same telescope
    array configuration and simulation settings to the MC files
    to add the NSB (simulation_file).

    This tool estimates the NSB level as follows:
    1) Exclude faulty pixels (broken and outliers) from the target observation
    using the median of the waveform variance per pixel.
    2) Compute the 90% quantile of the pixel charge for the target data (excluding
    faulty pixels) and NSB-only MC target. This quantity is estimated for different
    NSB levels for NSB-only simulations.
    3) Fit the NSB-only 90% quantile at different NSB levels with a parabola and
    find the closest NSB level to the target 90% quantile.

    The ratio between median pixel charge variance of the NSB-only file and target
    observation can be used to obtain an approximate NSB level to probe a limited
    NSB-level parameter space and reduce the computation time (variance method).
    Alternatively, a specific NSB range can be introduced.

    Note that if the simulation is not entirely realistic, the result may depend
    on the integrator and/or its parameters.
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

    simulation_file = Path(
        help="Path to a cosmics MC simulation simulated with the same settings as nsb_file (requires r1 waveforms)",
    ).tag(config=True)

    input_file = Path(
        help="Path to target observation (cosmics) to estimate the NSB level",
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

    # Other parameters (noise realisations and fit)
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
        ("i", "input-file"): "EstimateNSBlevelTool.input_file",
        ("n", "nsb-file"): "EstimateNSBlevelTool.nsb_file",
        ("s", "sim-file"): "EstimateNSBlevelTool.simulation_file",
        "output": "EstimateNSBlevelTool.output_path",
    }

    classes = [
        ImageExtractor,
        WaveformModifier,
    ] + classes_with_traits(ImageExtractor)

    def setup(self):
        if self.input_file is None:
            self.log.critical("Specifying 'input_file' is required.")
            self.exit(1)
        if self.nsb_file is None:
            self.log.critical("Specifying 'nsb_file' is required.")
            self.exit(1)

        self.check_output(self.output_path)

        with EventSource(self.input_file) as source:
            self.subarray = source.subarray
            # Only consider tel_ids in the input file
            self.tel_id_list = self.subarray.tel_ids

        self.image_extractor = ImageExtractor.from_name(
            self.image_extractor_type,
            subarray=self.subarray,
            parent=self,
        )

        # Verify that nsb_range only contains
        if self.nsb_range is not None and len(self.nsb_range) != 2:
            raise TraitError("`nsb_range` must contain two element: [lower, upper]")

        self.event_type_filter = PedestalEventTypeFilter(parent=self)

    def _extract_images_per_pixel_and_event(self, file, extractor):
        """
        Function to extract images (gain selected) per pixel and event for a certain extractor.

        The pixel charge is computed for each telescope, event, and pixel.

        Parameters
        ----------
        file: Path
            Path to file with waveform data.
        extractor: ImageExtractor
            Extractor to apply to waveforms.

        Returns
        -------
        pixel_data: dict with item's shape (n_events, n_pixels)
            Pixel-wise image information.
        broken_pixels: dict
            Information about broken pixels.
        """

        pixel_data = {}
        broken_pixels = {}

        with EventSource(file) as source:
            for event in source:
                # Simulation file
                if source.is_simulation:
                    self._process_sim_events(
                        source, event, extractor, pixel_data, broken_pixels
                    )
                else:
                    self._process_pedestal_events(
                        event, extractor, pixel_data, broken_pixels
                    )
        return pixel_data, broken_pixels

    def _process_sim_events(self, source, event, extractor, pixel_data, broken_pixels):
        """
        Auxiliary function to perform waveform integration for simulated events
        in noise-only pixels for the high gain channel.
        """
        for tel_id in event.trigger.tels_with_trigger:
            if tel_id not in self.tel_id_list:
                return

            true_pe = event.simulation.tel[tel_id].true_image
            wf = event.r1.tel[tel_id].waveform

            # select noise-only waveforms
            wf[:, true_pe > 0, :] = np.nan

            # use high gain for pedestals
            selected_gain_channel = np.broadcast_to(
                GainChannel.HIGH,
                source.subarray.tel[tel_id].camera.geometry.n_pixels,
            )
            broken_pixels[tel_id] = event.monitoring.tel[
                tel_id
            ].camera.coefficients.outlier_mask[GainChannel.HIGH]
            q_ev = extractor(
                waveforms=wf,
                tel_id=tel_id,
                selected_gain_channel=selected_gain_channel,
                broken_pixels=broken_pixels[tel_id],
            )

            if tel_id in pixel_data:
                pixel_data[tel_id].append(q_ev.image)
            else:
                pixel_data[tel_id] = [q_ev.image]

    def _process_pedestal_events(self, event, extractor, pixel_data, broken_pixels):
        """
        Auxiliary function to perform waveform integration for pedestal events.
        """
        if not self.event_type_filter(event):
            return

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

    def calculate_pixel_charge_ped(self):
        """
        Compute the pixel charge of the target observation.

        The pixel charge is computed for each telescope, event, and pixel.
        """
        self.ped_pixel_charge, self.bool_broken_pixels = (
            self._extract_images_per_pixel_and_event(
                self.input_file, self.image_extractor
            )
        )

    def calculate_pixel_charge_sim(self):
        """
        Compute the pixel charge of the MC simulation (cosmics file) associated to
        the NSB-only file.

        The pixel charge is computed for each telescope, event, and pixel.
        """
        self.sim_pixel_charge, _ = self._extract_images_per_pixel_and_event(
            self.simulation_file, self.image_extractor
        )

    def identify_faulty_pixels(self):
        """
        Identify outlier pixels of the target observation using the median variance of the waveform per pixel.
        Outliers are defined as pixels with modified Z-score statistics beyond the threshold mod_zscore_th.

        Faulty pixels = outlier pixels + broken pixels
        """

        variance_extractor = VarianceExtractor(subarray=self.subarray)
        # Compute the median variance per pixel
        shower_pixel_variance, broken_pixels = self._extract_images_per_pixel_and_event(
            self.input_file, variance_extractor
        )

        self.faulty_pixels = {}
        self.pixel_var = {}
        for tel_id in np.sort(self.tel_id_list):
            tel_pixel_variance = np.array(shower_pixel_variance[tel_id])
            # Make sure we only select the high-gain channel in case more than one channel
            # is available
            if tel_pixel_variance.ndim == 3:
                self.pixel_var[tel_id] = np.nanmedian(
                    tel_pixel_variance[:, GainChannel.HIGH, :],
                    axis=0,
                )
            else:
                self.pixel_var[tel_id] = np.nanmedian(tel_pixel_variance, axis=0)
            if broken_pixels[tel_id].ndim == 2:
                mask_broken = broken_pixels[tel_id][GainChannel.HIGH]
            else:
                mask_broken = broken_pixels[tel_id]

            # Compute modified Z-score
            mad_pix = median_abs_deviation(self.pixel_var[tel_id], axis=-1)
            median_pix = np.nanmedian(self.pixel_var[tel_id])
            abs_mod_zscore = np.abs(
                (self.pixel_var[tel_id] - median_pix) / (1.486 * mad_pix)
            )

            # Obtain outliers pixels
            mask_outliers = abs_mod_zscore > self.mod_zscore_th
            # Faulty pixels = outlier pixels + broken pixels
            self.faulty_pixels[tel_id] = mask_outliers | mask_broken

            # Checks
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
        Compute the charge from NSB-only waveforms for 'nevents' events
        for different NSB levels.
        The pixel charge is computed for each telescope, event, pixel and NSB.

        Output dimensions of nsb_pixel_charge: (n_nsb_levels, n_events, n_pixels)

        Parameters
        ----------
        nsb_levels: np.array
            Grid on NSB levels.
        nevents: Int
            Number of noise realisations in WaveformModifier.

        Returns
        -------
        nsb_pixel_charge: dict with item's shape (n_nsb_levels, n_events, n_pixels)
            Charges per telescope, NSB, event, and pixel.
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
                # Add NSB to waveforms
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
        Estimate the NSB from the ratio between variances of the pixel charge
        between the NSB-only simulation and the target observation.

        The NSB level is estimated wrt the first element in the NSB level dimension of
        nsb_lvl1_pixel_charge.

        Note: the obtained results may yield wrong estimated for biased extractors
        and small extractor windows.

        Parameters
        ----------
        nsb_lvl1_pixel_charge: dict with item's shape (n_nsb_levels, n_events, n_pixels).
            Pixel charges from NSB-only waveforms.

        Returns
        -------
        nsb_grid: np.array
            Grid on NSB levels.
        nsb_lvl: float
            Estimated NSB level.
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

        # Create the NSB grid
        nsb_min = np.floor(
            max(nsb_lvl[min(nsb_lvl, key=nsb_lvl.get)] - self.n_grid_points, 1)
        ).astype(int)
        nsb_max = np.ceil(
            nsb_lvl[max(nsb_lvl, key=nsb_lvl.get)] + self.n_grid_points + 1
        ).astype(int)
        nsb_grid = np.arange(nsb_min, nsb_max)
        return nsb_grid, nsb_lvl

    def compute_quantile(self, pixel_charge):
        """
        Compute the self.quantile_threshold quantile of the charge images
        using all usable pixels and events.

        Parameters
        ----------
        pixel_charge: dict
            Pixel charge information

        Returns
        -------
        charge_q: dict
            Quantile value of the pixel charge.

        """
        charge_q = {}
        for tel_id in self.tel_id_list:
            tel_pixel_charge = np.array(pixel_charge[tel_id])

            # Mask faulty pixels
            tel_pixel_charge = tel_pixel_charge[..., ~self.faulty_pixels[tel_id]]

            # Reshape the NSB-only pixel charge to (n_nsb_levels, n_events * n_pixels)
            if tel_pixel_charge.ndim == 3:
                shape_pixel_charge = np.shape(tel_pixel_charge)

                all_nsb_pixel_charge = np.reshape(
                    tel_pixel_charge,
                    (
                        shape_pixel_charge[0],
                        shape_pixel_charge[1] * shape_pixel_charge[2],
                    ),
                )
                # Compute quantile
                charge_q[tel_id] = np.nanquantile(
                    all_nsb_pixel_charge, self.quantile_threshold, axis=1
                )
            # Flatten the full array for cosmics pixel charge
            else:
                charge_q[tel_id] = np.nanquantile(
                    tel_pixel_charge.flatten(),
                    self.quantile_threshold,
                    axis=0,
                )
        return charge_q

    def quantile_curve_fit(self, charge_q):
        """
        Fit a parabola to the quantile vs NSB curve for NSB-only data and
        find the NSB level that matches the charge_q values.

        Parameters
        ----------
        charge_q: dict
            Charge quantile to estimate the NSB level.

        Returns
        -------
        required_nsb_level: dict
            Assigned NSB level per telescope.
        exact_nsb_level: dict
            Exact NSB levels per telescope.
        """

        required_nsb_level = {}
        exact_nsb_level = {}

        for tel_id in self.tel_id_list:
            # Fits a parabola
            coeffs = np.polyfit(self.qnsb_q[tel_id], self.nsb_levels, deg=2)
            model_fit = np.poly1d(coeffs)

            max_qnsb_q = np.max(self.qnsb_q[tel_id])
            min_qnsb_q = np.min(self.qnsb_q[tel_id])

            if charge_q[tel_id] > max_qnsb_q or charge_q[tel_id] < min_qnsb_q:
                self.log.info(
                    f"Warning: 90% quantile ({charge_q[tel_id]:1.3f}) is outside the range"
                    + f" of the MC files [{min_qnsb_q:1.3f}, {max_qnsb_q:1.3f}]"
                    + f"for tel_id={tel_id}."
                )
                self.log.info(
                    "The NSB level is extrapolated from the fitted parabola between"
                    + f" NSB levels [{self.nsb_levels[0]}, {self.nsb_levels[-1]}]."
                )
                required_nsb_level[tel_id] = round(model_fit(charge_q[tel_id]))

            else:
                arg_min = np.argmin(
                    np.abs(model_fit(charge_q[tel_id]) - self.nsb_levels)
                )
                required_nsb_level[tel_id] = self.nsb_levels[arg_min]

            exact_nsb_level[tel_id] = model_fit(charge_q[tel_id])

        return required_nsb_level, exact_nsb_level

    def start(self):
        """
        Calculate the pixel charge statistics and the NSB level.
        """
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

        self.calculate_pixel_charge_sim()

        qped_q = self.compute_quantile(self.ped_pixel_charge)
        self.qnsb_q = self.compute_quantile(self.nsb_pixel_charge)
        qsim_q = self.compute_quantile(self.sim_pixel_charge)

        self.nsb_level, self.exact_nsb_level = self.quantile_curve_fit(qped_q)
        self.nsb_level_sim, self.exact_nsb_level_sim = self.quantile_curve_fit(qsim_q)

        self.nsb_level_wrt_sim = {}
        self.exact_nsb_level_wrt_sim = {}
        for tel_id in self.tel_id_list:
            self.exact_nsb_level_wrt_sim[tel_id] = (
                self.exact_nsb_level[tel_id] - self.exact_nsb_level_sim[tel_id]
            )
            self.nsb_level_wrt_sim[tel_id] = round(self.exact_nsb_level_wrt_sim[tel_id])
            if self.nsb_level_wrt_sim[tel_id] < 0:
                self.log.info("The level of NSB wrt the simulation is lower than 0.")

    def finish(self):
        """
        Write the parameters to the output file.
        """
        output_info = {
            "nsb-only-file": str(self.nsb_file),
            "simulation-file": str(self.simulation_file),
            "input-file": str(self.input_file),
            "exact_nsb_level": {
                int(tel_id): float(val)
                for tel_id, val in sorted(self.exact_nsb_level.items())
            },
            "nsb_level": {
                int(tel_id): int(val) for tel_id, val in sorted(self.nsb_level.items())
            },
            "mode_nsb_level": int(mode(list(self.nsb_level.values())).mode),
            "exact_nsb_level_wrt_sim": {
                int(tel_id): float(val)
                for tel_id, val in sorted(self.exact_nsb_level_wrt_sim.items())
            },
            "nsb_level_wrt_sim": {
                int(tel_id): int(val)
                for tel_id, val in sorted(self.nsb_level_wrt_sim.items())
            },
            "mode_nsb_level_wrt_sim": int(
                mode(list(self.nsb_level_wrt_sim.values())).mode
            ),
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
