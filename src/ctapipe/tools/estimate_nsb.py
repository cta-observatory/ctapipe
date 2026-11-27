"""
Estimate the level of NSB required to match MC simulations to observations.
"""

import pathlib

import numpy as np
import tables
from astropy.table import Table
from traitlets import Bool, Float, Int, List, Path, TraitError

from ctapipe.calib.camera.gainselection import GainChannel
from ctapipe.core import Provenance, Tool
from ctapipe.core.traits import ComponentName, classes_with_traits
from ctapipe.image.extractor import ImageExtractor
from ctapipe.image.modifications import PedestalEventTypeFilter, WaveformModifier
from ctapipe.instrument import SubarrayDescription
from ctapipe.io import EventSource, read_table, write_table
from ctapipe.io.hdf5dataformat import DL1_PIXEL_STATISTICS_GROUP
from ctapipe.monitoring import StatisticsAggregator


class EstimateNSBlevelTool(Tool):
    """
    Tool to estimate the level of NSB required to match the diffuse NSB from the
    reference MC simulation (simulation_file) to a target observation (input_file)
    through NSB-only simulations (nsb_file).

    The NSB level (nsb_level) is defined as the number of (randomly selected) NSB-only
    waveforms from nsb_file to be added to each waveform in the reference MC simulation
    (simulation_file) in order to reach the NSB level of the target file (input_file).
    For example, if we consider an NSB-only simulation produced with 25% of dark NSB
    as the reference MC simulation and the target observation has dark NSB, the required
    NSB level to match the dark NSB is 3.

    The NSB-only file must be produced with the same telescope
    array configuration and simulation settings as the MC file
    to which the NSB is being added (simulation_file).

    This tool estimates the NSB level as follows:
    1) Exclude faulty pixels (broken and outliers) from the target observation
    using the median of the waveform variance per pixel.
    2) Compute the 90% quantile of the pixel charge for the target data (excluding
    faulty pixels) and NSB-only MC target. The later quantity is estimated for different
    NSB levels by randomly selecting waveforms from the NSB-only file.
    3) Fit the NSB-only 90% quantile at different NSB levels with a parabola and
    find the closest NSB level to the target 90% quantile.

    The ratio between median pixel charge variance of the target observation and
    the NSB-only file [Var(target) / Var(simulation)] can be used to obtain an
    approximate NSB level to probe a limited NSB-level parameter space and reduce
    the computation time (hereafter, variance method). Alternatively, a specific NSB
    range can be introduced.

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
        help=(
            "Path to a pedestal MC simulation simulated with the same settings "
            "as nsb_file (requires r1 waveforms)"
        ),
    ).tag(config=True)

    input_file = Path(
        help="Path to camera calibration file (with interleaved pedestal statistics) to estimate the NSB level",
    ).tag(config=True)

    # Optional parameters:
    output_path = Path(
        help="Output filename", default_value=pathlib.Path("nsb_params.h5")
    ).tag(config=True)

    min_npixels_valid = Int(
        default_value=1000,
        help=(
            "Minimum number of usable pixels (excluding outliers and broken pixels)"
            " needed to estimate the pedestal statistics"
        ),
    ).tag(config=True)

    min_nped_events = Int(
        default_value=100,
        help="Minimum number of pedestal events needed to estimate the pedestal statistics per pixel.",
    ).tag(config=True)

    # Other parameters (noise realisations and fit)
    is_simulated_pedestal = Bool(
        default_value=True,
        help=("Whether the simulation_file contains shower images or pedestal data."),
    ).tag(config=True)

    quantile_threshold = Float(
        default_value=0.9,
        help=(
            "The quantile threshold to compute the quantile of the pixel charge distribution. "
            "Only quantiles in StatisticsAggregator are allowed."
        ),
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

        self.subarray = SubarrayDescription.read(self.nsb_file)

        # Only consider tel_ids in the input file
        self.tel_id_list = []
        with tables.open_file(self.input_file, mode="r") as f:
            try:
                self.ped_type = "pedestal"
                pixel_ped_path = f"{DL1_PIXEL_STATISTICS_GROUP}/{self.ped_type}_image"
                nodes = f.iter_nodes(pixel_ped_path, classname="Leaf")
            except tables.NoSuchNodeError:
                self.ped_type = "sky_pedestal"
                pixel_ped_path = f"{DL1_PIXEL_STATISTICS_GROUP}/{self.ped_type}_image"
                nodes = f.iter_nodes(pixel_ped_path, classname="Leaf")

            for node in nodes:
                self.tel_id_list.append(int(node._v_name.split("_")[-1]))

        self.image_extractor = ImageExtractor.from_name(
            self.image_extractor_type,
            subarray=self.subarray,
            parent=self,
        )

        valid_quantiles = StatisticsAggregator._QUANTILES
        q_vals = list(valid_quantiles.values())
        q_keys = list(valid_quantiles.keys())
        bool_present = np.isclose(self.quantile_threshold, q_vals)
        if not np.any(bool_present):
            raise TraitError(
                f"`quantile_threshold` must be one of the following values: {q_vals}"
            )

        self.key_q_th = q_keys[np.argmax(bool_present)]

        # Verify that nsb_range only contains two elements
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
        pixel_data: dict
            Tel_id-wise, pixel-wise image information.
        broken_pixels: dict
            Tel_id-wise information about broken pixels.
        waveform_data:
            Tel_id-wise waveforms. The dict is only filled for the case of simulations.
        """

        pixel_data = {}
        waveform_data = {}
        broken_pixels = {}

        # skip_calibration_events is required to process simulated pedestals
        with EventSource(file, skip_calibration_events=False) as source:
            for event in source:
                self._process_sim_events(
                    source,
                    event,
                    extractor,
                    pixel_data,
                    waveform_data,
                    broken_pixels,
                )

        return pixel_data, waveform_data, broken_pixels

    def _process_sim_events(
        self, source, event, extractor, pixel_data, waveform_data, broken_pixels
    ):
        """
        Auxiliary function to perform waveform integration for simulated events
        of pedestals or showers with in noise-only pixels for the high gain channel.

        This is intended to enable the processing of simulation files containing showers,
        since they are those in which we want to increase the NSB level to better match the real data.
        """
        for tel_id, tel_event in event.tel.items():
            if tel_id not in self.tel_id_list:
                return

            wf = tel_event.r1.waveform

            if not self.is_simulated_pedestal:
                true_pe = tel_event.simulation.true_image
                # select noise-only waveforms
                wf[:, true_pe > 0, :] = np.nan
            else:
                if not self.event_type_filter(event):
                    return

            # use high gain for pedestals
            selected_gain_channel = np.broadcast_to(
                GainChannel.HIGH,
                source.subarray.tel[tel_id].camera.geometry.n_pixels,
            )
            broken_pixels[tel_id] = (
                tel_event.monitoring.camera.coefficients.outlier_mask[GainChannel.HIGH]
            )
            q_ev = extractor(
                waveforms=wf,
                tel_id=tel_id,
                selected_gain_channel=selected_gain_channel,
                broken_pixels=broken_pixels[tel_id],
            )

            if tel_id in pixel_data:
                pixel_data[tel_id].append(q_ev.image)
                waveform_data[tel_id].append(wf)
            else:
                pixel_data[tel_id] = [q_ev.image]
                waveform_data[tel_id] = [wf]

    def read_pixel_charge_stats(self):
        """
        Read the pixel charge information.

        The pixel charge is read for each telescope, data chunk, and pixel.
        """
        self.ped_pixel_charge_var = {}
        self.ped_pixel_charge_q = {}
        self.faulty_pixels = {}
        self.valid_chunks = {}

        for tel_id in self.tel_id_list:
            ped_stats = read_table(
                self.input_file,
                path=f"{DL1_PIXEL_STATISTICS_GROUP}/{self.ped_type}_image/tel_{tel_id:03d}",
            )

            # Select only the high-gain channel in case more than one channel is available
            self.ped_pixel_charge_var[tel_id] = (
                ped_stats["std"][:, GainChannel.HIGH, :].value ** 2
            )

            self.ped_pixel_charge_q[tel_id] = ped_stats[self.key_q_th][
                :, GainChannel.HIGH, :
            ].value
            outlier_mask = ped_stats["outlier_mask"][:, GainChannel.HIGH, :].value

            # Discard pixels with too few valid events
            pixels_few_stat_per_chunk_mask = (
                ped_stats["n_events"][:, GainChannel.HIGH, :].value
                < self.min_nped_events
            )
            self.faulty_pixels[tel_id] = outlier_mask | pixels_few_stat_per_chunk_mask

            # Only consider chunks with enough valid pixels
            n_valid_per_chunk = np.sum(~self.faulty_pixels[tel_id], axis=1)
            self.valid_chunks[tel_id] = n_valid_per_chunk > self.min_npixels_valid
            arg_chunks_invalid = np.where(~np.atleast_1d(self.valid_chunks[tel_id]))[0]

            if (~self.valid_chunks[tel_id]).any():
                self.log.warning(
                    "Too many pixels were discarded (either because of broken pixels or contained few pedestal events) "
                    + f"for tel_id={tel_id} and chunks {arg_chunks_invalid} to estimate pedestal statistics."
                    + "These chunks will be discarded from the analysis."
                )

                n_outliers_per_chunk = np.sum(outlier_mask, axis=1)
                n_pixels_with_few_stat_per_chunk = np.sum(
                    pixels_few_stat_per_chunk_mask, axis=1
                )

                self.log.warning(
                    f"Number of pixels tagged as outliers in discarded chunks: {n_outliers_per_chunk[~self.valid_chunks[tel_id]]}"
                )
                self.log.warning(
                    f"Number of pixels with too few pedestal in discarded chunks: {n_pixels_with_few_stat_per_chunk[~self.valid_chunks[tel_id]]}"
                )

    def calculate_pixel_charge_sim(self):
        """
        Compute the pixel charge of the reference MC simulation.

        The pixel charge is computed for each telescope, event, and pixel.
        """
        self.sim_pixel_charge, self.sim_waveforms, self.sim_broken_pixels = (
            self._extract_images_per_pixel_and_event(
                self.simulation_file, self.image_extractor
            )
        )

    def calculate_pixel_charge_nsb_only_grid(self, nsb_levels):
        """
        Compute the charge of the waveform that results from summing the reference simulation
        waveform and the NSB-only waveforms at different NSB levels.

        The pixel charge is computed for each telescope, NSB, event, and pixel.

        Parameters
        ----------
        nsb_levels: np.array
            Grid on NSB levels.

        Returns
        -------
        nsb_pixel_charge: dict
            Tel_id-wise, pixel-wise charge of shape (n_nsb_levels, n_events, n_pixels).
        """

        self.nsb_levels = nsb_levels
        self.nsb_added_pixel_charge = {}

        with EventSource(
            input_url=self.nsb_file, skip_calibration_events=False
        ) as source:
            max_nevents_tel_id = 0
            for tel_id in self.tel_id_list:
                camera_geom = self.subarray.tel[tel_id].camera.geometry
                nevents_tel_id = np.shape(self.sim_waveforms[tel_id])[0]

                # Consider the same number of events that exist in the reference simulation
                self.nsb_added_pixel_charge[tel_id] = np.zeros(
                    (len(self.nsb_levels), nevents_tel_id, camera_geom.n_pixels)
                )
                max_nevents_tel_id = max(nevents_tel_id, max_nevents_tel_id)

            for i_nsb, nsb in enumerate(self.nsb_levels):
                # Add NSB to waveforms
                wf_mod = WaveformModifier(
                    subarray=source.subarray,
                    nsb_file=self.nsb_file,
                    nsb_level=nsb,
                    n_noise_realizations=max_nevents_tel_id,  # Simulate maximum number of realisations
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

                    nevents_tel_id = np.shape(self.sim_waveforms[tel_id])[0]

                    for i_ev in range(nevents_tel_id):
                        # Sum the waveforms of the reference simulation and the NSB-only waveform
                        total_sim_wf = (
                            self.sim_waveforms[tel_id][i_ev]
                            + wf_mod.total_noise[tel_id][i_ev]
                        )
                        q_ped = self.image_extractor(
                            waveforms=total_sim_wf,
                            tel_id=tel_id,
                            selected_gain_channel=selected_gain_channel,
                            broken_pixels=bool_broken_pixels,
                        )
                        self.nsb_added_pixel_charge[tel_id][i_nsb][i_ev] = q_ped.image

        return self.nsb_added_pixel_charge

    def nsb_from_variance(self):
        """
        Estimate the NSB from the ratio between variances of the pixel charge
        between the target observation and reference simulation:

        Var(target) / Var(simulation)

        Note: the obtained results may yield wrong estimates for biased extractors
        and small extractor windows.

        Returns
        -------
        nsb_grid: np.array
            Grid on NSB levels.
        nsb_lvl: float
            Estimated NSB level.
        """
        nsb_lvl = {}
        for tel_id in self.tel_id_list:
            # obtain the pixel charge variance per pixel for simulations
            ped_sim_var_pix = np.nanvar(self.sim_pixel_charge[tel_id], axis=0)

            # All pixels are used to compute the median
            ped_sim_var_pix_median = np.median(ped_sim_var_pix)

            nsb_lvl[tel_id] = np.zeros(len(self.valid_chunks[tel_id]), dtype=float)

            for chunk_id in range(len(self.valid_chunks[tel_id])):
                if not self.valid_chunks[tel_id][chunk_id]:
                    nsb_lvl[tel_id][chunk_id] = np.nan
                else:
                    # median pixel charge variance
                    ped_var_pix_median = np.median(
                        self.ped_pixel_charge_var[tel_id][chunk_id][
                            ~self.faulty_pixels[tel_id][chunk_id]
                        ]
                    )

                    nsb_lvl[tel_id][chunk_id] = (
                        ped_var_pix_median / ped_sim_var_pix_median
                    )

        all_nsb_vals = np.concatenate(
            [nsb_lvl[tel_id].flatten() for tel_id in self.tel_id_list]
        )

        if np.isnan(all_nsb_vals).all():
            # Dummy grid when all chunks from all telescopes are invalid
            nsb_grid = np.arange(1, 5)
        else:
            min_val = np.nanmin(all_nsb_vals)
            max_val = np.nanmax(all_nsb_vals)
            nsb_min = np.floor(max(min_val - self.n_grid_points, 1)).astype(int)
            nsb_max = np.ceil(max_val + self.n_grid_points + 1).astype(int)
            nsb_grid = np.arange(nsb_min, nsb_max)

        return nsb_grid, nsb_lvl

    def compute_quantile(self):
        """
        Compute the self.quantile_threshold quantile of the charge images
        using all pixels.

        Returns
        -------
        charge_q: dict
            Quantile value of the pixel charge.

        """
        charge_q = {}
        for tel_id in self.tel_id_list:
            tel_pixel_charge = np.array(self.nsb_added_pixel_charge[tel_id])

            tel_pixel_charge = tel_pixel_charge[..., ~self.sim_broken_pixels[tel_id]]

            # Reshape the NSB-only pixel charge to (n_nsb_levels, n_events * n_pixels)
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

            required_nsb_level[tel_id] = np.zeros(
                len(self.valid_chunks[tel_id]), dtype=float
            )
            exact_nsb_level[tel_id] = np.zeros(
                len(self.valid_chunks[tel_id]), dtype=float
            )

            for chunk_id in range(len(self.valid_chunks[tel_id])):
                if not self.valid_chunks[tel_id][chunk_id]:
                    required_nsb_level[tel_id][chunk_id] = np.nan
                    exact_nsb_level[tel_id][chunk_id] = np.nan
                    continue

                if (
                    charge_q[tel_id][chunk_id] > max_qnsb_q
                    or charge_q[tel_id][chunk_id] < min_qnsb_q
                ):
                    self.log.info(
                        f"Warning: {self.quantile_threshold * 100:1.0f}% quantile ({charge_q[tel_id][chunk_id]:1.3f}) "
                        + f"is outside the range of the MC files [{min_qnsb_q:1.3f}, {max_qnsb_q:1.3f}]"
                        + f"for tel_id={tel_id} and chunk={chunk_id}."
                    )
                    self.log.info(
                        "The NSB level is extrapolated from the fitted parabola between"
                        + f" NSB levels [{self.nsb_levels[0]}, {self.nsb_levels[-1]}]."
                    )
                    required_nsb_level[tel_id][chunk_id] = round(
                        model_fit(charge_q[tel_id][chunk_id])
                    )

                else:
                    arg_min = np.argmin(
                        np.abs(model_fit(charge_q[tel_id][chunk_id]) - self.nsb_levels)
                    )
                    required_nsb_level[tel_id][chunk_id] = self.nsb_levels[arg_min]

                exact_nsb_level[tel_id][chunk_id] = model_fit(
                    charge_q[tel_id][chunk_id]
                )

        return required_nsb_level, exact_nsb_level

    def start(self):
        """
        Calculate the pixel charge statistics and the NSB level.
        """
        self.read_pixel_charge_stats()

        # compute median pixel charge quantile
        ped_median_charge_q = {}
        for tel_id in self.tel_id_list:
            ped_median_charge_q[tel_id] = np.zeros(
                len(self.valid_chunks[tel_id]), dtype=float
            )
            for chunk_id in range(len(self.valid_chunks[tel_id])):
                if not self.valid_chunks[tel_id][chunk_id]:
                    ped_median_charge_q[tel_id][chunk_id] = np.nan
                else:
                    ped_median_charge_q[tel_id][chunk_id] = np.median(
                        self.ped_pixel_charge_q[tel_id][chunk_id][
                            ~self.faulty_pixels[tel_id][chunk_id]
                        ]
                    )

        self.calculate_pixel_charge_sim()

        # estimate the NSB range from median variance
        if self.nsb_range is not None:
            nsb_grid = np.arange(self.nsb_range[0], self.nsb_range[1] + 1)
        else:
            nsb_grid, _ = self.nsb_from_variance()

        # combine the waveforms (reference + nsb-only waveforms)
        # and compute the pixel charge
        self.calculate_pixel_charge_nsb_only_grid(nsb_levels=nsb_grid)

        self.qnsb_q = self.compute_quantile()

        self.nsb_level, self.exact_nsb_level = self.quantile_curve_fit(
            ped_median_charge_q
        )

    def finish(self):
        """
        Write the parameters to the output file.
        """

        meta_info = {
            "nsb_only_file": str(self.nsb_file),
            "simulation_file": str(self.simulation_file),
            "input_file": str(self.input_file),
        }

        for tel_id in self.tel_id_list:
            data = {
                "nsb_level": self.nsb_level[tel_id],
                "exact_nsb_level": self.exact_nsb_level[tel_id],
                "is_valid": self.valid_chunks[tel_id],
            }
            nsb_tab = Table(data)
            nsb_tab.meta = meta_info

            write_table(
                table=nsb_tab,
                h5file=self.output_path,
                path=f"/dl1/monitoring/nsb/tel_{tel_id:03d}",
                overwrite=True,
            )

        self.log.info(f"NSB parameters were written to {self.output_path}")

        Provenance().add_output_file(self.output_path, role="NSB parameters")

        self.log.info("Tool is shutting down")


def main():
    tool = EstimateNSBlevelTool()
    tool.run()


if __name__ == "__main__":
    main()
