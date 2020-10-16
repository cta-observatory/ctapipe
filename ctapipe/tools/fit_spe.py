"""
Extract the Single Photoelectron spectrum parameters from the measured charges
stored in a DL1 file
"""
import tables
from tables.nodes import filenode
from traitlets import Dict, List, Unicode, Int, Tuple
from ctapipe.core import Provenance, Tool, traits
import numpy as np
from matplotlib import pyplot as plt

try:
    from spefit import CameraFitter, PDF, Cost
except ImportError:
    msg = ("This tool requires spefit:\n "
           "`conda install -c cta-observatory spefit` / `pip install spefit`")
    raise ImportError(msg)


def plot_pixel(pixel, pdf, fitter, charges):
    _, ax = plt.subplots()
    fitter._apply_pixel(charges, pixel)
    for i in range(fitter.n_illuminations):
        color = next(ax._get_lines.prop_cycler)['color']
        lambda_ = fitter.pixel_values[pixel][f"lambda_{i}"]
        lambda_err = fitter.pixel_errors[pixel][f"lambda_{i}"]
        arrays = fitter.pixel_arrays[pixel][i]
        ax.hist(
            arrays['charge_hist_x'],
            weights=arrays['charge_hist_y'],
            bins=arrays['charge_hist_edges'],
            density=True,
            histtype='step',
            color=color,
            label=f"λ = ({lambda_:.2f} ± {lambda_err:.2f}) p.e."
        )
        ax.plot(arrays['fit_x'], arrays['fit_y'], color=color)

        initial_array = np.array(list(pdf.initial.values()))
        initial_x = arrays['fit_x']
        initial_y = pdf(initial_x, initial_array, i)
        ax.plot(initial_x, initial_y, ls="--", color=color, alpha=0.3, label="Initial")

    ax.set_xlabel("Charge (A.U.)")
    ax.set_ylabel("Probability")
    ax.set_title(f"Pixel {pixel}")
    ax.legend()

    print(f"Pixel {pixel} fit results:")
    for param, value in fitter.pixel_values[pixel].items():
        error = fitter.pixel_errors[pixel][param]
        print(f"  {param} = {value:.2e} ± {error:.2e}")

    plt.show()


def append_to_table(pixel, table, result):
    row = table.row
    for key, value in result[pixel].items():
        row[key] = value
    row.append()


def append_illumination_to_table(illumination, pixel, table, result):
    row = table.row
    for key, value in result[pixel][illumination].items():
        row[key] = value
    row.append()


def write_spe_file(fitter, pdf, n_pixels, metadata, output_path):
    hist_size = fitter.pixel_arrays[0][0]['charge_hist_x'].size
    curve_size = fitter.pixel_arrays[0][0]['fit_x'].size

    # Prepare Table descriptions
    class ParamTableDesc(tables.IsDescription):
        pass
    for parameter in pdf.parameter_names:
        ParamTableDesc.columns[parameter] = tables.Float64Col()

    class ScoreTableDesc(tables.IsDescription):
        chi2 = tables.Float64Col()
        reduced_chi2 = tables.Float64Col()
        p_value = tables.Float64Col()

    class ArrayTableDesc(tables.IsDescription):
        charge_hist_x = tables.Float64Col(shape=(hist_size))
        charge_hist_y = tables.Float64Col(shape=(hist_size))
        charge_hist_edges = tables.Float64Col(shape=(hist_size+1))
        fit_x = tables.Float64Col(shape=(curve_size))
        fit_y = tables.Float64Col(shape=(curve_size))

    Provenance().add_output_file(output_path)
    with tables.File(output_path, mode='w') as output:
        results = output.create_group(output.root, "results", "results")
        arrays = output.create_group(output.root, "arrays", "arrays")
        values_table = output.create_table(
            results, "values", ParamTableDesc, "SPE Fit Parameter Values"
        )
        errors_table = output.create_table(
            results, "errors", ParamTableDesc, "SPE Fit Parameter Errors"
        )
        scores_table = output.create_table(
            results, "scores", ScoreTableDesc, "SPE Fit Scores"
        )

        for pixel in range(n_pixels):
            append_to_table(pixel, values_table, fitter.pixel_values)
            append_to_table(pixel, errors_table, fitter.pixel_errors)
            append_to_table(pixel, scores_table, fitter.pixel_scores)

        for i in range(fitter.n_illuminations):
            arrays_table = output.create_table(
                arrays,
                f"illumination_{i}",
                ArrayTableDesc,
                f"SPE Fit Illumination {i} Arrays"
            )
            for pixel in range(n_pixels):
                append_illumination_to_table(i, pixel, arrays_table, fitter.pixel_arrays)

        metadata_node = filenode.new_node(output, where='/', name='metadata')
        for key, value in metadata.items():
            metadata_node.attrs[key] = value


class SPEFitter(Tool):
    name = "SPEFitter"
    description = "Extract the Single Photoelectron spectrum parameters from the measured charges stored in a DL1 file"

    input_paths = List(
        traits.Path(exists=True, directory_ok=False),
        default_value=None,
        help="Paths to DL1 files"
    ).tag(config=True)
    telescope = Int(
        None,
        allow_none=True,
        help="Telescope to process",
    ).tag(config=True)
    output_path = traits.Path(
        help="Path to save results of SPE fit (HDF5 file)",
    ).tag(config=True)
    pdf_name = traits.create_class_enum_trait(
        PDF, default_value="PMTSingleGaussian"
    )
    cost_name = traits.create_class_enum_trait(
        Cost, default_value="BinnedNLL"
    )
    n_bins = Int(
        100,
        help="Number of bins in charge histogram",
    ).tag(config=True)
    range = Tuple(
        (-1, 5),
        help="Range for charge histogram"
    ).tag(config=True)
    n_processes = Int(
        1,
        help="Number of processes for processing pixels in parallel",
    ).tag(config=True)
    plot_pixel = Int(
        None,
        allow_none=True,
        help="If set, plot the SPE of a single pixel only",
    ).tag(config=True)
    initial = Dict(
        None,
        allow_none=True,
        help="Update the fit parameter initial values",
    ).tag(config=True)
    limits = Dict(
        None,
        allow_none=True,
        help="Update the fit parameter limits",
    ).tag(config=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def start(self):
        n_pixels = None
        charges = []
        for path in self.input_paths:
            with tables.open_file(path) as file:
                tel = f'tel_{self.telescope:03d}'
                charge_array = file.root.dl1.event.telescope.images[tel].col("image")
                _, n_pixels = charge_array.shape
            charges.append(charge_array)

        n_illuminations = len(charges)
        pdf = PDF.from_name(self.pdf_name, n_illuminations=n_illuminations)
        if self.initial is not None:
            pdf.update_parameters_initial(**self.initial)
        if self.limits is not None:
            pdf.update_parameters_limits(**self.limits)
        fitter = CameraFitter(pdf, self.n_bins, self.range, self.cost_name)

        if self.plot_pixel is not None:
            plot_pixel(self.plot_pixel, pdf, fitter, charges)
            return

        if self.n_processes == 1:
            fitter.process(charges)
        else:
            fitter.multiprocess(charges, self.n_processes)

        metadata = dict(
            paths=self.input_paths,
            telescope=self.telescope,
            pdf=self.pdf_name,
            cost=self.cost_name,
            n_bins=self.n_bins,
            range=self.range,
            initial=self.initial,
            limits=self.limits,
        )
        write_spe_file(fitter, pdf, n_pixels, metadata, self.output_path)


def main():
    exe = SPEFitter()
    exe.run()


if __name__ == "__main__":
    main()
