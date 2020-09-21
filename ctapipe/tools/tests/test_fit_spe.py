import pytest
from ctapipe.tools.tests.test_tools import run_tool
from ctapipe.tools.fit_spe import SPEFitter
from ctapipe.io import HDF5TableWriter
from ctapipe.containers import DL1CameraContainer, TelEventIndexContainer
from spefit import PMTSingleGaussian
import matplotlib as mpl
import numpy as np


@pytest.fixture(scope="module")
def example_pdf():
    pdf = PMTSingleGaussian(n_illuminations=2)
    pdf.update_parameters_initial(lambda_1=1)
    return pdf


@pytest.fixture(scope="module")
def example_params(example_pdf):
    return np.array(list(example_pdf.initial.values()))


@pytest.fixture(scope='function')
def example_spe_dl1_paths(tmp_path, example_pdf, example_params):
    n_pixels = 10
    n_events = 1000
    tel_id = 1
    rng = np.random.default_rng(seed=1)
    paths = []
    for i in range(example_pdf.n_illuminations):
        path = str(tmp_path / f"spe_{i}_dl1.h5")
        dl1 = DL1CameraContainer()
        with HDF5TableWriter(path) as writer:
            for _ in range(n_events):
                x = np.linspace(-1, 6, 10000)
                y = example_pdf(x, example_params, i)
                charge = rng.choice(x, p=y / y.sum(), size=n_pixels).astype(np.float32)

                dl1.image = charge
                table_name = f"tel_{tel_id:03d}"
                tel_index = TelEventIndexContainer(tel_id=np.int16(tel_id))
                writer.write(
                    table_name=f"dl1/event/telescope/images/{table_name}",
                    containers=[tel_index, dl1],
                )
        paths.append(path)
    return paths


def test_help():
    tool = SPEFitter()
    assert run_tool(tool, ["--help-all"]) == 0


def test_plot(example_spe_dl1_paths, tmp_path):
    mpl.use("Agg")
    output_path = tmp_path / "output.h5"
    argv = [
        f"--SPEFitter.input_paths={example_spe_dl1_paths}",
        "--SPEFitter.telescope=1",
        "--SPEFitter.pdf_name=PMTSingleGaussian",
        f"--SPEFitter.output_path={str(output_path)}",
        "--SPEFitter.plot_pixel=1"
    ]
    assert run_tool(SPEFitter(), argv=argv) == 0
    assert not output_path.exists()

    output_path = tmp_path / "output.h5"
    argv = [
        f"--SPEFitter.input_paths={example_spe_dl1_paths}",
        "--SPEFitter.telescope=1",
        "--SPEFitter.pdf_name=PMTSingleGaussian",
        f"--SPEFitter.output_path={str(output_path)}",
        "--SPEFitter.plot_pixel=10000"
    ]
    assert run_tool(SPEFitter(), argv=argv) == 1
    assert not output_path.exists()


def test_write(example_spe_dl1_paths, tmp_path):
    output_path = tmp_path / "output.h5"
    argv = [
        f"--SPEFitter.input_paths={example_spe_dl1_paths}",
        "--SPEFitter.telescope=1",
        "--SPEFitter.pdf_name=PMTSingleGaussian",
        f"--SPEFitter.output_path={str(output_path)}",
    ]
    assert run_tool(SPEFitter(), argv=argv) == 0
    assert output_path.exists()

    output_path = tmp_path / "output2.h5"
    argv = [
        f"--SPEFitter.input_paths={example_spe_dl1_paths}",
        "--SPEFitter.telescope=1000",
        "--SPEFitter.pdf_name=PMTSingleGaussian",
        f"--SPEFitter.output_path={str(output_path)}",
    ]
    assert run_tool(SPEFitter(), argv=argv) == 1
    assert not output_path.exists()
