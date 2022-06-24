from ctapipe.core import run_tool
from ctapipe.io import read_table


def test_process_apply_energy(tmp_path, energy_regressor_path):
    from ctapipe.tools.process import ProcessorTool

    output = tmp_path / "gamma_prod5.dl2_energy.h5"

    argv = [
        "--input=dataset://gamma_prod5.simtel.zst",
        f"--output={output}",
        "--write-images",
        "--write-stereo-shower",
        "--write-mono-shower",
        f"--energy-regressor={energy_regressor_path}"

    ]
    assert run_tool(ProcessorTool(), argv=argv, cwd=tmp_path) == 0

    print(read_table(output, '/dl2/event/telescope/energy/ExtraTreesRegressor/tel_004'))

