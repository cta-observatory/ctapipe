import json

from ctapipe.core import Provenance
from ctapipe.core.provenance import _ActivityProvenance
from ctapipe.io.metadata import Reference


def test_provenance_activity_names(provenance):
    provenance.start_activity("test1")
    provenance.add_input_file("input.txt")
    provenance.add_output_file("output.txt")
    provenance.start_activity("test2")
    provenance.add_input_file("input_a.txt")
    provenance.add_input_file("input_b.txt")
    provenance.finish_activity("test2")
    provenance.finish_activity("test1")
    assert set(provenance.finished_activity_names) == {"test2", "test1"}


def test_ActivityProvenance():
    prov = _ActivityProvenance()
    prov.start()
    prov.register_input("test.txt")
    prov.register_input("test2.txt")
    prov.register_output("out.txt")
    prov.sample_cpu_and_memory()
    prov.finish()


def test_provenence_contextmanager():
    prov = Provenance()

    with prov.activity("myactivity"):
        assert "myactivity" in prov.active_activity_names

    assert "myactivity" in prov.finished_activity_names
    assert "myactivity" not in prov.active_activity_names


def test_provenance_json(provenance: Provenance):
    provenance.start_activity("test1")
    provenance.finish_activity("test1")
    data = json.loads(provenance.as_json())

    activity = data[0]

    assert "python" in activity["system"]
    packages = activity["system"]["python"].get("packages")
    assert isinstance(packages, list)
    assert any(p["name"] == "numpy" for p in packages)


def test_provenance_input_reference_meta(provenance: Provenance, dl1_file):
    provenance.start_activity("test1")
    provenance.add_input_file(dl1_file, "events")
    provenance.finish_activity("test1")
    data = json.loads(provenance.as_json())

    inputs = data[0]["input"]
    assert len(inputs) == 1
    input_meta = inputs[0]
    assert "reference_meta" in input_meta
    assert "CTA PRODUCT ID" in input_meta["reference_meta"]
    Reference.from_dict(input_meta["reference_meta"])


def test_get_distribution_of_module():
    from ctapipe.core.provenance import get_distribution_of_module

    assert get_distribution_of_module("yaml").name == "PyYAML"
    assert get_distribution_of_module("sklearn").name == "scikit-learn"
    assert get_distribution_of_module("ctapipe_test_plugin").version == "0.1.0"
