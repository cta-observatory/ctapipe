import json

from ctapipe.core import Provenance
from ctapipe.core.provenance import _ActivityProvenance


def test_Provenance():
    prov = Provenance()
    prov.start_activity("test1")
    prov.add_input_file("input.txt")
    prov.add_output_file("output.txt")
    prov.start_activity("test2")
    prov.add_input_file("input_a.txt")
    prov.add_input_file("input_b.txt")
    assert len(prov.active_activity_names) == 2
    prov.finish_activity("test2")
    prov.finish_activity("test1")

    assert set(prov.finished_activity_names) == {'test2', 'test1'}

    return prov

def test_ActivityProvenance():
    prov = _ActivityProvenance()
    prov.start()
    prov.register_input('test.txt')
    prov.register_input('test2.txt')
    prov.register_output('out.txt')
    prov.sample_cpu_and_memory()
    prov.finish()

if __name__ == '__main__':

    import logging
    logging.basicConfig(level=logging.DEBUG)

    prov = test_Provenance()
    print(json.dumps(prov.provenance, indent=4))
