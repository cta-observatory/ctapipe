import json

from ctapipe.core import Provenance


def test_prov():
    prov = Provenance()
    prov.start()
    prov.register_input("input.txt")
    prov.register_output("output.txt")
    prov.finish()


if __name__ == '__main__':

    import time

    prov = Provenance()
    prov.start()
    prov.register_input('test.txt')
    prov.register_input('test2.txt')
    prov.register_output('out.txt')

    print("please wait...")
    for ii in range(3):
        print("sample", ii)
        time.sleep(1)
        prov.sample()

    prov.finish()
    print(json.dumps(prov.provenance, indent=4))
