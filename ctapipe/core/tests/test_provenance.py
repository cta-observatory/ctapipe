from ctapipe.core import provenance
from pprint import pprint

def test_prov():

    prov = provenance.Provenance()
    prov.start()
    prov.finish()



if __name__ == '__main__':

    import time

    prov = provenance.Provenance()
    prov.start()

    for ii in range(3):
        time.sleep(1)
        prov.sample()


    prov.finish()
    pprint(prov.provenance)
