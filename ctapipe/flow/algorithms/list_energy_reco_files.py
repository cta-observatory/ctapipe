from ctapipe.core import Component
from traitlets import Unicode
#from ctadata import common_ptime_preco_list


class ListEnergyRecoFiles(Component):

    """ListEnergyRecoFiles class represents a Producer for pipeline.
    It yields a dictionary containing two keys. Both values are filename
        - 'RecoEvent'
        - 'TimeHillas'
    """
    #source_dir = Unicode(help='directory containing data files').tag(config=True)
    source_dir = '/tmp'

    def init(self):
        self.log.info('----- ListEnergyRecoFiles init source_dir={}'.format(self.source_dir))
        self.hillas_list, self.recoevent_list = None, None#common_ptime_preco_list(self.source_dir)
        if len(self.hillas_list) != len(self.recoevent_list):
            self.log.error('input files list do not have the same size')
            return False
        return True

    def run(self):
        for hillas, reco in zip(self.hillas_list, self.recoevent_list):
            yield (hillas, reco)

        self.log.debug("\n--- ListEnergyRecoFiles Done ---")

    def finish(self):
        self.log.debug("--- ListEnergyRecoFiles finish ---")
        pass
