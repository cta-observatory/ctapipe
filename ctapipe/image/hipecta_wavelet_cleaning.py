from ctapipe.core import Component
import hipecta

#obsolete import, replace ASAP for injunction table computation
import hipectaold.core as core 


class WaveletCleaner (Component) :
    def __init__ (self) :
        self.tabInj_dict = {}
    
    def apply (self, cam_geom, image, alpha=2.0) :
        if not cam_geom.cam_id in self.tabInj_dict.keys() :
            self._compute_tabinj (cam_geom)
        return hipecta.image.hexwave_cleaning (
            image,
            self.tabInj_dict[cam_geom.cam_id][0], #tabinj
            self.tabInj_dict[cam_geom.cam_id][1], #nbr
            self.tabInj_dict[cam_geom.cam_id][2], #nbc
            alpha #wavelet threshold
        )
        
    def _compute_tabinj (self, cam_geom) :
        tabpix = empty((2*cam_geom.pix_x.value.shape[0]), dtype=np.float32)
 
        for index in range(cam_geom.pix_x.value.shape[0]) :
            tabpix[2*index] = cam_geom.pix_x.value[index]
            tabpix[2*index+1] = cam_geom.pix_y.value[index]
        
        tabpix = tabpix.reshape((cam_geom.pix_x.value.shape[0], 2))
        #TODO : Implement injunction table computation in the new hipecta
        tabInj, nbr, nbc = core.createAutoInjunctionTable(tabpix)
        self.tabInj_dict[cam_geom.cam_id] = (tabInj, nbr, nbc)

