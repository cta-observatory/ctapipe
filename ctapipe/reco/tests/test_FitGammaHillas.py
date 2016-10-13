from astropy import units as u

from ctapipe.instrument.InstrumentDescription import load_hessio

from ctapipe.utils.datasets import get_path

from ctapipe.reco.FitGammaHillas import FitGammaHillas
from ctapipe.reco.hillas import hillas_parameters
from ctapipe.reco.cleaning import tailcuts_clean, dilate

from ctapipe.io.hessio import hessio_event_source
from ctapipe.io import CameraGeometry


def test_FitGammaHillas():

    filename = get_path("gamma_test.simtel.gz")
    
    fit = FitGammaHillas()
    fit.setup_geometry( *load_hessio(filename),phi=180*u.deg,theta=20*u.deg )
    
    tel_geom = {}

    source = hessio_event_source(filename)

    for event in source:

        Hillas_Dict = {}
        for tel_id in set(event.trig.tels_with_trigger) & set(event.dl0.tels_with_data):
            

            if tel_id not in tel_geom:
                tel_geom[tel_id] = CameraGeometry.guess(fit.cameras(tel_id)['PixX'].to(u.m),
                                                        fit.cameras(tel_id)['PixY'].to(u.m),
                                                        fit.telescopes['FL'][tel_id-1] * u.m) 
                            
            
            pmt_signal = event.dl0.tel[tel_id].adc_sums[0]
            
            mask = tailcuts_clean(tel_geom[tel_id], pmt_signal, 1,picture_thresh=10.,boundary_thresh=5.)
            if True in mask:
                pmt_signal[mask==False] = 0
                        
            try:
                moments, moms2 = hillas_parameters(fit.cameras(tel_id)['PixX'],
                                                   fit.cameras(tel_id)['PixY'],
                                                   pmt_signal)
                Hillas_Dict[tel_id] = moments
            except Exception as e :
                print(e)
                continue
        
        if len(Hillas_Dict) < 2: continue
    
        fit_result = fit.predict(Hillas_Dict)
        
        print(fit_result)
        assert fit_result
        return 

if __name__ == "__main__":
    test_FitGammaHillas()