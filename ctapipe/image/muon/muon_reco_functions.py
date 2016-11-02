from ctapipe.io.containers import CalibratedCameraData, MuonRingParameter, MuonIntensityParameter
from astropy import log
from ctapipe.image.cleaning import tailcuts_clean
from ctapipe.coordinates import CameraFrame, NominalFrame
import numpy as np
from astropy import units as u

from IPython import embed

from ctapipe.image.muon.muon_ring_finder import ChaudhuriKunduRingFitter
from ctapipe.image.muon.muon_integrator import *
from ctapipe.image.muon.muon_diagnostic_plots import plot_muon_event


def analyze_muon_event(event, params=None, geom_dict=None):
    """
    Generic muon event analyzer.

    Parameters
    ----------
    event : ctapipe dl1 event container


    Returns
    -------
    muonringparam, muonintensityparam : MuonRingParameter and MuonIntensityParameter container event

    """

    muonringparam = None
    muonintensityparam = None

    for telid in event.dl0.tels_with_data:

        x, y = event.meta.pixel_pos[telid]

        image = event.dl1.tel[telid].pe_charge

        # Get geometry
        geom = None
        if geom_dict is not None and telid in geom_dict:
            geom = geom_dict[telid]
        else:
            log.debug("[calib] Guessing camera geometry")
            geom = CameraGeometry.guess(*event.meta.pixel_pos[telid],
                                        event.meta.optical_foclen[telid])
            log.debug("[calib] Camera geometry found")
            if geom_dict is not None:
                geom_dict[telid] = geom
        
        #embed()
        tailcuts = (5.,7.)
        #Try a higher threshold for FlashCam
        if event.meta.optical_foclen[telid] == 16.*u.m and event.dl0.tel[telid].num_pixels == 1764:
            tailcuts = (10.,12.)


        clean_mask = tailcuts_clean(geom,image,1,picture_thresh=tailcuts[0],boundary_thresh=tailcuts[1])#was 5,7 (1.5,2.5)

        camera_coord = CameraFrame(x=x,y=y,z=np.zeros(x.shape)*u.m)

        nom_coord = camera_coord.transform_to(NominalFrame(array_direction=[event.mc.alt, event.mc.az],pointing_direction=[event.mc.alt, event.mc.az],focal_length = event.meta.optical_foclen[telid])) # tel['TelescopeTable_VersionFeb2016'][tel['TelescopeTable_VersionFeb2016']['TelID']==telid]['FL'][0]*u.m))
        
        x = nom_coord.x.to(u.deg)
        y = nom_coord.y.to(u.deg)

        img = image*clean_mask
        noise = 5.
        weight = img / (img+noise)

        muonring = ChaudhuriKunduRingFitter(None)

        muonringparam = muonring.fit(x,y,image*clean_mask)
        #muonringparam = muonring.fit(x,y,weight)
        dist = np.sqrt(np.power(x-muonringparam.ring_center_x,2) + np.power(y-muonringparam.ring_center_y,2))
        ring_dist = np.abs(dist-muonringparam.ring_radius)
        #embed()
        #1/0
        muonringparam = muonring.fit(x,y,img*(ring_dist<muonringparam.ring_radius*0.3))

        dist = np.sqrt(np.power(x-muonringparam.ring_center_x,2) + np.power(y-muonringparam.ring_center_y,2))
        ring_dist = np.abs(dist-muonringparam.ring_radius)
        
        #embed()

        muonringparam = muonring.fit(x,y,img*(ring_dist<muonringparam.ring_radius*0.3))
        muonringparam.tel_id = telid
        muonringparam.run_id = event.dl1.run_id
        muonringparam.event_id = event.dl1.event_id
        dist_mask = np.abs(dist-muonringparam.ring_radius)<muonringparam.ring_radius*0.4

        #print("Fitted ring centre:",muonringparam.ring_center_x,muonringparam.ring_center_y)

        #embed()
        #1/0

        rad = list()
        cx = list()
        cy = list()
        
        mc_x = event.mc.core_x
        mc_y = event.mc.core_y
        pix_im = image*dist_mask
        nom_dist = np.sqrt(np.power(muonringparam.ring_center_x,2)+np.power(muonringparam.ring_center_y,2))
        numpix = event.dl0.tel[telid].num_pixels
        minpix = 0.06*numpix #or 8%

        mir_rad = np.sqrt(event.meta.mirror_dish_area[telid]/(np.pi))#need to consider units?

        if(np.sum(pix_im>5)>0.5*minpix and np.sum(pix_im)>minpix and nom_dist <1.*u.deg and muonringparam.ring_radius<1.5*u.deg and muonringparam.ring_radius>1.*u.deg):

            #Guess HESS is 0.16 - LST is 0.11?
            hess = MuonLineIntegrate(mir_rad,0.2*u.m,pixel_width=0.11*u.deg)

            if (image.shape[0]<2000):
                muonintensityoutput = hess.fit_muon(muonringparam.ring_center_x,muonringparam.ring_center_y,muonringparam.ring_radius,x[dist_mask],y[dist_mask],image[dist_mask])
                muonintensityoutput.tel_id = telid
                muonintensityoutput.run_id = event.dl1.run_id
                muonintensityoutput.event_id = event.dl1.event_id
                if( muonintensityoutput.impact_parameter < 0.9*mir_rad and muonintensityoutput.impact_parameter>0.2*u.m and muonintensityoutput.ring_width<0.08*u.deg and muonintensityoutput.ring_width>0.04*u.deg ):
                    muonintensityparam = muonintensityoutput
                else:
                    continue

        #print("Fitted ring centre (2):",muonringparam.ring_center_x,muonringparam.ring_center_y)


    return muonringparam, muonintensityparam

def analyze_muon_source(source, params=None, geom_dict=None, args=None):
    """
    Generator for analyzing all the muon events

    Parameters
    ----------
    source : generator
    A 'ctapipe' event generator as
    'ctapipe.io.hessio_event_source

    Returns
    -------
    analyzed_muon : container
    A ctapipe event container (MuonParameter) with muon information

    """
    log.info("[FUNCTION] {}".format(__name__))

    if geom_dict is None:
        geom_dict={}
        
    numev = 0
    for event in source:#Put a limit on number of events
        numev += 1
        analyzed_muon = analyze_muon_event(event, params, geom_dict)
        print("Analysed event number",numev)
        #
        plot_muon_event(event, analyzed_muon, geom_dict, args)

        if numev > 20: #Ugly, for testing purposes only
            break

        yield analyzed_muon
