from ctapipe.io.containers import  MuonRingParameter, MuonIntensityParameter
from astropy import log
from ctapipe.instrument import CameraGeometry
from ctapipe.image.cleaning import tailcuts_clean
from ctapipe.coordinates import CameraFrame, NominalFrame, HorizonFrame
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
    # Declare a dict to define the muon cuts (ASTRI and SCT missing)
    muon_cuts = {}
    names = ['LSTCam','NectarCam','FlashCam','SST-1m','CHEC']
    TailCuts = [(5,7),(5,7),(75,80),(30,35),(5,7)] #10,12?
    impact = [(0.2,0.9),(0.2,0.9),(0.2,0.9),(0.2,0.9),(0.2,0.9)]
    ringwidth = [(0.04,0.08),(0.04,0.08),(0.01,0.5),(0.01,0.5),(0.04,0.08)]
    TotalPix = [1855.,1855.,1764.,1296.,2048.]#8% (or 6%) as limit
    MinPix = [148.,148.,141.,104.,164.]
    #Need to either convert from the pixel area in m^2 or check the camera specs
    AngPixelWidth = [0.1,0.2,0.18,0.24,0.2] #Found from TDRs (or the pixel area)
    hole_rad = []#Need to check and implement
    cam_rad = [2.26,3.96,3.87,4.45,2.86]#Found from the field of view calculation
    sec_rad = [0.*u.m,0.*u.m,0.*u.m,0.*u.m,1.*u.m]
    sct = [False,False,False,False,True]


    muon_cuts = {'Name':names,'TailCuts':TailCuts,'Impact':impact,'RingWidth':ringwidth,'TotalPix':TotalPix,'MinPix':MinPix,'CamRad':cam_rad,'SecRad':sec_rad,'SCT':sct,'AngPixW':AngPixelWidth}
    #print(muon_cuts)

    muonringparam = None
    muonintensityparam = None

    for telid in event.dl0.tels_with_data:

        print("Analysing muon event for tel",telid)
        
        x, y = event.inst.pixel_pos[telid]

        #image = event.dl1.tel[telid].calibrated_image
        image = event.dl1.tel[telid].image[0]

        # Get geometry
        geom = None
        if geom_dict is not None and telid in geom_dict:
            geom = geom_dict[telid]
        else:
            log.debug("[calib] Guessing camera geometry")
            geom = CameraGeometry.guess(*event.inst.pixel_pos[telid],
                                        event.inst.optical_foclen[telid])
            log.debug("[calib] Camera geometry found")
            if geom_dict is not None:
                geom_dict[telid] = geom


        dict_index = muon_cuts['Name'].index(geom.cam_id)
        #print('found an index of',dict_index,'for camera',geom.cam_id)

        #tailcuts = (5.,7.)
        tailcuts = muon_cuts['TailCuts'][dict_index]

        #print("Tailcuts are",tailcuts[0],tailcuts[1])

        #rot_angle = 0.*u.deg
        #if event.inst.optical_foclen[telid] > 10.*u.m and event.dl0.tel[telid].num_pixels != 1764:
            #rot_angle = -100.14*u.deg

        clean_mask = tailcuts_clean(geom,image,picture_thresh=tailcuts[0],boundary_thresh=tailcuts[1])
        camera_coord = CameraFrame(x=x,y=y,z=np.zeros(x.shape)*u.m, focal_length = event.inst.optical_foclen[telid], rotation=geom.pix_rotation)

        #print("Camera",geom.cam_id,"focal length",event.inst.optical_foclen[telid],"rotation",geom.pix_rotation)
        #TODO: correct this hack for values over 90
        altval = event.mcheader.run_array_direction[1]
        if (altval > np.pi/2.):
            altval = np.pi/2.

        altaz = HorizonFrame(alt=altval*u.rad,az=event.mcheader.run_array_direction[0]*u.rad)
        nom_coord = camera_coord.transform_to(NominalFrame(array_direction=altaz,pointing_direction=altaz))

        
        x = nom_coord.x.to(u.deg)
        y = nom_coord.y.to(u.deg)

        img = image*clean_mask
        noise = 5.
        weight = img / (img+noise)

        muonring = ChaudhuriKunduRingFitter(None)

        #print("img:",np.sum(image),"mask:",np.sum(clean_mask), "x=",x,"y=",y)
        muonringparam = muonring.fit(x,y,image*clean_mask)

        #muonringparam = muonring.fit(x,y,weight)
        dist = np.sqrt(np.power(x-muonringparam.ring_center_x,2) + np.power(y-muonringparam.ring_center_y,2))
        ring_dist = np.abs(dist-muonringparam.ring_radius)
        muonringparam = muonring.fit(x,y,img*(ring_dist<muonringparam.ring_radius*0.4))

        dist = np.sqrt(np.power(x-muonringparam.ring_center_x,2) + np.power(y-muonringparam.ring_center_y,2))
        ring_dist = np.abs(dist-muonringparam.ring_radius)

        #print("1: x",muonringparam.ring_center_x,"y",muonringparam.ring_center_y,"radius",muonringparam.ring_radius)
        muonringparam = muonring.fit(x,y,img*(ring_dist<muonringparam.ring_radius*0.4))
        #print("2: x",muonringparam.ring_center_x,"y",muonringparam.ring_center_y,"radius",muonringparam.ring_radius)
        muonringparam.tel_id = telid
        muonringparam.run_id = event.dl0.run_id
        muonringparam.event_id = event.dl0.event_id
        dist_mask = np.abs(dist-muonringparam.ring_radius)<muonringparam.ring_radius*0.4

        rad = list()
        cx = list()
        cy = list()
        
        mc_x = event.mc.core_x
        mc_y = event.mc.core_y
        pix_im = image*dist_mask
        nom_dist = np.sqrt(np.power(muonringparam.ring_center_x,2)+np.power(muonringparam.ring_center_y,2))
        #numpix = event.dl0.tel[telid].num_pixels

        minpix = muon_cuts['MinPix'][dict_index]#0.06*numpix #or 8%

        mir_rad = np.sqrt(event.inst.mirror_dish_area[telid]/(np.pi))#need to consider units? (what about hole? Area is then less...)


        #Camera containment radius -  better than nothing - guess pixel diameter of 0.11, all cameras are perfectly circular   cam_rad = np.sqrt(numpix*0.11/(2.*np.pi))

        if(np.sum(pix_im>tailcuts[0])>0.1*minpix and np.sum(pix_im)>minpix and nom_dist < muon_cuts['CamRad'][dict_index]*u.deg and muonringparam.ring_radius<1.5*u.deg and muonringparam.ring_radius>1.*u.deg):

            #Guess HESS is 0.16 
            #sec_rad = 0.*u.m
            #sct = False
            #if numpix == 2048 and mir_rad > 2.*u.m and mir_rad < 2.1*u.m:
            #    sec_rad = 1.*u.m
            #    sct = True

            ctel = MuonLineIntegrate(mir_rad,0.2*u.m,pixel_width=muon_cuts['AngPixW'][dict_index]*u.deg,sct_flag=muon_cuts['SCT'][dict_index], secondary_radius=muon_cuts['SecRad'][dict_index])
          
            if (image.shape[0] == muon_cuts['TotalPix'][dict_index]):
                muonintensityoutput = ctel.fit_muon(muonringparam.ring_center_x,muonringparam.ring_center_y,muonringparam.ring_radius,x[dist_mask],y[dist_mask],image[dist_mask])

                muonintensityoutput.tel_id = telid
                muonintensityoutput.run_id = event.dl0.run_id
                muonintensityoutput.event_id = event.dl0.event_id
                muonintensityoutput.mask = dist_mask
                print("Impact parameter = ",muonintensityoutput.impact_parameter,"mir_rad",mir_rad,"ring_width=",muonintensityoutput.ring_width)

                #print("Cuts <",muon_cuts["Impact"][dict_index][1]*mir_rad,"cuts >",muon_cuts['Impact'][dict_index][0]*u.m,'ring_width < ',muon_cuts['RingWidth'][dict_index][1]*u.deg,'ring_width >',muon_cuts['RingWidth'][dict_index][0]*u.deg)
                if( muonintensityoutput.impact_parameter < muon_cuts['Impact'][dict_index][1]*mir_rad and muonintensityoutput.impact_parameter > muon_cuts['Impact'][dict_index][0]*u.m and muonintensityoutput.ring_width < muon_cuts['RingWidth'][dict_index][1]*u.deg and muonintensityoutput.ring_width > muon_cuts['RingWidth'][dict_index][0]*u.deg ):
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
        #   if analyzed_muon[1] is not None:
        #           plot_muon_event(event, analyzed_muon, geom_dict, args)
            
        #  if numev > 50: #for testing purposes only
        #          break

        yield analyzed_muon
