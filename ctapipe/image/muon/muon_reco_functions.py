from ctapipe.io.containers import MuonIntensityParameter
from astropy import log
from ctapipe.instrument import CameraGeometry
from ctapipe.image.cleaning import tailcuts_clean
from ctapipe.coordinates import CameraFrame, NominalFrame, HorizonFrame
import numpy as np
from astropy import units as u
from ctapipe.image.muon.muon_ring_finder import ChaudhuriKunduRingFitter
from ctapipe.image.muon.muon_integrator import MuonLineIntegrate

import logging
logger = logging.getLogger(__name__)


def analyze_muon_event(event, params=None, geom_dict=None):
    """
    Generic muon event analyzer. 

    Parameters
    ----------
    event : ctapipe dl1 event container


    Returns
    -------
    muonringparam, muonintensityparam : MuonRingParameter 
    and MuonIntensityParameter container event

    """
    # Declare a dict to define the muon cuts (ASTRI and SCT missing)
    muon_cuts = {}

    names = ['LST:LSTCam', 'MST:NectarCam', 'MST:FlashCam', 'MST-SCT:SCTCam',
             'SST-1M:DigiCam', 'SST-GCT:CHEC', 'SST-ASTRI:ASTRICam']
    TailCuts = [(5, 7), (5, 7), (10, 12), (5, 7), (5, 7), (5, 7), (5, 7)]  # 10,12?
    impact = [(0.2, 0.9), (0.1, 0.95), (0.2, 0.9), (0.2, 0.9),
              (0.1, 0.95), (0.1, 0.95), (0.1, 0.95)]
    ringwidth = [(0.04, 0.08), (0.02, 0.1), (0.01, 0.1), (0.02, 0.1),
                 (0.01, 0.5), (0.02, 0.2), (0.02, 0.2)]
    TotalPix = [1855., 1855., 1764., 11328., 1296., 2048., 2368.]  # 8% (or 6%) as limit
    MinPix = [148., 148., 141., 680., 104., 164., 142.]
    # Need to either convert from the pixel area in m^2 or check the camera specs
    AngPixelWidth = [0.1, 0.2, 0.18, 0.067, 0.24, 0.2, 0.17]
    # Found from TDRs (or the pixel area)
    # hole_rad = []   Need to check and implement
    cam_rad = [2.26, 3.96, 3.87, 4., 4.45, 2.86, 5.25]
    # Above found from the field of view calculation
    sec_rad = [0. * u.m, 0. * u.m, 0. * u.m, 2.7 * u.m,
               0. * u.m, 1. * u.m, 1.8 * u.m]
    sct = [False, False, False, True, False, True, True]


    muon_cuts = {'Name': names, 'TailCuts': TailCuts, 'Impact': impact,
                 'RingWidth': ringwidth, 'TotalPix': TotalPix,
                 'MinPix': MinPix, 'CamRad': cam_rad, 'SecRad': sec_rad,
                 'SCT': sct, 'AngPixW': AngPixelWidth}
    logger.debug(muon_cuts)

    muonringlist = []  # [None] * len(event.dl0.tels_with_data)
    muonintensitylist = []  # [None] * len(event.dl0.tels_with_data)
    tellist = []
    muon_event_param = {'TelIds': tellist,
                        'MuonRingParams': muonringlist,
                        'MuonIntensityParams': muonintensitylist}

    for telid in event.dl0.tels_with_data:

        logger.debug("Analysing muon event for tel %d", telid)
        muonringparam = None
        muonintensityparam = None
        image = event.dl1.tel[telid].image[0]

        # Get geometry
        teldes = event.inst.subarray.tel[telid]
        geom = teldes.camera
        x, y = geom.pix_x, geom.pix_y

        dict_index = muon_cuts['Name'].index(str(teldes))
        logger.debug('found an index of %d for camera %d',
                     dict_index, geom.cam_id)

        tailcuts = muon_cuts['TailCuts'][dict_index]

        logger.debug("Tailcuts are %s", tailcuts)

        clean_mask = tailcuts_clean(geom, image, picture_thresh=tailcuts[0],
                                    boundary_thresh=tailcuts[1])
        camera_coord = CameraFrame(
            x=x, y=y,
            focal_length=teldes.optics.equivalent_focal_length,
            rotation=geom.pix_rotation
        )

        # TODO: correct this hack for values over 90
        altval = event.mcheader.run_array_direction[1]
        if (altval > np.pi / 2.):
            altval = np.pi / 2.

        altaz = HorizonFrame(alt=altval * u.rad,
                             az=event.mcheader.run_array_direction[0] * u.rad)
        nom_coord = camera_coord.transform_to(
            NominalFrame(array_direction=altaz, pointing_direction=altaz)
        )
        x = nom_coord.x.to(u.deg)
        y = nom_coord.y.to(u.deg)

        img = image * clean_mask
        muonring = ChaudhuriKunduRingFitter(None)

        logger.debug("img: %s mask: %s, x=%s y= %s",np.sum(image),
                     np.sum(clean_mask), x, y)

        if not sum(img):  # Nothing left after tail cuts
            continue

        muonringparam = muonring.fit(x, y, image * clean_mask)

        dist = np.sqrt(np.power(x - muonringparam. ring_center_x, 2)
                       + np.power(y - muonringparam.ring_center_y, 2))
        ring_dist = np.abs(dist - muonringparam.ring_radius)

        muonringparam = muonring.fit(
            x, y, img * (ring_dist < muonringparam.ring_radius * 0.4)
        )

        dist = np.sqrt(np.power(x - muonringparam.ring_center_x, 2) +
                       np.power(y - muonringparam.ring_center_y, 2))
        ring_dist = np.abs(dist - muonringparam.ring_radius)

        muonringparam = muonring.fit(
            x, y, img * (ring_dist < muonringparam.ring_radius * 0.4)
        )

        muonringparam.tel_id = telid
        muonringparam.run_id = event.dl0.run_id
        muonringparam.event_id = event.dl0.event_id
        dist_mask = np.abs(dist - muonringparam.
                           ring_radius) < muonringparam.ring_radius * 0.4
        pix_im = image * dist_mask
        nom_dist = np.sqrt(np.power(muonringparam.ring_center_x,
                                    2) + np.power(muonringparam.ring_center_y, 2))


        minpix = muon_cuts['MinPix'][dict_index]  # 0.06*numpix #or 8%

        mir_rad = np.sqrt(teldes.optics.mirror_area.to("m2") / (np.pi))

        # Camera containment radius -  better than nothing - guess pixel
        # diameter of 0.11, all cameras are perfectly circular   cam_rad =
        # np.sqrt(numpix*0.11/(2.*np.pi))

        if(np.sum(pix_im > tailcuts[0]) > 0.1 * minpix
           and np.sum(pix_im) > minpix
           and nom_dist < muon_cuts['CamRad'][dict_index] * u.deg
           and muonringparam.ring_radius < 1.5 * u.deg
           and muonringparam.ring_radius > 1. * u.deg):

            #Guess HESS is 0.16
            #sec_rad = 0.*u.m
            #sct = False
            #if numpix == 2048 and mir_rad > 2.*u.m and mir_rad < 2.1*u.m:
            #    sec_rad = 1.*u.m
            #    sct = True
            #
            #Store muon ring parameters (passing cuts stage 1)
            #muonringlist[idx] = muonringparam

            tellist.append(telid)
            muonringlist.append(muonringparam)
            muonintensitylist.append(None)

            ctel = MuonLineIntegrate(
                mir_rad, 0.2 * u.m,
                pixel_width=muon_cuts['AngPixW'][dict_index] * u.deg,
                sct_flag=muon_cuts['SCT'][dict_index],
                secondary_radius=muon_cuts['SecRad'][dict_index]
            )

            if (image.shape[0] == muon_cuts['TotalPix'][dict_index]):
                muonintensityoutput = ctel.fit_muon(muonringparam.ring_center_x,
                                                    muonringparam.ring_center_y,
                                                    muonringparam.ring_radius,
                                                    x[dist_mask], y[dist_mask],
                                                    image[dist_mask])

                muonintensityoutput.tel_id = telid
                muonintensityoutput.run_id = event.dl0.run_id
                muonintensityoutput.event_id = event.dl0.event_id
                muonintensityoutput.mask = dist_mask

                logger.debug("Tel %d Impact parameter = %s mir_rad=%s "
                             "ring_width=%s", telid,
                             muonintensityoutput.impact_parameter, mir_rad,
                             muonintensityoutput.ring_width)

                conditions = [
                    muonintensityoutput.impact_parameter <
                    muon_cuts['Impact'][dict_index][1] * mir_rad,

                    muonintensityoutput.impact_parameter
                    > muon_cuts['Impact'][dict_index][0] * u.m,

                    muonintensityoutput.ring_width
                    <muon_cuts['RingWidth'][dict_index][1] * u.deg,

                    muonintensityoutput.ring_width
                    > muon_cuts['RingWidth'][dict_index][0] * u.deg
                ]

                if all(conditions):
                    muonintensityparam = muonintensityoutput
                    idx = tellist.index(telid)
                    muonintensitylist[idx] = muonintensityparam
                    logger.debug("Muon found in tel %d,  tels in event=%d",
                                 telid, len(event.dl0.tels_with_data))
                else:
                    continue

    return muon_event_param


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
        geom_dict = {}        
    numev = 0
    for event in source:  # Put a limit on number of events
        numev += 1
        analyzed_muon = analyze_muon_event(event, params, geom_dict)

        yield analyzed_muon
