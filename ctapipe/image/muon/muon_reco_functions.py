import logging
import warnings

import numpy as np
from astropy import log
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.utils.decorators import deprecated

from ctapipe.coordinates import CameraFrame, NominalFrame, HorizonFrame
from ctapipe.image.cleaning import tailcuts_clean
from ctapipe.image.muon.features import ring_containment
from ctapipe.image.muon.features import ring_completeness
from ctapipe.image.muon.features import npix_above_threshold
from ctapipe.image.muon.features import npix_composing_ring
from ctapipe.image.muon.muon_integrator import MuonLineIntegrate
from ctapipe.image.muon.muon_ring_finder import ChaudhuriKunduRingFitter

logger = logging.getLogger(__name__)


def analyze_muon_event(event):
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

    names = ['LST:LSTCam', 'MST:NectarCam', 'MST:FlashCam', 'MST-SCT:SCTCam',
             'SST-1M:DigiCam', 'SST-GCT:CHEC', 'SST-ASTRI:ASTRICam', 'SST-ASTRI:CHEC']
    tail_cuts = [(5, 7), (5, 7), (10, 12), (5, 7),
                (5, 7), (5, 7), (5, 7), (5, 7)]  # 10, 12?
    impact = [(0.2, 0.9), (0.1, 0.95), (0.2, 0.9), (0.2, 0.9),
              (0.1, 0.95), (0.1, 0.95), (0.1, 0.95), (0.1, 0.95)] * u.m
    ringwidth = [(0.04, 0.08), (0.02, 0.1), (0.01, 0.1), (0.02, 0.1),
                 (0.01, 0.5), (0.02, 0.2), (0.02, 0.2), (0.02, 0.2)] * u.deg
    total_pix = [1855., 1855., 1764., 11328., 1296., 2048., 2368., 2048]
    # 8% (or 6%) as limit
    min_pix = [148., 148., 141., 680., 104., 164., 142., 164]
    # Need to either convert from the pixel area in m^2 or check the camera specs
    ang_pixel_width = [0.1, 0.2, 0.18, 0.067, 0.24, 0.2, 0.17, 0.2, 0.163] * u.deg
    # Found from TDRs (or the pixel area)
    hole_rad = [0.308 * u.m, 0.244 * u.m, 0.244 * u.m,
                4.3866 * u.m, 0.160 * u.m, 0.130 * u.m,
                0.171 * u.m, 0.171 * u.m]  # Assuming approximately spherical hole
    cam_rad = [2.26, 3.96, 3.87, 4., 4.45, 2.86, 5.25, 2.86] * u.deg
    # Above found from the field of view calculation
    sec_rad = [0. * u.m, 0. * u.m, 0. * u.m, 2.7 * u.m,
               0. * u.m, 1. * u.m, 1.8 * u.m, 1.8 * u.m]
    sct = [False, False, False, True, False, True, True, True]
    # Added cleaning here. All these options should go to an input card
    cleaning = True

    muon_cuts = {'Name': names, 'tail_cuts': tail_cuts, 'Impact': impact,
                 'RingWidth': ringwidth, 'total_pix': total_pix,
                 'min_pix': min_pix, 'CamRad': cam_rad, 'SecRad': sec_rad,
                 'SCT': sct, 'AngPixW': ang_pixel_width, 'HoleRad': hole_rad}
    logger.debug(muon_cuts)

    muonringlist = []  # [None] * len(event.dl0.tels_with_data)
    muonintensitylist = []  # [None] * len(event.dl0.tels_with_data)
    tellist = []
    muon_event_param = {'TelIds': tellist,
                        'MuonRingParams': muonringlist,
                        'MuonIntensityParams': muonintensitylist}

    for telid in event.dl0.tels_with_data:

        logger.debug("Analysing muon event for tel %d", telid)
        image = event.dl1.tel[telid].image[0]

        # Get geometry
        teldes = event.inst.subarray.tel[telid]
        geom = teldes.camera
        x, y = geom.pix_x, geom.pix_y

        dict_index = muon_cuts['Name'].index(str(teldes))
        logger.debug('found an index of %d for camera %d',
                     dict_index, geom.cam_id)

        tailcuts = muon_cuts['tail_cuts'][dict_index]
        logger.debug("Tailcuts are %s", tailcuts)

        clean_mask = tailcuts_clean(geom, image, picture_thresh=tailcuts[0],
                                    boundary_thresh=tailcuts[1])

        # TODO: correct this hack for values over 90
        altval = event.mcheader.run_array_direction[1]
        if altval > Angle(90, unit=u.deg):
            warnings.warn('Altitude over 90 degrees')
            altval = Angle(90, unit=u.deg)

        telescope_pointing = SkyCoord(
            alt=altval,
            az=event.mcheader.run_array_direction[0],
            frame=HorizonFrame()
        )
        camera_coord = SkyCoord(
            x=x, y=y,
            frame=CameraFrame(
                focal_length=teldes.optics.equivalent_focal_length,
                rotation=geom.pix_rotation,
                telescope_pointing=telescope_pointing,
            )
        )

        nom_coord = camera_coord.transform_to(
            NominalFrame(origin=telescope_pointing)
        )
        x = nom_coord.delta_az.to(u.deg)
        y = nom_coord.delta_alt.to(u.deg)

        if(cleaning):
            img = image * clean_mask
        else:
            img = image

        muonring = ChaudhuriKunduRingFitter(None)

        logger.debug("img: %s mask: %s, x=%s y= %s", np.sum(image),
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
        muonringparam.obs_id = event.dl0.obs_id
        muonringparam.event_id = event.dl0.event_id
        dist_mask = np.abs(dist - muonringparam.
                           ring_radius) < muonringparam.ring_radius * 0.4
        pix_im = image * dist_mask
        nom_dist = np.sqrt(np.power(muonringparam.ring_center_x,
                                    2) + np.power(muonringparam.ring_center_y, 2))

        minpix = muon_cuts['min_pix'][dict_index]  # 0.06*numpix #or 8%

        mir_rad = np.sqrt(teldes.optics.mirror_area.to("m2") / np.pi)

        # Camera containment radius -  better than nothing - guess pixel
        # diameter of 0.11, all cameras are perfectly circular   cam_rad =
        # np.sqrt(numpix*0.11/(2.*np.pi))

        if(npix_above_threshold(pix_im, tailcuts[0]) > 0.1 * minpix
           and npix_composing_ring(pix_im) > minpix
           and nom_dist < muon_cuts['CamRad'][dict_index]
           and muonringparam.ring_radius < 1.5 * u.deg
           and muonringparam.ring_radius > 1. * u.deg):
            muonringparam.ring_containment = ring_containment(
                muonringparam.ring_radius,
                muon_cuts['CamRad'][dict_index],
                muonringparam.ring_center_x,
                muonringparam.ring_center_y)

            # Guess HESS is 0.16
            # sec_rad = 0.*u.m
            # sct = False
            # if numpix == 2048 and mir_rad > 2.*u.m and mir_rad < 2.1*u.m:
            #     sec_rad = 1.*u.m
            #     sct = True
            #
            # Store muon ring parameters (passing cuts stage 1)
            # muonringlist[idx] = muonringparam

            tellist.append(telid)
            muonringlist.append(muonringparam)
            muonintensitylist.append(None)

            ctel = MuonLineIntegrate(
                mir_rad, hole_radius=muon_cuts['HoleRad'][dict_index],
                pixel_width=muon_cuts['AngPixW'][dict_index],
                sct_flag=muon_cuts['SCT'][dict_index],
                secondary_radius=muon_cuts['SecRad'][dict_index]
            )

            if image.shape[0] == muon_cuts['total_pix'][dict_index]:
                muonintensityoutput = ctel.fit_muon(muonringparam.ring_center_x,
                                                    muonringparam.ring_center_y,
                                                    muonringparam.ring_radius,
                                                    x[dist_mask], y[dist_mask],
                                                    image[dist_mask])

                muonintensityoutput.tel_id = telid
                muonintensityoutput.obs_id = event.dl0.obs_id
                muonintensityoutput.event_id = event.dl0.event_id
                muonintensityoutput.mask = dist_mask

                idx_ring = np.nonzero(pix_im)
                muonintensityoutput.ring_completeness = ring_completeness(
                    x[idx_ring], y[idx_ring], pix_im[idx_ring],
                    muonringparam.ring_radius,
                    muonringparam.ring_center_x,
                    muonringparam.ring_center_y,
                    threshold=30,
                    bins=30)
                muonintensityoutput.ring_size = np.sum(pix_im)

                dist_ringwidth_mask = np.abs(dist - muonringparam.ring_radius
                                             ) < (muonintensityoutput.ring_width)
                pix_ringwidth_im = image * dist_ringwidth_mask
                idx_ringwidth = np.nonzero(pix_ringwidth_im)

                muonintensityoutput.ring_pix_completeness = npix_above_threshold(
                    pix_ringwidth_im[idx_ringwidth], tailcuts[0]) / len(
                    pix_im[idx_ringwidth])

                logger.debug("Tel %d Impact parameter = %s mir_rad=%s "
                             "ring_width=%s", telid,
                             muonintensityoutput.impact_parameter, mir_rad,
                             muonintensityoutput.ring_width)
                conditions = [
                    muonintensityoutput.impact_parameter * u.m <
                    muon_cuts['Impact'][dict_index][1] * mir_rad,

                    muonintensityoutput.impact_parameter
                    > muon_cuts['Impact'][dict_index][0],

                    muonintensityoutput.ring_width
                    < muon_cuts['RingWidth'][dict_index][1],

                    muonintensityoutput.ring_width
                    > muon_cuts['RingWidth'][dict_index][0]
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


@deprecated('0.6')
def analyze_muon_source(source):
    """
    Generator for analyzing all the muon events

    Parameters
    ----------
    source : ctapipe.io.EventSource
        input event source

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
        analyzed_muon = analyze_muon_event(event)

        yield analyzed_muon
