import logging
import warnings

import numpy as np
from astropy import log
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord, AltAz
from astropy.utils.decorators import deprecated

from ctapipe.coordinates import CameraFrame, TelescopeFrame
from ctapipe.image.cleaning import tailcuts_clean
from ctapipe.image.muon.features import ring_containment
from ctapipe.image.muon.features import ring_completeness
from ctapipe.image.muon.features import npix_above_threshold
from ctapipe.image.muon.features import npix_composing_ring
from ctapipe.image.muon.muon_integrator import MuonLineIntegrate
from ctapipe.image.muon.muon_ring_finder import MuonRingFitter

logger = logging.getLogger(__name__)


def transform_pixel_coords_from_meter_to_deg(x, y, foc_len, fast_but_bad=False):
    if fast_but_bad:
        delta_alt = np.rad2deg((x / foc_len).values) * u.deg
        delta_az = np.rad2deg((y / foc_len).values) * u.deg

    else:
        pixel_coords_in_telescope_frame = SkyCoord(
            x=x,
            y=y,
            frame=CameraFrame(focal_length=foc_len)
        ).transform_to(TelescopeFrame())
        delta_az = pixel_coords_in_telescope_frame.delta_az.deg
        delta_alt = pixel_coords_in_telescope_frame.delta_alt.deg

    return delta_az, delta_alt

def calc_nom_dist(ring_fit):
    nom_dist = np.sqrt(
        (ring_fit.ring_center_x)**2 +
        (ring_fit.ring_center_y)**2
    )
    return nom_dist

def calc_dist_and_ring_dist(x, y, ring_fit, parameter=0.4):
    dist = np.sqrt(
        (x - ring_fit.ring_center_x)**2 +
        (y - ring_fit.ring_center_y)**2
    )
    ring_dist = np.abs(dist - ring_fit.ring_radius)
    dist_mask = ring_dist < ring_fit.ring_radius * parameter

    return dist, ring_dist, dist_mask


def generate_muon_cuts_by_telescope_name():
    names = ['LST_LST_LSTCam', 'MST_MST_NectarCam', 'MST_MST_FlashCam', 'MST_SCT_SCTCam',
             'SST_1M_DigiCam', 'SST_GCT_CHEC', 'SST_ASTRI_ASTRICam', 'SST_ASTRI_CHEC']
    tail_cuts = [(5, 7), (5, 7), (10, 12), (5, 7),
                (5, 7), (5, 7), (5, 7), (5, 7)]  # 10, 12?
    impact = [(0.2, 0.9), (0.1, 0.95), (0.2, 0.9), (0.2, 0.9),
              (0.1, 0.95), (0.1, 0.95), (0.1, 0.95), (0.1, 0.95)] # in units of mirror radii
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


    muon_cuts = {'Name': names, 'tail_cuts': tail_cuts, 'Impact': impact,
                 'RingWidth': ringwidth, 'total_pix': total_pix,
                 'min_pix': min_pix, 'CamRad': cam_rad, 'SecRad': sec_rad,
                 'SCT': sct, 'AngPixW': ang_pixel_width, 'HoleRad': hole_rad}

    muon_cuts_list_of_dicts = [
        {k:v for k,v in zip(muon_cuts.keys(), values)}
        for values in zip(*muon_cuts.values())
    ]
    muon_cuts_by_name = {mc['Name']:mc for mc in muon_cuts_list_of_dicts}

    # replace tail_cuts tuples with more descriptive dicts.
    for muon_cut in muon_cuts_by_name.values():
        muon_cut['tail_cuts'] = {
            'picture_thresh': muon_cut['tail_cuts'][0],
            'boundary_thresh': muon_cut['tail_cuts'][1],
        }

    return muon_cuts_by_name

def is_something_good(pix_im, ring_fit, muon_cut):
    '''this is testing something on the image and the fit,
    but I do not really get it.
    '''
    return (
        npix_above_threshold(
            pix_im, muon_cut['tail_cuts']['picture_thresh']
        ) > 0.1 * muon_cut['min_pix']
        and npix_composing_ring(pix_im) > muon_cut['min_pix']
        and calc_nom_dist(ring_fit) < muon_cut['CamRad']
        and ring_fit.ring_radius < 1.5 * u.deg
        and ring_fit.ring_radius > 1. * u.deg
    )

def do_multi_ring_fit(x, y, image, clean_mask):
    # 1st fit
    ring_fit = muon_ring_fit(x, y, image, clean_mask)
    dist, ring_dist, dist_mask = calc_dist_and_ring_dist(x, y, ring_fit)
    mask = clean_mask * dist_mask
    # 2nd fit
    ring_fit = muon_ring_fit(x, y, image, mask)
    dist, ring_dist, dist_mask = calc_dist_and_ring_dist(x, y, ring_fit)
    mask *= dist_mask
    # 3rd fit
    ring_fit = muon_ring_fit(x, y, image, mask)
    dist, ring_dist, dist_mask = calc_dist_and_ring_dist(x, y, ring_fit)
    mask *= dist_mask

    return ring_fit, mask
    pix_im = image * mask


def analyze_muon_event(event):
    """
    Generic muon event analyzer.

    Parameters
    ----------
    event : ctapipe dl1 event container


    Returns
    -------
    ring_fit, muonintensityparam : MuonRingParameter
    and MuonIntensityParameter container event

    """
    muon_cuts_by_name = generate_muon_cuts_by_telescope_name()

    logger.debug(muon_cuts)

    output = []
    for telid in event.dl0.tels_with_data:
        logger.debug("Analysing muon event for tel %d", telid)
        image = event.dl1.tel[telid].image

        teldes = event.inst.subarray.tel[telid]
        foc_len = teldes.optics.equivalent_focal_length
        geom = teldes.camera
        optics = teldes.optics
        mirror_radius = optics.mirror_radius
        x, y = geom.pix_x, geom.pix_y

        muon_cut = muon_cuts_by_name[str(teldes)]
        tailcuts = muon_cut['tail_cuts']
        logger.debug("Tailcuts are %s", tailcuts)
        clean_mask = tailcuts_clean(geom, image, **tailcuts)

        x, y = transform_pixel_coords_from_meter_to_deg(
            x, y, foc_len, fast_but_bad=True)

        muon_ring_fit = MuonRingFitter(fit_method="chaudhuri_kundu")

        logger.debug("img: %s mask: %s, x=%s y= %s", np.sum(image),
                     np.sum(clean_mask), x, y)

        if not np.any(clean_mask):  # early bail out - safes time
            continue

        ring_fit, mask = do_multi_ring_fit(x, y, image, clean_mask)
        pix_im = image * mask

        if is_something_good(pix_im, ring_fit, muon_cut)
            ring_fit.ring_containment = ring_containment(
                ring_fit.ring_radius,
                muon_cut['CamRad'],
                ring_fit.ring_center_x,
                ring_fit.ring_center_y
            )

            ctel = MuonLineIntegrate(
                mirror_radius,
                hole_radius=muon_cut['HoleRad'],
                pixel_width=muon_cut['AngPixW'],
                sct_flag=muon_cut['SCT'],
                secondary_radius=muon_cut['SecRad'],
            )

            muonintensityoutput = ctel.fit_muon(
                ring_fit.ring_center_x,
                ring_fit.ring_center_y,
                ring_fit.ring_radius,
                x[mask],
                y[mask],
                image[mask]
            )

            muonintensityoutput.tel_id = telid
            muonintensityoutput.obs_id = event.dl0.obs_id
            muonintensityoutput.event_id = event.dl0.event_id
            muonintensityoutput.mask = mask

            idx_ring = np.nonzero(pix_im)
            muonintensityoutput.ring_completeness = ring_completeness(
                x[idx_ring],
                y[idx_ring],
                pix_im[idx_ring],
                ring_fit.ring_radius,
                ring_fit.ring_center_x,
                ring_fit.ring_center_y,
                threshold=30,
                bins=30)
            muonintensityoutput.ring_size = np.sum(pix_im)

            dist_ringwidth_mask = ring_dist < muonintensityoutput.ring_width
            pix_ringwidth_im = image * dist_ringwidth_mask
            idx_ringwidth = np.nonzero(pix_ringwidth_im)

            muonintensityoutput.ring_pix_completeness = (
                npix_above_threshold(pix_ringwidth_im[idx_ringwidth], tailcuts['picture_thresh'])
                / len(pix_im[idx_ringwidth])
            )

            logger.debug("Tel %d Impact parameter = %s mirror_radius=%s "
                         "ring_width=%s", telid,
                         muonintensityoutput.impact_parameter, mirror_radius,
                         muonintensityoutput.ring_width)
            conditions = [
                muonintensityoutput.impact_parameter <
                muon_cut['Impact'][1] * mirror_radius,

                muonintensityoutput.impact_parameter
                > muon_cut['Impact'][0] * mirror_radius,

                muonintensityoutput.ring_width
                < muon_cut['RingWidth'][1],

                muonintensityoutput.ring_width
                > muon_cut['RingWidth'][0]
            ]


        ring_fit.tel_id = telid
        ring_fit.obs_id = event.dl0.obs_id
        ring_fit.event_id = event.dl0.event_id

        output.append({
            'MuonRingParams': ring_fit,
            'MuonIntensityParams': muonintensityoutput,
            'muon_found': all(conditions),
            'mirror_radius': mirror_radius,
        })

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
    log.info(f"[FUNCTION] {__name__}")

    if geom_dict is None:
        geom_dict = {}
    numev = 0
    for event in source:  # Put a limit on number of events
        numev += 1
        analyzed_muon = analyze_muon_event(event)

        yield analyzed_muon
