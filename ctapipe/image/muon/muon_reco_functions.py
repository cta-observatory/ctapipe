import logging
import warnings

import numpy as np
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord, AltAz

from ctapipe.coordinates import CameraFrame, NominalFrame
from ctapipe.image.cleaning import tailcuts_clean
from ctapipe.image.muon.features import ring_containment
from ctapipe.image.muon.features import ring_completeness
from ctapipe.image.muon.features import npix_above_threshold
from ctapipe.image.muon.features import npix_composing_ring
from ctapipe.image.muon.intensity_fit import fit_muon
from ctapipe.image.muon.muon_ring_finder import MuonRingFitter

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

    names = [
        "LST_LST_LSTCam",
        "MST_MST_NectarCam",
        "MST_MST_FlashCam",
        "MST_SCT_SCTCam",
        "SST_1M_DigiCam",
        "SST_GCT_CHEC",
        "SST_ASTRI_ASTRICam",
        "SST_ASTRI_CHEC",
    ]
    tail_cuts = [
        (5, 7),
        (5, 7),
        (10, 12),
        (5, 7),
        (5, 7),
        (5, 7),
        (5, 7),
        (5, 7),
    ]  # 10, 12?
    impact = [
        (0.2, 0.9),
        (0.1, 0.95),
        (0.2, 0.9),
        (0.2, 0.9),
        (0.1, 0.95),
        (0.1, 0.95),
        (0.1, 0.95),
        (0.1, 0.95),
    ] * u.m
    ringwidth = [
        (0.04, 0.08),
        (0.02, 0.1),
        (0.01, 0.1),
        (0.02, 0.1),
        (0.01, 0.5),
        (0.02, 0.2),
        (0.02, 0.2),
        (0.02, 0.2),
    ] * u.deg
    total_pix = [1855.0, 1855.0, 1764.0, 11328.0, 1296.0, 2048.0, 2368.0, 2048]
    # 8% (or 6%) as limit
    min_pix = [148.0, 148.0, 141.0, 680.0, 104.0, 164.0, 142.0, 164]
    # Need to either convert from the pixel area in m^2 or check the camera specs
    ang_pixel_width = [0.1, 0.2, 0.18, 0.067, 0.24, 0.2, 0.17, 0.2, 0.163] * u.deg
    # Found from TDRs (or the pixel area)
    hole_rad = [
        0.308 * u.m,
        0.244 * u.m,
        0.244 * u.m,
        4.3866 * u.m,
        0.160 * u.m,
        0.130 * u.m,
        0.171 * u.m,
        0.171 * u.m,
    ]  # Assuming approximately spherical hole
    cam_rad = [2.26, 3.96, 3.87, 4.0, 4.45, 2.86, 5.25, 2.86] * u.deg
    # Above found from the field of view calculation
    sec_rad = [
        0.0 * u.m,
        0.0 * u.m,
        0.0 * u.m,
        2.7 * u.m,
        0.0 * u.m,
        1.0 * u.m,
        1.8 * u.m,
        1.8 * u.m,
    ]
    sct = [False, False, False, True, False, True, True, True]
    # Added cleaning here. All these options should go to an input card
    cleaning = True

    muon_cuts = {
        "Name": names,
        "tail_cuts": tail_cuts,
        "Impact": impact,
        "RingWidth": ringwidth,
        "total_pix": total_pix,
        "min_pix": min_pix,
        "CamRad": cam_rad,
        "SecRad": sec_rad,
        "SCT": sct,
        "AngPixW": ang_pixel_width,
        "HoleRad": hole_rad,
    }
    logger.debug(muon_cuts)

    muonringlist = []  # [None] * len(event.dl0.tels_with_data)
    muonintensitylist = []  # [None] * len(event.dl0.tels_with_data)
    tellist = []
    muon_event_param = {
        "TelIds": tellist,
        "MuonRingParams": muonringlist,
        "MuonIntensityParams": muonintensitylist,
    }

    for telid in event.dl0.tels_with_data:

        logger.debug("Analysing muon event for tel %d", telid)
        image = event.dl1.tel[telid].image

        # Get geometry
        teldes = event.inst.subarray.tel[telid]
        geom = teldes.camera.geometry
        x, y = geom.pix_x, geom.pix_y

        dict_index = muon_cuts["Name"].index(str(teldes))
        logger.debug("found an index of %d for camera %d", dict_index, geom.camera_name)

        tailcuts = muon_cuts["tail_cuts"][dict_index]
        logger.debug("Tailcuts are %s", tailcuts)

        clean_mask = tailcuts_clean(
            geom, image, picture_thresh=tailcuts[0], boundary_thresh=tailcuts[1]
        )

        # TODO: correct this hack for values over 90
        altval = event.mcheader.run_array_direction[1]
        if altval > Angle(90, unit=u.deg):
            warnings.warn("Altitude over 90 degrees")
            altval = Angle(90, unit=u.deg)

        telescope_pointing = SkyCoord(
            alt=altval, az=event.mcheader.run_array_direction[0], frame=AltAz()
        )
        camera_coord = SkyCoord(
            x=x,
            y=y,
            frame=CameraFrame(
                focal_length=teldes.optics.equivalent_focal_length,
                rotation=geom.pix_rotation,
                telescope_pointing=telescope_pointing,
            ),
        )

        nom_coord = camera_coord.transform_to(NominalFrame(origin=telescope_pointing))
        x = nom_coord.delta_az.to(u.deg)
        y = nom_coord.delta_alt.to(u.deg)

        if cleaning:
            img = image * clean_mask
        else:
            img = image

        muon_ring_fit = MuonRingFitter(fit_method="kundu_chaudhuri")

        logger.debug(
            "img: %s mask: %s, x=%s y= %s", np.sum(image), np.sum(clean_mask), x, y
        )

        if img.sum() <= 0:  # Nothing left after tail cuts
            continue

        muonringparam = muon_ring_fit(x, y, image, clean_mask)

        for i in range(2):
            dist = np.sqrt(
                (x - muonringparam.ring_center_x)**2
                + (y - muonringparam.ring_center_y)**2
            )
            ring_dist = np.abs(dist - muonringparam.ring_radius)

            muonringparam = muon_ring_fit(
                x, y, img, (ring_dist < muonringparam.ring_radius * 0.4)
            )

        dist_mask = (
            np.abs(dist - muonringparam.ring_radius) < muonringparam.ring_radius * 0.4
        )
        pix_im = image * dist_mask
        nom_dist = np.sqrt(
            (muonringparam.ring_center_x)**2
            + (muonringparam.ring_center_y)**2
        )

        minpix = muon_cuts["min_pix"][dict_index]  # 0.06*numpix #or 8%

        mir_rad = np.sqrt(teldes.optics.mirror_area.to("m2") / np.pi)

        # Camera containment radius -  better than nothing - guess pixel
        # diameter of 0.11, all cameras are perfectly circular   cam_rad =
        # np.sqrt(numpix*0.11/(2.*np.pi))

        if (
            npix_above_threshold(pix_im, tailcuts[0]) > 0.1 * minpix
            and npix_composing_ring(pix_im) > minpix
            and nom_dist < muon_cuts["CamRad"][dict_index]
            and muonringparam.ring_radius < 1.5 * u.deg
            and muonringparam.ring_radius > 1.0 * u.deg
        ):
            muonringparam.ring_containment = ring_containment(
                muonringparam.ring_radius,
                muon_cuts["CamRad"][dict_index],
                muonringparam.ring_center_x,
                muonringparam.ring_center_y,
            )

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

            if image.shape[0] == muon_cuts["total_pix"][dict_index]:
                muonintensityoutput = fit_muon(
                    muonringparam.ring_center_x,
                    muonringparam.ring_center_y,
                    muonringparam.ring_radius,
                    x[dist_mask],
                    y[dist_mask],
                    image[dist_mask],
                )

                muonintensityoutput.tel_id = telid
                muonintensityoutput.obs_id = event.dl0.obs_id
                muonintensityoutput.event_id = event.dl0.event_id
                muonintensityoutput.mask = dist_mask

                idx_ring = np.nonzero(pix_im)
                muonintensityoutput.ring_completeness = ring_completeness(
                    x[idx_ring],
                    y[idx_ring],
                    pix_im[idx_ring],
                    muonringparam.ring_radius,
                    muonringparam.ring_center_x,
                    muonringparam.ring_center_y,
                    threshold=30,
                    bins=30,
                )
                muonintensityoutput.ring_size = np.sum(pix_im)

                dist_ringwidth_mask = np.abs(dist - muonringparam.ring_radius) < (
                    muonintensityoutput.ring_width
                )
                pix_ringwidth_im = image * dist_ringwidth_mask
                idx_ringwidth = np.nonzero(pix_ringwidth_im)

                muonintensityoutput.ring_pix_completeness = npix_above_threshold(
                    pix_ringwidth_im[idx_ringwidth], tailcuts[0]
                ) / len(pix_im[idx_ringwidth])

                logger.debug(
                    "Tel %d Impact parameter = %s mir_rad=%s " "ring_width=%s",
                    telid,
                    muonintensityoutput.impact_parameter,
                    mir_rad,
                    muonintensityoutput.ring_width,
                )
                conditions = [
                    muonintensityoutput.impact_parameter * u.m
                    < muon_cuts["Impact"][dict_index][1] * mir_rad,
                    muonintensityoutput.impact_parameter
                    > muon_cuts["Impact"][dict_index][0],
                    muonintensityoutput.ring_width
                    < muon_cuts["RingWidth"][dict_index][1],
                    muonintensityoutput.ring_width
                    > muon_cuts["RingWidth"][dict_index][0],
                ]

                if all(conditions):
                    muonintensityparam = muonintensityoutput
                    idx = tellist.index(telid)
                    muonintensitylist[idx] = muonintensityparam
                    logger.debug(
                        "Muon found in tel %d,  tels in event=%d",
                        telid,
                        len(event.dl0.tels_with_data),
                    )
                else:
                    continue

    return muon_event_param
