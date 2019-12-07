"""
Set of diagnostic plots relating to muons
For generic use with all muon algorithms
"""

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.table import Table
from matplotlib import colors
from scipy.stats import norm

from astropy.coordinates import SkyCoord, AltAz
from ctapipe.coordinates import CameraFrame, NominalFrame
from ctapipe.image.cleaning import tailcuts_clean
from ctapipe.plotting.camera import CameraPlotter
from ctapipe.utils.fitshistogram import Histogram

import logging

logger = logging.getLogger(__name__)


def plot_muon_efficiency(outputpath):
    """
    Plot the muon efficiencies
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    figip, axip = plt.subplots(1, 1, figsize=(10, 10))
    figrw, axrw = plt.subplots(1, 1, figsize=(10, 10))

    nbins = 16
    t = Table.read(str(outputpath) + '_muontable.fits')
    logger.info('Reading muon efficiency from table "%s"', outputpath)

    if len(t['MuonEff']) < 1:
        logger.warning("No muon events to plot")
        return
    else:
        logger.info("Found %d muon events", len(t['MuonEff']))

    (mu, sigma) = norm.fit(t['MuonEff'])

    logger.debug('Gaussian fit with mu=%f, sigma=%f', mu, sigma)

    conteff = ax.hist(t['MuonEff'], nbins)
    ax.set_xlim(0.2 * min(t['MuonEff']), 1.2 * max(t['MuonEff']))

    xtest = np.linspace(min(t['MuonEff']), max(t['MuonEff']), nbins)
    yg = mlab.normpdf(xtest, mu, sigma)
    logger.debug('mu=%f sigma=%f yg=%f', mu, sigma, yg)
    ax.plot(xtest, yg, 'r', linewidth=2)

    ax.set_ylim(0., 1.2 * max(conteff[0]))
    ax.set_xlabel('Muon Efficiency')
    plt.draw()

    contimp = axip.hist(t['ImpactP'], nbins)
    axip.set_xlim(0.2 * min(t['ImpactP']), 1.2 * max(t['ImpactP']))
    axip.set_ylim(0., 1.2 * max(contimp[0]))
    axip.set_xlabel('Impact Parameter (m)')

    plt.draw()

    heffimp = Histogram(nbins=[16, 16],
                        ranges=[(min(t['MuonEff']), max(t['MuonEff'])),
                                (min(t['ImpactP']), max(t[
                                    'ImpactP']))])  #
    #  ,axisNames=["MuonEfficiency","ImpactParameter"])

    heffimp.fill([t['MuonEff'], t['ImpactP']])
    heffimp.draw_2d()

    contrw = axrw.hist(t['RingWidth'], nbins)
    axrw.set_xlim(0.2 * min(t['RingWidth']), 1.2 * max(t['RingWidth']))
    axrw.set_ylim(0., 1.2 * max(contrw[0]))
    axrw.set_xlabel(r'Ring Width ($^\circ$)')

    plt.draw()

    if outputpath is not None:
        logger.info("saving figure to '%s'", outputpath)
        fig.savefig(str(outputpath) + '_MuonEff.png')
        figip.savefig(str(outputpath) + '_ImpactParameter.png')
        figrw.savefig(str(outputpath) + '_RingWidth.png')
    else:
        logger.info("Not saving figure, no outputpath")
        plt.show()


def plot_muon_event(event, muonparams):
    if muonparams['MuonRingParams'] is not None:

        # Plot the muon event and overlay muon parameters
        fig = plt.figure(figsize=(16, 7))

        colorbar = None
        colorbar2 = None

        subarray = event.inst.subarray

        # for tel_id in event.dl0.tels_with_data:
        for tel_id in muonparams['TelIds']:
            idx = muonparams['TelIds'].index(tel_id)

            if not muonparams['MuonRingParams'][idx]:
                continue

            # otherwise...
            npads = 2
            # Only create two pads if there is timing information extracted
            # from the calibration
            ax1 = fig.add_subplot(1, npads, 1)
            plotter = CameraPlotter(event)
            image = event.dl1.tel[tel_id].image
            geom = event.inst.subarray.tel[tel_id].camera

            tailcuts = (5., 7.)
            # Try a higher threshold for
            if geom.cam_id == 'FlashCam':
                tailcuts = (10., 12.)

            clean_mask = tailcuts_clean(geom, image,
                                        picture_thresh=tailcuts[0],
                                        boundary_thresh=tailcuts[1])

            signals = image * clean_mask

            rotr_angle = geom.pix_rotation
# The following two lines have been commented out to avoid a rotation error.
#            if geom.cam_id == 'LSTCam' or geom.cam_id == 'NectarCam':

#                rotr_angle = 0. * u.deg

            # Convert to camera frame (centre & radius)
            altaz = AltAz(alt=event.mc.alt, az=event.mc.az)

            ring_nominal = SkyCoord(
                delta_az=muonparams['MuonRingParams'][idx].ring_center_x,
                delta_alt=muonparams['MuonRingParams'][idx].ring_center_y,
                frame=NominalFrame(origin=altaz)
            )

            flen = subarray.tel[tel_id].optics.equivalent_focal_length
            ring_camcoord = ring_nominal.transform_to(CameraFrame(
                pointing_direction=altaz,
                focal_length=flen,
                rotation=rotr_angle))

            centroid = (ring_camcoord.x.value, ring_camcoord.y.value)

            radius = muonparams['MuonRingParams'][idx].ring_radius
            ringrad_camcoord = 2 * radius.to(u.rad) * flen  # But not FC?

            px = subarray.tel[tel_id].camera.pix_x
            py = subarray.tel[tel_id].camera.pix_y
            camera_coord = SkyCoord(
                x=px,
                y=py,
                frame=CameraFrame(
                    focal_length=flen,
                    rotation=geom.pix_rotation,
                )
            )

            nom_coord = camera_coord.transform_to(
                NominalFrame(origin=altaz)
            )

            px = nom_coord.delta_az.to(u.deg)
            py = nom_coord.delta_alt.to(u.deg)
            dist = np.sqrt(np.power(px - muonparams['MuonRingParams'][idx].ring_center_x,
                                    2) + np.power(py - muonparams['MuonRingParams'][idx].
                                                  ring_center_y, 2))
            ring_dist = np.abs(dist - muonparams['MuonRingParams'][idx].ring_radius)
            pix_rmask = ring_dist < muonparams['MuonRingParams'][idx].ring_radius * 0.4

            if muonparams['MuonIntensityParams'][idx] is not None:
                signals *= muonparams['MuonIntensityParams'][idx].mask
            elif muonparams['MuonIntensityParams'][idx] is None:
                signals *= pix_rmask

            camera1 = plotter.draw_camera(tel_id, signals, ax1)

            cmaxmin = (max(signals) - min(signals))
            cmin = min(signals)
            if not cmin:
                cmin = 1.
            if not cmaxmin:
                cmaxmin = 1.

            cmap_charge = colors.LinearSegmentedColormap.from_list(
                'cmap_c', [(0 / cmaxmin, 'darkblue'),
                           (np.abs(cmin) / cmaxmin, 'black'),
                           (2.0 * np.abs(cmin) / cmaxmin, 'blue'),
                           (2.5 * np.abs(cmin) / cmaxmin, 'green'),
                           (1, 'yellow')]
            )
            camera1.pixels.set_cmap(cmap_charge)
            if not colorbar:
                camera1.add_colorbar(ax=ax1, label=" [photo-electrons]")
                colorbar = camera1.colorbar
            else:
                camera1.colorbar = colorbar
            camera1.update(True)

            camera1.add_ellipse(centroid, ringrad_camcoord.value,
                                ringrad_camcoord.value, 0., 0., color="red")

            if muonparams['MuonIntensityParams'][idx] is not None:

                ringwidthfrac = muonparams['MuonIntensityParams'][idx].ring_width / \
                    muonparams['MuonRingParams'][idx].ring_radius
                ringrad_inner = ringrad_camcoord * (1. - ringwidthfrac)
                ringrad_outer = ringrad_camcoord * (1. + ringwidthfrac)
                camera1.add_ellipse(centroid, ringrad_inner.value,
                                    ringrad_inner.value, 0., 0.,
                                    color="magenta")
                camera1.add_ellipse(centroid, ringrad_outer.value,
                                    ringrad_outer.value, 0., 0.,
                                    color="magenta")
                npads = 2
                ax2 = fig.add_subplot(1, npads, npads)
                pred = muonparams['MuonIntensityParams'][idx].prediction

                if len(pred) != np.sum(
                        muonparams['MuonIntensityParams'][idx].mask):
                    logger.warning("Lengths do not match...len(pred)=%s len("
                                   "mask)=", len(pred),
                                   np.sum(muonparams['MuonIntensityParams'][idx].mask))

                # Numpy broadcasting - fill in the shape
                plotpred = np.zeros(image.shape)
                truelocs = np.where(muonparams['MuonIntensityParams'][idx].mask == True)
                plotpred[truelocs] = pred

                camera2 = plotter.draw_camera(tel_id, plotpred, ax2)

                if np.isnan(max(plotpred)) or np.isnan(min(plotpred)):
                    logger.debug("nan prediction, skipping...")
                    continue

                c2maxmin = (max(plotpred) - min(plotpred))
                if not c2maxmin:
                    c2maxmin = 1.

                c2map_charge = colors.LinearSegmentedColormap.from_list(
                    'c2map_c', [(0 / c2maxmin, 'darkblue'),
                                (np.abs(min(plotpred)) / c2maxmin, 'black'),
                                (
                                2.0 * np.abs(min(plotpred)) / c2maxmin, 'blue'),
                                (2.5 * np.abs(min(plotpred)) / c2maxmin,
                                 'green'),
                                (1, 'yellow')]
                )
                camera2.pixels.set_cmap(c2map_charge)
                if not colorbar2:
                    camera2.add_colorbar(ax=ax2, label=" [photo-electrons]")
                    colorbar2 = camera2.colorbar
                else:
                    camera2.colorbar = colorbar2
                camera2.update(True)
                plt.pause(1.)  # make shorter

            # plt.pause(0.1)
            # if pp is not None:
            #    pp.savefig(fig)
            # fig.savefig(str(args.output_path) + "_" +
            #            str(event.dl0.event_id) + '.png')


            plt.close()
