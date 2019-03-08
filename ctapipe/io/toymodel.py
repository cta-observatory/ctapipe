# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Create a toymodel event stream of array events
"""
import logging

import numpy as np
from ctapipe.image import toymodel
from scipy.stats import norm
import astropy.units as u

from .containers import DataContainer

logger = logging.getLogger(__name__)


def toymodel_event_source(geoms, max_events=100, single_tel=False, n_channels=1,
                          n_samples=25, p_trigger=0.3):
    """
    An event source that produces array
    Parameters
    ----------
    geoms : list of CameraGeometry instances
        Geometries for the telescopes to simulate
    max_events : int, default: 100
        maximum number of events to create
    n_channels : int
        how many channels per telescope
    n_samples : int
        how many adc samples per pixel
    p_trigger : float
        mean trigger probability for the telescopes
    """
    n_telescopes = len(geoms)
    container = DataContainer()
    container.meta['toymodel__max_events'] = max_events
    container.meta['source'] = "toymodel"
    tel_ids = np.arange(n_telescopes)

    for event_id in range(max_events):

        n_triggered = np.random.poisson(n_telescopes * 0.3)
        if n_triggered > n_telescopes:
            n_triggered = n_telescopes

        triggered_tels = np.random.choice(tel_ids, n_triggered, replace=False)

        container.r0.event_id = event_id
        container.r0.tels_with_data = triggered_tels
        container.count = event_id

        # handle single-telescope case (ignore others:
        if single_tel:
            if single_tel not in container.r0.tels_with_data:
                continue
            container.r0.tels_with_data = [single_tel, ]

        container.r0.tel.reset()  # clear the previous telescopes
        t = np.arange(n_samples)

        for tel_id in container.r0.tels_with_data:
            geom = geoms[tel_id]

            # fill pixel position dictionary, if not already done:
            if tel_id not in container.inst.pixel_pos:
                container.inst.pixel_pos[tel_id] = (
                    geom.pix_x.value,
                    geom.pix_y.value,
                )

            x, y = np.random.uniform(geom.pix_x.min(), geom.pix_y.max(), 2)
            length = np.random.uniform(0.02, 0.2)
            width = np.random.uniform(0.01, length)
            psi = np.random.randint(0, 360)
            intensity = np.random.poisson(int(10000 * width * length))
            model = toymodel.Gaussian(
                x=x * u.m,
                y=y * u.m,
                length=length * u.m,
                width=width * u.m,
                psi=f'{psi}d',
            )
            image, _, _ = model.generate_image(
                geom,
                intensity,
            )

            # container.r0.tel[tel_id] = R0CameraContainer()
            container.inst.num_channels[tel_id] = n_channels
            n_pix = len(geom.pix_id)
            means = np.random.normal(15, 1, (n_pix, 1))
            stds = np.random.uniform(3, 6, (n_pix, 1))
            samples = image[:, np.newaxis] * norm.pdf(t, means, stds)

            for chan in range(n_channels):
                container.r0.tel[tel_id].waveform[chan] = samples
                container.r0.tel[tel_id].image[chan] = image

        yield container
