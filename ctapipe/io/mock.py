# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Create a mock event stream of array events
"""
import logging

from .containers import RawData, RawCameraData
from ctapipe.core import Container
from ctapipe.reco import mock
import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)


def mock_event_source(geoms, events=100, single_tel=False, n_channels=1, n_samples=25,  p_trigger=0.3):
    """
    An event source that produces array
    Parameters
    ----------
    geoms : list of CameraGeometry instances
        Geometries for the telescopes to simulate
    events : int, default: 100
        maximum number of events to create
    n_channels : int
        how many channels per telescope
    n_samples : int
        how many adc samples per pixel
    p_trigger : float
        mean trigger probability for the telescopes
    """
    n_telescopes = len(geoms)
    container = EventContainer()
    container.meta.add_item('mock__max_events', events)
    container.meta.source = "mock"
    tel_ids = np.arange(n_telescopes)

    for event_id in range(events):

        n_triggered = np.random.poisson(n_telescopes * 0.3)
        if n_triggered > n_telescopes:
            n_triggered = n_telescopes

        triggered_tels = np.random.choice(tel_ids, n_triggered, replace=False)

        container.dl0.event_id = event_id
        container.dl0.tels_with_data = triggered_tels
        container.count = event_id

        # handle single-telescope case (ignore others:
        if single_tel:
            if single_tel not in container.dl0.tels_with_data:
                continue
            container.dl0.tels_with_data = [single_tel, ]

        container.dl0.tel = dict()  # clear the previous telescopes
        t = np.arange(n_samples)

        for tel_id in container.dl0.tels_with_data:
            geom = geoms[tel_id]

            # fill pixel position dictionary, if not already done:
            if tel_id not in container.meta.pixel_pos:
                container.meta.pixel_pos[tel_id] = (
                    geom.pix_x.value,
                    geom.pix_y.value,
                )

            centroid = np.random.uniform(geom.pix_x.min(), geom.pix_y.max(), 2)
            length = np.random.uniform(0.02, 0.2)
            width = np.random.uniform(0.01, length)
            psi = np.random.randint(0, 360)
            intensity = np.random.poisson(int(10000 * width * length))
            model = mock.generate_2d_shower_model(
                centroid,
                width,
                length,
                '{}d'.format(psi)
            )
            image, _, _ = mock.make_mock_shower_image(
                geom,
                model.pdf,
                intensity,
            )

            container.dl0.tel[tel_id] = RawCameraData(tel_id)
            container.dl0.tel[tel_id].num_channels = n_channels
            n_pix = len(geom.pix_id)
            samples = np.empty((n_pix, n_samples))
            means = np.random.normal(15, 1, (n_pix, 1))
            stds = np.random.uniform(3, 6, (n_pix, 1))
            samples = image[:, np.newaxis] * norm.pdf(t, means, stds)

            for chan in range(n_channels):
                container.dl0.tel[tel_id].adc_samples[chan] = samples
                container.dl0.tel[tel_id].adc_sums[chan] = image

        yield container
