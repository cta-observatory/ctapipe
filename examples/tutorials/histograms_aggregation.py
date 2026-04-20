"""
Histogram aggregation with HistogramsAggregator
===============================================

This tutorial shows how to:

1. Build an event table with camera-like data (images and peak times) and some invalid values.
2. Configure and run HistogramsAggregator in chunks.
3. Access counts, bin edges, and valid-event counts (n_events).
4. Plot one pixel histogram from the selected chunks and both gain channels for both image and peak_time columns.
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from astropy.time import Time
from traitlets.config import Config

from ctapipe.monitoring.aggregator import HistogramsAggregator


# -------------------------------------------------------------------
# Create synthetic event-wise camera data
# -------------------------------------------------------------------
rng = np.random.default_rng(42)

n_events = 2000
n_channels = 2
n_pixels = 100

times = Time(
    np.linspace(60117.911, 60117.9258, num=n_events),
    scale="tai",
    format="mjd",
)
event_ids = np.arange(n_events)
images = rng.normal(loc=77.0, scale=10.0, size=(n_events, n_channels, n_pixels))
peak_time = rng.normal(loc=20.0, scale=5.0, size=(n_events, n_channels, n_pixels))

# Add a few invalid values to demonstrate n_events behavior.
images[3, 0, 10] = np.nan
images[15, 0, 10] = np.nan
peak_time[5, 0, 10] = np.nan
peak_time[35, 1, 10] = np.nan

# Optional static mask over sample dimensions (channel, pixel).
# Here we exclude channel 1, pixel 99 for all events.
masked_elements_of_sample = np.zeros((n_channels, n_pixels), dtype=bool)
masked_elements_of_sample[1, 99] = True

table = Table(
    [times, event_ids, images, peak_time],
    names=("time", "event_id", "image", "peak_time"),
)


# -------------------------------------------------------------------
# Configure and run the histogram aggregator
# -------------------------------------------------------------------
config = Config(
    {
        "HistogramsAggregator": {"chunking_type": "SizeChunking"},
        "SizeChunking": {"chunk_size": 1000},
    }
)

aggregator = HistogramsAggregator(config=config)
n_bins = 50
result = aggregator(
    table=table,
    col_name="image",
    masked_elements_of_sample=masked_elements_of_sample,
    bins=n_bins,
)

result_peak_time = aggregator(
    table=table,
    col_name="peak_time",
    masked_elements_of_sample=masked_elements_of_sample,
    bins=n_bins,
)

print(f"Number of chunks: {len(result)}")
print(f"counts shape per chunk: {result[0]['counts'].shape}")
print(f"bins shape per chunk: {result[0]['bins'].shape}")
print(f"n_events shape per chunk: {result[0]['n_events'].shape}")


# -------------------------------------------------------------------
# Inspect one pixel histogram in both chunks and both gain channels
# -------------------------------------------------------------------
pixel_index = 10
gain_label = {0: "High Gain", 1: "Low Gain"}

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for chunk_index, ax in enumerate(axes):
    bins = result[chunk_index]["bins"]

    for channel_index in range(n_channels):
        counts = result[chunk_index]["counts"][:, channel_index, pixel_index]
        valid_events = result[chunk_index]["n_events"][channel_index, pixel_index]
        ax.step(
            bins[:-1],
            counts,
            where="post",
            label=f"{gain_label[channel_index]} (n_events={valid_events})",
        )

    ax.set_title(f"Chunk {chunk_index}, pixel {pixel_index}")
    ax.set_xlabel("image value")
    ax.set_ylabel("Counts")
    ax.legend(loc="upper right", fontsize=8)

plt.show()


# -------------------------------------------------------------------
# Plot peak_time histograms in a separate figure
# -------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for chunk_index, ax in enumerate(axes):
    bins = result_peak_time[chunk_index]["bins"]

    for channel_index in range(n_channels):
        counts = result_peak_time[chunk_index]["counts"][:, channel_index, pixel_index]
        valid_events = result_peak_time[chunk_index]["n_events"][
            channel_index, pixel_index
        ]
        ax.step(
            bins[:-1],
            counts,
            where="post",
            label=f"{gain_label[channel_index]} (n_events={valid_events})",
        )

    ax.set_title(f"Peak Time - Chunk {chunk_index}, pixel {pixel_index}")
    ax.set_xlabel("peak_time value")
    ax.set_ylabel("Counts")
    ax.legend(loc="upper right", fontsize=8)

plt.show()
