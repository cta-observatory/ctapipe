"""
Histogram aggregation with HistogramAggregator
==============================================

This tutorial shows how to:

1. Build an event table with camera-like data (images and peak times) and some invalid values.
2. Configure and run HistogramAggregator in chunks.
3. Access histogram counts, bin edges, summary statistics, and valid-event counts (n_events).
4. Plot one pixel histogram from the selected chunks and both gain channels for both image and peak_time columns.
5. Overlay mean, median, and std on top of the histogram curves.
6. Plot the same histogram using Hist's built-in plotting functionality after filling a Hist object with the aggregated histogram counts and variances.
"""

import matplotlib.pyplot as plt
import numpy as np
import hist
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from astropy.table import Table
from astropy.time import Time
from traitlets.config import Config

from ctapipe.monitoring.aggregator import HistogramAggregator
from hist import Hist


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
peak_time = rng.normal(loc=20.0, scale=2.0, size=(n_events, n_channels, n_pixels))

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
# Configure and run histogram aggregation
# -------------------------------------------------------------------
config_image = Config(
    {
        "HistogramAggregator": {
            "chunking_type": "SizeChunking",
            "axis_definition": {
                "class_name": "Regular",
                "bins": 50,
                "start": 40.0,
                "stop": 110.0,
            },
        },
        "SizeChunking": {"chunk_size": 1000},
    }
)

aggregator_image = HistogramAggregator(config=config_image)
result = aggregator_image(
    table=table,
    col_name="image",
    masked_elements_of_sample=masked_elements_of_sample,
)

config_peak_time = Config(
    {
        "HistogramAggregator": {
            "chunking_type": "SizeChunking",
            "axis_definition": {
                "class_name": "Regular",
                "bins": 50,
                "start": 2.0,
                "stop": 38.0,
            },
        },
        "SizeChunking": {"chunk_size": 1000},
    }
)

aggregator_peak_time = HistogramAggregator(config=config_peak_time)
result_peak_time = aggregator_peak_time(
    table=table,
    col_name="peak_time",
    masked_elements_of_sample=masked_elements_of_sample,
)

print(f"Number of chunks: {len(result)}")
print(f"histogram shape per chunk: {result[0]['histogram'].shape}")
print(f"bin edges shape per chunk: {result[0].meta['bin_edges'].shape}")
print(f"bin centers shape per chunk: {result[0].meta['bin_centers'].shape}")
print(f"n_events shape per chunk: {result[0]['n_events'].shape}")


# -------------------------------------------------------------------
# Plot the histograms for one pixel with two gain channels
# -------------------------------------------------------------------
# We aggreagted the histograms in two chunks of 1000 events each, so we have two histograms per gain channel
# for the selected pixel. We will plot both chunks for the selected pixel and gain channels
# on the same axes for comparison, and then do the same for the peak_time column in a separate figure.

pixel_index = 10
gain_label = {0: "High Gain", 1: "Low Gain"}

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for chunk_index, ax in enumerate(axes):
    bin_edges = result[chunk_index].meta["bin_edges"]
    bin_centers = result[chunk_index].meta["bin_centers"]
    channel_handles = []

    for channel_index in range(n_channels):
        counts = result[chunk_index]["histogram"][:, channel_index, pixel_index]
        valid_events = result[chunk_index]["n_events"][channel_index, pixel_index]
        mean_val = result[chunk_index]["mean"][channel_index, pixel_index]
        median_val = result[chunk_index]["median"][channel_index, pixel_index]
        std_val = result[chunk_index]["std"][channel_index, pixel_index]

        line = ax.step(
            bin_edges[:-1],
            counts,
            where="post",
            label=f"{gain_label[channel_index]} (n_events={valid_events})",
        )[0]
        channel_handles.append(line)
        color = line.get_color()

        # Plot bin variances as error bars (use sqrt of variance for error) at bin centers
        bin_errors = np.sqrt(counts)
        ax.errorbar(
            bin_centers,
            counts,
            yerr=bin_errors,
            fmt="none",
            color=color,
            elinewidth=1.0,
            capsize=3,
            alpha=0.6,
        )

        ax.axvline(mean_val, color=color, linestyle="--", linewidth=1.2)
        ax.axvline(median_val, color=color, linestyle=":", linewidth=1.2)
        ax.axvspan(
            mean_val - std_val,
            mean_val + std_val,
            color=color,
            alpha=0.12,
        )

    ax.set_title(f"Chunk {chunk_index}, pixel {pixel_index}")
    ax.set_xlabel("image value")
    ax.set_ylabel("Counts")
    stat_handles = [
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.2, label="Mean"),
        Line2D([0], [0], color="black", linestyle=":", linewidth=1.2, label="Median"),
        Patch(facecolor="gray", alpha=0.12, label="Mean ± Std"),
    ]
    ax.legend(
        handles=channel_handles + stat_handles,
        loc="upper left",
        fontsize=8,
    )

plt.show()


# -------------------------------------------------------------------
# Plot peak_time histograms in a separate figure
# -------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for chunk_index, ax in enumerate(axes):
    bin_edges = result_peak_time[chunk_index].meta["bin_edges"]
    bin_centers = result_peak_time[chunk_index].meta["bin_centers"]

    channel_handles = []

    for channel_index in range(n_channels):
        counts = result_peak_time[chunk_index]["histogram"][
            :, channel_index, pixel_index
        ]
        valid_events = result_peak_time[chunk_index]["n_events"][
            channel_index, pixel_index
        ]
        mean_val = result_peak_time[chunk_index]["mean"][channel_index, pixel_index]
        median_val = result_peak_time[chunk_index]["median"][channel_index, pixel_index]
        std_val = result_peak_time[chunk_index]["std"][channel_index, pixel_index]

        line = ax.step(
            bin_edges[:-1],
            counts,
            where="post",
            label=f"{gain_label[channel_index]} (n_events={valid_events})",
        )[0]
        channel_handles.append(line)
        color = line.get_color()

        # Plot bin variances as error bars (use sqrt of variance for error) at bin centers
        bin_errors = np.sqrt(counts)
        ax.errorbar(
            bin_centers,
            counts,
            yerr=bin_errors,
            fmt="none",
            color=color,
            elinewidth=1.0,
            capsize=3,
            alpha=0.6,
        )

        ax.axvline(mean_val, color=color, linestyle="--", linewidth=1.2)
        ax.axvline(median_val, color=color, linestyle=":", linewidth=1.2)
        ax.axvspan(
            mean_val - std_val,
            mean_val + std_val,
            color=color,
            alpha=0.12,
        )

    ax.set_title(f"Peak Time - Chunk {chunk_index}, pixel {pixel_index}")
    ax.set_xlabel("peak_time value")
    ax.set_ylabel("Counts")
    stat_handles = [
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.2, label="Mean"),
        Line2D([0], [0], color="black", linestyle=":", linewidth=1.2, label="Median"),
        Patch(facecolor="gray", alpha=0.12, label="Mean ± Std"),
    ]
    ax.legend(
        handles=channel_handles + stat_handles,
        loc="upper left",
        fontsize=8,
    )

plt.show()


# -------------------------------------------------------------------
# Initialize hist, fill it and plot via Hist functionality
# -------------------------------------------------------------------

# Create a Hist object with the same binning as the aggregator
bin_edges = result[0].meta["bin_edges"]
h = Hist(
    hist.axis.Regular(len(bin_edges) - 1, bin_edges[0], bin_edges[-1], name="value")
)

# Get the histogram counts and variances for the selected pixel and channel
chunk_index = 0
counts = result[0]["histogram"][:, chunk_index, pixel_index]

# Set the histogram values using the view interface
h.view(flow=False)[:] = counts

# Plot the histogram with error bars using Hist's built-in plotting functionality
# Requires 'hist[plot]' to be installed in the environment.
h.plot(yerr=True)
plt.title(
    f"Chunk {chunk_index}, Pixel {pixel_index} (High Gain) histogram from Hist object"
)
plt.xlabel("image value")
plt.ylabel("Counts")
plt.show()
