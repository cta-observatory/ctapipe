"""
Histogram aggregation with HistogramAggregator
==============================================

This tutorial shows how to:

1. Build an event table with camera-like data (images and peak times) and some invalid values.
2. Configure and run HistogramAggregator in chunks.
3. Access histogram counts, bin edges, summary statistics, and valid-event counts (n_events).
4. Plot one pixel histogram from the selected chunks and both gain channels for both image and peak_time columns.
5. Plot the same histogram using Hist's built-in plotting functionality after filling a Hist object with the aggregated histogram counts and variances.
6. Plot the integral over all pixels for both channels using Hist's built-in plotting functionality.
7. Compare no-flow, underflow-only, overflow-only, and both-flow histograms to see how the outer bins behave.
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from astropy.time import Time
from traitlets.config import Config

from ctapipe.containers import ChunkHistogramContainer
from ctapipe.monitoring.aggregator import HistogramAggregator


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
images = rng.normal(loc=85.0, scale=10.0, size=(n_events, n_channels, n_pixels))
images[:, 1, :] -= 15  # Simulate lower gain channel by shifting the mean down by 15
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
                "name": "image",
            },
            "axis_names": ["channel", "pixel"],
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
                "name": "peak_time",
            },
            "axis_names": ["channel", "pixel"],
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

        line = ax.stairs(
            counts,
            bin_edges,
            label=f"{gain_label[channel_index]} (n_events={valid_events})",
        )
        channel_handles.append(line)
        color = line.get_edgecolor()

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

    ax.set_title(f"Chunk {chunk_index}, pixel {pixel_index}")
    ax.set_xlabel("image value")
    ax.set_ylabel("Counts")

    ax.legend(
        handles=channel_handles,
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

        line = ax.stairs(
            counts,
            bin_edges,
            label=f"{gain_label[channel_index]} (n_events={valid_events})",
        )
        channel_handles.append(line)
        color = line.get_edgecolor()

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

    ax.set_title(f"Peak Time - Chunk {chunk_index}, pixel {pixel_index}")
    ax.set_xlabel("peak_time value")
    ax.set_ylabel("Counts")
    ax.legend(
        handles=channel_handles,
        loc="upper left",
        fontsize=8,
    )

plt.show()


# -------------------------------------------------------------------
# Build a Hist object from serialized axis metadata and plot it
# -------------------------------------------------------------------

# Reconstruct the histogram axis from metadata stored by HistogramAggregator.
# In this tutorial, axis_definition uses hist.axis.Regular.
chunk_index = 0
chunk_histograms_container = ChunkHistogramContainer(
    **dict(zip(result[chunk_index].colnames, result[chunk_index]))
)
chunk_histograms_container.meta = result.meta

# Plot three nearby pixels using Hist's built-in plotting functionality.
# Requires 'hist[plot]' to be installed in the environment. Reconstruct
# the full histogram as a Hist object for the chunk using hist_from_container method.
full_hist = HistogramAggregator.hist_from_container(chunk_histograms_container)
pixels_to_plot = [pixel_index, pixel_index + 1, pixel_index + 2]
fig, axes = plt.subplots(1, len(pixels_to_plot), figsize=(15, 4), sharey=True)

for ax, pixel_to_plot in zip(axes, pixels_to_plot):
    for channel_index in range(n_channels):
        h = full_hist[{"channel": channel_index, "pixel": pixel_to_plot}]
        h.name = gain_label[channel_index]

        plt.sca(ax)
        h.plot(histtype="step", yerr=True, label=h.name)

    ax.set_title(f"Chunk {chunk_index}, Pixel {pixel_to_plot}")
    ax.set_xlabel("image value")
    ax.legend(fontsize=8, loc="upper left")

axes[0].set_ylabel("Counts")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------------------------
# Plot the integral over all pixels for both channels using Hist's built-in plotting functionality
# ------------------------------------------------------------------------------------------------
h = full_hist

fig, ax = plt.subplots(
    1,
    3,
    figsize=(20, 4),
    gridspec_kw={"width_ratios": [1.15, 1.15, 1.4]},
)
fig.suptitle(f"Chunk {chunk_index}")
h[:, 0, :].plot2d(ax=ax[0], norm="log")
h[:, 1, :].plot2d(ax=ax[1], norm="log")
channel_stack = h.integrate("pixel").stack("channel")
channel_stack[0].name = "High Gain"
channel_stack[1].name = "Low Gain"
channel_stack.plot(ax=ax[2], legend=True)

ax[0].set_title(channel_stack[0].name)
ax[1].set_title(channel_stack[1].name)
ax[2].set_title("Integral over all pixels")
for heatmap_ax in ax[:2]:
    heatmap_ax.set_xlabel("image value")
    heatmap_ax.set_ylabel("Pixel")
    heatmap_ax.set_yticks([25, 50, 75, 100], labels=["25", "50", "75", "100"])

ax[2].set_xlabel("image value")
fig.subplots_adjust(wspace=0.5, top=0.88)
plt.show()

# ----------------------------------------------------------------------
# Demonstrate underflow/overflow via HistogramAggregator axis_definition
# ----------------------------------------------------------------------

FLOW_CONFIGS = {
    "No flow bins": {"underflow": False, "overflow": False},
    "Underflow only": {"underflow": True, "overflow": False},
    "Overflow only": {"underflow": False, "overflow": True},
    "With flow bins": {"underflow": True, "overflow": True},
}

BASE_AXIS = {
    "class_name": "Regular",
    "bins": 20,
    "start": 75.0,
    "stop": 95.0,
}

CHUNKING = {
    "chunking_type": "SizeChunking",
    "SizeChunking": {"chunk_size": 1000},
}
# Run all configurations and extract histograms/counts
results = {}
histograms = {}
flow_counts = {}

for label, flow_options in FLOW_CONFIGS.items():
    config = Config(
        {
            "HistogramAggregator": {
                "chunking_type": "SizeChunking",
                "axis_definition": {
                    **BASE_AXIS,
                    **flow_options,
                },
                "axis_names": ["channel", "pixel"],
            },
            "SizeChunking": {"chunk_size": 1000},
        }
    )

    # Aggregate the histogram which will return an astropy table
    result = HistogramAggregator(config=config)(
        table=table,
        col_name="image",
        masked_elements_of_sample=masked_elements_of_sample,
    )
    # Create a Hist object from the aggregated histogram and
    # metadata for the selected chunk using the hist_from_tablerow method.
    histogram_cont = HistogramAggregator.hist_from_tablerow(result[chunk_index])
    histogram = histogram_cont[{"channel": 0, "pixel": pixel_index}]

    flow_view = histogram.view(flow=True)
    axis_kwargs = result.meta["axis_kwargs"]

    results[label] = result
    histograms[label] = histogram
    flow_counts[label] = {
        "underflow": (int(flow_view[0]) if axis_kwargs.get("underflow") else 0),
        "in_range": int(histogram.values().sum()),
        "overflow": (int(flow_view[-1]) if axis_kwargs.get("overflow") else 0),
    }

valid_events = results["With flow bins"][chunk_index]["n_events"][0, pixel_index]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

styles = {
    "No flow bins": "-",
    "Underflow only": "--",
    "Overflow only": "-.",
    "With flow bins": ":",
}

# Left: histogram comparison
for label, histogram in histograms.items():
    axes[0].stairs(
        histogram.values(),
        histogram.axes[0].edges,
        linestyle=styles[label],
        label=label,
    )

axes[0].set_title(
    f"HistogramAggregator output: chunk {chunk_index}, pixel {pixel_index}"
)
axes[0].set_xlabel("image value")
axes[0].set_ylabel("Counts")

axis_margin = 0.05 * (BASE_AXIS["stop"] - BASE_AXIS["start"])
axes[0].set_xlim(
    BASE_AXIS["start"] - axis_margin,
    BASE_AXIS["stop"] + axis_margin,
)

axes[0].legend(fontsize=8, loc="upper left")

# Right: flow-bin behavior
x = np.arange(3)
labels = ["underflow", "in-range", "overflow"]

bar_offsets = {
    "No flow bins": -0.18,
    "Underflow only": 0.00,
    "Overflow only": 0.18,
    "With flow bins": 0.36,
}

bar_width = 0.18

for label, offset in bar_offsets.items():
    counts = flow_counts[label]

    axes[1].bar(
        x + offset,
        [
            counts["underflow"],
            counts["in_range"],
            counts["overflow"],
        ],
        width=bar_width,
        label=label,
    )

axes[1].set_xticks(x + 0.09, labels)
axes[1].set_ylabel("Event count")
axes[1].set_title("Under/overflow behavior")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.show()
