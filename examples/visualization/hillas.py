"""
Event Display with Hillas Reconstruction
========================================

This script displays events with Hillas parameter reconstruction and visualization of all parameters on the camera image.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Arc
from matplotlib.lines import Line2D

from ctapipe.instrument import SubarrayDescription

# from ctapipe.coordinates import TelescopeFrame
from ctapipe.image.toymodel import Gaussian
from ctapipe.visualization import CameraDisplay
from ctapipe.image.cleaning import tailcuts_clean
from ctapipe.image.hillas import hillas_parameters, HillasParameterizationError

import astropy.units as u

######################################################################
# ## Camera Display with Hillas Parameters annotated


def display_event_with_annotated_hillas(
    image, geom, picture_thresh=10, boundary_thresh=5
):
    """
    Display an event with detailed annotations showing what each Hillas parameter represents.
    """

    # Clean the image
    mask = tailcuts_clean(
        geom, image, picture_thresh=picture_thresh, boundary_thresh=boundary_thresh
    )

    # Calculate Hillas parameters
    try:
        hillas = hillas_parameters(geom[mask], image[mask])
    except HillasParameterizationError:
        print("Could not parametrize event")
        return None

    fig, ax = plt.subplots(figsize=(14, 12))

    # Display the camera image
    display = CameraDisplay(geom, ax=ax, cmap="gray")
    display.image = image * mask
    # display.highlight_pixels(mask, color='red', linewidth=2, alpha=0.4)
    display.add_colorbar(ax=ax, label="Charge [p.e.]")

    # Define colors for different elements using a colorblind-friendly palette
    cog_color = "red"
    ellipse_color = "#D55E00"  # vermilion
    length_color = "#0072B2"  # blue
    width_color = "#009E73"  # green
    angle_color = "#CC79A7"  # pink
    radial_color = "#E69F00"  # orange

    # x = hillas.fov_lon
    # y = hillas.fov_lat
    x = hillas.x
    y = hillas.y

    # 1. Center of Gravity (x, y)
    ax.plot(
        x.value,
        y.value,
        "o",
        color=cog_color,
        markersize=15,
        markeredgewidth=3,
        markerfacecolor="none",
        label="COG (x, y)",
    )

    ax.plot(
        x.value,
        y.value,
        "o",
        color=cog_color,
        markersize=15,
        markeredgewidth=3,
        markerfacecolor="none",
        label="COG (FOV lon, lat)",
    )

    # 2. Hillas Ellipse (shows length and width)
    # Note: Ellipse angle is rotation of the width (horizontal) axis
    # Since height=length (major axis) should be along psi, we add 90 degrees
    ellipse = Ellipse(
        xy=(x.value, y.value),
        width=hillas.width.value * 2,
        height=hillas.length.value * 2,
        angle=np.degrees(hillas.psi.value) + 90,
        fill=False,
        color=ellipse_color,
        linewidth=3,
        linestyle="--",
        label="Hillas Ellipse",
    )
    ax.add_patch(ellipse)

    # 3. Length axis (major axis)
    length_end_x = x.value + hillas.length.value * np.cos(hillas.psi.value)
    length_end_y = y.value + hillas.length.value * np.sin(hillas.psi.value)
    length_start_x = x.value
    length_start_y = y.value

    ax.plot(
        [length_start_x, length_end_x],
        [length_start_y, length_end_y],
        color=length_color,
        linewidth=3,
        label="Length",
        zorder=10,
    )

    long_axis_end_x = x.value + 2 * hillas.length.value * np.cos(hillas.psi.value)
    long_axis_end_y = y.value + 2 * hillas.length.value * np.sin(hillas.psi.value)
    long_axis_start_x = x.value - 2 * hillas.length.value * np.cos(hillas.psi.value)
    long_axis_start_y = y.value - 2 * hillas.length.value * np.sin(hillas.psi.value)

    ax.plot(
        [long_axis_start_x, long_axis_end_x],
        [long_axis_start_y, long_axis_end_y],
        color=length_color,
        linewidth=3,
        label="long-axis",
        zorder=10,
        alpha=0.5,
        ls="--",
    )

    # Annotate length
    mid_length_x = x.value + 0.7 * hillas.length.value * np.cos(hillas.psi.value)
    mid_length_y = y.value + 0.7 * hillas.length.value * np.sin(hillas.psi.value)
    ax.annotate(
        f"Length\n{hillas.length:.3f}",
        xy=(mid_length_x, mid_length_y),
        xytext=(mid_length_x + 0.3, mid_length_y - 0.2),
        color=length_color,
        fontsize=16,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=length_color, lw=2),
        bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=1),
    )

    # 4. Width axis (minor axis)
    width_end_x = x.value + hillas.width.value * np.sin(hillas.psi.value)
    width_end_y = y.value - hillas.width.value * np.cos(hillas.psi.value)
    width_start_x = x.value
    width_start_y = y.value

    ax.plot(
        [width_start_x, width_end_x],
        [width_start_y, width_end_y],
        color=width_color,
        linewidth=3,
        label="Width",
        zorder=10,
    )

    # Annotate width
    mid_width_x = x.value + 0.7 * hillas.width.value * np.sin(hillas.psi.value)
    mid_width_y = y.value - 0.7 * hillas.width.value * np.cos(hillas.psi.value)
    ax.annotate(
        f"Width\n{hillas.width:.3f}",
        xy=(mid_width_x, mid_width_y),
        xytext=(mid_width_x - 0.35, mid_width_y + 0.25),
        color=width_color,
        fontsize=16,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=width_color, lw=2),
        bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=1),
    )

    # 5. Angle psi (orientation of major axis)
    # Draw the angle between the length axis and the x-axis (horizontal)
    # We'll draw this at the end of the length axis

    # Draw a horizontal reference line at the end of the length axis
    psi_ref_length = 0.25
    ref_line_x_start = length_end_x
    ref_line_x_end = length_end_x + psi_ref_length
    ref_line_y = length_end_y

    ax.plot(
        [ref_line_x_start, ref_line_x_end],
        [ref_line_y, ref_line_y],
        color=angle_color,
        linewidth=3,
        linestyle="--",
        alpha=1,
        zorder=10,
    )

    # Draw arc showing the psi angle at the end of the length axis
    arc_radius = 0.12
    # The arc should go from the horizontal (0°) to the length axis direction (psi)
    angle_to_cog = np.degrees(hillas.psi.value)

    # Normalize to 0-360
    while angle_to_cog < 0:
        angle_to_cog += 360
    while angle_to_cog >= 360:
        angle_to_cog -= 360

    # Draw arc from horizontal (0°) to the direction pointing back to COG
    arc = Arc(
        (length_end_x, length_end_y),
        2 * arc_radius,
        2 * arc_radius,
        angle=0,
        theta1=0,
        theta2=angle_to_cog,
        color=angle_color,
        linewidth=2.5,
        zorder=10,
    )
    ax.add_patch(arc)

    # Annotate psi angle - place it along the bisector of the arc
    psi_bisector = angle_to_cog / 2
    psi_label_x = length_end_x + arc_radius * 1.4 * np.cos(np.radians(psi_bisector))
    psi_label_y = length_end_y + arc_radius * 1.4 * np.sin(np.radians(psi_bisector))
    ax.annotate(
        f"ψ = {np.degrees(hillas.psi.value):.1f}°",
        xy=(psi_label_x, psi_label_y),
        color=angle_color,
        fontsize=16,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.7),
    )

    # 6. Radial distance r and angle phi from camera center
    camera_center_x = 0
    camera_center_y = 0

    # Draw line from camera center to COG
    ax.plot(
        [camera_center_x, x.value],
        [camera_center_y, y.value],
        color=radial_color,
        linewidth=2.5,
        linestyle=":",
        label="r (radial)",
        zorder=5,
    )

    # Mark camera center
    ax.plot(
        camera_center_x,
        camera_center_y,
        "x",
        color=radial_color,
        markersize=20,
        markeredgewidth=3,
    )

    # Annotate camera center
    ax.annotate(
        "Camera\nCenter",
        xy=(camera_center_x, camera_center_y),
        xytext=(-0.25, -0.15),
        color=radial_color,
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.7),
    )

    # Annotate r
    mid_r_x = x.value / 2
    mid_r_y = y.value / 2
    ax.annotate(
        f"r = {hillas.r:.3f}",
        xy=(mid_r_x, mid_r_y),
        xytext=(mid_r_x + 0.2, mid_r_y - 0),
        color=radial_color,
        fontsize=16,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=radial_color, lw=2),
        bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.7),
    )

    # Draw arc for phi angle
    phi_arc_radius = 0.1
    phi_arc = Arc(
        (camera_center_x, camera_center_y),
        2 * phi_arc_radius,
        2 * phi_arc_radius,
        angle=0,
        theta1=0,
        theta2=np.degrees(hillas.phi.value),
        color=radial_color,
        linewidth=2,
        linestyle="--",
        zorder=5,
    )
    ax.add_patch(phi_arc)

    # Annotate phi
    phi_label_angle = hillas.phi.value / 2
    phi_label_x = phi_arc_radius * 1.8 * np.cos(phi_label_angle)
    phi_label_y = phi_arc_radius * 1.8 * np.sin(phi_label_angle)
    ax.annotate(
        f"φ = {np.degrees(hillas.phi.value):.1f}°",
        xy=(phi_label_x, phi_label_y),
        color=radial_color,
        fontsize=16,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.7),
    )

    # Add reference x-axis line for angle measurement
    ax.plot(
        [-1.2, 1.2],
        [camera_center_y, camera_center_y],
        "w--",
        linewidth=2,
        alpha=0.8,
        zorder=1,
    )

    # Add legend with all Hillas parameters
    param_text = (
        f"Hillas Parameters\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Intensity: {hillas.intensity:.1f} p.e.\n"
        f"\n"
        f"x: {x:.4f}\n"
        f"y: {y:.4f}\n"
        f"r: {hillas.r:.4f}\n"
        f"φ: {np.degrees(hillas.phi.value):.2f}°\n"
        f"\n"
        f"Length: {hillas.length:.4f}\n"
        f"Width: {hillas.width:.4f}\n"
        f"ψ: {np.degrees(hillas.psi.value):.2f}°\n"
        f"\n"
        f"Skewness: {hillas.skewness:.3f}\n"
        f"Kurtosis: {hillas.kurtosis:.3f}"
    )

    ax.text(
        0.02,
        0.98,
        param_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        family="monospace",
    )

    ax.set_title("Annotated Hillas Parameters", fontsize=14, pad=20, fontweight="bold")

    # Custom legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="none",
            markeredgecolor=cog_color,
            markersize=10,
            markeredgewidth=2,
            label="COG (x, y)",
            linestyle="None",
        ),
        Line2D(
            [0],
            [0],
            color=ellipse_color,
            linewidth=2,
            linestyle="--",
            label="Hillas Ellipse",
        ),
        Line2D([0], [0], color=length_color, linewidth=2, label="Length"),
        Line2D([0], [0], color=width_color, linewidth=2, label="Width"),
        Line2D([0], [0], color=angle_color, linewidth=2, label="Angle ψ"),
        Line2D(
            [0],
            [0],
            color=radial_color,
            linewidth=2,
            linestyle=":",
            label="r, φ (radial)",
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=10, framealpha=0.9)

    plt.tight_layout()

    return fig, hillas


######################################################################
# ## Simulate and display an event

subarray = SubarrayDescription.read("dataset://gamma_prod5.simtel.zst")
geom = subarray.tel[1].camera.geometry  # .transform_to(TelescopeFrame())

# Define Gaussian model parameters
x0 = -0.2 * u.m
y0 = 0.5 * u.m
sigma_length = 0.4 * u.m
sigma_width = 0.1 * u.m
psi = 65.0 * u.deg

model = Gaussian(x0, y0, sigma_length, sigma_width, psi)
image = model.generate_image(geom, intensity=10000)[0]


CameraDisplay(geom, image=image)

fig, hillas = display_event_with_annotated_hillas(
    image, geom, picture_thresh=20, boundary_thresh=10
)


plt.savefig("hillas_annotated_event.png", dpi=300)
plt.show()

######################################################################
# ## Attributes
#
# Hillas parameters and their meaning.
# They are calculated **after image cleaning**.
#
# | Attribute | Description |
# |---|---|
# | **intensity** | total intensity (size) |
# | **skewness** | measure of the asymmetry |
# | **kurtosis** | measure of the tailedness |
# | **fov_lon** | longitude angle in a spherical system centered on the pointing position (deg) |
# | **fov_lat** | latitude angle in a spherical system centered on the pointing position (deg) |
# | **r** | radial coordinate of centroid (deg) |
# | **phi** | polar coordinate of centroid (deg) |
# | **length** | standard deviation along the major-axis (deg) |
# | **length_uncertainty** | uncertainty of length (deg) |
# | **width** | standard spread along the minor-axis (deg) |
# | **width_uncertainty** | uncertainty of width (deg) |
# | **psi** | rotation angle of ellipse (deg) |
# | **psi_uncertainty** | uncertainty of psi (deg) |
#
