"""
PSF model usage in ctapipe
==========================

"""

import astropy.units as u
import numpy as np
from itertools import product
from ctapipe.instrument.optics import ComaPSFModel, ZernikePSFModel
from ctapipe.instrument import SubarrayDescription
import matplotlib.pyplot as plt

# %matplotlib inline


# make plots and fonts larger
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 16


######################################################################
# Set up the PSF model
# ~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# This sets up the PSF models describing PSF effect for the LSTs.
# The parameters for Coma PSF model are taken from
# :cite:p:`startracker`, which was original given in polar coordinates
# in the camera frame. We here manually convert the parameters using
# the plate scale of LSTs to get the parameters in the TelescopeFrame.
#

subarray = SubarrayDescription.read("dataset://gamma_prod5.simtel.zst")

lst_plate_scale_deg = np.rad2deg(
    1.0 / subarray.tel[1].optics.effective_focal_length.to_value(u.m)
)

lst1 = subarray.select_subarray([1])

coma_psf_model = ComaPSFModel(
    subarray=lst1,
    asymmetry_max=0.49244797,
    asymmetry_decay_rate=9.23573115 / lst_plate_scale_deg,
    asymmetry_linear_term=0.15216096 / lst_plate_scale_deg,
    radial_scale_offset=0.01409259 * lst_plate_scale_deg,
    radial_scale_linear=0.02947208,
    radial_scale_quadratic=0.06000271 / lst_plate_scale_deg**1,
    radial_scale_cubic=-0.02969355 / lst_plate_scale_deg**2,
    polar_scale_amplitude=0.24271557,
    polar_scale_decay=7.5511501 / lst_plate_scale_deg,
    polar_scale_offset=0.02037972 * lst_plate_scale_deg,
)

zernike_psf_model = ZernikePSFModel(
    subarray=lst1,
    pupil_size=512,
    psf_reference=[("type", "*", 0.5 * u.deg)],
    pupil_diameter_fraction=0.12,
    pupil_edge_softness=0.08,
    focal_plane_smoothing_sigma_pix=3.0,
    wavelength_min=320e-9 * u.m,
    wavelength_max=550e-9 * u.m,
    wavelength_samples=17,
    cherenkov_spectrum_index=2.0,
    z4=[("type", "*", 1.013e-07 * u.m)],
    z5=[("type", "*", 0.0 * u.m)],
    z6=[("type", "*", 0.0 * u.m)],
    z11=[("type", "*", 3.648e-08 * u.m)],
    z7_theta=[("type", "*", 2.332e-08 * u.m / u.deg)],
    z8_theta=[("type", "*", 1.919e-07 * u.m / u.deg)],
    z5_theta2=[("type", "*", 7.913e-08 * u.m / u.deg**2)],
    z6_theta2=[("type", "*", 2.397e-08 * u.m / u.deg**2)],
)

######################################################################
# calculate PSF at different positions in the field of view
#


lon0 = 1 * lst_plate_scale_deg * u.deg
lat0 = 1 * lst_plate_scale_deg * u.deg

edges_x = (
    np.linspace(-0.1 * lst_plate_scale_deg, 0.1 * lst_plate_scale_deg, 250) * u.deg
)
edges_y = (
    np.linspace(-0.1 * lst_plate_scale_deg, 0.1 * lst_plate_scale_deg, 250) * u.deg
)
centers_x = 0.5 * (edges_x[:-1] + edges_x[1:])
centers_y = 0.5 * (edges_y[:-1] + edges_y[1:])
x, y = np.meshgrid(centers_x, centers_y)

psf_center_coma = coma_psf_model.pdf(
    tel_id=1, lon=x, lat=y, lon0=0.0 * u.deg, lat0=0.0 * u.deg
)
psf_center_zernike = zernike_psf_model.pdf(
    tel_id=1, lon=x, lat=y, lon0=0.0 * u.deg, lat0=0.0 * u.deg
)
psf_border_coma = coma_psf_model.pdf(
    tel_id=1,
    lon=x + 1 * lst_plate_scale_deg * u.deg,
    lat=y + 1 * lst_plate_scale_deg * u.deg,
    lon0=lon0,
    lat0=lat0,
)
psf_border_zernike = zernike_psf_model.pdf(
    tel_id=1,
    lon=x + 1 * lst_plate_scale_deg * u.deg,
    lat=y + 1 * lst_plate_scale_deg * u.deg,
    lon0=lon0,
    lat0=lat0,
)


######################################################################
# Plot the results
# ----------------
#

fig, axes = plt.subplots(
    2,
    2,
    layout="constrained",
    figsize=(10, 8),
)

plots = [
    (axes[0, 0], psf_center_coma, "Coma PSF at (0°, 0°)"),
    (axes[0, 1], psf_center_zernike, "Zernike PSF at (0°, 0°)"),
    (
        axes[1, 0],
        psf_border_coma,
        f"Coma PSF at ({lon0.to_value(u.deg):.2f}°, {lat0.to_value(u.deg):.2f}°)",
    ),
    (
        axes[1, 1],
        psf_border_zernike,
        f"Zernike PSF at ({lon0.to_value(u.deg):.2f}°, {lat0.to_value(u.deg):.2f}°)",
    ),
]

for ax, psf, title in plots:
    plot_vmax = max(float(np.percentile(psf, 99.5)), 1e-12)
    ax.pcolormesh(
        edges_x.to_value(u.deg),
        edges_y.to_value(u.deg),
        psf,
        cmap="inferno",
        vmin=0.0,
        vmax=plot_vmax,
        shading="auto",
    )
    ax.set(
        aspect=1,
        title=title,
        xlabel="lon [deg]",
        ylabel="lat [deg]",
    )

plt.show()


######################################################################
# Stacked PSF over multiple source positions
# -----------------------------------------------------
#

lons = np.linspace(-1.0, 1.0, 11) * lst_plate_scale_deg * u.deg
lats = np.linspace(-1.0, 1.0, 11) * lst_plate_scale_deg * u.deg

edges_x_stack = np.linspace(-2.5 * u.deg, 2.5 * u.deg, 500)
edges_y_stack = np.linspace(-2.5 * u.deg, 2.5 * u.deg, 500)
centers_x_stack = 0.5 * (edges_x_stack[:-1] + edges_x_stack[1:])
centers_y_stack = 0.5 * (edges_y_stack[:-1] + edges_y_stack[1:])
x_stack, y_stack = np.meshgrid(centers_x_stack, centers_y_stack)

psf_stacked_coma = np.zeros(x_stack.shape)
psf_stacked_zernike = np.zeros(x_stack.shape)
for source_lon, source_lat in product(lons, lats):
    psf_stacked_coma += coma_psf_model.pdf(
        tel_id=1,
        lon=x_stack,
        lat=y_stack,
        lon0=source_lon,
        lat0=source_lat,
    )
    psf_stacked_zernike += zernike_psf_model.pdf(
        tel_id=1,
        lon=x_stack,
        lat=y_stack,
        lon0=source_lon,
        lat0=source_lat,
    )

fig_stack, axes = plt.subplots(
    1,
    2,
    layout="constrained",
    figsize=(10, 5),
)

axes[0].pcolormesh(
    edges_x_stack.to_value(u.deg),
    edges_y_stack.to_value(u.deg),
    psf_stacked_coma,
    cmap="inferno",
    shading="auto",
    vmin=0.0,
    vmax=np.percentile(psf_stacked_coma, 99.5),
)
axes[0].set(
    aspect=1,
    title="Stacked Coma PSF",
)

axes[1].pcolormesh(
    edges_x_stack.to_value(u.deg),
    edges_y_stack.to_value(u.deg),
    psf_stacked_zernike,
    cmap="inferno",
    shading="auto",
    vmin=0.0,
    vmax=np.percentile(psf_stacked_zernike, 99.5),
)
axes[1].set(
    aspect=1,
    title="Stacked Zernike PSF",
)

plt.show()
