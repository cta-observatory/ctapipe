#!/usr/bin/env python3
"""
Example of drawing a Camera using a toymodel shower image.
"""

import astropy.units as u
import matplotlib.pylab as plt

from ctapipe.image import hillas_parameters, tailcuts_clean, toymodel
from ctapipe.instrument import SubarrayDescription
from ctapipe.visualization import CameraDisplay

if __name__ == "__main__":

    # Load the camera
    subarray = SubarrayDescription.read("dataset://gamma_prod5.simtel.zst")
    geom = subarray.tel[1].camera.geometry
    disp = CameraDisplay(geom)
    disp.add_colorbar()

    # Create a fake camera image to display:
    model = toymodel.Gaussian(
        x=0.2 * u.m, y=0.0 * u.m, width=0.05 * u.m, length=0.15 * u.m, psi="35d"
    )

    image, sig, bg = model.generate_image(geom, intensity=1500, nsb_level_pe=2)

    # Apply image cleaning
    cleanmask = tailcuts_clean(geom, image, picture_thresh=10, boundary_thresh=5)
    clean = image.copy()
    clean[~cleanmask] = 0.0

    # Calculate image parameters
    hillas = hillas_parameters(geom, clean)
    print(hillas)

    # Show the camera image and overlay Hillas ellipse and clean pixels
    disp.image = image
    disp.cmap = "inferno"
    disp.highlight_pixels(cleanmask, color="crimson")
    disp.overlay_moments(hillas, color="cyan", linewidth=1)

    plt.show()
