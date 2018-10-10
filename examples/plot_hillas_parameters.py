#!/usr/bin/env python3
"""
Example of drawing a Camera using a toymodel shower image.
"""

import matplotlib.pylab as plt

from ctapipe.image import toymodel, hillas_parameters, tailcuts_clean
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay


if __name__ == '__main__':

    # Load the camera
    geom = CameraGeometry.from_name("LSTCam")
    disp = CameraDisplay(geom)
    disp.add_colorbar()

    # Create a fake camera image to display:
    model = toymodel.generate_2d_shower_model(
        centroid=(0.2, 0.0), width=0.05, length=0.15, psi='35d'
    )

    image, sig, bg = toymodel.make_toymodel_shower_image(
        geom, model.pdf, intensity=1500, nsb_level_pe=3
    )

    # Apply image cleaning
    cleanmask = tailcuts_clean(
        geom, image, picture_thresh=10, boundary_thresh=5
    )

    # Calculate image parameters
    hillas = hillas_parameters(geom[cleanmask], image[cleanmask])

    # Show the camera image and overlay Hillas ellipse and clean pixels
    disp.image = image
    disp.highlight_pixels(cleanmask, color='crimson')
    disp.overlay_moments(hillas, color='cyan', linewidth=3)

    plt.show()
