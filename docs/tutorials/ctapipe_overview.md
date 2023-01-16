---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"pycharm": {}}

# Analyzing Events Using ctapipe

+++ {"pycharm": {}, "slideshow": {"slide_type": "slide"}}

<div align="center" style="font-size: 2rem">

<img heoght="300px" src="https://cta-observatory.github.io/ctapipe/_images/ctapipe_logo.png" alt="ctapipe"/>


<p style="text-align: center;">Initially presented @ LST Analysis Bootcamp</p>

<p style="text-align: center">Padova, 26.11.2018</p>

<p style="text-align: center">Maximilian Nöthe (@maxnoe) & Kai A. Brügge (@mackaiver)</p>

</div>

```{code-cell} ipython3
---
pycharm: {}
slideshow:
  slide_type: skip
---
import matplotlib.pyplot as plt
import numpy as np

%matplotlib inline
```

```{code-cell} ipython3
---
pycharm: {}
slideshow:
  slide_type: skip
---
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 14
plt.rcParams["figure.figsize"]
```

+++ {"pycharm": {}, "slideshow": {"slide_type": "slide"}}

<h1 id="tocheading">Table of Contents</h1>
<div id="toc"></div>

+++ {"pycharm": {}, "slideshow": {"slide_type": "slide"}}

## General Information

+++ {"pycharm": {}, "slideshow": {"slide_type": "slide"}}

### Design

* DL0 → DL3 analysis

* Currently some R0 → DL2 code to be able to analyze simtel files

* ctapipe is built upon the Scientific Python Stack, core dependencies are
  * numpy
  * scipy
  * astropy
  * numba

+++ {"pycharm": {}, "slideshow": {"slide_type": "slide"}}

### Developement

* ctapipe is developed as Open Source Software (BSD 3-Clause License) at <https://github.com/cta-observatory/ctapipe>

* We use the "Github-Workflow": 
  * Few people (e.g. @kosack, @maxnoe) have write access to the main repository
  * Contributors fork the main repository and work on branches
  * Pull Requests are merged after Code Review and automatic execution of the test suite

* Early developement stage ⇒ backwards-incompatible API changes might and will happen 

+++ {"pycharm": {}, "slideshow": {"slide_type": "slide"}}

### What's there?

* Reading simtel simulation files
* Simple calibration, cleaning and feature extraction functions
* Camera and Array plotting
* Coordinate frames and transformations 
* Stereo-reconstruction using line intersections
  
 

+++ {"pycharm": {}, "slideshow": {"slide_type": "slide"}}

### What's still missing?

* Good integration with machine learning techniques
* IRF calculation 
* Documentation, e.g. formal definitions of coordinate frames  

+++ {"pycharm": {}, "slideshow": {"slide_type": "slide"}}

### What can you do?

* Report issues
  * Hard to get started? Tell us where you are stuck
  * Tell user stories
  * Missing features

* Start contributing
  * ctapipe needs more workpower
  * Implement new reconstruction features

+++ {"pycharm": {}, "slideshow": {"slide_type": "slide"}}

## A simple hillas analysis

+++ {"pycharm": {}}

### Reading in simtel files

```{code-cell} ipython3
:pycharm: {}

from ctapipe.io import EventSource
from ctapipe.utils.datasets import get_dataset_path

input_url = get_dataset_path("gamma_prod5.simtel.zst")

# EventSource() automatically detects what kind of file we are giving it,
# if already supported by ctapipe
source = EventSource(input_url, max_events=5)

print(type(source))
```

```{code-cell} ipython3
:pycharm: {}

for event in source:
    print(
        "Id: {}, E = {:1.3f}, Telescopes: {}".format(
            event.count, event.simulation.shower.energy, len(event.r0.tel)
        )
    )
```

+++ {"pycharm": {}}

Each event is a `DataContainer` holding several `Field`s of data, which can be containers or just numbers.
Let's look a one event:

```{code-cell} ipython3
:pycharm: {}

event
```

```{code-cell} ipython3
:pycharm: {}

source.subarray.camera_types
```

```{code-cell} ipython3
:pycharm: {}

len(event.r0.tel), len(event.r1.tel)
```

+++ {"pycharm": {}}

### Data calibration

The `CameraCalibrator` calibrates the event (obtaining the `dl1` images).

```{code-cell} ipython3
:pycharm: {}

from ctapipe.calib import CameraCalibrator

calibrator = CameraCalibrator(subarray=source.subarray)
```

```{code-cell} ipython3
:pycharm: {}

calibrator(event)
```

+++ {"pycharm": {}}

### Event displays

Let's use ctapipe's plotting facilities to plot the telescope images

```{code-cell} ipython3
:pycharm: {}

event.dl1.tel.keys()
```

```{code-cell} ipython3
:pycharm: {}

tel_id = 130
```

```{code-cell} ipython3
:pycharm: {}

geometry = source.subarray.tel[tel_id].camera.geometry
dl1 = event.dl1.tel[tel_id]

geometry, dl1
```

```{code-cell} ipython3
:pycharm: {}

dl1.image
```

```{code-cell} ipython3
:pycharm: {}

from ctapipe.visualization import CameraDisplay

display = CameraDisplay(geometry)

# right now, there might be one image per gain channel.
# This will change as soon as
display.image = dl1.image
display.add_colorbar()
```

+++ {"pycharm": {}}

### Image Cleaning

```{code-cell} ipython3
:pycharm: {}

from ctapipe.image.cleaning import tailcuts_clean
```

```{code-cell} ipython3
:pycharm: {}

# unoptimized cleaning levels
cleaning_level = {
    "CHEC": (2, 4, 2),
    "LSTCam": (3.5, 7, 2),
    "FlashCam": (3.5, 7, 2),
    "NectarCam": (4, 8, 2),
}
```

```{code-cell} ipython3
:pycharm: {}

boundary, picture, min_neighbors = cleaning_level[geometry.name]

clean = tailcuts_clean(
    geometry,
    dl1.image,
    boundary_thresh=boundary,
    picture_thresh=picture,
    min_number_picture_neighbors=min_neighbors,
)
```

```{code-cell} ipython3
:pycharm: {}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

d1 = CameraDisplay(geometry, ax=ax1)
d2 = CameraDisplay(geometry, ax=ax2)

ax1.set_title("Image")
d1.image = dl1.image
d1.add_colorbar(ax=ax1)

ax2.set_title("Pulse Time")
d2.image = dl1.peak_time - np.average(dl1.peak_time, weights=dl1.image)
d2.cmap = "RdBu_r"
d2.add_colorbar(ax=ax2)
d2.set_limits_minmax(-20, 20)

d1.highlight_pixels(clean, color="red", linewidth=1)
```

+++ {"pycharm": {}}

### Image Parameters

```{code-cell} ipython3
:pycharm: {}

from ctapipe.image import (
    hillas_parameters,
    leakage_parameters,
    concentration_parameters,
)
from ctapipe.image import timing_parameters
from ctapipe.image import number_of_islands
from ctapipe.image import camera_to_shower_coordinates
```

```{code-cell} ipython3
:pycharm: {}

hillas = hillas_parameters(geometry[clean], dl1.image[clean])

print(hillas)
```

```{code-cell} ipython3
:pycharm: {}

display = CameraDisplay(geometry)

# set "unclean" pixels to 0
cleaned = dl1.image.copy()
cleaned[~clean] = 0.0

display.image = cleaned
display.add_colorbar()

display.overlay_moments(hillas, color="xkcd:red")
```

```{code-cell} ipython3
:pycharm: {}

timing = timing_parameters(geometry, dl1.image, dl1.peak_time, hillas, clean)

print(timing)
```

```{code-cell} ipython3
:pycharm: {}

long, trans = camera_to_shower_coordinates(
    geometry.pix_x, geometry.pix_y, hillas.x, hillas.y, hillas.psi
)

plt.plot(long[clean], dl1.peak_time[clean], "o")
plt.plot(long[clean], timing.slope * long[clean] + timing.intercept)
```

```{code-cell} ipython3
:pycharm: {}

l = leakage_parameters(geometry, dl1.image, clean)
print(l)
```

```{code-cell} ipython3
:pycharm: {}

disp = CameraDisplay(geometry)
disp.image = dl1.image
disp.highlight_pixels(geometry.get_border_pixel_mask(1), linewidth=2, color="xkcd:red")
```

```{code-cell} ipython3
:pycharm: {}

n_islands, island_id = number_of_islands(geometry, clean)

print(n_islands)
```

```{code-cell} ipython3
:pycharm: {}

conc = concentration_parameters(geometry, dl1.image, hillas)
print(conc)
```

+++ {"pycharm": {}}

### Putting it all together / Stereo reconstruction


All these steps are now unified in several components configurable through the config system, mainly:

* CameraCalibrator for DL0 → DL1 (Images)
* ImageProcessor for DL1 (Images) → DL1 (Parameters)
* ShowerProcessor for stereo reconstruction of the shower geometry
* DataWriter for writing data into HDF5

A command line tool doing these steps and writing out data in HDF5 format is available as `ctapipe-process`

```{code-cell} ipython3
:pycharm: {}

import astropy.units as u
from astropy.coordinates import SkyCoord, AltAz

from ctapipe.containers import ImageParametersContainer
from ctapipe.io import EventSource
from ctapipe.utils.datasets import get_dataset_path

from ctapipe.calib import CameraCalibrator

from ctapipe.image import ImageProcessor
from ctapipe.reco import ShowerProcessor

from ctapipe.io import DataWriter

from copy import deepcopy
import tempfile

from traitlets.config import Config


image_processor_config = Config(
    {
        "ImageProcessor": {
            "image_cleaner_type": "TailcutsImageCleaner",
            "TailcutsImageCleaner": {
                "picture_threshold_pe": [
                    ("type", "LST_LST_LSTCam", 7.5),
                    ("type", "MST_MST_FlashCam", 8),
                    ("type", "MST_MST_NectarCam", 8),
                    ("type", "SST_ASTRI_CHEC", 7),
                ],
                "boundary_threshold_pe": [
                    ("type", "LST_LST_LSTCam", 5),
                    ("type", "MST_MST_FlashCam", 4),
                    ("type", "MST_MST_NectarCam", 4),
                    ("type", "SST_ASTRI_CHEC", 4),
                ],
            },
        }
    }
)

input_url = get_dataset_path("gamma_prod5.simtel.zst")
source = EventSource(input_url)

calibrator = CameraCalibrator(subarray=source.subarray)
image_processor = ImageProcessor(
    subarray=source.subarray, config=image_processor_config
)
shower_processor = ShowerProcessor(subarray=source.subarray)
horizon_frame = AltAz()

f = tempfile.NamedTemporaryFile(suffix=".hdf5")

with DataWriter(
    source, output_path=f.name, overwrite=True, write_showers=True
) as writer:

    for event in source:
        energy = event.simulation.shower.energy
        n_telescopes_r1 = len(event.r1.tel)
        event_id = event.index.event_id
        print(f"Id: {event_id}, E = {energy:1.3f}, Telescopes (R1): {n_telescopes_r1}")

        calibrator(event)
        image_processor(event)
        shower_processor(event)

        stereo = event.dl2.stereo.geometry["HillasReconstructor"]
        if stereo.is_valid:
            print("  Alt: {:.2f}°".format(stereo.alt.deg))
            print("  Az: {:.2f}°".format(stereo.az.deg))
            print("  Hmax: {:.0f}".format(stereo.h_max))
            print("  CoreX: {:.1f}".format(stereo.core_x))
            print("  CoreY: {:.1f}".format(stereo.core_y))
            print("  Multiplicity: {:d}".format(len(stereo.telescopes)))

        # save a nice event for plotting later
        if event.count == 3:
            plotting_event = deepcopy(event)

        writer(event)
```

```{code-cell} ipython3
:pycharm: {}

from astropy.coordinates.angle_utilities import angular_separation
import pandas as pd

from ctapipe.io import TableLoader

loader = TableLoader(f.name, load_dl2=True, load_simulated=True)

events = loader.read_subarray_events()
```

```{code-cell} ipython3
:pycharm: {}

theta = angular_separation(
    events["HillasReconstructor_az"].quantity,
    events["HillasReconstructor_alt"].quantity,
    events["true_az"].quantity,
    events["true_alt"].quantity,
)

plt.hist(theta.to_value(u.deg) ** 2, bins=25, range=[0, 0.3])
plt.xlabel(r"$\theta² / deg²$")
None
```

+++ {"pycharm": {}}

## ArrayDisplay

```{code-cell} ipython3
:pycharm: {}

from ctapipe.visualization import ArrayDisplay


angle_offset = plotting_event.pointing.array_azimuth

plotting_hillas = {
    tel_id: dl1.parameters.hillas for tel_id, dl1 in plotting_event.dl1.tel.items()
}

plotting_core = {
    tel_id: dl1.parameters.core.psi for tel_id, dl1 in plotting_event.dl1.tel.items()
}


disp = ArrayDisplay(source.subarray)

disp.set_line_hillas(plotting_hillas, plotting_core, 500)

plt.scatter(
    plotting_event.simulation.shower.core_x,
    plotting_event.simulation.shower.core_y,
    s=200,
    c="k",
    marker="x",
    label="True Impact",
)
plt.scatter(
    plotting_event.dl2.stereo.geometry["HillasReconstructor"].core_x,
    plotting_event.dl2.stereo.geometry["HillasReconstructor"].core_y,
    s=200,
    c="r",
    marker="x",
    label="Estimated Impact",
)

plt.legend()
# plt.xlim(-400, 400)
# plt.ylim(-400, 400)
```

+++ {"pycharm": {}}

### Reading the LST dl1 data

```{code-cell} ipython3
loader = TableLoader(f.name, load_simulated=True, load_dl1_parameters=True)

dl1_table = loader.read_telescope_events(["LST_LST_LSTCam"])
```

```{code-cell} ipython3
:pycharm: {}

plt.scatter(
    np.log10(dl1_table["true_energy"].quantity / u.TeV),
    np.log10(dl1_table["hillas_intensity"]),
)
plt.xlabel("log10(E / TeV)")
plt.ylabel("log10(intensity)")
None
```

+++ {"pycharm": {}}

## Isn't python slow?

* Many of you might have heard: "Python is slow".
* That's trueish.
* All python objects are classes living on the heap, even integers.
* Looping over lots of "primitives" is quite slow compared to other languages.

⇒ Vectorize as much as possible using numpy  
⇒ Use existing interfaces to fast C / C++ / Fortran code  
⇒ Optimize using numba  

**But: "Premature Optimization is the root of all evil" — Donald Knuth**

So profile to find exactly what is slow.

### Why use python then?

* Python works very well as *glue* for libraries of all kinds of languages
* Python has a rich ecosystem for data science, physics, algorithms, astronomy

### Example: Number of Islands

Find all groups of pixels, that survived the cleaning

```{code-cell} ipython3
:pycharm: {}

from ctapipe.image import toymodel
from ctapipe.instrument import SubarrayDescription


geometry = loader.subarray.tel[1].camera.geometry
```

+++ {"pycharm": {}}

Let's create a toy images with several islands;

```{code-cell} ipython3
:pycharm: {}

np.random.seed(42)

image = np.zeros(geometry.n_pixels)


for i in range(9):

    model = toymodel.Gaussian(
        x=np.random.uniform(-0.8, 0.8) * u.m,
        y=np.random.uniform(-0.8, 0.8) * u.m,
        width=np.random.uniform(0.05, 0.075) * u.m,
        length=np.random.uniform(0.1, 0.15) * u.m,
        psi=np.random.uniform(0, 2 * np.pi) * u.rad,
    )

    new_image, sig, bg = model.generate_image(
        geometry, intensity=np.random.uniform(1000, 3000), nsb_level_pe=5
    )
    image += new_image
```

```{code-cell} ipython3
:pycharm: {}

clean = tailcuts_clean(
    geometry,
    image,
    picture_thresh=10,
    boundary_thresh=5,
    min_number_picture_neighbors=2,
)
```

```{code-cell} ipython3
:pycharm: {}

disp = CameraDisplay(geometry)
disp.image = image
disp.highlight_pixels(clean, color="xkcd:red", linewidth=1.5)
disp.add_colorbar()
```

```{code-cell} ipython3
:pycharm: {}

def num_islands_python(camera, clean):
    """A breadth first search to find connected islands of neighboring pixels in the cleaning set"""

    # the camera geometry has a [n_pixel, n_pixel] boolean array
    # that is True where two pixels are neighbors
    neighbors = camera.neighbor_matrix

    island_ids = np.zeros(camera.n_pixels)
    current_island = 0

    # a set to remember which pixels we already visited
    visited = set()

    # go only through the pixels, that survived cleaning
    for pix_id in np.where(clean)[0]:
        if pix_id not in visited:
            # remember that we already checked this pixel
            visited.add(pix_id)

            # if we land in the outer loop again, we found a new island
            current_island += 1
            island_ids[pix_id] = current_island

            # now check all neighbors of the current pixel recursively
            to_check = set(np.where(neighbors[pix_id] & clean)[0])
            while to_check:
                pix_id = to_check.pop()

                if pix_id not in visited:
                    visited.add(pix_id)
                    island_ids[pix_id] = current_island

                    to_check.update(np.where(neighbors[pix_id] & clean)[0])

    n_islands = current_island
    return n_islands, island_ids
```

```{code-cell} ipython3
:pycharm: {}

n_islands, island_ids = num_islands_python(geometry, clean)
```

```{code-cell} ipython3
:pycharm: {}

from matplotlib.colors import ListedColormap

cmap = plt.get_cmap("Paired")
cmap = ListedColormap(cmap.colors[:n_islands])
cmap.set_under("k")

disp = CameraDisplay(geometry)
disp.image = island_ids
disp.cmap = cmap
disp.set_limits_minmax(0.5, n_islands + 0.5)
disp.add_colorbar()
```

```{code-cell} ipython3
:pycharm: {}

%timeit num_islands_python(geometry, clean)
```

```{code-cell} ipython3
:pycharm: {}

from scipy.sparse.csgraph import connected_components


def num_islands_scipy(geometry, clean):
    neighbors = geometry.neighbor_matrix_sparse

    clean_neighbors = neighbors[clean][:, clean]
    num_islands, labels = connected_components(clean_neighbors, directed=False)

    island_ids = np.zeros(geometry.n_pixels)
    island_ids[clean] = labels + 1

    return num_islands, island_ids
```

```{code-cell} ipython3
:pycharm: {}

n_islands_s, island_ids_s = num_islands_scipy(geometry, clean)
```

```{code-cell} ipython3
:pycharm: {}

disp = CameraDisplay(geometry)
disp.image = island_ids_s
disp.cmap = cmap
disp.set_limits_minmax(0.5, n_islands_s + 0.5)
disp.add_colorbar()
```

```{code-cell} ipython3
:pycharm: {}

%timeit num_islands_scipy(geometry, clean)
```

+++ {"pycharm": {}}

**A lot less code, and a factor 3 speed improvement**

+++

Finally, current ctapipe implementation is using numba:

```{code-cell} ipython3
%timeit number_of_islands(geometry, clean)
```