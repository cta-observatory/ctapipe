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

# Convert camera images to pixels on a s square grid

```{code-cell} ipython3
from ctapipe.utils import get_dataset_path
from ctapipe.visualization import CameraDisplay
from ctapipe.instrument import SubarrayDescription
from ctapipe.io import EventSource
from ctapipe.image.toymodel import Gaussian
import matplotlib.pyplot as plt
import astropy.units as u
```

```{code-cell} ipython3
# get the subarray from an example file
subarray = SubarrayDescription.read("dataset://gamma_prod5.simtel.zst")
```

## Geometries with square pixels

Define a camera geometry and generate a dummy image:

```{code-cell} ipython3
geom = subarray.tel[40].camera.geometry
model = Gaussian(
    x=0.05 * u.m,
    y=0.05 * u.m,
    width=0.01 * u.m,
    length=0.05 * u.m,
    psi="30d",
)
_, image, _ = model.generate_image(geom, intensity=500, nsb_level_pe=3)
```

```{code-cell} ipython3
CameraDisplay(geom, image)
```

The `CameraGeometry` has functions to convert the 1d image arrays to 2d arrays and back to the 1d array:

```{code-cell} ipython3
image_square = geom.image_to_cartesian_representation(image)
```

```{code-cell} ipython3
plt.imshow(image_square)
```

```{code-cell} ipython3
image_1d = geom.image_from_cartesian_representation(image_square)
```

```{code-cell} ipython3
CameraDisplay(geom, image_1d)
```

## Geometries with hexagonal pixels

Define a camera geometry and generate a dummy image:

```{code-cell} ipython3
geom = subarray.tel[1].camera.geometry
model = Gaussian(
    x=0.5 * u.m,
    y=0.5 * u.m,
    width=0.1 * u.m,
    length=0.2 * u.m,
    psi="30d",
)
_, image, _ = model.generate_image(geom, intensity=5000)
```

```{code-cell} ipython3
CameraDisplay(geom, image)
```

```{code-cell} ipython3
image_square = geom.image_to_cartesian_representation(image)
```

## Conversion into square geometry

Since the resulting array has square pixels, the pixel grid has to be rotated and distorted.
This is reversible (The `image_from_cartesian_representation` method takes care of this):

```{code-cell} ipython3
plt.imshow(image_square)
```

```{code-cell} ipython3
image_1d = geom.image_from_cartesian_representation(image_square)
```

```{code-cell} ipython3
disp = CameraDisplay(geom, image_1d)
```
