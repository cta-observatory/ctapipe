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

# Working with Instrumental Descriptions

the instrumental description is loaded by the event source, and consists of a hierarchy of classes in the ctapipe.instrument module, the base of which is the `SubarrayDescription`

```{code-cell} ipython3
from ctapipe.utils.datasets import get_dataset_path
from ctapipe.io import EventSource
import numpy as np

filename = get_dataset_path("gamma_prod5.simtel.zst")

with EventSource(filename, max_events=1) as source:
    subarray = source.subarray
```

## the SubarrayDescription:

```{code-cell} ipython3
subarray.info()
```

```{code-cell} ipython3
subarray.to_table()
```

```{code-cell} ipython3

```

You can also get a table of just the `OpticsDescriptions` (`CameraGeometry` is more complex and can't be stored on a single table row, so each one can be converted to a table separately)

```{code-cell} ipython3
subarray.to_table(kind="optics")
```

Make a sub-array with only SC-type telescopes:

```{code-cell} ipython3
sc_tels = [tel_id for tel_id, tel in subarray.tel.items() if tel.optics.n_mirrors == 2]
newsub = subarray.select_subarray(sc_tels, name="SCTels")
newsub.info()
```

can also do this by using `Table.group_by`

+++

## Explore some of the details of the telescopes

```{code-cell} ipython3
tel = subarray.tel[1]
tel
```

```{code-cell} ipython3
tel.optics.mirror_area
```

```{code-cell} ipython3
tel.optics.n_mirror_tiles
```

```{code-cell} ipython3
tel.optics.equivalent_focal_length
```

```{code-cell} ipython3
tel.camera
```

```{code-cell} ipython3
tel.camera.geometry.pix_x
```

```{code-cell} ipython3
%matplotlib inline
from ctapipe.visualization import CameraDisplay

CameraDisplay(tel.camera.geometry)
```

```{code-cell} ipython3
CameraDisplay(subarray.tel[98].camera.geometry)
```

## Plot the subarray

We'll make a subarray by telescope type and plot each separately, so they appear in different colors.  We also calculate the radius using the mirror area (and exagerate it a bit).

This is just for debugging and info, for any "real" use, a `visualization.ArrayDisplay` should be used

```{code-cell} ipython3
subarray.peek()
```

```{code-cell} ipython3
subarray.footprint
```

## Get info about the subarray in general

```{code-cell} ipython3
subarray.telescope_types
```

```{code-cell} ipython3
subarray.camera_types
```

```{code-cell} ipython3
subarray.optics_types
```

```{code-cell} ipython3
from astropy.coordinates import SkyCoord
from ctapipe.coordinates import GroundFrame

center = SkyCoord("10.0 m", "2.0 m", "0.0 m", frame="groundframe")
coords = subarray.tel_coords  # a flat list of coordinates by tel_index
coords.separation(center)
```

## Telescope IDs vs Indices

Note that `subarray.tel` is a dict mapped by `tel_id` (the indentifying number of a telescope).  It is  possible to have telescope IDs that do not start at 0, are not contiguouous (e.g. if a subarray is selected).  Some functions and properties like `tel_coords` are numpy arrays (not dicts) so they are not mapped to the telescope ID, but rather the *index* within this SubarrayDescription. To convert between the two concepts you can do:

```{code-cell} ipython3
subarray.tel_ids_to_indices([1, 5, 23])
```

or you can get the indexing array directly in numpy or dict form:

```{code-cell} ipython3
subarray.tel_index_array
```

```{code-cell} ipython3
subarray.tel_index_array[[1, 5, 23]]
```

```{code-cell} ipython3
subarray.tel_indices[
    1
]  # this is a dict of tel_id -> tel_index, so we can only do one at once
```

```{code-cell} ipython3
ids = subarray.get_tel_ids_for_type(subarray.telescope_types[0])
ids
```

```{code-cell} ipython3
idx = subarray.tel_ids_to_indices(ids)
idx
```

```{code-cell} ipython3
subarray.tel_coords[idx]
```

so, with that method you can quickly get many telescope positions at once (the alternative is to use the dict `positions` which maps `tel_id` to a position on the ground

```{code-cell} ipython3
subarray.positions[1]
```
