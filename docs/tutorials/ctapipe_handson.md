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

# Getting Started with ctapipe

This hands-on was presented at the Paris CTA Consoritum meeting (K. Kosack)

+++

## Part 1: load and loop over data

```{code-cell} ipython3
from ctapipe.io import EventSource
from ctapipe import utils
from matplotlib import pyplot as plt
import numpy as np
%matplotlib inline
```

```{code-cell} ipython3
path = utils.get_dataset_path("gamma_prod5.simtel.zst")
```

```{code-cell} ipython3
source = EventSource(path, max_events=5)

for event in source:
    print(event.count, event.index.event_id, event.simulation.shower.energy)
```

```{code-cell} ipython3
event
```

```{code-cell} ipython3
event.r1
```

```{code-cell} ipython3
for event in EventSource(path, max_events=5):
    print(event.count, event.r1.tel.keys())
```

```{code-cell} ipython3
event.r0.tel[3]
```

```{code-cell} ipython3
r0tel = event.r0.tel[3]
```

```{code-cell} ipython3
r0tel.waveform
```

```{code-cell} ipython3
r0tel.waveform.shape
```

note that this is ($N_{channels}$, $N_{pixels}$, $N_{samples}$)

```{code-cell} ipython3
plt.pcolormesh(r0tel.waveform[0])
```

```{code-cell} ipython3
brightest_pixel = np.argmax(r0tel.waveform[0].sum(axis=1))
print(f"pixel {brightest_pixel} has sum {r0tel.waveform[0,1535].sum()}")
```

```{code-cell} ipython3
plt.plot(r0tel.waveform[0,brightest_pixel], label="channel 0 (high-gain)")
plt.plot(r0tel.waveform[1,brightest_pixel], label="channel 1 (low-gain)")
plt.legend()
```

```{code-cell} ipython3
from ipywidgets import interact

@interact
def view_waveform(chan=0, pix_id=brightest_pixel):
    plt.plot(r0tel.waveform[chan, pix_id])
```

try making this compare 2 waveforms

+++

## Part 2: Explore the instrument description
This is all well and good, but we don't really know what camera or telescope this is... how do we get instrumental description info?

Currently this is returned *inside* the event (it will soon change to be separate in next version or so)

```{code-cell} ipython3
subarray = source.subarray 
```

```{code-cell} ipython3
subarray
```

```{code-cell} ipython3
subarray.peek()
```

```{code-cell} ipython3
subarray.to_table()
```

```{code-cell} ipython3
subarray.tel[2]
```

```{code-cell} ipython3
subarray.tel[2].camera
```

```{code-cell} ipython3
subarray.tel[2].optics
```

```{code-cell} ipython3
tel = subarray.tel[2]
```

```{code-cell} ipython3
tel.camera
```

```{code-cell} ipython3
tel.optics
```

```{code-cell} ipython3
tel.camera.geometry.pix_x
```

```{code-cell} ipython3
tel.camera.geometry.to_table()
```

```{code-cell} ipython3
tel.optics.mirror_area
```

```{code-cell} ipython3
from ctapipe.visualization import CameraDisplay
```

```{code-cell} ipython3
disp = CameraDisplay(tel.camera.geometry)
```

```{code-cell} ipython3
disp = CameraDisplay(tel.camera.geometry)
disp.image = r0tel.waveform[0,:,10]  # display channel 0, sample 0 (try others like 10)
```

 ** aside: ** show demo using a CameraDisplay in interactive mode in ipython rather than notebook

+++

## Part 3: Apply some calibration and trace integration

```{code-cell} ipython3
from ctapipe.calib import CameraCalibrator
```

```{code-cell} ipython3
calib = CameraCalibrator(subarray=subarray)
```

```{code-cell} ipython3
for event in EventSource(path, max_events=5):
    calib(event) # fills in r1, dl0, and dl1
    print(event.dl1.tel.keys())
```

```{code-cell} ipython3
event.dl1.tel[3]
```

```{code-cell} ipython3
dl1tel = event.dl1.tel[3]
```

```{code-cell} ipython3
dl1tel.image.shape # note this will be gain-selected in next version, so will be just 1D array of 1855
```

```{code-cell} ipython3
dl1tel.peak_time
```

```{code-cell} ipython3
CameraDisplay(tel.camera.geometry, image=dl1tel.image)
```

```{code-cell} ipython3
CameraDisplay(tel.camera.geometry, image=dl1tel.peak_time)
```

Now for Hillas Parameters

```{code-cell} ipython3
from ctapipe.image import hillas_parameters, tailcuts_clean
```

```{code-cell} ipython3
image = dl1tel.image
mask = tailcuts_clean(tel.camera.geometry, image, picture_thresh=10, boundary_thresh=5)
mask
```

```{code-cell} ipython3
CameraDisplay(tel.camera.geometry, image=mask)
```

```{code-cell} ipython3
cleaned = image.copy()
cleaned[~mask] = 0 
```

```{code-cell} ipython3
disp = CameraDisplay(tel.camera.geometry, image=cleaned)
disp.cmap = plt.cm.coolwarm
disp.add_colorbar()
plt.xlim(0.5, 1.0)
plt.ylim(-1.0, 0.0)
```

```{code-cell} ipython3
params = hillas_parameters(tel.camera.geometry, cleaned)
print(params)
```

```{code-cell} ipython3
disp = CameraDisplay(tel.camera.geometry, image=cleaned)
disp.cmap = plt.cm.coolwarm
disp.add_colorbar()
plt.xlim(0.5, 1.0)
plt.ylim(-1.0, 0.0)
disp.overlay_moments(params, color='white', lw=2)
```

## Part 4:  Let's put it all together: 
- loop over events, selecting only telescopes of the same type (e.g. LST:LSTCam)
- for each event, apply calibration/trace integration
- calculate Hillas parameters 
- write out all hillas paremeters to a file that can be loaded with Pandas

+++

first let's select only those telescopes with LST:LSTCam

```{code-cell} ipython3
subarray.telescope_types
```

```{code-cell} ipython3
subarray.get_tel_ids_for_type("LST_LST_LSTCam")
```

Now let's write out program

```{code-cell} ipython3
data = utils.get_dataset_path("gamma_prod5.simtel.zst") 
source = EventSource(data) # remove the max_events limit to get more stats
```

```{code-cell} ipython3
for event in source:
    calib(event)
    
    for tel_id, tel_data in event.dl1.tel.items():
        tel = source.subarray.tel[tel_id]
        mask = tailcuts_clean(tel.camera.geometry, tel_data.image)
        if np.count_nonzero(mask) > 0:
            params = hillas_parameters(tel.camera.geometry[mask], tel_data.image[mask])
```

```{code-cell} ipython3
from ctapipe.io import HDF5TableWriter
```

```{code-cell} ipython3
with HDF5TableWriter(filename='hillas.h5', group_name='dl1', overwrite=True) as writer:
    
    source = EventSource(data, allowed_tels=[1,2,3,4],  max_events=10)
    for event in source:
        calib(event)
    
        for tel_id, tel_data in event.dl1.tel.items():
            tel = source.subarray.tel[tel_id]
            mask = tailcuts_clean(tel.camera.geometry, tel_data.image)
            params = hillas_parameters(tel.camera.geometry[mask], tel_data.image[mask])
            writer.write("hillas", params)
```

### We can now load in the file we created and plot it

```{code-cell} ipython3
!ls *.h5
```

```{code-cell} ipython3
import pandas as pd

hillas = pd.read_hdf("hillas.h5", key='/dl1/hillas')
hillas
```

```{code-cell} ipython3
_ = hillas.hist(figsize=(8,8))
```

If you do this yourself, chose a larger file to loop over more events to get better statistics
