---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Using the ctapipe Provenance service

The provenance functionality is used automatically when you use most of ctapipe functionality (particularly `ctapipe.core.Tool` and functions in `ctapipe.io` and `ctapipe.utils`), so normally you don't have to work with it directly. It tracks both input and output files, as well as details of the machine and software environment on which a Tool executed. 

Here we show some very low-level functions of this system:

```{code-cell} ipython3
from ctapipe.core import Provenance
from pprint import pprint
```

## Activities

The basis of Provenance is an *activity*, which is generally an executable or step in a script. Activities can be nested (e.g. with sub-activities), as shown below, but normally this is not required:

```{code-cell} ipython3
p = Provenance()  # note this is a singleton, so only ever one global provenence object
p.clear()
p.start_activity()
p.add_input_file("test.txt")

p.start_activity("sub")
p.add_input_file("subinput.txt")
p.add_input_file("anothersubinput.txt")
p.add_output_file("suboutput.txt")
p.finish_activity("sub")

p.start_activity("sub2")
p.add_input_file("sub2input.txt")
p.finish_activity("sub2")

p.finish_activity()
```

```{code-cell} ipython3
p.finished_activity_names
```

Activities have associated input and output *entities*  (files or other objects)

```{code-cell} ipython3
[ (x['activity_name'], x['input']) for x in p.provenance]
```

Activities track when they were started and finished:

```{code-cell} ipython3
[ (x['activity_name'],x['duration_min']) for x in p.provenance]
```

## Full provenance

The provence object is a list of activitites, and for each lots of details are collected:

```{code-cell} ipython3
p.provenance[0]
```

This can be better represented in JSON:

```{code-cell} ipython3
print(p.as_json(indent=2))
```

## Storing provenance info in output files

* already this can be stored in something like an HDF5 file header, which allows hierarchies.
* Try to flatted the data so it can be stored in a key=value header in a **FITS file** (using the FITS extended keyword convention to allow >8 character keywords), or as a table 

```{code-cell} ipython3
def flatten_dict(y):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '.')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '.')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(y)
    return out
```

```{code-cell} ipython3
d = dict(activity=p.provenance)
```

```{code-cell} ipython3
pprint(flatten_dict(d))
```
