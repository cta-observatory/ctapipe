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

# Using Container classes

`ctapipe.core.Container` is the base class for all event-wise data classes in ctapipe. It works like a object-relational mapper, in that it defines a set of `Fields` along with their metadata (description, unit, default), which can be later translated automatially into an output table using a `ctapipe.io.TableWriter`.

```{code-cell} ipython3
from ctapipe.core import Container, Field, Map
import numpy as np
from astropy import units as u
from functools import partial
```

Let's define a few example containers with some dummy fields in them:

```{code-cell} ipython3
class SubContainer(Container):
    junk = Field(-1, "Some junk")
    value = Field(0.0, "some value", unit=u.deg)

    
class TelContainer(Container):
    # defaults should match the other requirements, e.g. the defaults
    # should have the correct unit. It most often also makes sense to use
    # an invalid value marker like nan for floats or -1 for positive integers
    # as default
    tel_id = Field(-1, "telescope ID number")
    
        
    # For mutable structures like lists, arrays or containers, use a `default_factory` function or class
    # not an instance to assure each container gets a fresh instance and there is no hidden 
    # shared state between containers.
    image = Field(default_factory=lambda: np.zeros(10), description="camera pixel data")


class EventContainer(Container):
    event_id = Field(-1,"event id number")

    tels_with_data = Field(default_factory=list, description="list of telescopes with data")
    sub = Field(default_factory=SubContainer, description="stuff")  # a sub-container in the hierarchy

    # A Map is like a defaultdictionary with a specific container type as default.
    # This can be used to e.g. store a container per telescope
    # we use partial here to automatically get a function that creates a map with the correct container type
    # as default
    tel = Field(default_factory=partial(Map, TelContainer) , description="telescopes")  
```

## Basic features

```{code-cell} ipython3
ev = EventContainer()
```

Check that default values are automatically filled in

```{code-cell} ipython3
print(ev.event_id)
print(ev.sub)
print(ev.tel)
print(ev.tel.keys())

# default dict access will create container:
print(ev.tel[1])
```

print the dict representation

```{code-cell} ipython3
print(ev)
```

We also get docstrings "for free"

```{code-cell} ipython3
EventContainer?
```

```{code-cell} ipython3
SubContainer?
```

values can be set as normal for a class:

```{code-cell} ipython3
ev.event_id = 100
ev.event_id
```

```{code-cell} ipython3
ev.as_dict()  # by default only shows the bare items, not sub-containers (See later)
```

```{code-cell} ipython3
ev.as_dict(recursive=True)
```

and we can add a few of these to the parent container inside the tel dict:

```{code-cell} ipython3
ev.tel[10] = TelContainer()
ev.tel[5] = TelContainer()
ev.tel[42] = TelContainer()
```

```{code-cell} ipython3
# because we are using a default_factory to handle mutable defaults, the images are actually different:
ev.tel[42].image is ev.tel[32]
```

Be careful to use the `default_factory` mechanism for mutable fields, see this **negative** example:

```{code-cell} ipython3
class DangerousContainer(Container):
    image = Field(
        np.zeros(10),
        description="Attention!!!! Globally mutable shared state. Use default_factory instead"
    )
    
    
c1 = DangerousContainer()
c2 = DangerousContainer()

c1.image[5] = 9999

print(c1.image)
print(c2.image)
print(c1.image is c2.image)
```

```{code-cell} ipython3
ev.tel
```

## Converion to dictionaries

```{code-cell} ipython3
ev.as_dict()
```

```{code-cell} ipython3
ev.as_dict(recursive=True, flatten=False)
```

for serialization to a table, we can even flatten the output into a single set of columns

```{code-cell} ipython3
ev.as_dict(recursive=True, flatten=True)
```

## Setting and clearing values

```{code-cell} ipython3
ev.tel[5].image[:] = 9
print(ev)
```

```{code-cell} ipython3
ev.reset()
ev.as_dict(recursive=True)
```

## look at a pre-defined Container

```{code-cell} ipython3
from ctapipe.containers import SimulatedShowerContainer
```

```{code-cell} ipython3
SimulatedShowerContainer?
```

```{code-cell} ipython3
shower = SimulatedShowerContainer()
shower
```

## Container prefixes

To store the same container in the same table in a file or give more information, containers support setting
a custom prefix:

```{code-cell} ipython3
c1 = SubContainer(junk=5, value=3, prefix="foo")
c2 = SubContainer(junk=10, value=9001, prefix="bar")

# create a common dict with data from both containers:
d = c1.as_dict(add_prefix=True)
d.update(c2.as_dict(add_prefix=True))
d
```
