"""
Using Container classes
=======================

``ctapipe.core.Container`` is the base class for all event-wise data
classes in ctapipe. It works like a object-relational mapper, in that it
defines a set of ``Fields`` along with their metadata (description,
unit, default), which can be later translated automatically into an
output table using a ``ctapipe.io.TableWriter``.

"""

from functools import partial

import numpy as np
from astropy import units as u

from ctapipe.containers import SimulatedShowerContainer
from ctapipe.core import Container, Field, Map

######################################################################
# Let’s define a few example containers with some dummy fields in them:
#


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
    event_id = Field(-1, "event id number")

    tels_with_data = Field(
        default_factory=list, description="list of telescopes with data"
    )
    sub = Field(
        default_factory=SubContainer, description="stuff"
    )  # a sub-container in the hierarchy

    # A Map is like a defaultdictionary with a specific container type as default.
    # This can be used to e.g. store a container per telescope
    # we use partial here to automatically get a function that creates a map with the correct container type
    # as default
    tel = Field(default_factory=partial(Map, TelContainer), description="telescopes")


######################################################################
# Basic features
# --------------
#

ev = EventContainer()


######################################################################
# Check that default values are automatically filled in
#

print(ev.event_id)
print(ev.sub)
print(ev.tel)
print(ev.tel.keys())

# default dict access will create container:
print(ev.tel[1])


######################################################################
# print the dict representation
#

print(ev)


######################################################################
# We also get docstrings “for free”
#
help(EventContainer)

######################################################################
help(SubContainer)

######################################################################
# values can be set as normal for a class:
#

ev.event_id = 100
ev.event_id

######################################################################
ev.as_dict()  # by default only shows the bare items, not sub-containers (See later)

######################################################################
ev.as_dict(recursive=True)


######################################################################
# and we can add a few of these to the parent container inside the tel
# dict:
#

ev.tel[10] = TelContainer()
ev.tel[5] = TelContainer()
ev.tel[42] = TelContainer()

######################################################################
# because we are using a default_factory to handle mutable defaults, the images are actually different:
ev.tel[42].image is ev.tel[32]


######################################################################
# Be careful to use the ``default_factory`` mechanism for mutable fields,
# see this **negative** example:
#


class DangerousContainer(Container):
    image = Field(
        np.zeros(10),
        description=(
            "Attention!!!! Globally mutable shared state. Use default_factory instead"
        ),
    )


c1 = DangerousContainer()
c2 = DangerousContainer()

c1.image[5] = 9999

print(c1.image)
print(c2.image)
print(c1.image is c2.image)

######################################################################
ev.tel


######################################################################
# Conversion to dictionaries
# --------------------------
#

ev.as_dict()

######################################################################
ev.as_dict(recursive=True, flatten=False)


######################################################################
# for serialization to a table, we can even flatten the output into a
# single set of columns
#

ev.as_dict(recursive=True, flatten=True)


######################################################################
# Setting and clearing values
# ---------------------------
#

ev.tel[5].image[:] = 9
print(ev)

######################################################################
ev.reset()
ev.as_dict(recursive=True)


######################################################################
# look at a pre-defined Container
# -------------------------------
#


help(SimulatedShowerContainer)

######################################################################
shower = SimulatedShowerContainer()
shower


######################################################################
# Container prefixes
# ------------------
#
# To store the same container in the same table in a file or give more
# information, containers support setting a custom prefix:
#

c1 = SubContainer(junk=5, value=3, prefix="foo")
c2 = SubContainer(junk=10, value=9001, prefix="bar")

# create a common dict with data from both containers:
d = c1.as_dict(add_prefix=True)
d.update(c2.as_dict(add_prefix=True))
d
