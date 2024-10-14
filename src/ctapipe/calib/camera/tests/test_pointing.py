"""
Tests for StatisticsExtractor and related functions
"""

import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.table import QTable
from astropy.time import Time

from ctapipe.calib.camera.pointing import PointingCalculator
from ctapipe.instrument.camera.geometry import CameraGeometry

# let's prepare a StarTracer to make realistic images
# we need a maximum magnitude
max_magnitude = 2.0

# then some time period. let's take the first minute of 2023

times = [
    Time("2023-01-01T00:00:" + str(t), format="isot", scale="utc")
    for t in range(10, 40, 5)
]

# then the location of the first LST

location = {"longitude": 342.108612, "latitude": 28.761389, "elevation": 2147}

LST = EarthLocation(
    lat=location["latitude"] * u.deg,
    lon=location["longitude"] * u.deg,
    height=location["elevation"] * u.m,
)

# some weather data

meteo_parameters = {"relative_humidity": 0.5, "temperature": 10, "pressure": 790}

# some wavelength

obswl = 0.35 * u.micron

# pointing to the crab nebula

crab = SkyCoord.from_name("crab nebula")

alt = []
az = []

# get the local pointing values

for t in times:
    contemporary_crab = crab.transform_to(
        AltAz(
            obstime=t,
            location=LST,
            relative_humidity=meteo_parameters["relative_humidity"],
            temperature=meteo_parameters["temperature"] * u.deg_C,
            pressure=meteo_parameters["pressure"] * u.hPa,
        )
    )
    alt.append(contemporary_crab.alt.to_value())
    az.append(contemporary_crab.az.to_value())

# next i make the pointing table for the fake data generator

pointing_table = QTable(
    [alt, az, times],
    names=["telescope_pointing_altitude", "telescope_pointing_azimuth", "time"],
)

# the LST geometry

geometry = CameraGeometry.from_name(name="LSTCam")

# now set up the PointingCalculator

pc = PointingCalculator(geometry)

# now make fake mispointed data

pc.make_mispointed_data((1.0, -1.0), pointing_table)

pc.fit()
