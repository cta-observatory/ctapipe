"""
Tests for StatisticsExtractor and related functions
"""

import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

from ctapipe.calib.camera.pointing import StarTracer

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

obswl = 0.42 * u.micron

# pointing to the crab nebula

crab = SkyCoord.from_name("crab nebula")

alt = []
az = []

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

# the LST geometry

# the LST focal length

st = StarTracer.from_lookup(max_magnitude, az, alt, times, meteo_parameters, obswl)
