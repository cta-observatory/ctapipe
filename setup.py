from setuptools import setup, find_packages

version = {}
with open("./ctapipe_resources/VERSION.py") as fp:
    exec(fp.read(), version)

setup(
    name='ctapipe-extra',
    version=version['__version__'],
    packages=find_packages(),
    package_data={'ctapipe_resources': '*'},
)
