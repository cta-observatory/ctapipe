from setuptools import setup, find_packages


setup(
    name='ctapipe-extra',
    version='0.2.1',
    packages=find_packages(),
    package_data={'ctapipe_resources': '*'},
)
