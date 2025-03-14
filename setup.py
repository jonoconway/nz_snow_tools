import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="nz_snow_tools",
    version="0.1.0",
    author='Jono Conway',
    author_email='jono.conway@niwa.co.nz',
    description='Tools to run and evaluate snow models',
    license="GPL-3.0",
    packages=["nz_snow_tools",
            "nz_snow_tools.eval", 
            "nz_snow_tools.hpc_runs",
            "nz_snow_tools.met", 
            "nz_snow_tools.snow", 
            "nz_snow_tools.util",
            "nz_snow_tools.util.SIN",
            "nz_snow_tools.util.idealised",],
    long_description=read('README.rst'),
    url='https://github.com/jonoconway/nz_snow_tools',  # use the URL to the github repo
    download_url='https://github.com/jonoconway/nz_snow_tools/archive/master.zip',
    install_requires=[
        'matplotlib',
        'netCDF4',
        'numpy',
        'pyshp',
        'pillow',
        'pyproj',
        'f90nml',
        'pyyaml',
        'scipy',
        'cartopy',
        'geopandas',
        'fiona',
    ],
)
