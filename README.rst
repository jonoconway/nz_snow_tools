nz_snow_tools
=============

A suite of python code to run and evaluate snow models developed as part of the Deep South National Science Challenge "Frozen Water" project. Forked from https://github.com/bodekerscientific/nz_snow_tools



Installation
============

Install the latest version of the package from github:

https://github.com/jonoconway/nz_snow_tools

once you have downloaded, check the requirements.txt file and install the needed packages. It is safer to use conda, rather than pip to install these. run setup.py to install (checks what other packages are necessary).


Getting Started
===============

The package was originally written for Python 2.7, but some parts have been upgraded to Python 3.6.

The snow model is called from run_snow_model (for 2D met input) or run_snow_model_simple (for 1D met input)

A configuration dictionary can be used to change the default parameter values in the snow model. This dictionary specifies the parameter values, which will override the default values specified in the functions.