#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='PSF-Model-Fits',
      author='Minghan Chen',
      version='1.0',
      python_requires='>=3.6',
      packages=find_packages(),
      package_dir={'PSF_fits':'PSF_fits'},
      #setup_requires=['pytest-runner'],
      install_requires=['astropy>=2.0', 'scipy>=1.0.0', 'numpy>=1.16']
      #tests_require=['pytest>=3.5']
      )