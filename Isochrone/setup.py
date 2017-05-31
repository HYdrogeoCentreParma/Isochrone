from __future__ import with_statement

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import Isochrone

Isochrone_classifiers = [
    "Programming Language :: Python :: 2",
    "Programming Language :: Python :: 3",
    "Intended Audience :: User",
]

setup(name="Isochrone",
      version=Isochrone.__version__,
      author="A. Feo, A. Zanini, E. Petrella, F. Celico",
      author_email="alessandra.feo@unipr.it",
      url="http://....../",
      py_modules=["Isochrone"],
      description="Python 2 and 3 utility for Modflow2005 postprocessing",
      long_description="Isochrone",
      license="GPLv2",
      classifiers=Isochrone_classifiers
      )
