"""Sphinx configuration for the StrikeTwice documentation."""

from importlib.metadata import version as package_version
from os import environ

project = "StrikeTwice"
author = "Gwenn Guichaoua and Elodie Laine"
release = package_version("StrikeTwice")
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "nbsphinx",
]

exclude_patterns = ["_build", "**/.ipynb_checkpoints"]
html_theme = "sphinx_rtd_theme"
html_logo = "images/striketwice-logo.png"

nbsphinx_execute = "always"
nbsphinx_timeout = 120

# Use CPU for generated timings unless the build explicitly selects a backend.
environ.setdefault("STRIKETWICE_TASK2_DEVICE", "cpu")
