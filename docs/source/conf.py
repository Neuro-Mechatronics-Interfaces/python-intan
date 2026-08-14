# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # Ensure intan/ is discoverable

from intan import __version__


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'python-intan'
copyright = '2024-2026, Jonathan Shulgach'
author = 'Jonathan Shulgach'
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "m2r2",
]

templates_path = ['_templates']
exclude_patterns = []
autodoc_typehints = 'description'
autodoc_member_order = 'bysource'
add_module_names = False


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_logo = "../figs/logo.png"

# For Markdown support:
source_suffix = ['.rst', '.md']

# Set your master doc:
master_doc = 'index'
