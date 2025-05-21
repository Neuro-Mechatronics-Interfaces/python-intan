# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # Ensure intan/ is discoverable


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'python-intan'
copyright = '2025, Jonathan Shulgach'
author = 'Jonathan Shulgach'
release = '0.0.3'

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

#html_theme = 'alabaster'
#html_static_path = ['_static']
# -- Options for HTML output --
html_theme = "sphinx_rtd_theme"
html_logo = "../figs/logo.png"  # or path to your logo if desired

# For Markdown support:
source_suffix = ['.rst', '.md']

# Set your master doc:
master_doc = 'index'
