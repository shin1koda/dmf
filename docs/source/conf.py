# Configuration file for the Sphinx documentation builder.

import os
import sys

# --- Correct path settings ----------------------------------------------------
# Add the project root and src/ to sys.path so autodoc can import `dmf`
sys.path.insert(0, os.path.abspath('../../src'))
sys.path.insert(0, os.path.abspath('../../'))

# --- Project information -------------------------------------------------------
project = 'dmf'
author = 'Shin-ichi Koda'
copyright = '2025, Shin-ichi Koda'
release = '1.0.0'

# --- General configuration -----------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'sphinx_autodoc_typehints',
]

templates_path = ['_templates']
exclude_patterns = []

# Napoleon settings (Google/NumPy docstring support)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = False
napoleon_custom_sections = [('Returns', 'params_style')]

# autodoc settings
autodoc_member_order = 'bysource'

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'inherited-members': True,
    'show-inheritance': True,
}

# --- Options for HTML output ---------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

