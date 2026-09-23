"""Sphinx configuration for the ROTE documentation."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rote_benchmark import __version__

project = "ROTE"
author = "Xinye Chen"
version = release = __version__
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]
autodoc_typehints = "description"
napoleon_numpy_docstring = True
html_theme = "furo"
html_title = "ROTE documentation"
html_theme_options = {
    "source_repository": "https://github.com/chenxinye/slearn/",
    "source_branch": "main",
    "source_directory": "docs/source/",
    "light_css_variables": {
        "color-brand-primary": "#006c70",
        "color-brand-content": "#006c70",
    },
    "dark_css_variables": {
        "color-brand-primary": "#6cddd0",
        "color-brand-content": "#6cddd0",
    },
}
