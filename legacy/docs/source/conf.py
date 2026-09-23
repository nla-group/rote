import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

project = 'slearn'
author = 'Roberto Cahuantzi, Xinye Chen, and Stefan Guettel'
copyright = '2021-2026, Numerical Linear Algebra Group, The University of Manchester'

try:
    from slearn import __version__
except Exception:
    __version__ = '0.2.9'

release = __version__
version = __version__

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

autosummary_generate = True
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
napoleon_google_docstring = False
napoleon_numpy_docstring = True

source_suffix = '.rst'
templates_path = ['_templates']
exclude_patterns = ['build', 'Thumbs.db', '.DS_Store', '_autosummary']
locale_dirs = ['locale/']
gettext_compact = False
pygments_style = 'lovelace'

try:
    import furo  # noqa: F401
    html_theme = 'furo'
    html_theme_options = {
        'navigation_with_keys': True,
        'top_of_page_button': 'edit',
        'source_repository': 'https://github.com/nla-group/slearn/',
        'source_branch': 'main',
        'source_directory': 'docs/source/',
        'light_css_variables': {
            'color-brand-primary': '#2454a6',
            'color-brand-content': '#2454a6',
            'color-api-name': '#12366f',
            'color-api-pre-name': '#0f766e',
            'font-stack': "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
            'font-stack--monospace': "'SFMono-Regular', Consolas, 'Liberation Mono', monospace",
        },
        'dark_css_variables': {
            'color-brand-primary': '#8ab4ff',
            'color-brand-content': '#8ab4ff',
            'color-api-name': '#b7cdfc',
            'color-api-pre-name': '#7dd3c7',
        },
    }
except Exception:
    html_theme = 'sphinx_rtd_theme'
    html_theme_options = {'navigation_depth': 5}

html_title = 'slearn documentation'
html_short_title = 'slearn'
html_static_path = ['_static']
html_css_files = ['custom.css']
