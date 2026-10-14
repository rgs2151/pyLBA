"""Sphinx configuration for pyLBA."""
from importlib.metadata import version as package_version

project = "pyLBA"
author = "Rudramani Singha"
copyright = "2026, Rudramani Singha"
release = package_version("pyLBA")
extensions = [
    "sphinx.ext.autodoc", "sphinx.ext.napoleon",
    "sphinx.ext.githubpages",
]
html_theme = "pydata_sphinx_theme"
html_title = "pyLBA"
html_logo = "../logo/pylba.svg"
html_baseurl = "https://rgs2151.github.io/pyLBA/"
html_context = {"default_mode": "light"}
html_theme_options = {
    "github_url": "https://github.com/rgs2151/pyLBA",
    "logo": {"text": "pyLBA"},
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "search_bar_text": "Search documentation",
    "search_as_you_type": True,
    "show_nav_level": 2,
}
exclude_patterns = ["_build", "**/.ipynb_checkpoints"]
autodoc_member_order = "bysource"
autodoc_typehints = "description"
