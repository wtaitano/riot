# ========================================================================================
#  (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
#
#  This program was produced under U.S. Government contract 89233218CNA000001 for Los
#  Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
#  for the U.S. Department of Energy/National Nuclear Security Administration. All rights
#  in the program are reserved by Triad National Security, LLC, and the U.S. Department
#  of Energy/National Nuclear Security Administration. The Government is granted for
#  itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
#  license in this material to reproduce, prepare derivative works, distribute copies to
#  the public, perform publicly and display publicly, and to permit others to do so.
# ========================================================================================
# This file was made in part with generative AI.

# Sphinx configuration for the ckm docs. Mirrors riot's doc/sphinx/conf.py so the
# two sites look alike; ckm is not (yet) published with sphinx-multiversion.

project = "ckm"
copyright = "2026. Triad National Security"
author = "The ckm Team"

extensions = [
    "sphinx.ext.autosectionlabel",
    "sphinx_rtd_theme",
]
autosectionlabel_prefix_document = True

templates_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["table-wrap.css"]
