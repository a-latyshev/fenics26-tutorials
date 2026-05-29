# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: python3
#     language: python
#     name: python3
# ---

import pyvista as pv
from pyvista import examples
mesh = examples.download_bunny()
plotter = pv.Plotter()
plotter.add_mesh(mesh, color='lightblue')
plotter.export_html('scene.html')

# %% [markdown]
# :::{iframe} scene.html
# :width: 100%
# :title: Bunny
# :::
# %%
