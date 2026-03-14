---
title: Embed PyVista via export_html
---

# Embed PyVista via `export_html`

PyVista’s interactive Jupyter backends (e.g. ipywidgets/trame) need a live Python kernel when the page is being viewed.
On GitHub Pages you only get static files, so the robust approach is:

1. **Export the scene to a standalone HTML file** using `pyvista.Plotter.export_html()`.
2. **Embed that HTML file** into your MyST/Jupyter Book page with an `<iframe>`.

This keeps the interactivity (rotate/zoom/pan, UI widgets shipped in the HTML) while avoiding any runtime kernel requirement.

## Minimal embedding

This page embeds a standalone HTML file that is deployed as a static asset.
In this repo, the tutorial code exports plain HTML filenames (e.g. `pyvista_scene.html`).
During deployment, CI gathers these exported HTML files and copies them into the built site (e.g. under `/pyvista/`).

:::{iframe} ../pyvista/pyvista_scene.html
:width: 100%
:title: PyVista exported scene
:::

Fallback link (also helps ensure the file is included in the static build):
[Open the exported scene in a new tab](/pyvista/pyvista_scene.html).

## How to generate `pyvista_scene.html`

Run something like the following **locally** (or in CI) to generate the standalone HTML file.

```python
import pyvista as pv

plotter = pv.Plotter()
plotter.add_mesh(pv.Sphere(), color="lightgray")
plotter.add_axes()
plotter.export_html("pyvista_scene.html")
```

Notes:

- `export_html` requires `trame` (and typically `trame-vtk`).
- For GitHub Pages, prefer keeping the exported HTML **self-contained** (no external CDNs) if you need offline/reproducible builds.
- You can regenerate the HTML during book builds too, but then your build environment must have PyVista + trame available.

## Limitations

- This gives you **client-side interactivity** only. Anything that requires Python callbacks at view-time (e.g. updating the mesh based on new computations) still needs a live kernel.
