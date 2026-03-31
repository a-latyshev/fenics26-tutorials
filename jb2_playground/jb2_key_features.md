---
authors:
  - name: "Andrey Latyshev"
    email: andrey.latyshev@uni.lu
    url: https://www.uni.lu/fstm-en/people/andrey-latyshev/
    affiliations: University of Luxembourg
---

# Jupyter Book 2 - key features

## Enable PyVista interactive scenes in JB2

PyVista’s interactive Jupyter backends (e.g. ipywidgets/trame) need a live Python kernel when the page is being viewed.
On GitHub Pages you only get static files, so the robust approach is:

1. **Export the scene to a standalone HTML file** using `pyvista.Plotter.export_html()`.
2. **Embed that HTML file** into your MyST/Jupyter Book page with an `<iframe>`.

This keeps the interactivity (rotate/zoom/pan, UI widgets shipped in the HTML) while avoiding any runtime kernel requirement.

### Minimal embedding

This page embeds a standalone HTML file that is deployed as a static asset. In
this repo, the tutorial code exports plain HTML filenames (e.g.
`fundamentals_mesh.html`). During deployment, CI gathers these exported HTML
files and copies them into the built site (e.g. under `/pyvista/`).

:::{iframe} ../pyvista/fundamentals_solution.html
:width: 100%
:title: PyVista exported scene
:::

Fallback link (also helps ensure the file is included in the static build):
[Open the exported scene in a new tab](/pyvista/fundamentals_solution.html).

This particular html is a part of the [Poisson
tutorial](fundamentals_code.ipynb)
([Source](https://jsdokken.com/dolfinx-tutorial/chapter1/fundamentals_code.html)).

### How to generate `fundamentals_mesh.html`

Run something like the following **locally** (or in CI) to generate the standalone HTML file.

```{code-block} python
:linenos:
:caption: Exporting `html`-based static scene with PyVista.

def plot_pyvista():
    pyvista.set_jupyter_backend("static")
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    plotter.view_xy()
    plotter.show()
    plotter.close()
    plotter.deep_clean()
plot_pyvista()

# setting the backend to `html` again
pyvista.set_jupyter_backend("html")

plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()
mesh_html = "fundamentals_mesh.html"
plotter.export_html(mesh_html)
```

Notes: `export_html` requires `trame` (and typically `trame-vtk`).

## Existing FEniCSx tutorials converted to JB2:

Here authors may find some examples of FEniCSx demos migrated from JB1 to JB2.

* [Solving the Poisson equation](fundamentals_code.ipynb)
  ([Source](https://jsdokken.com/dolfinx-tutorial/chapter1/fundamentals_code.html))
* [Solving von Mises Plasticity via Numba](demo_plasticity_von_mises.ipynb)
  ([Source](https://a-latyshev.github.io/dolfinx-external-operator/demo/demo_plasticity_von_mises.html))
* https://dolfiny.uni.lu/

## MyST Markdown features

MyST Markdown provides fancy-looking features, try them out:

- https://jupyterbook.org/stable/authoring/mystmd/
- https://mystmd.org/guide/quickstart-myst-documents
- https://mystmd.org/guide/typography (check out all pages in Authoring)