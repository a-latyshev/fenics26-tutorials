# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: python3
#     language: python
#     name: python3
# ---
#
# # The shifted boundary method in FEniCSx
#
# In this tutorial, we will focus on the [shifted boundary method](https://doi.org/10.1016/j.jcp.2017.10.026).
#
# The idea of the method is to shift the boundary conditions from a true boundary $\Gamma$ that is not aligned with the
# mesh to a surrogate boundary $\bar\Gamma$.
#
# We will first illustrate some of the core concepts require for the method in detail, before we move on to the variational
# formulation and how it can be implemented in FEniCSx.

# ## Creating the real boundary $\Gamma$
#
# First, we start by defining a simple domain $\Omega$, defined through its boundary $\bar\Gamma=\partial\Omega$.
#
# The boundary of our real object will be an ellipsoid, which we will define through arbitrary order Lagrange line segments.
# First we import the required libaries

from mpi4py import MPI
import dolfinx
import numpy as np
import basix.ufl
import ufl

# Next, we define the characteristics of the line mesh, i.e. the center $(c_x, c_y)$, the radii $(R_x, R_y)$, and the number of elements $M$
# and the Lagrange degree of the elements.

line_degree = 3
center = (0.4, 0.3)
radii = (0.2, 0.13)
M = 5

# We define the number of nodes in the mesh, which depends on the number of elements and the degree of the Lagrange polynomials.
# We then define the coordinates of the nodes, which are placed equidistantly on the ellipse.
#  Finally, we define the connectivity of the mesh, which describes how the nodes are connected to form elements.

num_nodes = (M - 1) * (line_degree) + line_degree
nodes = np.zeros((num_nodes, 2), dtype=np.float64)
theta = np.linspace(0, 2 * np.pi, nodes.shape[0] + 1, endpoint=True)[:-1]
nodes[:, 0] = center[0] + radii[0] * np.cos(theta)
nodes[:, 1] = center[1] + radii[1] * np.sin(theta)

# Next, we can define the connecitivty of the mesh, which can be easily done with numpy tiling.

single_cell = np.empty(line_degree + 1, dtype=np.int64)
single_cell[0] = 0
single_cell[1] = line_degree
single_cell[2:] = np.arange(1, line_degree)
multiplier = np.arange(M) * line_degree
connectivity = np.tile(single_cell, (M, 1))
connectivity += multiplier[:, None]
connectivity[-1, 1] = 0

# Next, we define the symbolic representation of the mesh, which we in turn can pass into
# the DOLFINx mesh creator

c_el = ufl.Mesh(
    basix.ufl.element(
        "Lagrange",
        basix.CellType.interval,
        line_degree,
        shape=(nodes.shape[1],),
        lagrange_variant=basix.LagrangeVariant.equispaced,
    )
)
ghost_mode = dolfinx.mesh.GhostMode.none
max_facet_to_cell_links = 2
surface_comm = MPI.COMM_SELF
line_mesh = dolfinx.mesh.create_mesh(
    comm=surface_comm,
    x=nodes,
    cells=connectivity,
    e=c_el,
    partitioner=dolfinx.mesh.create_cell_partitioner(
        ghost_mode, max_facet_to_cell_links=2
    ),
    max_facet_to_cell_links=2,
)
line_mesh.name = "line"
assert line_mesh.topology.index_map(1).size_global == M
assert line_mesh.geometry.index_map().size_global == num_nodes

# ```{admonition} Important input parameters
# There are three very important input parameters that we send into the mesh creator:
# - `surface_comm`: The true boundary $\Gamma$ is represented on every process and is not distributed it across processes, which ensured by using `MPI.COMM_SELF` as the communicator for the mesh.
# - `ghost_mode`: As the mesh is not distributed, we set the `ghost_mode` to `GhostMode.none`.
# - `max_facet_to_cell_links`: Indicates that at max two cells can be connected to **any** facet, which is the case for
# this mesh, but not for T-joint grids or graph based meshes.
# ```

# Furthermore, we create the structured grid we will perform simulations on
domain = ((0.17, 0), (0.7, 0.6))
nx = ny = 21
mesh = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD, domain, (nx, ny), cell_type=dolfinx.mesh.CellType.quadrilateral
)

# To illustrate that the true boundary is curved, we use the function `interpolate_geometry` to interpolate
# the geometry into a first order space, i.e. remove all nodes that are not vertices

linear_cmap = dolfinx.fem.coordinate_element(line_mesh.topology.cell_type, 1)
linear_lines = dolfinx.fem.interpolate_geometry(line_mesh, linear_cmap)

# +[tags="hide-output"]
import pyvista as pv

surface_grid = pv.UnstructuredGrid(*dolfinx.plot.vtk_mesh(mesh))
obstacle_grid = pv.UnstructuredGrid(*dolfinx.plot.vtk_mesh(line_mesh))
linear_obstacle_grid = pv.UnstructuredGrid(*dolfinx.plot.vtk_mesh(linear_lines))
plotter = pv.Plotter()
plotter.add_mesh(
    surface_grid,
    color="darkblue",
    show_edges=True,
    opacity=0.5,
    label="Background mesh",
)
plotter.add_mesh(
    linear_obstacle_grid,
    color="blue",
    style="points",
    point_size=15.0,
    label="Mesh vertices",
)
plotter.add_mesh(
    obstacle_grid, color="red", style="points", point_size=10.0, label="Mesh nodes"
)
tessellated = obstacle_grid.tessellate()
tessellated.clear_data()
plotter.add_mesh(
    tessellated,
    color="black",
    style="wireframe",
    label="True boundary",
)
plotter.add_legend()
plotter.view_xy()
plotter.export_html("pyvista_true_boundary.html")

# -

# %% [markdown]
# :::{iframe} ../pyvista/pyvista_true_boundary.html
# :width: 100%
# :title: True boundary $\Gamma$
# :::
# %%
