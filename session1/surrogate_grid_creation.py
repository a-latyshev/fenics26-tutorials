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

# ---
# authors:
# - jsdokken
# ---
# # Computing the surrogate grid from a higher order surface mesh
#
# ## Creating the real boundary $\Gamma$
#
# First, we start by defining a simple domain $\Omega$, defined through its boundary $\bar\Gamma=\partial\Omega$.
#
# The boundary of our real object will be the [fifth parametric heart curve](https://mathworld.wolfram.com/HeartCurve.html),
# which we will define through arbitrary order Lagrange line segments.
# First we import the required libaries

from mpi4py import MPI
import dolfinx
import numpy as np
import basix.ufl
import ufl

# Next, we define the characteristics of the line mesh and the number of elements $M$
# and the Lagrange degree of the elements.

line_degree = 3
center = (1.1, 0.9)
scale = 0.05
M = 12

# We define the number of nodes in the mesh, which depends on the number of elements and the degree of the Lagrange polynomials.
# We then define the coordinates of the `nodes`, which are placed equidistantly on the ellipse.
#  Finally, we define the `connectivity` of the mesh, which describes how the nodes are connected to form elements.

num_nodes = (M - 1) * (line_degree) + line_degree
nodes = np.zeros((num_nodes, 2), dtype=np.float64)
theta = np.linspace(0, 2 * np.pi, nodes.shape[0] + 1, endpoint=True)[:-1]
nodes[:, 0] = center[0] + (scale * 16 * np.sin(theta) ** 3)
nodes[:, 1] = center[1] + scale * (
    13 * np.cos(theta)
    - 5 * np.cos(2 * theta)
    - 2 * np.cos(3 * theta)
    - 1 * np.cos(4 * theta)
)

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
# the [create_mesh](xref:dolfinx#dolfinx.mesh.create_mesh) function to create the DOLFINx mesh representing the true boundary $\Gamma$.

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
true_surface = dolfinx.mesh.create_mesh(
    comm=surface_comm,
    x=nodes,
    cells=connectivity,
    e=c_el,
    partitioner=dolfinx.mesh.create_cell_partitioner(
        ghost_mode, max_facet_to_cell_links=2
    ),
    max_facet_to_cell_links=2,
)
true_surface.name = "true_surface"
assert true_surface.topology.index_map(1).size_global == M
assert true_surface.geometry.index_map().size_global == num_nodes

# ```{admonition} Important input parameters
# There are three very important input parameters that we send into the mesh creator:
# - `surface_comm`: The true boundary $\Gamma$ is represented on every process and is not distributed it across processes,
# which ensured by using `MPI.COMM_SELF` as the communicator for the mesh.
# - `ghost_mode`: As the mesh is not distributed, we set it to [GhostMode.none](xref:dolfinx#dolfinx.mesh.GhostMode).
# - `max_facet_to_cell_links`: Indicates that at max two cells can be connected to **any** facet, which is the case for
# this mesh, but not for T-joint grids or graph based meshes.
# ```

# Furthermore, we create the structured grid we will perform simulations on

domain = ((-0.1, -0.2), (2.1, 2.0))
nx = ny = 37
mesh = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD, domain, (nx, ny), cell_type=dolfinx.mesh.CellType.quadrilateral
)

# To illustrate that the true boundary is curved, we use the function
# [interpolate_geometry](xref:dolfinx#dolfinx.fem.interpolate_geometry)
# to interpolate the geometry into a first order space, i.e. remove all nodes that are not vertices

linear_cmap = dolfinx.fem.coordinate_element(true_surface.topology.cell_type, 1)
linear_lines = dolfinx.fem.interpolate_geometry(true_surface, linear_cmap)

# + tags=["hide-input", "hide-output"]
import pyvista as pv

topology, cell_types, geometry = dolfinx.plot.vtk_mesh(mesh)
cell_types[:] = (
    pv.CellType.QUAD
    if mesh.topology.cell_type == dolfinx.mesh.CellType.quadrilateral
    else cell_types
)
surface_grid = pv.UnstructuredGrid(topology, cell_types, geometry)
true_surface_grid = pv.UnstructuredGrid(*dolfinx.plot.vtk_mesh(true_surface))
linearized_surface_grid = pv.UnstructuredGrid(*dolfinx.plot.vtk_mesh(linear_lines))
plotter = pv.Plotter()
plotter.add_mesh(
    surface_grid,
    color="darkblue",
    show_edges=True,
    opacity=0.5,
    label="Background mesh",
)
plotter.add_mesh(
    linearized_surface_grid,
    color="blue",
    style="points",
    point_size=15.0,
    label="Mesh vertices",
)
plotter.add_mesh(
    true_surface_grid, color="red", style="points", point_size=10.0, label="Mesh nodes"
)
tessellated = true_surface_grid.tessellate(max_n_subdivide=10)
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

# # Removing non-intersecting cells
# The next step is to remove all cells that are not intersected by the true boundary $\Gamma$.

# We start by creating a set of [axis-aligned bounding boxes](https://doi.org/10.1137/11085949X)
# for the cells in the background mesh using [bb_tree](xref:dolfinx#dolfinx.geometry.bb_tree),
# which we can use to quickly query which cells are intersected by the true boundary $\Gamma$.

tol = 10 * np.finfo(mesh.geometry.x.dtype).eps
bulk_cells = mesh.topology.index_map(mesh.topology.dim)
num_cells_local = bulk_cells.size_local + bulk_cells.num_ghosts
bbox_bulk = dolfinx.geometry.bb_tree(
    mesh,
    mesh.topology.dim,
    entities=np.arange(num_cells_local, dtype=np.int32),
    padding=tol,
)

# Next, we create a similar bounding box tree for the true surface.

bbox_surface = dolfinx.geometry.bb_tree(
    true_surface, true_surface.topology.dim, padding=tol
)
bbox_surface_glob = bbox_surface.create_global_tree(true_surface.comm)

# And we compute the intersection between the two trees, which gives us the indices of the background cells that
# are within the bounding boxes enclosing the whole true surface.

bulk_elements = dolfinx.geometry.compute_collisions_trees(bbox_bulk, bbox_surface_glob)
colliding_cells = np.unique(bulk_elements[:, 0])

# We visualize this
OUTSIDE_MARKER = 0
KEEP_MARKER = 2
values = np.full(num_cells_local, OUTSIDE_MARKER, dtype=np.int32)
values[colliding_cells] = KEEP_MARKER
initial_tag = dolfinx.mesh.meshtags(
    mesh, mesh.topology.dim, np.arange(num_cells_local, dtype=np.int32), values
)
surface_grid.cell_data["colliding"] = values
plotter_collision = pv.Plotter()
plotter_collision.add_mesh(
    surface_grid,
    scalars="colliding",
    show_edges=True,
    cmap="coolwarm",
    label="Colliding cells",
)
plotter_collision.add_mesh(
    tessellated, color="red", style="wireframe", label="True boundary"
)
plotter_collision.add_legend()
plotter_collision.view_xy()
plotter_collision.export_html("pyvista_bb_collisions.html")

# %% [markdown]
# :::{iframe} ../pyvista/pyvista_bb_collisions.html
# :width: 100%
# :title: Colliding cells
# :::
# %%

# We observe that this is not a very good estimate as to which cells are intersected by the true boundary, but at least it
# gives us a good starting point for filtering out the non-intersecting cells.

# As the surface is quite curved, we will create a point cloud of points on the true boundary.
# We do this by creating a quadrature space, as it dof coordinates will make a suitable point cloud representing the true boundary.

reference_points = basix.create_lattice(
    true_surface.basix_cell(),
    50,
    basix.LatticeType.equispaced,
    exterior=True,
    method=basix.LatticeSimplexMethod.none,
)
q_manifold = basix.ufl.quadrature_element(
    true_surface.basix_cell(),
    points=reference_points,
    weights=np.ones(reference_points.shape[0]),
    scheme="gauss_jacobi",
    value_shape=(),
)
Q = dolfinx.fem.functionspace(true_surface, q_manifold)
refined_surface_nodes = Q.tabulate_dof_coordinates()

# + tags=["hide-input"]
point_cloud = pv.PolyData(refined_surface_nodes)
plotter.add_mesh(point_cloud, label="Point cloud", point_size=10.0, color="green")
plotter.export_html("pyvista_pc_boundary.html")
# -

# %% [markdown]
# :::{iframe} ../pyvista/pyvista_pc_boundary.html
# :width: 100%
# :title: True boundary $\Gamma$
# :::
# %%

# Now that we have a good representation of the curved boundary,
# to find all cells that are intersected by the true boundary $\Gamma$.
# We will use the [GJK distance algorithm](https://doi.org/10.1109/56.2083) to compute
# the distance between th convex hull that makes up each cell of the background mesh
# and the point cloud that represents the true boundary.

background_cell_nodes = mesh.geometry.x[mesh.geometry.dofmaps[0]][colliding_cells]
cell_indicator = dolfinx.la.vector(
    mesh.topology.index_map(mesh.topology.dim), bs=1, dtype=np.int32
)
INTERFACE_MARKER = 3
tol = 10 * np.finfo(refined_surface_nodes.dtype).eps
cell_indicator.array[colliding_cells] = KEEP_MARKER

# For each node in the interface, we find which cells that contains the nodes.

for i, node in enumerate(refined_surface_nodes):
    distance_vector = dolfinx.geometry.compute_distances_gjk(
        list(background_cell_nodes), node.reshape(-1, 3), num_threads=2
    )
    close_cells = np.linalg.norm(distance_vector, axis=1) < tol
    cell_indicator.array[colliding_cells[close_cells]] = INTERFACE_MARKER
cell_indicator.scatter_forward()

# Once we have the interface, we can create an iterative algorithm that starts with the cells that are outside
# the initial bounding box tree collision and iteratively mark all cells connected to these
# by using [compute_incident_entities](xref:dolfinx#dolfinx.mesh.compute_incident_entities)
# until we have marked all cells that are not intersected by the true boundary $\Gamma$.

num_new_cells = 1
sweep = 0
mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
while num_new_cells > 0:
    outside_cells = np.flatnonzero(cell_indicator.array == OUTSIDE_MARKER)
    # Find all cells connected to these by facet
    facets = dolfinx.mesh.compute_incident_entities(
        mesh.topology, outside_cells, mesh.topology.dim, mesh.topology.dim - 1
    )
    adjacent_cells = dolfinx.mesh.compute_incident_entities(
        mesh.topology, facets, mesh.topology.dim - 1, mesh.topology.dim
    )
    new_outside_cells = cell_indicator.array[adjacent_cells] != INTERFACE_MARKER
    cell_indicator.array[adjacent_cells[new_outside_cells]] = OUTSIDE_MARKER
    num_new_cells = np.sum(cell_indicator.array == OUTSIDE_MARKER) - len(outside_cells)
    sweep += 1
    print(f"Sweep {sweep}: Marked {num_new_cells} new cells as outside.")
cell_tag = dolfinx.mesh.meshtags(
    mesh,
    mesh.topology.dim,
    np.arange(len(cell_indicator.array), dtype=np.int32),
    cell_indicator.array,
)

# Next we plot the new set of tagged cells, which is what we will use as the set of colliding cells for the rest of the tutorial.

# + tags=["hide-input"]
surface_grid["refined_collisions"] = cell_tag.values
plotter_refined_collision = pv.Plotter()
plotter_refined_collision.add_mesh(
    surface_grid,
    scalars="refined_collisions",
    show_edges=True,
    cmap="coolwarm",
    label="Refined collisions",
)
plotter_refined_collision.add_mesh(
    tessellated, color="red", style="wireframe", label="True boundary"
)
plotter_refined_collision.add_legend()
plotter_refined_collision.view_xy()
plotter_refined_collision.export_html("pyvista_refined_collision.html")
# -

# %% [markdown]
# :::{iframe} ../pyvista/pyvista_refined_collision.html
# :width: 100%
# :title: Refined colliding cells
# :::
# %%

## Creating the reduced mesh
# We use [create_submesh](xref:dolfinx#dolfinx.mesh.create_submesh)
# to restrict the problem to the marked cells only

submesh, entity_map, vertex_map, _ = dolfinx.mesh.create_submesh(
    mesh, mesh.topology.dim, cell_tag.find(KEEP_MARKER)
)
submesh.topology.create_connectivity(submesh.topology.dim - 1, submesh.topology.dim)
submesh_facets = dolfinx.mesh.exterior_facet_indices(submesh.topology)

# + tags=["hide-input"]
submesh_topology, submesh_cell_types, submesh_geometry = dolfinx.plot.vtk_mesh(submesh)
submesh_cell_types[:] = (
    pv.CellType.QUAD
    if submesh.topology.cell_type == dolfinx.mesh.CellType.quadrilateral
    else submesh_cell_types
)
submesh_grid = pv.UnstructuredGrid(
    submesh_topology, submesh_cell_types, submesh_geometry
)
plotter_submesh = pv.Plotter()
plotter_submesh.add_mesh(
    submesh_grid,
    show_edges=True,
)
plotter_submesh.add_mesh(
    tessellated, color="red", style="wireframe", label="True boundary"
)
plotter_submesh.add_legend()
plotter_submesh.view_xy()
plotter_submesh.export_html("pyvista_submesh.html")
# -

# %% [markdown]
# :::{iframe} ../pyvista/pyvista_submesh.html
# :width: 100%
# :title: Submesh
# :::
# %%

facet_submesh, facet_map = dolfinx.mesh.create_submesh(
    submesh, submesh.topology.dim - 1, submesh_facets
)[:2]
