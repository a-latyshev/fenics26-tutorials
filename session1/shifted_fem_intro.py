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
# We will aim to solve the Poisson problem with Dirichlet boundary conditions on a domain $\Omega$ which has a boundary
# $\Gamma$ that doesn't align with our background mesh.
#
# $$
# \begin{aligned}
# -\Delta u &= f \quad \text{in } \Omega, \\
# u &= u_G \quad \text{on } \Gamma.
# \end{aligned}
# $$
#
# The idea of the method is to shift the boundary conditions from a true boundary $\Gamma$ that is not aligned with the
# mesh to a surrogate boundary $\bar\Gamma$.
#
# %% [markdown]
# :::{iframe} ../pyvista/pyvista_submesh.html
# :width: 100%
# :title: Submesh
# :::
# %%
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

linear_cmap = dolfinx.fem.coordinate_element(true_surface.topology.cell_type, 1)
linear_lines = dolfinx.fem.interpolate_geometry(true_surface, linear_cmap)

# +[tags="hide-output"]
import pyvista as pv

surface_grid = pv.UnstructuredGrid(*dolfinx.plot.vtk_mesh(mesh))
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
# for the cells in the background mesh

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

values = np.full(num_cells_local, 0.0)
values[colliding_cells] = 1.0
initial_tag = dolfinx.mesh.meshtags(
    mesh, mesh.topology.dim, np.arange(num_cells_local, dtype=np.int32), values
)
cell_grid = pv.UnstructuredGrid(*dolfinx.plot.vtk_mesh(mesh))
cell_grid.cell_data["colliding"] = values
plotter_collision = pv.Plotter()
plotter_collision.add_mesh(
    cell_grid,
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
    25,
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

point_cloud = pv.PolyData(refined_surface_nodes)
plotter.add_mesh(point_cloud, label="Point cloud", point_size=10.0, color="green")
plotter.export_html("pyvista_pc_boundary.html")

# %% [markdown]
# :::{iframe} ../pyvista/pyvista_pc_boundary.html
# :width: 100%
# :title: True boundary $\Gamma$
# :::
# %%

# Now that we have a good representation of the curved boundary,
# to find all cells that are intersected by the true boundary $\Gamma$.
# We will use the [GJK distance algorithm](https://doi.org/10.1109/56.2083) to compute
# the distance between th convex hull that makes up each cell and the point cloud
# that represents the true boundary. If the distance is small, then we check
# if all the vertices of the cell are within the convex hull of the true boundary.

background_cell_nodes = mesh.geometry.x[mesh.geometry.dofmap][colliding_cells]
cell_indicator = dolfinx.la.vector(
    mesh.topology.index_map(mesh.topology.dim), bs=1, dtype=np.int32
)
KEEP_MARKER = 2
tol = 10 * np.finfo(refined_surface_nodes.dtype).eps
for cell, cell_geom in zip(colliding_cells, background_cell_nodes):
    distance = dolfinx.geometry.compute_distance_gjk(cell_geom, refined_surface_nodes)
    cell_indicator.array[cell] = np.linalg.norm(distance) < tol

    # For each cell that is marked, check that all vertices are inside the surface
    if cell_indicator.array[cell]:
        inside = True
        for point in cell_geom:
            sub_distance = dolfinx.geometry.compute_distance_gjk(
                point, refined_surface_nodes
            )
            if np.linalg.norm(sub_distance) > tol:
                inside = False
                break
        cell_indicator.array[cell] = inside
cell_indicator.scatter_reverse(dolfinx.la.InsertMode.add)
cell_indicator.scatter_forward()
cell_indicator = KEEP_MARKER * (cell_indicator.array > 0).astype(np.int32)
cell_tag = dolfinx.mesh.meshtags(
    mesh,
    mesh.topology.dim,
    np.arange(len(cell_indicator), dtype=np.int32),
    cell_indicator,
)

# Next we plot the new set of tagged cells, which is what we will use as the set of colliding cells for the rest of the tutorial.

cell_grid["refined_collisions"] = cell_tag.values
plotter_refined_collision = pv.Plotter()
plotter_refined_collision.add_mesh(
    cell_grid,
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

# ```{note}
# Note that GJK is only correct for convex shapes. Therefore, to adapt the method to non-convex shapes,
# one can for instance adapt the sub distance measure to use a
# [closest point projection method](https://scientificcomputing.github.io/scifem/docs/api.html#scifem.closest_point_projection)
# (for instance implemented in scifem) in combination with a signed distance or winding number computation.
# However, this is out of the scope of this tutorial.
# ```

# %% [markdown]
# :::{iframe} ../pyvista/pyvista_refined_collision.html
# :width: 100%
# :title: Refined colliding cells
# :::
# %%

## Creating the reduced mesh
# We use {py:func}`dolfinx.mesh.create_submesh` to restrict the problem to the marked cells only

submesh, entity_map, vertex_map, _ = dolfinx.mesh.create_submesh(
    mesh, mesh.topology.dim, cell_tag.find(KEEP_MARKER)
)
submesh.topology.create_connectivity(submesh.topology.dim - 1, submesh.topology.dim)
submesh_facets = dolfinx.mesh.exterior_facet_indices(submesh.topology)

submesh_grid = pv.UnstructuredGrid(*dolfinx.plot.vtk_mesh(submesh))
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

# %% [markdown]
# :::{iframe} ../pyvista/pyvista_submesh.html
# :width: 100%
# :title: Submesh
# :::
# %%

# ## The shifted boundary method
#
# We start creating $u_G$, the boundary condition on the true boundary $\Gamma$.
# Furthermore, we also define the tangential deriviative along the true boundary, which we will require in the variational formulation.

J = ufl.Jacobian(true_surface)
tangent_unscaled = ufl.as_vector([J[i, 0] for i in range(true_surface.geometry.dim)])
t = tangent_unscaled / ufl.sqrt(ufl.dot(tangent_unscaled, tangent_unscaled))
x_s, y_s = ufl.SpatialCoordinate(true_surface)
uG = x_s * ufl.cos(y_s)
duG_dt = ufl.dot(ufl.grad(uG), t)

# In the shifted boundary method, we require a map $M: \bar\Gamma \to \Gamma$ that maps points on the surrogate boundary $\bar\Gamma$ to the true boundary $\Gamma$.
# As we are working with finite element methods, we actually only need this map at the quadrature points of the surrogate boundary.
# We therefore create another submesh of the restricted mesh, which only contains the exterior facets of the submesh, which will be our surrogate boundary $\bar\Gamma$.

# Create a vector and scalar quadrature space for integration on the submesh
facet_qdeg = 4
facet_submesh, facet_map = dolfinx.mesh.create_submesh(
    submesh, submesh.topology.dim - 1, submesh_facets
)[:2]
q_surface = basix.ufl.quadrature_element(
    facet_submesh.basix_cell(),
    degree=facet_qdeg,
    value_shape=(facet_submesh.geometry.dim,),
)
Q_facet = dolfinx.fem.functionspace(facet_submesh, q_surface)
Q_facet_coords = Q_facet.tabulate_dof_coordinates()
d = dolfinx.fem.Function(Q_facet)

q_scalar_surface = basix.ufl.quadrature_element(
    facet_submesh.basix_cell(), degree=facet_qdeg, value_shape=()
)
Q_scalar_facet = dolfinx.fem.functionspace(facet_submesh, q_scalar_surface)
np.testing.assert_allclose(Q_scalar_facet.tabulate_dof_coordinates(), Q_facet_coords)
