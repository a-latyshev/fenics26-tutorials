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
# The mathematical formulation of the problem can be written as follows:
# Given a domain $\tilde\Omega$ with exterior boundary $\bar\Gamma$ and a map $M: \bar\Gamma \to \Gamma$,
# we define:
#
# $$
# \begin{aligned}
# \nb(\xtilde) & \equiv \mathbf{n}(M(\xtilde)), \\
# \bartau(\xtilde) & \equiv\boldsymbol{\tau}_i(M(\xtilde)), \\
# \dM(\xtilde) & = M(\xtilde) - \xtilde, \\
# \psi_{,\bar z}(\xtilde) = \nabla \psi(\xtilde) \cdot \bar{\mathbf{z}}
# & = \nabla \psi(\xtilde) \cdot \mathbf{z}(M(\xtilde)),
# \quad \text{for } \mathbf{z} = \bar{\mathbf{n}}, \bar{\boldsymbol{\tau}}_i.
# \end{aligned}
# $$
#
# where $\mathbf{n}$ is the normal vector and $i$th tangent vector on the true boundary $\Gamma$.
#
# The extension operator of a function $\phi$ on $\Gamma$ to $\bar\Gamma$ is defined as
# $\bar{\phi}(\mathbf{\tilde x}) \equiv \phi(M(\xtilde))$.
#
# and therefore for the boundary condition, we can write
#
# $$
# \uG(\mathbf{\tilde x}) = u_G(M(\xtilde)),\\
# \duGtau(\mathbf{\tilde x}) = \nabla \uG(\xtilde) \cdot \bartau(\xtilde).
# $$
#
# The full derivation of the variational formulation can be found in [the original paper](https://doi.org/10.1016/j.jcp.2017.10.026),
# and can be written as:
# Find $u^h \in V(\tilde\Omega)$ such that
#
# $$
# \begin{aligned}
# a(u^h, v^h) &= L(v^h) \quad \forall v^h \in V(\tilde\Omega), \\
# a(u, v) & = \intO{\nabla u}{\nabla v}
# - \intG{\nabla u \cdot \nt}{v + \nabla v \cdot \dM}\\
# &- \intG{u + \nabla u \cdot \dM}{\nabla v \cdot \nt}
# + \intG{(\nb\cdot \nt)/\vert\vert \dM\vert\vert \nabla u \cdot \dM}{\nabla v \cdot \dM}, \\
# &+ \intG{\alpha/h u + \nabla u \cdot \dM}{v + \nabla v \cdot \dM}, \\
# L(v) & = \intO{f}{v} - \intG{\uG}{\nabla v \cdot \nt}
# - \intG{\duGtau(\bartau \cdot \nt)}{\nabla v \cdot \dM}\\
# &+ \intG{\alpha/h \uG}{v + \nabla v \cdot \dM}.
# \end{aligned}
# $$

# Assuming we have the mesh above, we can start to implement this in UFL as follows

# +
import ufl
import basix.ufl

cell = "quadrilateral"
c_el = basix.ufl.element("Lagrange", cell, 1, shape=(2,))
OmegaG = ufl.Mesh(c_el)
el = basix.ufl.element("Lagrange", cell, 1)
V = ufl.FunctionSpace(OmegaG, el)

u = ufl.TrialFunction(V)
w = ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(w)) * ufl.dx

x = ufl.SpatialCoordinate(OmegaG)
f = ufl.sin(x[0]) * ufl.cos(x[1])  # Some spatially varying expression
L = ufl.inner(f, w) * ufl.dx
# -

# However, we require $\dM$, $\nt$, $\bartau$, $\uG$ and $\duGtau$ to be implemented on the boundary of the domain.
# For this we will use a submesh of the original mesh, which only contain the exterior facets.
# For now, this can symbolically be defined as

facet = "interval"
f_el = basix.ufl.element("Lagrange", facet, 1, shape=(2,))
GammaG = ufl.Mesh(f_el)

# Furthermore, we only require these quantities at the quadrature points used in the numerical integration,
# and therefore we define the following abstract functions

# +
quadrature_degree = 4
q_el = basix.ufl.quadrature_element(facet, degree=quadrature_degree, value_shape=(2,))
q_scalar_el = basix.ufl.quadrature_element(
    facet, degree=quadrature_degree, value_shape=()
)

Q_scalar = ufl.FunctionSpace(GammaG, q_scalar_el)
uG = ufl.Coefficient(Q_scalar)
duG_t = ufl.Coefficient(Q_scalar)

Q = ufl.FunctionSpace(GammaG, q_el)
dM = ufl.Coefficient(Q)
t_bar = ufl.Coefficient(Q)
# -

# The normal vector on the true boundary is derived from the closest point project from the surrogate boundary to the true boundary.

d_scalar = ufl.sqrt(ufl.dot(dM, dM))
n_bar = dM / d_scalar

# Next we can define the integration measure over the surrogate boundary

nt = ufl.FacetNormal(OmegaG)
dsG = ufl.Measure(
    "ds", domain=OmegaG, metadata={"quadrature_degree": quadrature_degree}
)

# and define the remainder of the variational formulation

alpha = ufl.Constant(OmegaG)
h = ufl.Coefficient(Q_scalar)


def shift(z, d):
    return z + ufl.dot(ufl.grad(z), d)


a -= ufl.inner(ufl.dot(ufl.grad(u), nt), shift(w, dM)) * dsG
a -= ufl.inner(shift(u, dM), ufl.dot(ufl.grad(w), nt)) * dsG
a += (
    ufl.dot(nt, n_bar)
    / d_scalar
    * ufl.inner(ufl.dot(ufl.grad(u), dM), ufl.dot(ufl.grad(w), dM))
    * dsG
)

a += alpha / h * ufl.inner(shift(u, dM), shift(w, dM)) * dsG
L -= ufl.inner(uG, ufl.dot(ufl.grad(w), nt)) * dsG
L += alpha / h * ufl.inner(uG, shift(w, dM)) * dsG
L -= ufl.inner(ufl.dot(duG_t * t_bar, nt), ufl.dot(ufl.grad(w), dM)) * dsG


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

# Next, we define the characteristics of the line mesh and the number of elements $M$
# and the Lagrange degree of the elements.

line_degree = 3
center = (1.1, 0.9)
scale = 0.05
M = 12

# We define the number of nodes in the mesh, which depends on the number of elements and the degree of the Lagrange polynomials.
# We then define the coordinates of the nodes, which are placed equidistantly on the ellipse.
#  Finally, we define the connectivity of the mesh, which describes how the nodes are connected to form elements.

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
# - `surface_comm`: The true boundary $\Gamma$ is represented on every process and is not distributed it across processes,
# which ensured by using `MPI.COMM_SELF` as the communicator for the mesh.
# - `ghost_mode`: As the mesh is not distributed, we set the `ghost_mode` to `GhostMode.none`.
# - `max_facet_to_cell_links`: Indicates that at max two cells can be connected to **any** facet, which is the case for
# this mesh, but not for T-joint grids or graph based meshes.
# ```

# Furthermore, we create the structured grid we will perform simulations on
domain = ((-0.1, -0.2), (2.1, 2.0))
nx = ny = 37
mesh = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD, domain, (nx, ny), cell_type=dolfinx.mesh.CellType.quadrilateral
)

# To illustrate that the true boundary is curved, we use the function `interpolate_geometry` to interpolate
# the geometry into a first order space, i.e. remove all nodes that are not vertices

linear_cmap = dolfinx.fem.coordinate_element(true_surface.topology.cell_type, 1)
linear_lines = dolfinx.fem.interpolate_geometry(true_surface, linear_cmap)

# +[tags="hide-output"]
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
# the distance between th convex hull that makes up each cell of the background mesh
# and the point cloud that represents the true boundary.

background_cell_nodes = mesh.geometry.x[mesh.geometry.dofmap][colliding_cells]
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
# We use [create_submesh](xref:dolfinx#mesh.create_submesh) to restrict the problem to the marked cells only

submesh, entity_map, vertex_map, _ = dolfinx.mesh.create_submesh(
    mesh, mesh.topology.dim, cell_tag.find(KEEP_MARKER)
)
submesh.topology.create_connectivity(submesh.topology.dim - 1, submesh.topology.dim)
submesh_facets = dolfinx.mesh.exterior_facet_indices(submesh.topology)

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
