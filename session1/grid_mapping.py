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
#
# # Computing the closest point mapping from the surrogate boundary to the true boundary
# In the previous sections, we have shown how to create the surrogate mesh, and how the variational form would look like on this grid.
# However, we have not talked about how to transfer data from one grid to the other.
# Recall that we would like to compute
#
# $$
# \begin{aligned}
# {\color{#009988}{\bar{\mathbf{n}}}}(\mathbf{\tilde x}) & \equiv \mathbf{n}(M(\mathbf{\tilde x})), \\
# {\color{#EE3377}\bar{\boldsymbol{\tau}}_i}(\mathbf{\tilde x}) & \equiv\boldsymbol{\tau}_i(M(\mathbf{\tilde x})), \\
# {\color{#56B4E9}\mathbf{d_M}}(\mathbf{\tilde x}) & = M(\mathbf{\tilde x}) - \mathbf{\tilde x}, \\
# {\color{#E69F00}\bar{u}_G}(\mathbf{\tilde x}) = u_G(M(\mathbf{\tilde x})),\\
# {\color{#DDCC77}\bar{u}_{G,\bar{\boldsymbol{\tau}}_i}}(\mathbf{\tilde x}) = \nabla {\color{#E69F00}\bar{u}_G}(\mathbf{\tilde x}) \cdot {\color{#EE3377}\bar{\boldsymbol{\tau}}_i}(\mathbf{\tilde x}).
# \end{aligned}
# $$
#
# where we require a map $M: \bar\Gamma \to \Gamma$ that maps points on the surrogate boundary $\bar\Gamma$ to the true boundary $\Gamma$.
# Therefore, in this section we will create the map $M(x)$ from the surrogate boundary to the real boundary.

# + tags=["remove-output"]

from mpi4py import MPI
from surrogate_grid_creation import (
    true_surface,
    facet_map,
    surrogate_mesh,
    surrogate_facetmesh,
    surrogate_mesh_pv,
    tessellated,
    linearized_surrogate_pv,
)
import tempfile
import ufl
import basix.ufl
import dolfinx
import numpy as np

# -

# As we are working with finite element methods, we actually only need this map at the quadrature points of the surrogate boundary.
# We therefore create another `surrogate_mesh` of the restricted mesh, which only contains the exterior facets of the `surrogate_mesh`,
# which will be our surrogate boundary $\bar\Gamma$.

# Create a vector and scalar quadrature space for integration on the `surrogate_mesh`

# +

facet_qdeg = 3
q_surface = basix.ufl.quadrature_element(
    surrogate_facetmesh.basix_cell(),
    degree=facet_qdeg,
    value_shape=(surrogate_facetmesh.geometry.dim,),
)
Q_facet = dolfinx.fem.functionspace(surrogate_facetmesh, q_surface)
Q_facet_coords = Q_facet.tabulate_dof_coordinates()
d = dolfinx.fem.Function(Q_facet)

q_scalar_surface = basix.ufl.quadrature_element(
    surrogate_facetmesh.basix_cell(), degree=facet_qdeg, value_shape=()
)
Q_scalar_facet = dolfinx.fem.functionspace(surrogate_facetmesh, q_scalar_surface)
np.testing.assert_allclose(Q_scalar_facet.tabulate_dof_coordinates(), Q_facet_coords)

# -

# For each of the quadrature points on the surrogate boundary, we can now use
# [compute_closest_entity](xref:dolfinx#dolfinx.geometry.compute_closest_entity) to compute the closest point on the true boundary.
# ```{note}
# Compute closest entity only gives you accurate results for truly convex shapes,
# as it relies on the [GJK distance algorithm](https://doi.org/10.1109/56.2083).
# Therefore, this only gives us a good estimation of the closest point on the true boundary, we
# will use a closest point project method.
# ```

surface_cells = np.arange(
    true_surface.topology.index_map(true_surface.topology.dim).size_local,
    dtype=np.int32,
)
surface_bb = dolfinx.geometry.bb_tree(
    true_surface, true_surface.topology.dim, entities=surface_cells, padding=1e-8
)
surface_midpoint_bb = dolfinx.geometry.create_midpoint_tree(
    true_surface, true_surface.topology.dim, entities=surface_cells
)
closest_surface_cell = dolfinx.geometry.compute_closest_entity(
    surface_bb, surface_midpoint_bb, true_surface, Q_facet_coords
)

# We now use the [closest point projection](xref:scifem#scifem.closest_point_projection) method from scifem,
# that leverages the Goldstein-Levitin-Polyak Gradient projection method,
# where potential simplex constraints are handled by an exact projection
# [Held 1974](https://doi.org/10.1007/BF01580223),
# [Bertsekas 1976](https://doi.org/10.1109/TAC.1976.1101194) and
# [Condat 2016](https://doi.org/10.1007/s10107-015-0946-6).

# +
from scifem import closest_point_projection

closest_points, reference_points = closest_point_projection(
    true_surface,
    closest_surface_cell,
    Q_facet_coords[:, : true_surface.geometry.dim].copy(),
    tol_x=1e-7,
)
# -

# As we talked about in the previous sections, we can now define ${\color{#56B4E9}\mathbf{d_M}}$

dM = dolfinx.fem.Function(Q_facet, name="dM")
dM.x.array[:] = (
    closest_points
    - Q_facet_coords[:, : true_surface.geometry.dim].reshape(
        -1, true_surface.geometry.dim
    )
).flatten()

# + tags=["hide-input"]

import pyvista as pv

q_cloud = pv.PolyData(Q_facet_coords)
dM_padded = np.zeros((Q_facet_coords.shape[0], 3))
dM_padded[:, : true_surface.geometry.dim] = dM.x.array[:].reshape(
    -1, true_surface.geometry.dim
)
q_cloud["dM"] = dM_padded

q_cloud["magnitude"] = np.linalg.norm(q_cloud["dM"], axis=1)

pl = pv.Plotter()

pl.add_mesh(tessellated, color="red", line_width=4, show_edges=True)
pl.add_mesh(surrogate_mesh_pv, color="black", style="wireframe", show_edges=True)
pl.add_mesh(
    linearized_surrogate_pv,
    color="red",
    style="points",
    point_size=25,
)
pl.add_mesh(q_cloud, point_size=10, color="blue", label="Quadrature points")

geom = pv.Arrow()
glyphs = q_cloud.glyph(orient="dM", scale="magnitude", factor=1, geom=geom)
pl.add_mesh(glyphs, color="red")
pl.view_xy()

pl.export_html("pyvista_closest_point.html")
# -

# %% [markdown]
# :::{iframe} ../pyvista/pyvista_closest_point.html
# :width: 100%
# :title: Closest points on $\Gamma$
# :::
# %%


# ## Defining data on $\Gamma$
#
# We start creating ${\color{#E69F00}\bar{u}_G}$, the boundary condition on the true boundary $\Gamma$.
# Furthermore, we also define the tangential derivative along the true boundary, which we will require in the variational formulation.

# +
def u_exact(x, mod):
    return x[0] + mod.sin(0.5 * mod.pi * x[1])


x_s = ufl.SpatialCoordinate(true_surface)
uG = u_exact(x_s, ufl)
# -

# Furthermore, as we require ${\color{#EE3377}\bar{\boldsymbol{\tau}}_i}$ and ${\color{#DDCC77}\bar{u}_{G,\bar{\boldsymbol{\tau}}_i}}$ in the variational formulation,
# we also need to compute the tangent vector along the true boundary,
# which we can do with the Jacobian of the surface mesh.

J = ufl.Jacobian(true_surface)
tangent_unscaled = ufl.as_vector([J[i, 0] for i in range(true_surface.geometry.dim)])
t = tangent_unscaled / ufl.sqrt(ufl.dot(tangent_unscaled, tangent_unscaled))
duG_dt = ufl.dot(ufl.grad(uG), t)


# Finally, we can now transfer the data from the true boundary to the surrogate boundary by using
# the closest point mapping $M$.
# There are multiple ways of doing this

# 1. Transfer [ufl-expressions](xref:ufl#ufl.core.expr.Expr) to a [Function](xref:dolfinx#dolfinx.fem.Function) on the true boundary, then use [Function.eval](xref:dolfinx#dolfinx.fem.Function.eval) to evaluate the function at the quadrature points on the surrogate boundary.

# Create functions to store the data in on the true boundary

bar_tangent_func = dolfinx.fem.Function(Q_facet, name="true_tangent")
bar_duG_t = dolfinx.fem.Function(Q_scalar_facet, name="true_duGt")
bar_uG = dolfinx.fem.Function(Q_scalar_facet, name="true_uG")

# Create UFL expressions that can be used with [interpolation points](xref:dolfinx#dolfinx.fem.Function.interpolate)

gdim = true_surface.geometry.dim
grid_cmap = true_surface.geometry.cmaps[0]
T_tmp = dolfinx.fem.functionspace(true_surface, ("DG", grid_cmap.degree, (gdim,)))
t_expr = dolfinx.fem.Expression(t, T_tmp.element.interpolation_points)
t_approx = dolfinx.fem.Function(T_tmp)
t_approx.interpolate(t_expr)
T_tmp_scalar = dolfinx.fem.functionspace(true_surface, ("DG", 4))
uG_expr = dolfinx.fem.Expression(uG, T_tmp_scalar.element.interpolation_points)
duG_t_expr = dolfinx.fem.Expression(
    duG_dt,
    T_tmp_scalar.element.interpolation_points,
)
uG_approx = dolfinx.fem.Function(T_tmp_scalar)
uG_approx.interpolate(uG_expr)
duG_t_approx = dolfinx.fem.Function(T_tmp_scalar)
duG_t_approx.interpolate(duG_t_expr)

# Transfer functions from true boundary to surrogate boundary

padded_closest_points = np.zeros((closest_points.shape[0], 3))
padded_closest_points[:, : true_surface.geometry.dim] = closest_points
bar_tangent_func.x.array[:] = (
    t_approx.eval(padded_closest_points, closest_surface_cell)
).flatten()

# As we used an intermediate space for the tangent, it doesn't necessarily have magnitude 1.
# Therefore we rescale it.

btf = bar_tangent_func.x.array[:].reshape(-1, gdim)
btf /= np.linalg.norm(bar_tangent_func.x.array.reshape(-1, gdim), axis=1)[:, None]

bar_duG_t.x.array[:] = duG_t_approx.eval(
    padded_closest_points,
    closest_surface_cell,
).flatten()
bar_uG.x.array[:] = uG_approx.eval(
    padded_closest_points, closest_surface_cell
).flatten()

# + tags=["hide-input"]
pl_data = pv.Plotter()
pl_data.add_mesh(tessellated, color="red", style="wireframe", show_edges=True)
pl_data.add_mesh(surrogate_mesh_pv, color="black", style="wireframe", show_edges=True)
tangent_padded = np.zeros((Q_facet_coords.shape[0], 3))
tangent_padded[:, : true_surface.geometry.dim] = btf
q_cloud["tangent"] = tangent_padded
glyphs = q_cloud.glyph(orient="tangent", scale="tangent", factor=0.05, geom=geom)
pl_data.add_mesh(glyphs)
pl_data.view_xy()
pl_data.export_html("pyvista_closest_point_data.html")
# -
# %% [markdown]
# :::{iframe} ../pyvista/pyvista_closest_point_data.html
# :width: 100%
# :title: Data on $\Gamma$
# :::


# 2. Create a [dolfinx.fem.Expression](xref:dolfinx#dolfinx.fem.Expression) that can be evaluated at the quadrature points on the surrogate boundary,
# and directly evaluate the expression at these points.

# If we do not want to go through an intermediate space, we can compile the [UFL-expression](xref:ufl#ufl.core.expr.Expr)
# directly into a [dolfinx.fem.Expression](xref:dolfinx#dolfinx.fem.Expression) and evaluate it at the quadrature points on the surrogate boundary.
# However, this procedure is quite costly, as we need to compile a separate expression for each quadrature point stemming from the
# closest point projection.
# Furthermore, as each process will require different points, we have to compile unique expressions per process.
# We therefore use the `MPI.COMM_SELF` communicator when initializing the expression, as well as creating a temporary cache
# directory for the generated C++ code, that is unique to each process. We use a temporary directory to avoid
# littering the system with tons of entries, which would reduce the overall performance of form compilation.

bar_tangent_func_expr = dolfinx.fem.Function(Q_facet, name="true_tangent")
bar_duG_t_expr = dolfinx.fem.Function(Q_scalar_facet, name="true_duGt")
bar_uG_expr = dolfinx.fem.Function(Q_scalar_facet, name="true_uG")
gdim = true_surface.geometry.dim
with tempfile.TemporaryDirectory(ignore_cleanup_errors=False, delete=True) as cache_dir:
    jit_options = {"cache_dir": cache_dir}
    for i, (ref_point, cell) in enumerate(zip(reference_points, closest_surface_cell)):
        # Combine expression to speed up compilation
        vec = [t[k] for k in range(gdim)]
        vec.append(uG)
        vec.append(duG_dt)
        combined_expr = ufl.as_vector(vec)
        combined_local_expr = dolfinx.fem.Expression(
            combined_expr, ref_point, comm=MPI.COMM_SELF, jit_options=jit_options
        )
        combined_vals = combined_local_expr.eval(
            true_surface, np.asarray([cell], dtype=np.int32)
        ).flatten()
        bar_tangent_func_expr.x.array[i * gdim : (i + 1) * gdim] = combined_vals[:gdim]
        bar_uG_expr.x.array[i] = combined_vals[gdim]
        bar_duG_t_expr.x.array[i] = combined_vals[gdim + 1]

# + tags=["hide-input"]
pl_data = pv.Plotter()
pl_data.add_mesh(tessellated, color="red", style="wireframe", show_edges=True)
pl_data.add_mesh(surrogate_mesh_pv, color="black", style="wireframe", show_edges=True)
tangent_expr_padded = np.zeros((Q_facet_coords.shape[0], 3))
tangent_expr_padded[:, : true_surface.geometry.dim] = bar_tangent_func_expr.x.array[
    :
].reshape(-1, gdim)
q_cloud_expr = pv.PolyData(Q_facet_coords)
q_cloud_expr["tangent_expr"] = tangent_expr_padded
glyphs_expr = q_cloud_expr.glyph(
    orient="tangent_expr", scale="tangent_expr", factor=0.05, geom=geom
)
pl_data.add_mesh(glyphs_expr)
pl_data.view_xy()
pl_data.export_html("pyvista_closest_point_data_expr.html")
# -

# Now that we have moved all the data to the surrogate boundary, we can use this data in the variational formulation!

# %% [markdown]
# :::{iframe} ../pyvista/pyvista_closest_point_data_expr.html
# :width: 100%
# :title: Data on $\Gamma$
# :::
