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
# # Putting it all together
# We can now use the data from the previous sections to solve the Poisson problem with the shifted boundary method.
#
# Find $u^h \in V(\tilde\Omega)$ such that
#
# $$
# \begin{aligned}
# a(u^h, v^h) &= L(v^h) \quad \forall v^h \in V(\tilde\Omega), \\
# a(u, v) & = \left(\nabla u, \nabla v\right)_{\tilde\Omega}
# - \left(\nabla u \cdot \tilde{\mathbf{n}}, v + \nabla v \cdot {\color{#56B4E9}\mathbf{d_M}}\right)_{\bar\Gamma}\\
# &- \left(u + \nabla u \cdot {\color{#56B4E9}\mathbf{d_M}}, \nabla v \cdot \tilde{\mathbf{n}}\right)_{\bar\Gamma}
# + \left(({\color{#009988}{\bar{\mathbf{n}}}}\cdot \tilde{\mathbf{n}})/\vert\vert {\color{#56B4E9}\mathbf{d_M}}\vert\vert \nabla u \cdot {\color{#56B4E9}\mathbf{d_M}}, \nabla v \cdot {\color{#56B4E9}\mathbf{d_M}}\right)_{\bar\Gamma}, \\
# &+ \left(\alpha/h u + \nabla u \cdot {\color{#56B4E9}\mathbf{d_M}}, v + \nabla v \cdot {\color{#56B4E9}\mathbf{d_M}}\right)_{\bar\Gamma}, \\
# L(v) & = \left(f, v\right)_{\tilde\Omega} - \left({\color{#E69F00}\bar{u}_G}, \nabla v \cdot \tilde{\mathbf{n}}\right)_{\bar\Gamma}
# - \left({\color{#DDCC77}\bar{u}_{G,\bar{\boldsymbol{\tau}}_i}}({\color{#EE3377}\bar{\boldsymbol{\tau}}_i} \cdot \tilde{\mathbf{n}}), \nabla v \cdot {\color{#56B4E9}\mathbf{d_M}}\right)_{\bar\Gamma}\\
# &+ \left(\alpha/h {\color{#E69F00}\bar{u}_G}, v + \nabla v \cdot {\color{#56B4E9}\mathbf{d_M}}\right)_{\bar\Gamma}.
# \end{aligned}
# $$

from mpi4py import MPI
from grid_mapping import (
    surrogate_mesh,
    facet_map,
    tessellated,
    bar_tangent_func_expr,
    bar_duG_t_expr,
    bar_uG_expr,
    u_exact,
    dM,
    facet_qdeg,
)
import dolfinx
import ufl
import numpy as np
import pyvista as pv

# +
V = dolfinx.fem.functionspace(surrogate_mesh, ("Lagrange", 1))

u = ufl.TrialFunction(V)
w = ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(w)) * ufl.dx

x = ufl.SpatialCoordinate(surrogate_mesh)
f = -ufl.div(ufl.grad(u_exact(x, ufl)))
L = ufl.inner(f, w) * ufl.dx

t_bar = bar_tangent_func_expr
duG_t = bar_duG_t_expr
uG = bar_uG_expr

# -

# The normal vector on the true boundary is derived from the closest point project from the surrogate boundary to the true boundary.

d_scalar = ufl.sqrt(ufl.dot(dM, dM))
n_bar = dM / d_scalar

# Next we can define the integration measure over the surrogate boundary

nt = ufl.FacetNormal(surrogate_mesh)
dsG = ufl.Measure(
    "ds", domain=surrogate_mesh, metadata={"quadrature_degree": facet_qdeg}
)

# and define the remainder of the variational formulation as done in [the introduction](./shifted_fem_intro)


# +
def shift(z, d):
    return z + ufl.dot(ufl.grad(z), d)


alpha = dolfinx.fem.Constant(surrogate_mesh, 10.0)
sub_cell_map = surrogate_mesh.topology.index_map(surrogate_mesh.topology.dim)
h_vars = surrogate_mesh.h(
    surrogate_mesh.topology.dim,
    np.arange(sub_cell_map.size_local + sub_cell_map.num_ghosts, dtype=np.int32),
)
Q = dolfinx.fem.functionspace(surrogate_mesh, ("DG", 0))
h = dolfinx.fem.Function(Q)
h.x.array[:] = h_vars

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
# -

# We use the standard DOLFINx solver interface, with the addition of the [EntityMap](xref:dolfinx#dolfinx.mesh.EntityMap)

problem = dolfinx.fem.petsc.LinearProblem(
    a,
    L,
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "ksp_error_if_not_converged": True,
    },
    petsc_options_prefix="surrogate_solver",
    entity_maps=[facet_map],
)
_ = problem.solve()

# We compute the L2 error within the surrogate domain and visualize it

# +
u_ex = u_exact(x, ufl)
diff = problem.u - u_ex
L2_error = dolfinx.fem.form(ufl.inner(diff, diff) * ufl.dx)

local_error = dolfinx.fem.assemble_scalar(L2_error)
global_error = np.sqrt(surrogate_mesh.comm.allreduce(local_error, op=MPI.SUM))

print(f"L2-error: {global_error:.4e}")
# -

# + tags=["hide-input"]
sg_t, sg_ct, sg_g = dolfinx.plot.vtk_mesh(V)
sg_ct[:] = (
    pv.CellType.QUAD
    if surrogate_mesh.topology.cell_type == dolfinx.mesh.CellType.quadrilateral
    else sg_ct
)
sol_plt = pv.UnstructuredGrid(sg_t, sg_ct, sg_g)
sol_plt.point_data["u"] = problem.u.x.array
plotter_sol = pv.Plotter(shape=(1, 2))
plotter_sol.subplot(0, 0)
plotter_sol.add_mesh(sol_plt, show_edges=True, cmap="Reds")

tessellated.point_data["uG"] = u_exact(tessellated.points.T, np)
tessellated.set_active_scalars("uG")
plotter_sol.add_mesh(tessellated, line_width=10, cmap="Reds")
plotter_sol.view_xy()

plotter_sol.subplot(0, 1)
diff = dolfinx.fem.Function(V)
diff.interpolate(dolfinx.fem.Expression(u_ex, V.element.interpolation_points))
diff.x.array[:] -= problem.u.x.array
sol_plt.point_data["|error|"] = np.abs(diff.x.array)
sol_plt.set_active_scalars("|error|")
plotter_sol.add_mesh(sol_plt, show_edges=True, cmap="Reds")

plotter_sol.link_views()
plotter_sol.export_html("pyvista_surrogate_sol.html")
# -

# %% [markdown]
# :::{iframe} ../pyvista/pyvista_surrogate_sol.html
# :width: 100%
# :title: Closest points on $\Gamma$
# :::
# %%
