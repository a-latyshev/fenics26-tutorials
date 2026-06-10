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
# Therefore, in this section we will create the map $M(x)$ from the surrogate boundary to the real boundary.

# + tags=["remove-output"]

from surrogate_grid_creation import (
    true_surface,
    submesh,
    facet_submesh,
    facet_map,
    submesh_grid,
    tessellated,
    linearized_surface_grid,
)
import ufl
import basix.ufl
import dolfinx
import numpy as np

# -

# ## Defining data on $\Gamma$
#
# We start creating $\uG$, the boundary condition on the true boundary $\Gamma$.
# Furthermore, we also define the tangential deriviative along the true boundary, which we will require in the variational formulation.

x_s, y_s = ufl.SpatialCoordinate(true_surface)
uG = x_s * ufl.cos(y_s)

# Furthermore, as we require $\bartau$ and $\duGtau$ in the variational formulation,
# we also need to compute the tangent vector along the true boundary,
# which we can do with the Jacobian of the surface mesh.

J = ufl.Jacobian(true_surface)
tangent_unscaled = ufl.as_vector([J[i, 0] for i in range(true_surface.geometry.dim)])
t = tangent_unscaled / ufl.sqrt(ufl.dot(tangent_unscaled, tangent_unscaled))
duG_dt = ufl.dot(ufl.grad(uG), t)

# In the shifted boundary method, we require a map $M: \bar\Gamma \to \Gamma$ that maps points on the surrogate boundary $\bar\Gamma$ to the true boundary $\Gamma$.
# As we are working with finite element methods, we actually only need this map at the quadrature points of the surrogate boundary.
# We therefore create another submesh of the restricted mesh, which only contains the exterior facets of the submesh, which will be our surrogate boundary $\bar\Gamma$.

# Create a vector and scalar quadrature space for integration on the submesh

facet_qdeg = 3
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

# For each of the quadrature points on the surrogate boundary, we can now use
# [compute_closest_entity](xref:dolfinx#dolfinx.geometry.compute_closest_entity) to compute the closest point on the true boundary.
# ```{note}
# Compute closest entity only gives you accurate results for truely convex shapes,
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

from scifem import closest_point_projection

closest_points, reference_points = closest_point_projection(
    true_surface,
    closest_surface_cell,
    Q_facet_coords[:, : true_surface.geometry.dim].copy(),
    tol_x=1e-8,
)

# + tags=["hide-input"]

import pyvista as pv

padded_closest_points = np.zeros((closest_points.shape[0], 3))
padded_closest_points[:, : true_surface.geometry.dim] = closest_points
q_cloud = pv.PolyData(Q_facet_coords)
q_cloud["vectors"] = padded_closest_points - Q_facet_coords

q_cloud["magnitude"] = np.linalg.norm(q_cloud["vectors"], axis=1)

pl = pv.Plotter()

pl.add_mesh(tessellated, color="red", line_width=4, show_edges=True)
pl.add_mesh(submesh_grid, color="black", style="wireframe", show_edges=True)
pl.add_mesh(
    linearized_surface_grid,
    color="red",
    style="points",
    point_size=25,
)
pl.add_mesh(q_cloud, point_size=10, color="blue", label="Quadrature points")

geom = pv.Arrow()
glyphs = q_cloud.glyph(orient="vectors", scale="magnitude", factor=1, geom=geom)
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
