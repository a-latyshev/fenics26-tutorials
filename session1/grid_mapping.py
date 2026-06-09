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

from surrogate_grid_creation import true_surface, submesh, facet_submesh, facet_map
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

facet_qdeg = 4
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
