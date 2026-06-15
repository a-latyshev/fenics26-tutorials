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
# :::{iframe} ../pyvista/pyvista_surrogate_mesh.html
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

# However, we require $\dM$, $\nb$, $\bartau$, $\uG$ and $\duGtau$ to be implemented on the boundary of the domain.
# For this we will use a `surrogate_mesh` of the original mesh, which only contain the exterior facets.
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


# +
def shift(z, d):
    return z + ufl.dot(ufl.grad(z), d)


alpha = ufl.Constant(OmegaG)
h = ufl.Coefficient(Q_scalar)
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

# With this formulation we are in theory ready to solve the problem with a shifted method.
# However, three questions remain:
# 1. How do we create the surrogate mesh $\tilde{\Omega}$ from a background mesh $\mathcal{K}_h$ and
#    the true boundary $\Gamma_h$?
# 2. How do we compute the map $M$ mapping each quadrature point on the surrogate boundary to the closest point on
#    $\Gamma_h$?
# 3. How do we transfer the associated quantities $\dM$, $\nb$, $\bartau$, $\uG$ and $\duGtau$?
#    to the facet surrogate_mesh?
