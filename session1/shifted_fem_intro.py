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
# {\color{#009988}{\bar{\mathbf{n}}}}(\mathbf{\tilde x}) & \equiv \mathbf{n}(M(\mathbf{\tilde x})), \\
# {\color{#EE3377}\bar{\boldsymbol{\tau}}_i}(\mathbf{\tilde x}) & \equiv\boldsymbol{\tau}_i(M(\mathbf{\tilde x})), \\
# {\color{#56B4E9}\mathbf{d_M}}(\mathbf{\tilde x}) & = M(\mathbf{\tilde x}) - \mathbf{\tilde x}, \\
# \psi_{,\bar z}(\mathbf{\tilde x}) = \nabla \psi(\mathbf{\tilde x}) \cdot \bar{\mathbf{z}}
# & = \nabla \psi(\mathbf{\tilde x}) \cdot \mathbf{z}(M(\mathbf{\tilde x})),
# \quad \text{for } \mathbf{z} = \bar{\mathbf{n}}, \bar{\boldsymbol{\tau}}_i.
# \end{aligned}
# $$
#
# where $\mathbf{n}$ is the normal vector and $i$th tangent vector on the true boundary $\Gamma$.
#
# The extension operator of a function $\phi$ on $\Gamma$ to $\bar\Gamma$ is defined as
# $\bar{\phi}(\mathbf{\tilde x}) \equiv \phi(M(\mathbf{\tilde x}))$.
#
# and therefore for the boundary condition, we can write
#
# $$
# {\color{#E69F00}\bar{u}_G}(\mathbf{\tilde x}) = u_G(M(\mathbf{\tilde x})),\\
# {\color{#DDCC77}\bar{u}_{G,\bar{\boldsymbol{\tau}}_i}}(\mathbf{\tilde x}) = \nabla {\color{#E69F00}\bar{u}_G}(\mathbf{\tilde x}) \cdot {\color{#EE3377}\bar{\boldsymbol{\tau}}_i}(\mathbf{\tilde x}).
# $$
#
# The full derivation of the variational formulation can be found in [the original paper](https://doi.org/10.1016/j.jcp.2017.10.026),
# and can be written as:
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

# However, we require ${\color{#56B4E9}\mathbf{d_M}}$, ${\color{#009988}{\bar{\mathbf{n}}}}$, ${\color{#EE3377}\bar{\boldsymbol{\tau}}_i}$, ${\color{#E69F00}\bar{u}_G}$ and ${\color{#DDCC77}\bar{u}_{G,\bar{\boldsymbol{\tau}}_i}}$ to be implemented on the boundary of the domain.
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
# 3. How do we transfer the associated quantities ${\color{#56B4E9}\mathbf{d_M}}$, ${\color{#009988}{\bar{\mathbf{n}}}}$, ${\color{#EE3377}\bar{\boldsymbol{\tau}}_i}$, ${\color{#E69F00}\bar{u}_G}$ and ${\color{#DDCC77}\bar{u}_{G,\bar{\boldsymbol{\tau}}_i}}$?
#    to the facet surrogate_mesh?
