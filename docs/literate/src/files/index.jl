# # Tutorials for Trixi.jl

# The tutorial section for [Trixi.jl](https://github.com/trixi-framework/Trixi.jl) also contains
# interactive step-by-step explanations via [Binder](https://mybinder.org).

# Right now, you are using the classic documentation. The corresponding interactive notebooks can
# be opened in [Binder](https://mybinder.org/) and viewed in [nbviewer](https://nbviewer.jupyter.org/)
# via the icons ![](https://mybinder.org/badge_logo.svg) and ![](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)
# in the respective tutorial.
# You can download the raw notebooks from GitHub via ![](https://camo.githubusercontent.com/aea75103f6d9f690a19cb0e17c06f984ab0f472d9e6fe4eadaa0cc438ba88ada/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f646f776e6c6f61642d6e6f7465626f6f6b2d627269676874677265656e).
# **Note:** To improve responsiveness via caching, the notebooks are updated only once a week. They are only
# available for the latest stable release of Trixi at the time of caching.
# There are tutorials for the following topics:

# ### [1 Introduction to DG methods](@ref scalar_linear_advection_1d)
#-
# This tutorial gives an introduction to discontinuous Galerkin (DG) methods with the example of the 
# scalar linear advection equation in 1D. Starting with some theoretical explanations, we first implement 
# a raw version of a discontinuous Galerkin spectral element method (DGSEM). Then, we will show how
# to use features of Trixi.jl to achieve the same result.

# ### 2 Adding a new equation
# #### [2.1 Scalar conservation law](@ref cubic_conservation_law)
#-
# This tutorial explains how to add a new physics model using the example of the cubic conservation
# law. First, we define the equation using a `struct` `CubicEquation` and the physical flux. Then,
# the corresponding standard setup in Trixi.jl (`mesh`, `solver`, `semi` and `ode`) is implemented
# and the ODE problem is solved by OrdinaryDiffEq's `solve` method.

# #### [2.2 Nonconservative advection](@ref nonconservative_advection)
#-
# In this part, another physics model is implemented, the nonconservative linear advection equation.
# We run two different simulations with different levels of refinement and compare the resulting errors.

# ### [3 Differentiable programming](@ref differentiable_programming)
#-
# This part deals with some basic differentiable programming topics. For example, a Jacobian, its
# eigenvalues and a curve of total energy (through the simulation) are calculated and plotted for
# a few semidiscretizations. Moreover, we calculate an example for propagating errors with Measurement.jl
# at the end.

# ### [4 Unstructured meshes with HOHQMesh.jl](@ref hohqmesh_tutorial)
#-
# The purpose of this tutorial is to demonstrate how to use the [`UnstructuredMesh2D`](@ref)
# functionality of Trixi.jl. This begins by running and visualizing an available unstructured
# quadrilateral mesh example. Then, the tutorial will demonstrate how to conceptualize a problem
# with curved boundaries, generate a curvilinear mesh using the available [HOHQMesh](https://github.com/trixi-framework/HOHQMesh)
# software in the Trixi.jl ecosystem, and then run a simulation using Trixi.jl on said mesh.
