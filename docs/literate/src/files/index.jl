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

# ### [2 DGSEM with flux differencing](@ref DGSEM_FluxDiff)
#-
# To improve stability often the flux differencing formulation of the DGSEM (split form) is used.
# We want to present the idea and formulation on a basic 1D level. Then, we show how this formulation
# can be implemented in Trixi and analyse entropy conservation for two different flux combinations.

# ### [3 Shock capturing with flux differencing and stage limiter](@ref shock_capturing)
#-
# Using the flux differencing formulation, a simple procedure to capture shocks is a hybrid blending
# of a high-order DG method and a low-order subcell finite volume (FV) method. We present the idea on a
# very basic level and show the implementation in Trixi. Then, a positivity preserving limiter is
# explained and added to an exemplary simulation of the Sedov blast wave with the 2D compressible Euler
# equations.

# ### [4 Curved structured mesh with mapping](@ref structured_mesh_mapping)
#-
# In this tutorial, the use of Trixi's structured curved mesh type [`StructuredMesh`](@ref) is explained.
# We present the two basic option to initialize such a mesh. First, the curved domain boundaries
# of a circular cylinder are set by explicit boundary functions. Then, a fully curved mesh is
# defined by passing the transformation mapping.

# ### 5 Adding a new equation
# #### [5.1 Scalar conservation law](@ref cubic_conservation_law)
#-
# This tutorial explains how to add a new physics model using the example of the cubic conservation
# law. First, we define the equation using a `struct` `CubicEquation` and the physical flux. Then,
# the corresponding standard setup in Trixi.jl (`mesh`, `solver`, `semi` and `ode`) is implemented
# and the ODE problem is solved by OrdinaryDiffEq's `solve` method.

# #### [5.2 Nonconservative advection](@ref nonconservative_advection)
#-
# In this part, another physics model is implemented, the nonconservative linear advection equation.
# We run two different simulations with different levels of refinement and compare the resulting errors.

# ### [6 Differentiable programming](@ref differentiable_programming)
#-
# This part deals with some basic differentiable programming topics. For example, a Jacobian, its
# eigenvalues and a curve of total energy (through the simulation) are calculated and plotted for
# a few semidiscretizations. Moreover, we calculate an example for propagating errors with Measurement.jl
# at the end.

# ### [7 Unstructured meshes with HOHQMesh.jl](@ref hohqmesh_tutorial)
#-
# The purpose of this tutorial is to demonstrate how to use the [`UnstructuredMesh2D`](@ref)
# functionality of Trixi.jl. This begins by running and visualizing an available unstructured
# quadrilateral mesh example. Then, the tutorial will demonstrate how to conceptualize a problem
# with curved boundaries, generate a curvilinear mesh using the available [HOHQMesh](https://github.com/trixi-framework/HOHQMesh)
# software in the Trixi.jl ecosystem, and then run a simulation using Trixi.jl on said mesh.


# ## Examples in Trixi
# Trixi already contains several more coding examples, the so-called `elixirs`. You can find them
# in the folder [`examples`](https://github.com/trixi-framework/Trixi.jl/blob/main/examples/).
# They are structured by the underlying mesh type and the respective number of spatial dimensions.
# The name of an elixir is composed of the underlying system of conservation equations (for instance
# `advection` or `euler`) and other special characteristics like the initial condition
# (e.g. `gauss`, `astro_jet`, `blast_wave`) or the included simulation features
# (e.g. `amr`, `shockcapturing`).
