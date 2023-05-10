# # Tutorials for Trixi.jl

# The tutorial section for [Trixi.jl](https://github.com/trixi-framework/Trixi.jl) also contains
# interactive step-by-step explanations via [Binder](https://mybinder.org).

# Right now, you are using the classic documentation. The corresponding interactive notebooks can
# be opened in [Binder](https://mybinder.org/) and viewed in [nbviewer](https://nbviewer.jupyter.org/)
# via the icons ![](https://mybinder.org/badge_logo.svg) and ![](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)
# in the respective tutorial.
# You can download the raw notebooks from GitHub via ![](https://camo.githubusercontent.com/aea75103f6d9f690a19cb0e17c06f984ab0f472d9e6fe4eadaa0cc438ba88ada/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f646f776e6c6f61642d6e6f7465626f6f6b2d627269676874677265656e).

# **Note:** To improve responsiveness via caching, the notebooks are updated only once a week. They are only
# available for the latest stable release of Trixi.jl at the time of caching.

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
# can be implemented in Trixi.jl and analyse entropy conservation for two different flux combinations.

# ### [3 Shock capturing with flux differencing and stage limiter](@ref shock_capturing)
#-
# Using the flux differencing formulation, a simple procedure to capture shocks is a hybrid blending
# of a high-order DG method and a low-order subcell finite volume (FV) method. We present the idea on a
# very basic level and show the implementation in Trixi.jl. Then, a positivity preserving limiter is
# explained and added to an exemplary simulation of the Sedov blast wave with the 2D compressible Euler
# equations.

# ### [4 Non-periodic boundary conditions](@ref non_periodic_boundaries)
#-
# Thus far, all examples used periodic boundaries. In Trixi.jl, you can also set up a simulation with
# non-periodic boundaries. This tutorial presents the implementation of the classical Dirichlet
# boundary condition with a following example. Then, other non-periodic boundaries are mentioned.

# ### [5 DG schemes via `DGMulti` solver](@ref DGMulti_1)
#-
# This tutorial is about the more general DG solver [`DGMulti`](@ref), introduced [here](@ref DGMulti).
# We are showing some examples for this solver, for instance with discretization nodes by Gauss or
# triangular elements. Moreover, we present a simple way to include pre-defined triangulate meshes for
# non-Cartesian domains using the package [StartUpDG.jl](https://github.com/jlchan/StartUpDG.jl).

# ### [6 Other SBP schemes (FD, CGSEM) via `DGMulti` solver](@ref DGMulti_2)
#-
# Supplementary to the previous tutorial about DG schemes via the `DGMulti` solver we now present
# the possibility for `DGMulti` to use other SBP schemes via the package
# [SummationByPartsOperators.jl](https://github.com/ranocha/SummationByPartsOperators.jl).
# For instance, we show how to set up a finite differences (FD) scheme and a continuous Galerkin
# (CGSEM) method.

# ### [7 Upwind FD SBP schemes](@ref upwind_fdsbp)
#-
# General SBP schemes can not only be used via the [`DGMulti`](@ref) solver but
# also with a general `DG` solver. In particular, upwind finite difference SBP
# methods can be used together with the `TreeMesh`. Similar to general SBP
# schemes in the `DGMulti` framework, the interface is based on the package
# [SummationByPartsOperators.jl](https://github.com/ranocha/SummationByPartsOperators.jl).

# ### [8 Adding a new scalar conservation law](@ref adding_new_scalar_equations)
#-
# This tutorial explains how to add a new physics model using the example of the cubic conservation
# law. First, we define the equation using a `struct` `CubicEquation` and the physical flux. Then,
# the corresponding standard setup in Trixi.jl (`mesh`, `solver`, `semi` and `ode`) is implemented
# and the ODE problem is solved by OrdinaryDiffEq's `solve` method.

# ### [9 Adding a non-conservative equation](@ref adding_nonconservative_equation)
#-
# In this part, another physics model is implemented, the nonconservative linear advection equation.
# We run two different simulations with different levels of refinement and compare the resulting errors.

# ### [10 Adaptive mesh refinement](@ref adaptive_mesh_refinement)
#-
# Adaptive mesh refinement (AMR) helps to increase the accuracy in sensitive or turbolent regions while
# not wasting resources for less interesting parts of the domain. This leads to much more efficient
# simulations. This tutorial presents the implementation strategy of AMR in Trixi.jl, including the use of
# different indicators and controllers.

# ### [11 Structured mesh with curvilinear mapping](@ref structured_mesh_mapping)
#-
# In this tutorial, the use of Trixi.jl's structured curved mesh type [`StructuredMesh`](@ref) is explained.
# We present the two basic option to initialize such a mesh. First, the curved domain boundaries
# of a circular cylinder are set by explicit boundary functions. Then, a fully curved mesh is
# defined by passing the transformation mapping.

# ### [12 Unstructured meshes with HOHQMesh.jl](@ref hohqmesh_tutorial)
#-
# The purpose of this tutorial is to demonstrate how to use the [`UnstructuredMesh2D`](@ref)
# functionality of Trixi.jl. This begins by running and visualizing an available unstructured
# quadrilateral mesh example. Then, the tutorial will demonstrate how to conceptualize a problem
# with curved boundaries, generate a curvilinear mesh using the available [HOHQMesh](https://github.com/trixi-framework/HOHQMesh)
# software in the Trixi.jl ecosystem, and then run a simulation using Trixi.jl on said mesh.
# In the end, the tutorial briefly explains how to simulate an example using AMR via `P4estMesh`.

# ### [13 Explicit time stepping](@ref time_stepping)
#-
# This tutorial is about time integration using [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl).
# It explains how to use their algorithms and presents two types of time step choices - with error-based
# and CFL-based adaptive step size control.

# ### [14 Differentiable programming](@ref differentiable_programming)
#-
# This part deals with some basic differentiable programming topics. For example, a Jacobian, its
# eigenvalues and a curve of total energy (through the simulation) are calculated and plotted for
# a few semidiscretizations. Moreover, we calculate an example for propagating errors with Measurement.jl
# at the end.



# ## Examples in Trixi.jl
# Trixi.jl already contains several more coding examples, the so-called `elixirs`. You can find them
# in the folder [`examples`](https://github.com/trixi-framework/Trixi.jl/blob/main/examples/).
# They are structured by the underlying mesh type and the respective number of spatial dimensions.
# The name of an elixir is composed of the underlying system of conservation equations (for instance
# `advection` or `euler`) and other special characteristics like the initial condition
# (e.g. `gauss`, `astro_jet`, `blast_wave`) or the included simulation features
# (e.g. `amr`, `shockcapturing`).
