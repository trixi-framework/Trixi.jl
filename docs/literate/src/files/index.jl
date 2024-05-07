# # Tutorials for Trixi.jl

# The tutorial section for [Trixi.jl](https://github.com/trixi-framework/Trixi.jl) also contains
# interactive step-by-step explanations via [Binder](https://mybinder.org).

# Right now, you are using the classic documentation. The corresponding interactive notebooks can
# be opened in [Binder](https://mybinder.org/) and viewed in [nbviewer](https://nbviewer.jupyter.org/)
# via the icons ![](https://mybinder.org/badge_logo.svg) and ![](https://img.shields.io/badge/render-nbviewer-f37726)
# in the respective tutorial.
# You can also open the raw notebook files via ![](https://img.shields.io/badge/raw-notebook-4cc61e).

# **Note:** To improve responsiveness via caching, the notebooks are updated only once a week. They are only
# available for the latest stable release of Trixi.jl at the time of caching.

# There are tutorials for the following topics:

#src Note to developers: Use "{ index }" (but without spaces, see next line) to enable automatic indexing
# ### [{index} First steps in Trixi.jl](@ref getting_started)
#-
# This tutorial provides guidance for getting started with Trixi.jl, and Julia as well. It outlines
# the installation procedures for both Julia and Trixi.jl, the execution of Trixi.jl elixirs, the
# fundamental structure of a Trixi.jl setup, the visualization of results, and the development
# process for Trixi.jl.

#src Note to developers: Use "{ index }" (but without spaces, see next line) to enable automatic indexing
# ### [{index} Behind the scenes of a simulation setup](@ref behind_the_scenes_simulation_setup)
#-
# This tutorial will guide you through a simple Trixi.jl setup ("elixir"), giving an overview of
# what happens in the background during the initialization of a simulation. While the setup
# described herein does not cover all details, it involves relatively stable parts of Trixi.jl that
# are unlikely to undergo significant changes in the near future. The goal is to clarify some of
# the more fundamental, *technical* concepts that are applicable to a variety of
# (also more complex) configurations.s

#src Note to developers: Use "{ index }" (but without spaces, see next line) to enable automatic indexing
# ### [{index} Introduction to DG methods](@ref scalar_linear_advection_1d)
#-
# This tutorial gives an introduction to discontinuous Galerkin (DG) methods with the example of the
# scalar linear advection equation in 1D. Starting with some theoretical explanations, we first implement
# a raw version of a discontinuous Galerkin spectral element method (DGSEM). Then, we will show how
# to use features of Trixi.jl to achieve the same result.

#src Note to developers: Use "{ index }" (but without spaces, see next line) to enable automatic indexing
# ### [{index} DGSEM with flux differencing](@ref DGSEM_FluxDiff)
#-
# To improve stability often the flux differencing formulation of the DGSEM (split form) is used.
# We want to present the idea and formulation on a basic 1D level. Then, we show how this formulation
# can be implemented in Trixi.jl and analyse entropy conservation for two different flux combinations.

#src Note to developers: Use "{ index }" (but without spaces, see next line) to enable automatic indexing
# ### [{index} Shock capturing with flux differencing and stage limiter](@ref shock_capturing)
#-
# Using the flux differencing formulation, a simple procedure to capture shocks is a hybrid blending
# of a high-order DG method and a low-order subcell finite volume (FV) method. We present the idea on a
# very basic level and show the implementation in Trixi.jl. Then, a positivity preserving limiter is
# explained and added to an exemplary simulation of the Sedov blast wave with the 2D compressible Euler
# equations.

#src Note to developers: Use "{ index }" (but without spaces, see next line) to enable automatic indexing
# ### [{index} Subcell limiting with the IDP Limiter](@ref subcell_shock_capturing)
#-
# Trixi.jl features a subcell-wise limiting strategy utilizing an Invariant Domain-Preserving (IDP)
# approach. This IDP approach computes a blending factor that balances the high-order
# discontinuous Galerkin (DG) method with a low-order subcell finite volume (FV) method for each
# node within an element. This localized approach minimizes the application of dissipation,
# resulting in less limiting compared to the element-wise strategy. Additionally, the framework
# supports both local bounds, which are primarily used for shock capturing, and global bounds.
# The application of global bounds ensures the minimal necessary limiting to meet physical
# admissibility conditions, such as ensuring the non-negativity of variables.

#src Note to developers: Use "{ index }" (but without spaces, see next line) to enable automatic indexing
# ### [{index} Non-periodic boundary conditions](@ref non_periodic_boundaries)
#-
# Thus far, all examples used periodic boundaries. In Trixi.jl, you can also set up a simulation with
# non-periodic boundaries. This tutorial presents the implementation of the classical Dirichlet
# boundary condition with a following example. Then, other non-periodic boundaries are mentioned.

#src Note to developers: Use "{ index }" (but without spaces, see next line) to enable automatic indexing
# ### [{index} DG schemes via `DGMulti` solver](@ref DGMulti_1)
#-
# This tutorial is about the more general DG solver [`DGMulti`](@ref), introduced [here](@ref DGMulti).
# We are showing some examples for this solver, for instance with discretization nodes by Gauss or
# triangular elements. Moreover, we present a simple way to include pre-defined triangulate meshes for
# non-Cartesian domains using the package [StartUpDG.jl](https://github.com/jlchan/StartUpDG.jl).

#src Note to developers: Use "{ index }" (but without spaces, see next line) to enable automatic indexing
# ### [{index} Other SBP schemes (FD, CGSEM) via `DGMulti` solver](@ref DGMulti_2)
#-
# Supplementary to the previous tutorial about DG schemes via the `DGMulti` solver we now present
# the possibility for `DGMulti` to use other SBP schemes via the package
# [SummationByPartsOperators.jl](https://github.com/ranocha/SummationByPartsOperators.jl).
# For instance, we show how to set up a finite differences (FD) scheme and a continuous Galerkin
# (CGSEM) method.

#src Note to developers: Use "{ index }" (but without spaces, see next line) to enable automatic indexing
# ### [{index} Upwind FD SBP schemes](@ref upwind_fdsbp)
#-
# General SBP schemes can not only be used via the [`DGMulti`](@ref) solver but
# also with a general `DG` solver. In particular, upwind finite difference SBP
# methods can be used together with the `TreeMesh`. Similar to general SBP
# schemes in the `DGMulti` framework, the interface is based on the package
# [SummationByPartsOperators.jl](https://github.com/ranocha/SummationByPartsOperators.jl).

#src Note to developers: Use "{ index }" (but without spaces, see next line) to enable automatic indexing
# ### [{index} Adding a new scalar conservation law](@ref adding_new_scalar_equations)
#-
# This tutorial explains how to add a new physics model using the example of the cubic conservation
# law. First, we define the equation using a `struct` `CubicEquation` and the physical flux. Then,
# the corresponding standard setup in Trixi.jl (`mesh`, `solver`, `semi` and `ode`) is implemented
# and the ODE problem is solved by OrdinaryDiffEq's `solve` method.

#src Note to developers: Use "{ index }" (but without spaces, see next line) to enable automatic indexing
# ### [{index} Adding a non-conservative equation](@ref adding_nonconservative_equation)
#-
# In this part, another physics model is implemented, the nonconservative linear advection equation.
# We run two different simulations with different levels of refinement and compare the resulting errors.

#src Note to developers: Use "{ index }" (but without spaces, see next line) to enable automatic indexing
# ### [{index} Parabolic terms](@ref parabolic_terms)
#-
# This tutorial describes how parabolic terms are implemented in Trixi.jl, e.g.,
# to solve the advection-diffusion equation.

#src Note to developers: Use "{ index }" (but without spaces, see next line) to enable automatic indexing
# ### [{index} Adding new parabolic terms](@ref adding_new_parabolic_terms)
#-
# This tutorial describes how new parabolic terms can be implemented using Trixi.jl.

#src Note to developers: Use "{ index }" (but without spaces, see next line) to enable automatic indexing
# ### [{index} Adaptive mesh refinement](@ref adaptive_mesh_refinement)
#-
# Adaptive mesh refinement (AMR) helps to increase the accuracy in sensitive or turbolent regions while
# not wasting resources for less interesting parts of the domain. This leads to much more efficient
# simulations. This tutorial presents the implementation strategy of AMR in Trixi.jl, including the use of
# different indicators and controllers.

#src Note to developers: Use "{ index }" (but without spaces, see next line) to enable automatic indexing
# ### [{index} Structured mesh with curvilinear mapping](@ref structured_mesh_mapping)
#-
# In this tutorial, the use of Trixi.jl's structured curved mesh type [`StructuredMesh`](@ref) is explained.
# We present the two basic option to initialize such a mesh. First, the curved domain boundaries
# of a circular cylinder are set by explicit boundary functions. Then, a fully curved mesh is
# defined by passing the transformation mapping.

#src Note to developers: Use "{ index }" (but without spaces, see next line) to enable automatic indexing
# ### [{index} Unstructured meshes with HOHQMesh.jl](@ref hohqmesh_tutorial)
#-
# The purpose of this tutorial is to demonstrate how to use the [`UnstructuredMesh2D`](@ref)
# functionality of Trixi.jl. This begins by running and visualizing an available unstructured
# quadrilateral mesh example. Then, the tutorial will demonstrate how to conceptualize a problem
# with curved boundaries, generate a curvilinear mesh using the available [HOHQMesh](https://github.com/trixi-framework/HOHQMesh)
# software in the Trixi.jl ecosystem, and then run a simulation using Trixi.jl on said mesh.
# In the end, the tutorial briefly explains how to simulate an example using AMR via `P4estMesh`.

#src Note to developers: Use "{ index }" (but without spaces, see next line) to enable automatic indexing
# ### [{index} P4est mesh from gmsh](@ref p4est_from_gmsh)
#-
# This tutorial describes how to obtain a [`P4estMesh`](@ref) from an existing mesh generated
# by [`gmsh`](https://gmsh.info/) or any other meshing software that can export to the Abaqus
# input `.inp` format. The tutorial demonstrates how edges/faces can be associated with boundary conditions based on the physical nodesets.

#src Note to developers: Use "{ index }" (but without spaces, see next line) to enable automatic indexing
# ### [{index} Explicit time stepping](@ref time_stepping)
#-
# This tutorial is about time integration using [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl).
# It explains how to use their algorithms and presents two types of time step choices - with error-based
# and CFL-based adaptive step size control.

#src Note to developers: Use "{ index }" (but without spaces, see next line) to enable automatic indexing
# ### [{index} Differentiable programming](@ref differentiable_programming)
#-
# This part deals with some basic differentiable programming topics. For example, a Jacobian, its
# eigenvalues and a curve of total energy (through the simulation) are calculated and plotted for
# a few semidiscretizations. Moreover, we calculate an example for propagating errors with Measurement.jl
# at the end.

#src Note to developers: Use "{ index }" (but without spaces, see next line) to enable automatic indexing
# ### [{index} Custom semidiscretization](@ref custom_semidiscretization)
#-
# This tutorial describes the [semidiscretiations](@ref overview-semidiscretizations) of Trixi.jl
# and explains how to extend them for custom tasks.



# ## Examples in Trixi.jl
# Trixi.jl already contains several more coding examples, the so-called `elixirs`. You can find them
# in the folder [`examples/`](https://github.com/trixi-framework/Trixi.jl/blob/main/examples/).
# They are structured by the underlying mesh type and the respective number of spatial dimensions.
# The name of an elixir is composed of the underlying system of conservation equations (for instance
# `advection` or `euler`) and other special characteristics like the initial condition
# (e.g. `gauss`, `astro_jet`, `blast_wave`) or the included simulation features
# (e.g. `amr`, `shockcapturing`).
