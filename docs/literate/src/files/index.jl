# # Tutorials for Trixi.jl

# This repository contains a tutorial section for [Trixi.jl](https://github.com/trixi-framework/Trixi.jl),
# with interactive step-by-step explanations via [Binder](https://mybinder.org).

#md # Right now, you are using the classic documentation. The corresponding notebooks can be viewed in
#md # [nbviewer](https://nbviewer.jupyter.org/) and opened in [Binder](https://mybinder.org/) via this
#md # icon ![](https://mybinder.org/badge_logo.svg) in the respective tutorial.
    
# There are tutorials for the following topics:

# ### 1 Adding a new equation
# #### [1.1 Scalar conservation law](@ref cubic_conservation_law_literate)
#-
# This tutorial explains how to add a new physics model using the example of the cubic conservation
# law. First, we define the equation using a `struct` `CubicEquation` and the physical flux. Then,
# the corresponding standard setup in Trixi.jl (`mesh`, `solver`, `semi` and `ode`) is implemented
# and the ODE problem is solved by OrdinaryDiffEq's `solve` method.

# #### [1.2 Nonconservative advection](@ref nonconservative_advection_literate)
#-
# In this part, another physics model is implemented, the nonconservative linear advection equation.
# We run two different simulations with a different level of refinement and compare the resulting errors.

# ### [2 Differentiable programming](@ref differentiable_programming_literate)
#-
# This part deals with some basic differentiable programming topics. For example, a Jacobian, it
# eigenvalues and a curve of total energy (through the simulation) are calculated and plotted for
# a few semidiscretizations. Moreover, an example for propagating errors with Measurement.jl is given
# at the end.
