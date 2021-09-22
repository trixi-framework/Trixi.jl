# # Tutorials for Trixi.jl

# The tutorial section for [Trixi.jl](https://github.com/trixi-framework/Trixi.jl) also contains
# interactive step-by-step explanations via [Binder](https://mybinder.org).

# Right now, you are using the classic documentation. The corresponding interactive notebooks can
# be viewed in [nbviewer](https://nbviewer.jupyter.org/) and opened in [Binder](https://mybinder.org/)
# via the icons ![]("https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg")
# and ![](https://mybinder.org/badge_logo.svg) in the respective tutorial.

# You can also download the notebook files via ![]("https://camo.githubusercontent.com/aea75103f6d9f690a19cb0e17c06f984ab0f472d9e6fe4eadaa0cc438ba88ada/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f646f776e6c6f61642d6e6f7465626f6f6b2d627269676874677265656e").
    
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
