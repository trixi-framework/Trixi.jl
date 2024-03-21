#src # Subcell limiting with the IDP Limiter

# In the previous tutorial, the element-wise limiting with [`IndicatorHennemannGassner`](@ref)
# and [`VolumeIntegralShockCapturingHG`](@ref) was explained. This tutorial contains a short
# introduction to the idea and implementation of subcell shock capturing approaches in Trixi.jl,
# which is also based on the DGSEM scheme in flux differencing formulation.
# Trixi.jl contains the a-posteriori invariant domain-preserving (IDP) limiter which was
# introduced by [Rueda-Ramírez, Pazner, Gassner (2022)](https://doi.org/10.1016/j.compfluid.2022.105627)
# and [Pazner (2020)](https://doi.org/10.1016/j.cma.2021.113876). It is a flux-corrected
# transport-type (FCT) limiter and is implemented using [`SubcellLimiterIDP`](@ref) and
# [`VolumeIntegralSubcellLimiting`](@ref).
# Since it is an a-posteriori limiter you have to apply a correction stage after each Runge-Kutta
# stage. This is done by passing the stage callback [`SubcellLimiterIDPCorrection`](@ref) to the
# time integration method.

# ## Time integration method
# As mentioned before, the IDP limiting is an a-posteriori limiter. Its limiting process
# guarantees the target bounds for a simple Euler evolution. To still achieve a high-order
# approximation, the implementation uses strong-stability preserving (SSP) Runge-Kutta method
# which can be written as convex combination of these forward Euler steps.
#-
# Due to this functionality of the limiting procedure the correcting stage and therefore the stage
# callbacks has to be applied to the solution after the forward Euler step and before further
# computation. Unfortunately, the `solve(...)` routines of
# [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl), which is normally used in
# Trixi.jl for the time integration, does not support calculations via callback at this point
# in the simulation.
#-
# Therefore, subcell limiting with the IDP limiter requires the use of a Trixi-intern
# time integration SSPRK method called with
# ````julia
# Trixi.solve(ode, method(stage_callbacks = stage_callbacks); ...)`.
# ````
#-
# Right now, only the third-order SSPRK method [`Trixi.SimpleSSPRK33`](@ref) is implemented.

# TODO: Some comments about
# - parameters of Newton method (max_iterations_newton = 10, newton_tolerances = (1.0e-12, 1.0e-14), gamma_constant_newton = 2 * ndims(equations)))
# - positivity_correction_factor (Maybe show calculation of bounds, also of local bounds)

using Trixi

# # `SubcellLimiterIDP`
# The IDP limiter supports several options of limiting which are passed very flexible as parameters to
# the limiter individually.

# ## Global bounds
# First, there is the use of global bounds. If enabled, they enforce physical admissibility
# conditions, such as non-negativity of variables.
# This can be done for conservative variables, where the limiter is of a one-sided Zalesak-type, and
# general non-linear variables, where a Newton-bisection algorithm is used to enforce the bounds.

# ### Conservative variables
# The procedure to enforce global bounds for a conservative variables is as follows:
# If you want to guarantee non-negativity for the density of compressible Euler equations,
# you pass the specific quantity name of the conservative variable.
equations = CompressibleEulerEquations2D(1.4)

# The quantity name of the density is `rho` shich is how we enable its limiting.
# ````julia
# positivity_variables_cons = ["rho"]
# ````

# The quantity names are passed as a vector to allow several quantities.
# This is for instance used if you want to limit the density of two different components using
# the multicomponent compressible Euler equations.
equations = CompressibleEulerMulticomponentEquations2D(gammas = (1.4, 1.648),
                                                       gas_constants = (0.287, 1.578))

# Then, we just pass both quantity names.
# ````julia
# positivity_variables_cons = ["rho1", "rho2"]
# ````

# Alternatively, it is possible to all limit all density variables with a general command using
# ````julia
# positivity_variables_cons = ["rho" * string(i) for i in eachcomponent(equations)]
# ````

# ### Non-linear variables
# To allow limitation for all possible non-linear variables including on-the-fly defined ones,
# you directly pass function here.
# For instance, if you want to enforce non-negativity for the pressure, do as follows.
# ````julia
# positivity_variables_nonlinear = [pressure]
# ````

# ## Local bounds (Shock capturing)
# Second, Trixi.jl supports the limiting with local bounds for conservative variables. They
# allow to avoid spurious  oscillations within the global bounds and to improve the
# shock-capturing capabilities of the method. The corresponding numerical admissibility
# conditions are frequently formulated as local maximum or minimum principles.
# There are different option to choose local bounds. We calculate them using the low-order FV
# solution.

# As for the limiting with global bounds you are passing the quantity names of the conservative
# variables you want to limit. So, to limit the density with lower and upper local bounds pass
# the following.
# ````julia
# local_minmax_variables_cons = ["rho"]
# ````

# ## Exemplary simulation
# TODO
