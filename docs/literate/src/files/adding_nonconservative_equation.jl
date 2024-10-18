#src # Adding a new equation: nonconservative linear advection
using Test: @test #src

# If you want to use Trixi.jl for your own research, you might be interested in
# a new physics model that is not present in Trixi.jl. In this tutorial,
# we will implement the nonconservative linear advection equation in a periodic domain
# ```math
# \left\{
# \begin{aligned}&\partial_t u(t,x) + a(x) \partial_x u(t,x) = 0 \\
# &u(0,x)=\sin(x) \\
# &u(t,-\pi)=u(t,\pi)
# \end{aligned}
# \right.
# ```
# where $a(x) = 2 + \cos(x)$. The analytic solution is
# ```math
# u(t,x)=-\sin \left(2 \tan ^{-1}\left(\sqrt{3} \tan \left(\frac{\sqrt{3} t}{2}-\tan ^{-1}\left(\frac{1}{\sqrt{3}}\tan \left(\frac{x}{2}\right)\right)\right)\right)\right)
# ```
# In Trixi.jl, such a mathematical model
# is encoded as a subtype of [`Trixi.AbstractEquations`](@ref).

# ## Basic setup

# Since there is no native support for variable coefficients, we need to transform the PDE to the following system:
# ```math
# \left\{
# \begin{aligned}&\partial_t \begin{pmatrix}u(t,x)\\a(t,x) \end{pmatrix} +\begin{pmatrix} a(t,x) \partial_x u(t,x) \\ 0 \end{pmatrix} = 0 \\
# &u(0,x)=\sin(x) \\
# &a(0,x)=2+\cos(x) \\
# &u(t,-\pi)=u(t,\pi)
# \end{aligned}
# \right.
# ```

## Define new physics
using Trixi
using Trixi: AbstractEquations, get_node_vars
import Trixi: varnames, default_analysis_integrals, flux, max_abs_speed_naive,
              have_nonconservative_terms

## Since there is no native support for variable coefficients, we use two
## variables: one for the basic unknown `u` and another one for the coefficient `a`
struct NonconservativeLinearAdvectionEquation <: AbstractEquations{1, # spatial dimension
                                                                   2} # two variables (u,a)
end

function varnames(::typeof(cons2cons), ::NonconservativeLinearAdvectionEquation)
    ("scalar", "advection_velocity")
end

default_analysis_integrals(::NonconservativeLinearAdvectionEquation) = ()

## The conservative part of the flux is zero
flux(u, orientation, equation::NonconservativeLinearAdvectionEquation) = zero(u)

## Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
function max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
                             ::NonconservativeLinearAdvectionEquation)
    _, advection_velocity_ll = u_ll
    _, advection_velocity_rr = u_rr

    return max(abs(advection_velocity_ll), abs(advection_velocity_rr))
end

## We use nonconservative terms
have_nonconservative_terms(::NonconservativeLinearAdvectionEquation) = Trixi.True()

## This "nonconservative numerical flux" implements the nonconservative terms.
## In general, nonconservative terms can be written in the form
##   g(u) ∂ₓ h(u)
## Thus, a discrete difference approximation of this nonconservative term needs
## - `u mine`:  the value of `u` at the current position (for g(u))
## - `u_other`: the values of `u` in a neighborhood of the current position (for ∂ₓ h(u))
function flux_nonconservative(u_mine, u_other, orientation,
                              equations::NonconservativeLinearAdvectionEquation)
    _, advection_velocity = u_mine
    scalar, _ = u_other

    return SVector(advection_velocity * scalar, zero(scalar))
end

# The implementation of nonconservative terms uses a single "nonconservative flux"
# function `flux_nonconservative`. It will basically be applied in a loop of the
# form
# ```julia
# du_m(D, u) = sum(D[m, l] * flux_nonconservative(u[m], u[l], 1, equations)) # orientation 1: x
# ```
# where `D` is the derivative matrix and `u` contains the nodal solution values.

# Now, we can run a simple simulation using a DGSEM discretization.

## Create a simulation setup
using Trixi
using OrdinaryDiffEq

equation = NonconservativeLinearAdvectionEquation()

## You can derive the exact solution for this setup using the method of
## characteristics
function initial_condition_sine(x, t, equation::NonconservativeLinearAdvectionEquation)
    x0 = -2 * atan(sqrt(3) * tan(sqrt(3) / 2 * t - atan(tan(x[1] / 2) / sqrt(3))))
    scalar = sin(x0)
    advection_velocity = 2 + cos(x[1])
    SVector(scalar, advection_velocity)
end

## Create a uniform mesh in 1D in the interval [-π, π] with periodic boundaries
mesh = TreeMesh(-Float64(π), Float64(π), # min/max coordinates
                initial_refinement_level = 4, n_cells_max = 10^4)

## Create a DGSEM solver with polynomials of degree `polydeg`
## Remember to pass a tuple of the form `(conservative_flux, nonconservative_flux)`
## as `surface_flux` and `volume_flux` when working with nonconservative terms
volume_flux = (flux_central, flux_nonconservative)
surface_flux = (flux_lax_friedrichs, flux_nonconservative)
solver = DGSEM(polydeg = 3, surface_flux = surface_flux,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

## Setup the spatial semidiscretization containing all ingredients
semi = SemidiscretizationHyperbolic(mesh, equation, initial_condition_sine, solver)

## Create an ODE problem with given time span
tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

## Set up some standard callbacks summarizing the simulation setup and computing
## errors of the numerical solution
summary_callback = SummaryCallback()
analysis_callback = AnalysisCallback(semi, interval = 50)
callbacks = CallbackSet(summary_callback, analysis_callback)

## OrdinaryDiffEq's `solve` method evolves the solution in time and executes
## the passed callbacks
sol = solve(ode, Tsit5(), abstol = 1.0e-6, reltol = 1.0e-6,
            save_everystep = false, callback = callbacks)

## Print the timer summary
summary_callback()

## Plot the numerical solution at the final time
using Plots: plot
plot(sol)

# You see a plot of the final solution.

# We can check whether everything fits together by refining the grid and comparing
# the numerical errors. First, we look at the error using the grid resolution
# above.

error_1 = analysis_callback(sol).l2 |> first
@test isapprox(error_1, 0.00029609575838969394) #src
# Next, we increase the grid resolution by one refinement level and run the
# simulation again.

mesh = TreeMesh(-Float64(π), Float64(π), # min/max coordinates
                initial_refinement_level = 5, n_cells_max = 10^4)

semi = SemidiscretizationHyperbolic(mesh, equation, initial_condition_sine, solver)

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan);

summary_callback = SummaryCallback()
analysis_callback = AnalysisCallback(semi, interval = 50)
callbacks = CallbackSet(summary_callback, analysis_callback);

sol = solve(ode, Tsit5(), abstol = 1.0e-6, reltol = 1.0e-6,
            save_everystep = false, callback = callbacks);
summary_callback()

#nb #-
error_2 = analysis_callback(sol).l2 |> first
@test isapprox(error_2, 1.860295931682964e-5, rtol = 0.05) #src
#-
error_1 / error_2
@test isapprox(error_1 / error_2, 15.916970234784808, rtol = 0.05) #src
# As expected, the new error is roughly reduced by a factor of 16, corresponding
# to an experimental order of convergence of 4 (for polynomials of degree 3).

# ## Summary of the code

# Here is the complete code that we used (without the callbacks since these
# create a lot of unnecessary output in the doctests of this tutorial).
# In addition, we create the `struct` inside the new module `NonconservativeLinearAdvection`.
# That ensures that we can re-create `struct`s defined therein without having to
# restart Julia.

# Define new physics
module NonconservativeLinearAdvection

using Trixi
using Trixi: AbstractEquations, get_node_vars
import Trixi: varnames, default_analysis_integrals, flux, max_abs_speed_naive,
              have_nonconservative_terms

## Since there is not yet native support for variable coefficients, we use two
## variables: one for the basic unknown `u` and another one for the coefficient `a`
struct NonconservativeLinearAdvectionEquation <: AbstractEquations{1, # spatial dimension
                                                                   2} # two variables (u,a)
end

function varnames(::typeof(cons2cons), ::NonconservativeLinearAdvectionEquation)
    ("scalar", "advection_velocity")
end

default_analysis_integrals(::NonconservativeLinearAdvectionEquation) = ()

## The conservative part of the flux is zero
flux(u, orientation, equation::NonconservativeLinearAdvectionEquation) = zero(u)

## Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
function max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
                             ::NonconservativeLinearAdvectionEquation)
    _, advection_velocity_ll = u_ll
    _, advection_velocity_rr = u_rr

    return max(abs(advection_velocity_ll), abs(advection_velocity_rr))
end

## We use nonconservative terms
have_nonconservative_terms(::NonconservativeLinearAdvectionEquation) = Trixi.True()

## This "nonconservative numerical flux" implements the nonconservative terms.
## In general, nonconservative terms can be written in the form
##   g(u) ∂ₓ h(u)
## Thus, a discrete difference approximation of this nonconservative term needs
## - `u mine`:  the value of `u` at the current position (for g(u))
## - `u_other`: the values of `u` in a neighborhood of the current position (for ∂ₓ h(u))
function flux_nonconservative(u_mine, u_other, orientation,
                              equations::NonconservativeLinearAdvectionEquation)
    _, advection_velocity = u_mine
    scalar, _ = u_other

    return SVector(advection_velocity * scalar, zero(scalar))
end

end # module

## Create a simulation setup
import .NonconservativeLinearAdvection
using Trixi
using OrdinaryDiffEq

equation = NonconservativeLinearAdvection.NonconservativeLinearAdvectionEquation()

## You can derive the exact solution for this setup using the method of
## characteristics
function initial_condition_sine(x, t,
                                equation::NonconservativeLinearAdvection.NonconservativeLinearAdvectionEquation)
    x0 = -2 * atan(sqrt(3) * tan(sqrt(3) / 2 * t - atan(tan(x[1] / 2) / sqrt(3))))
    scalar = sin(x0)
    advection_velocity = 2 + cos(x[1])
    SVector(scalar, advection_velocity)
end

## Create a uniform mesh in 1D in the interval [-π, π] with periodic boundaries
mesh = TreeMesh(-Float64(π), Float64(π), # min/max coordinates
                initial_refinement_level = 4, n_cells_max = 10^4)

## Create a DGSEM solver with polynomials of degree `polydeg`
## Remember to pass a tuple of the form `(conservative_flux, nonconservative_flux)`
## as `surface_flux` and `volume_flux` when working with nonconservative terms
volume_flux = (flux_central, NonconservativeLinearAdvection.flux_nonconservative)
surface_flux = (flux_lax_friedrichs, NonconservativeLinearAdvection.flux_nonconservative)
solver = DGSEM(polydeg = 3, surface_flux = surface_flux,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

## Setup the spatial semidiscretization containing all ingredients
semi = SemidiscretizationHyperbolic(mesh, equation, initial_condition_sine, solver)

## Create an ODE problem with given time span
tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan);

## Set up some standard callbacks summarizing the simulation setup and computing
## errors of the numerical solution
summary_callback = SummaryCallback()
analysis_callback = AnalysisCallback(semi, interval = 50)
callbacks = CallbackSet(summary_callback, analysis_callback);

## OrdinaryDiffEq's `solve` method evolves the solution in time and executes
## the passed callbacks
sol = solve(ode, Tsit5(), abstol = 1.0e-6, reltol = 1.0e-6,
            save_everystep = false);

## Plot the numerical solution at the final time
using Plots: plot
plot(sol);

# ## Package versions

# These results were obtained using the following versions.

using InteractiveUtils
versioninfo()

using Pkg
Pkg.status(["Trixi", "OrdinaryDiffEq", "Plots"],
           mode = PKGMODE_MANIFEST)
