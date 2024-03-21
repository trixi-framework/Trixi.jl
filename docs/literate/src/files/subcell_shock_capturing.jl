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
# Right now, only the canonical three-stage, third-order SSPRK method (Shu-Osher)
# [`Trixi.SimpleSSPRK33`](@ref) is implemented.

# TODO: Some comments about
# - parameters of Newton method (max_iterations_newton = 10, newton_tolerances = (1.0e-12, 1.0e-14), gamma_constant_newton = 2 * ndims(equations)))
# - positivity_correction_factor (Maybe show calculation of bounds, also of local bounds)

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
using Trixi
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
# How to set up a simulation using the IDP limiting becomes clearer when lokking at a exemplary
# setup. This will be a simplyfied version of `tree_2d_dgsem/elixir_euler_blast_wave_sc_subcell.jl`.
# Since the setup is mostly very similar to a pure DGSEM setup as in
# `tree_2d_dgsem/elixir_euler_blast_wave.jl`, the equivalent parts are without any explanation
# here.
using OrdinaryDiffEq
using Trixi

equations = CompressibleEulerEquations2D(1.4)

"""
    initial_condition_blast_wave(x, t, equations::CompressibleEulerEquations2D)

A medium blast wave taken from
- Sebastian Hennemann, Gregor J. Gassner (2020)
  A provably entropy stable subcell shock capturing approach for high order split form DG
  [arXiv: 2008.12044](https://arxiv.org/abs/2008.12044)
"""
function initial_condition_blast_wave(x, t, equations::CompressibleEulerEquations2D)
    # Modified From Hennemann & Gassner JCP paper 2020 (Sec. 6.3) -> "medium blast wave"
    # Set up polar coordinates
    inicenter = SVector(0.0, 0.0)
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    r = sqrt(x_norm^2 + y_norm^2)
    phi = atan(y_norm, x_norm)
    sin_phi, cos_phi = sincos(phi)

    # Calculate primitive variables
    rho = r > 0.5 ? 1.0 : 1.1691
    v1 = r > 0.5 ? 0.0 : 0.1882 * cos_phi
    v2 = r > 0.5 ? 0.0 : 0.1882 * sin_phi
    p = r > 0.5 ? 1.0E-3 : 1.245

    return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_blast_wave

###############################################################################
# TODO: Some explanation
surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha
basis = LobattoLegendreBasis(3)
limiter_idp = SubcellLimiterIDP(equations, basis;
                                local_minmax_variables_cons = ["rho"])
volume_integral = VolumeIntegralSubcellLimiting(limiter_idp;
                                                volume_flux_dg = volume_flux,
                                                volume_flux_fv = surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)


coordinates_min = (-2.0, -2.0)
coordinates_max = (2.0, 2.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 5,
                n_cells_max = 10_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 500
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 200,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 0.3)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# As explained above, the IDP limiter works a-posteriori and requires the additional use of a
# correction stage implemented with the stage callback [`SubcellLimiterIDPCorrection`](@ref).
# This callback is passed within a tuple to the time integration method.
#-
# Moreover, as mentioned before as well, simulations with subcell limiting require a Trixi-intern
# SSPRK time integration methods with passed stage callbacks and a Trixi-intern `Trixi.solve(...)`
# routine.
stage_callbacks = (SubcellLimiterIDPCorrection(),)

sol = Trixi.solve(ode, Trixi.SimpleSSPRK33(stage_callbacks = stage_callbacks);
                  dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  callback = callbacks);
summary_callback() # print the timer summary


# ## Visualizaton
# As for a standard simulation in Trixi.jl, it is possible to visualize the solution using Plots
# `plot` routine.
using Plots
plot(sol)

# To get an additional look at the amount of limiting that is used, you can use the visualization
# approach using the [`SaveSolutionCallback`](@ref), [`Trixi2Vtk`](https://github.com/trixi-framework/Trixi2Vtk.jl)
# and [ParaView](https://www.paraview.org/download/). More details about this procedure
# can be found in the [visualization documentation](@ref visualization).
# Unfortunately, the support for subcell limiting data is not yet merge into the main branch
# of Trixi2Vtk but lies in the branch `bennibolm/node-variables`.
#-
# With that implementation and the standard procedure used for Trixi2Vtk you get the following
# dropdown menu in ParaView.
# ![ParaView_Dropdownmenu](https://github.com/trixi-framework/Trixi.jl/assets/74359358/70d15f6a-059b-4349-8291-68d9ab3af43e)

# The resulting visualization of the density and the limiting parameter then looks like this.
# ![blast_wave_paraview](https://github.com/trixi-framework/Trixi.jl/assets/74359358/e5808bed-c8ab-43bf-af7a-050fe43dd630)

# You can see that the limiting coefficient does not lie in the interval [0,1], what actually was
# expected due to its calculation.
# TODO: Did I write something about this calculation?
# This is due to the reconstruction functionality which is defaultly enabled in Trixi2Vtk.
# You can disabled it with `reinterpolate=false` within the call of `trixi2vtk(...)` and get the
# following visualization.
# ![blast_wave_paraview_reinterpolate=false](https://github.com/trixi-framework/Trixi.jl/assets/74359358/39274f18-0064-469c-b4da-bac4b843e116)


# ## Target bounds checking
# TODO
