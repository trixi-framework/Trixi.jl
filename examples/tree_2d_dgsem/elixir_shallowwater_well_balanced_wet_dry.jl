
using OrdinaryDiffEq
using Trixi
using Printf: @printf, @sprintf

###############################################################################
# Semidiscretization of the shallow water equations
#
# TODO: TrixiShallowWater: wet/dry example elixir

equations = ShallowWaterEquations2D(gravity_constant = 9.812)

"""
    initial_condition_well_balanced_chen_noelle(x, t, equations:: ShallowWaterEquations2D)

Initial condition with a complex (discontinuous) bottom topography to test the well-balanced
property for the [`hydrostatic_reconstruction_chen_noelle`](@ref) including dry areas within the
domain. The errors from the analysis callback are not important but the error for this
lake-at-rest test case `∑|H0-(h+b)|` should be around machine roundoff.

The initial condition is taken from Section 5.2 of the paper:
- Guoxian Chen and Sebastian Noelle (2017)
  A new hydrostatic reconstruction scheme based on subcell reconstructions
  [DOI:10.1137/15M1053074](https://dx.doi.org/10.1137/15M1053074)
"""
function initial_condition_complex_bottom_well_balanced(x, t,
                                                        equations::ShallowWaterEquations2D)
    v1 = 0
    v2 = 0
    b = sin(4 * pi * x[1]) + 3

    if x[1] >= 0.5
        b = sin(4 * pi * x[1]) + 1
    end

    H = max(b, 2.5)
    if x[1] >= 0.5
        H = max(b, 1.5)
    end

    # It is mandatory to shift the water level at dry areas to make sure the water height h
    # stays positive. The system would not be stable for h set to a hard 0 due to division by h in
    # the computation of velocity, e.g., (h v1) / h. Therefore, a small dry state threshold
    # with a default value of 500*eps() ≈ 1e-13 in double precision, is set in the constructor above
    # for the ShallowWaterEquations and added to the initial condition if h = 0.
    # This default value can be changed within the constructor call depending on the simulation setup.
    H = max(H, b + equations.threshold_limiter)
    return prim2cons(SVector(H, v1, v2, b), equations)
end

initial_condition = initial_condition_complex_bottom_well_balanced

###############################################################################
# Get the DG approximation space

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
surface_flux = (FluxHydrostaticReconstruction(flux_hll_chen_noelle,
                                              hydrostatic_reconstruction_chen_noelle),
                flux_nonconservative_chen_noelle)

basis = LobattoLegendreBasis(3)

indicator_sc = IndicatorHennemannGassnerShallowWater(equations, basis,
                                                     alpha_max = 0.5,
                                                     alpha_min = 0.001,
                                                     alpha_smooth = true,
                                                     variable = waterheight_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

solver = DGSEM(basis, surface_flux, volume_integral)

###############################################################################
# Create the TreeMesh for the domain [0, 1]^2

coordinates_min = (0.0, 0.0)
coordinates_max = (1.0, 1.0)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                n_cells_max = 10_000)

# create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 50.0)
ode = semidiscretize(semi, tspan)

###############################################################################
# Workaround to set a discontinuous water and bottom topography for
# debugging and testing. Essentially, this is a slight augmentation of the
# `compute_coefficients` where the `x` node value passed here is slightly
# perturbed to the left / right in order to set a true discontinuity that avoids
# the doubled value of the LGL nodes at a particular element interface.
#
# Note! The errors from the analysis callback are not important but the error
# for this lake at rest test case `∑|H0-(h+b)|` should be near machine roundoff.

# point to the data we want to augment
u = Trixi.wrap_array(ode.u0, semi)
# reset the initial condition
for element in eachelement(semi.solver, semi.cache)
    for j in eachnode(semi.solver), i in eachnode(semi.solver)
        x_node = Trixi.get_node_coords(semi.cache.elements.node_coordinates, equations,
                                       semi.solver, i, j, element)
        # We know that the discontinuity is a vertical line. Slightly augment the x value by a factor
        # of unit roundoff to avoid the repeted value from the LGL nodes at at interface.
        if i == 1
            x_node = SVector(nextfloat(x_node[1]), x_node[2])
        elseif i == nnodes(semi.solver)
            x_node = SVector(prevfloat(x_node[1]), x_node[2])
        end
        u_node = initial_condition_complex_bottom_well_balanced(x_node, first(tspan),
                                                                equations)
        Trixi.set_node_vars!(u, u_node, equations, semi.solver, i, j, element)
    end
end

###############################################################################
# Callbacks

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     save_analysis = false)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 1000,
                                     save_initial_solution = true,
                                     save_final_solution = true)

stepsize_callback = StepsizeCallback(cfl = 2.0)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution,
                        stepsize_callback)

stage_limiter! = PositivityPreservingLimiterShallowWater(variables = (Trixi.waterheight,))

###############################################################################
# run the simulation

sol = solve(ode, SSPRK43(stage_limiter!); dt = 1.0,
            ode_default_options()..., callback = callbacks, adaptive = false);

summary_callback() # print the timer summary

###############################################################################
# Workaround to compute the well-balancedness error for this particular problem
# that has two reference water heights. One for a lake to the left of the
# discontinuous bottom topography `H0_upper = 2.5` and another for a lake to the
# right of the discontinuous bottom topography `H0_lower = 1.5`.

# Declare a special version of the function to compute the lake-at-rest error
# OBS! The reference water height values are hardcoded for convenience.
function lake_at_rest_error_two_level(u, x, equations::ShallowWaterEquations2D)
    h, _, _, b = u

    # For well-balancedness testing with possible wet/dry regions the reference
    # water height `H0` accounts for the possibility that the bottom topography
    # can emerge out of the water as well as for the threshold offset to avoid
    # division by a "hard" zero water heights as well.

    if x[1] < 0.5
        H0_wet_dry = max(2.5, b + equations.threshold_limiter)
    else
        H0_wet_dry = max(1.5, b + equations.threshold_limiter)
    end

    return abs(H0_wet_dry - (h + b))
end

# point to the data we want to analyze
u = Trixi.wrap_array(sol[end], semi)
# Perform the actual integration of the well-balancedness error over the domain
l1_well_balance_error = Trixi.integrate_via_indices(u, mesh, equations, semi.solver,
                                                    semi.cache;
                                                    normalize = true) do u, i, j, element,
                                                                         equations, solver
    x_node = Trixi.get_node_coords(semi.cache.elements.node_coordinates, equations, solver,
                                   i, j, element)
    # We know that the discontinuity is a vertical line. Slightly augment the x value by a factor
    # of unit roundoff to avoid the repeted value from the LGL nodes at at interface.
    if i == 1
        x_node = SVector(nextfloat(x_node[1]), x_node[2])
    elseif i == nnodes(semi.solver)
        x_node = SVector(prevfloat(x_node[1]), x_node[2])
    end
    u_local = Trixi.get_node_vars(u, equations, solver, i, j, element)
    return lake_at_rest_error_two_level(u_local, x_node, equations)
end

# report the well-balancedness lake-at-rest error to the screen
println("─"^100)
println(" Lake-at-rest error for '", Trixi.get_name(equations), "' with ", summary(solver),
        " at final time " * @sprintf("%10.8e", tspan[end]))

@printf(" %-12s:", Trixi.pretty_form_utf(lake_at_rest_error))
@printf("  % 10.8e", l1_well_balance_error)
println()
println("─"^100)
