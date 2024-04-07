
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the shallow water equations with a discontinuous
# bottom topography function

equations = ShallowWaterEquations1D(gravity_constant = 9.81, H0 = 3.25)

# Setup a truly discontinuous bottom topography function for this academic
# testcase of well-balancedness. The errors from the analysis callback are
# not important but the error for this lake-at-rest test case
# `∑|H0-(h+b)|` should be around machine roundoff.
# Works as intended for TreeMesh1D with `initial_refinement_level=3`. If the mesh
# refinement level is changed the initial condition below may need changed as well to
# ensure that the discontinuities lie on an element interface.
function initial_condition_discontinuous_well_balancedness(x, t,
                                                           equations::ShallowWaterEquations1D)
    # Set the background values
    H = equations.H0
    v = 0.0
    b = 0.0

    # Setup a discontinuous bottom topography
    if x[1] >= 0.5 && x[1] <= 0.75
        b = 2.0 + 0.5 * sin(2.0 * pi * x[1])
    end

    return prim2cons(SVector(H, v, b), equations)
end

initial_condition = initial_condition_discontinuous_well_balancedness

###############################################################################
# Get the DG approximation space

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
surface_flux = (flux_fjordholm_etal, flux_nonconservative_fjordholm_etal)
solver = DGSEM(polydeg = 4, surface_flux = surface_flux,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

###############################################################################
# Get the TreeMesh and setup a periodic mesh

coordinates_min = -1.0
coordinates_max = 1.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                n_cells_max = 10_000)

# Create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solver

tspan = (0.0, 100.0)
ode = semidiscretize(semi, tspan)

###############################################################################
# Callbacks

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_integrals = (lake_at_rest_error,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 1000,
                                     save_initial_solution = true,
                                     save_final_solution = true)

stepsize_callback = StepsizeCallback(cfl = 1.0)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
