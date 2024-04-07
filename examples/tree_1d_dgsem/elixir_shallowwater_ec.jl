
using OrdinaryDiffEq
using Trixi

###############################################################################
# Semidiscretization of the shallow water equations with a discontinuous
# bottom topography function

equations = ShallowWaterEquations1D(gravity_constant = 9.81)

# Initial condition with a truly discontinuous water height, velocity, and bottom
# topography function as an academic testcase for entropy conservation.
# The errors from the analysis callback are not important but `∑∂S/∂U ⋅ Uₜ` should
# be around machine roundoff.
# Works as intended for TreeMesh1D with `initial_refinement_level=4`. If the mesh
# refinement level is changed the initial condition below may need changed as well to
# ensure that the discontinuities lie on an element interface.
function initial_condition_ec_discontinuous_bottom(x, t, equations::ShallowWaterEquations1D)
    # Set the background values
    H = 4.25
    v = 0.0
    b = sin(x[1]) # arbitrary continuous function

    # Setup the discontinuous water height and velocity
    if x[1] >= 0.125 && x[1] <= 0.25
        H = 5.0
        v = 0.1882
    end

    # Setup a discontinuous bottom topography
    if x[1] >= -0.25 && x[1] <= -0.125
        b = 2.0 + 0.5 * sin(2.0 * pi * x[1])
    end

    return prim2cons(SVector(H, v, b), equations)
end

initial_condition = initial_condition_ec_discontinuous_bottom

###############################################################################
# Get the DG approximation space

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
solver = DGSEM(polydeg = 4,
               surface_flux = (flux_fjordholm_etal, flux_nonconservative_fjordholm_etal),
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

###############################################################################
# Get the TreeMesh and setup a periodic mesh

coordinates_min = -1.0
coordinates_max = 1.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 10_000)

# Create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solver

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

###############################################################################
# Callbacks

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
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
