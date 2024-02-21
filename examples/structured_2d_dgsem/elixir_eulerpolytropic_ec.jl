
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the polytropic Euler equations

gamma = 1.4
kappa = 0.5     # Scaling factor for the pressure.
equations = PolytropicEulerEquations2D(gamma, kappa)

initial_condition = initial_condition_weak_blast_wave

###############################################################################
# Get the DG approximation space

volume_flux = flux_winters_etal
solver = DGSEM(polydeg = 4, surface_flux = flux_winters_etal,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

###############################################################################
# Get the curved quad mesh from a mapping function

# Mapping as described in https://arxiv.org/abs/2012.12040, but reduced to 2D
function mapping(xi_, eta_)
    # Transform input variables between -1 and 1 onto [0,3]
    xi = 1.5 * xi_ + 1.5
    eta = 1.5 * eta_ + 1.5

    y = eta + 3 / 8 * (cos(1.5 * pi * (2 * xi - 3) / 3) *
                       cos(0.5 * pi * (2 * eta - 3) / 3))

    x = xi + 3 / 8 * (cos(0.5 * pi * (2 * xi - 3) / 3) *
                      cos(2 * pi * (2 * y - 3) / 3))

    return SVector(x, y)
end

cells_per_dimension = (16, 16)

# Create curved mesh with 16 x 16 elements
mesh = StructuredMesh(cells_per_dimension, mapping)

###############################################################################
# create the semi discretization object

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true)

stepsize_callback = StepsizeCallback(cfl = 1.0)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
