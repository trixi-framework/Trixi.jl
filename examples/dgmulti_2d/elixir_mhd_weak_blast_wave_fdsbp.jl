using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible ideal GLM-MHD equations

c_h = 5.0
equations = IdealGlmMhdEquations2D(1.4, c_h)

initial_condition = initial_condition_weak_blast_wave

# Create a 1D global FDSBP operator with periodic BCs
# In multiple space dimensions, tensor products of the 1D operator are used.
volume_flux = (flux_hindenlang_gassner, flux_nonconservative_powell)
dg = DGMulti(element_type = Quad(),
             approximation_type = periodic_derivative_operator(derivative_order = 1,
                                                               accuracy_order = 4,
                                                               xmin = 0.0, xmax = 1.0,
                                                               N = 40),
             volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

mesh = DGMultiMesh(dg,
                   coordinates_min = (-2.0, -2.0), coordinates_max = (2.0, 2.0))
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg;
                                    boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.2)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval, uEltype = real(dg))
alive_callback = AliveCallback(analysis_interval = analysis_interval)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback)

###############################################################################
# run the simulation

sol = solve(ode, RDPK3SpFSAL49(); abstol = 1.0e-8, reltol = 1.0e-8,
            save_everystep = false, callback = callbacks)
