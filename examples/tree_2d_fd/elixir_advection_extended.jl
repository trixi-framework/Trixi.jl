
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advectionvelocity = (0.2, -0.3)
equations = LinearScalarAdvectionEquation2D(advectionvelocity)

initial_condition = initial_condition_convergence_test

D_SBP = derivative_operator(MattssonNordström2004(),
                            derivative_order=1, accuracy_order=4,
                            xmin=0.0, xmax=1.0,
                            N=100,
                            # TOOD: FD. Can be removed once
                            # https://github.com/JuliaArrays/ArrayInterface.jl/issues/157
                            # https://github.com/JuliaArrays/ArrayInterface.jl/issues/158
                            # are fixed
                            parallel=Val(:plain))
solver = DG(D_SBP, nothing #= mortar =#,
            SurfaceIntegralStrongForm(flux_lax_friedrichs),
            VolumeIntegralStrongForm())

coordinates_min = (-1.0, -1.0)
coordinates_max = ( 1.0,  1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=1,
                n_cells_max=30_000,
                periodicity=true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan);

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     extra_analysis_integrals=(energy_total,))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback)


###############################################################################
# run the simulation

# TODO: FD.
sol = solve(ode, RDPK3SpFSAL49(), abstol=1.0e-6, reltol=1.0e-6,
# sol = solve(ode, Tsit5(), abstol=1.0e-6, reltol=1.0e-6,
            save_everystep=false, callback=callbacks, maxiters=1e5)
summary_callback()
