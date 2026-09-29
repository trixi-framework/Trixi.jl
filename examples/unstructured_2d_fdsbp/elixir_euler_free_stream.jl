using OrdinaryDiffEqSSPRK
using Trixi
using TrixiData # for the mesh file

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

# Free-stream initial condition
initial_condition = initial_condition_constant

# Boundary conditions for free-stream testing
boundary_condition_free_stream = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = (; Body = boundary_condition_free_stream,
                       Button1 = boundary_condition_free_stream,
                       Button2 = boundary_condition_free_stream,
                       Eye1 = boundary_condition_free_stream,
                       Eye2 = boundary_condition_free_stream,
                       Smile = boundary_condition_free_stream,
                       Bowtie = boundary_condition_free_stream)

###############################################################################
# Get the FDSBP approximation space

D_SBP = derivative_operator(SummationByPartsOperators.MattssonAlmquistVanDerWeide2018Accurate(),
                            derivative_order = 1, accuracy_order = 4,
                            xmin = -1.0, xmax = 1.0, N = 12)
solver = FDSBP(D_SBP,
               surface_integral = SurfaceIntegralStrongForm(flux_hll),
               volume_integral = VolumeIntegralStrongForm())

###############################################################################
# Get the curved quad mesh from a file (downloaded by TrixiData.jl if not available locally)
mesh_file = mesh_gingerbread_man()

mesh = UnstructuredMesh2D(mesh_file)

###############################################################################
# create the semi discretization object

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 5.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true)

callbacks = CallbackSet(summary_callback, analysis_callback,
                        alive_callback, save_solution)

###############################################################################
# run the simulation

# set small tolerances for the free-stream preservation test
sol = solve(ode, SSPRK43(), abstol = 1.0e-12, reltol = 1.0e-12;
            ode_default_options()..., callback = callbacks)
