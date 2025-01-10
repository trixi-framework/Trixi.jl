# !!! warning "Experimental implementation (upwind SBP)"
#     This is an experimental feature and may change in future releases.
# Similar to unstructured_2d_fdsbp/elixir_euler_free_stream_upwind.jl
# but using Float32 instead of the default Float64

using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4f0)

initial_condition = initial_condition_constant

# Boundary conditions for free-stream preservation test
boundary_condition_free_stream = BoundaryConditionDirichlet(initial_condition)

boundary_conditions = Dict(:outerCircle => boundary_condition_free_stream,
                           :cone1 => boundary_condition_free_stream,
                           :cone2 => boundary_condition_free_stream,
                           :iceCream => boundary_condition_free_stream)

###############################################################################
# Get the Upwind FDSBP approximation space

# TODO: FDSBP
# Note, one must set `xmin=-1` and `xmax=1` due to the reuse
# of interpolation routines from `calc_node_coordinates!` to create
# the physical coordinates in the mappings.
D_upw = upwind_operators(SummationByPartsOperators.Mattsson2017,
                         derivative_order = 1,
                         accuracy_order = 8,
                         xmin = -1.0f0, xmax = 1.0f0,
                         N = 17)

flux_splitting = splitting_vanleer_haenel
solver = FDSBP(D_upw,
               surface_integral = SurfaceIntegralStrongForm(FluxUpwind(flux_splitting)),
               volume_integral = VolumeIntegralUpwind(flux_splitting))

###############################################################################
# Get the curved quad mesh from a file (downloads the file if not available locally)

# Mesh with second-order boundary polynomials requires an upwind SBP operator
# with (at least) 4th order boundary closure to guarantee the approximation is
# free-stream preserving
mesh_file = Trixi.download("https://gist.githubusercontent.com/andrewwinters5000/ec9a345f09199ebe471d35d5c1e4e08f/raw/15975943d8642e42f8292235314b6f1b30aa860d/mesh_inner_outer_boundaries.mesh",
                           joinpath(@__DIR__, "mesh_inner_outer_boundaries.mesh"))

mesh = UnstructuredMesh2D(mesh_file, RealT = Float32)

###############################################################################
# create the semi discretization object

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0f0, 5.0f0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 1000,
                                     save_initial_solution = true,
                                     save_final_solution = true)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        save_solution,
                        alive_callback)

###############################################################################
# run the simulation

# set small tolerances for the free-stream preservation test
sol = solve(ode, SSPRK43(), abstol = 1.0f-6, reltol = 1.0f-6,
            save_everystep = false, callback = callbacks)

summary_callback() # print the timer summary
