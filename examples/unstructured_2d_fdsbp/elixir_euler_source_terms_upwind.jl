# !!! warning "Experimental implementation (upwind SBP)"
#     This is an experimental feature and may change in future releases.

using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_convergence_test

source_term = source_terms_convergence_test

boundary_condition_eoc = BoundaryConditionDirichlet(initial_condition)

boundary_conditions = Dict(:Top => boundary_condition_eoc,
                           :Bottom => boundary_condition_eoc,
                           :Right => boundary_condition_eoc,
                           :Left => boundary_condition_eoc)

###############################################################################
# Get the Upwind FDSBP approximation space

# TODO: FDSBP
# Note, one must set `xmin=-1` and `xmax=1` due to the reuse
# of interpolation routines from `calc_node_coordinates!` to create
# the physical coordinates in the mappings.
D_upw = upwind_operators(SummationByPartsOperators.Mattsson2017,
                         derivative_order = 1,
                         accuracy_order = 4,
                         xmin = -1.0, xmax = 1.0,
                         N = 9)

flux_splitting = splitting_drikakis_tsangaris
solver = FDSBP(D_upw,
               surface_integral = SurfaceIntegralStrongForm(FluxUpwind(flux_splitting)),
               volume_integral = VolumeIntegralUpwind(flux_splitting))

###############################################################################
# Get the curved quad mesh from a file (downloads the file if not available locally)

# Mesh with first-order boundary polynomials requires an upwind SBP operator
# with (at least) 2nd order boundary closure to guarantee the approximation is
# free-stream preserving
mesh_file = Trixi.download("https://gist.githubusercontent.com/andrewwinters5000/a4f4743008bf3233957a9ea6ac7a62e0/raw/8b36cc6649153fe0a5723b200368a210a1d74eaf/mesh_refined_box.mesh",
                           joinpath(@__DIR__, "mesh_refined_box.mesh"))

mesh = UnstructuredMesh2D(mesh_file)

###############################################################################
# create the semidiscretization object

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_term,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
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

sol = solve(ode, SSPRK43(), abstol = 1.0e-6, reltol = 1.0e-6,
            save_everystep = false, callback = callbacks)

summary_callback() # print the timer summary
