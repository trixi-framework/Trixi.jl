
using Downloads: download
using OrdinaryDiffEq
using Trixi

# This is a DGMulti version of the UnstructuredMesh2D elixir `elixir_euler_basic.jl`,
# which can be found at `examples/unstructured_2d_dgsem/elixir_euler_basic.jl`.

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_convergence_test
source_terms = source_terms_convergence_test

boundary_condition_convergence_test = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = (; :Slant  => boundary_condition_convergence_test,
                         :Bezier => boundary_condition_convergence_test,
                         :Right  => boundary_condition_convergence_test,
                         :Bottom => boundary_condition_convergence_test,
                         :Top    => boundary_condition_convergence_test )

###############################################################################
# Get the DG approximation space

dg = DGMulti(polydeg = 8, element_type = Quad(), approximation_type = SBP(),
             surface_integral = SurfaceIntegralWeakForm(flux_lax_friedrichs),
             volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha))

###############################################################################
# Get the curved quad mesh from a file (downloads the file if not available locally)

default_mesh_file = joinpath(@__DIR__, "mesh_trixi_unstructured_mesh_docs.mesh")
isfile(default_mesh_file) || download("https://gist.githubusercontent.com/andrewwinters5000/52056f1487853fab63b7f4ed7f171c80/raw/9d573387dfdbb8bce2a55db7246f4207663ac07f/mesh_trixi_unstructured_mesh_docs.mesh",
                                       default_mesh_file)
mesh_file = default_mesh_file

mesh = DGMultiMesh(dg, mesh_file)

###############################################################################
# create the semi discretization object

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg,
                                    source_terms=source_terms,
                                    boundary_conditions=boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 3.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, uEltype=real(dg))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback)

###############################################################################
# run the simulation

time_int_tol = 1e-8
sol = solve(ode, RDPK3SpFSAL49(); abstol=time_int_tol, reltol=time_int_tol,
            dt = time_int_tol, ode_default_options()..., callback=callbacks)

summary_callback() # print the timer summary
