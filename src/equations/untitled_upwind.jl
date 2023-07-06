
using Downloads: download
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

# Free-stream initial condition
initial_condition = initial_condition_constant

# Boundary conditions for free-stream testing
boundary_condition_free_stream = BoundaryConditionDirichlet(initial_condition)

# boundary_conditions = Dict( :Body    => boundary_condition_free_stream,
#                             :Button1 => boundary_condition_free_stream,
#                             :Button2 => boundary_condition_free_stream,
#                             :Eye1    => boundary_condition_free_stream,
#                             :Eye2    => boundary_condition_free_stream,
#                             :Smile   => boundary_condition_free_stream,
#                             :Bowtie  => boundary_condition_free_stream )

# for a warped style mesh
# boundary_conditions = Dict( :Top    => boundary_condition_free_stream,
#                             :Bottom => boundary_condition_free_stream,
#                             :Right  => boundary_condition_free_stream,
#                             :Left   => boundary_condition_free_stream )

# for a warped style mesh
boundary_conditions = Dict(:Top => boundary_condition_free_stream,
                           :Bottom => boundary_condition_free_stream,
                           :Right => boundary_condition_free_stream,
                           :Slant => boundary_condition_free_stream,
                           :Bezier => boundary_condition_free_stream)

# for the outer circle mesh
# boundary_conditions = Dict(:OuterCircle => boundary_condition_free_stream)

# for 3 circles mesh
# boundary_conditions = Dict(:outer => boundary_condition_free_stream,
#                            :Circle1 => boundary_condition_free_stream,
#                            :Circle2 => boundary_condition_free_stream,
#                            :Circle3 => boundary_condition_free_stream)

###############################################################################
# Get the Upwind FDSBP approximation space

D_upw = upwind_operators(SummationByPartsOperators.Mattsson2017,
                         derivative_order = 1,
                         accuracy_order = 4,
                         xmin = -1.0, xmax = 1.0,
                         N = 18)

flux_splitting = splitting_lax_friedrichs
solver = FDSBP(D_upw,
               surface_integral = SurfaceIntegralStrongForm(FluxUpwind(flux_splitting)),
               #    volume_integral=VolumeIntegralStrongForm())
               volume_integral = VolumeIntegralUpwind(flux_splitting))

###############################################################################
# Get the curved quad mesh from a file (downloads the file if not available locally)
default_mesh_file = joinpath(@__DIR__, "mesh_multiple_flips.mesh")
isfile(default_mesh_file) ||
    download("https://gist.githubusercontent.com/andrewwinters5000/b434e724e3972a9c4ee48d58c80cdcdb/raw/9a967f066bc5bf081e77ef2519b3918e40a964ed/mesh_multiple_flips.mesh",
             default_mesh_file)

# MADE A NEW FLIPPED ORIENTAION MESH
#default_mesh_file = joinpath(@__DIR__, "mesh_new_flip.mesh")
# default_mesh_file = joinpath(@__DIR__, "mesh_new_flip_corners.mesh")
# default_mesh_file = joinpath(@__DIR__, "WarpedMesh_00.mesh")

# Circular mesh with flips
default_mesh_file = joinpath(@__DIR__, "mesh_docs.mesh")
# default_mesh_file = joinpath(@__DIR__, "mesh_outer_circle.mesh")
# default_mesh_file = joinpath(@__DIR__, "Circles3.mesh")

mesh_file = default_mesh_file

mesh = UnstructuredMesh2D(mesh_file)

###############################################################################
# create the semi discretization object

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 5.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 1000,
                                     save_initial_solution = true,
                                     save_final_solution = true)

callbacks = CallbackSet(summary_callback, analysis_callback,
                        alive_callback, save_solution)

###############################################################################
# run the simulation

# set small tolerances for the free-stream preservation test
sol = solve(ode, SSPRK43(), abstol = 1.0e-12, reltol = 1.0e-12,
            save_everystep = false, callback = callbacks)
summary_callback() # print the timer summary
