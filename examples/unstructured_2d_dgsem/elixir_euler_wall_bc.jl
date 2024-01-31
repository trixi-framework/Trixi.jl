
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

@inline function uniform_flow_state(x, t, equations::CompressibleEulerEquations2D)

    # set the freestream flow parameters
    rho_freestream = 1.0
    u_freestream = 0.3
    p_freestream = inv(equations.gamma)

    theta = pi / 90.0 # analogous with a two degree angle of attack
    si, co = sincos(theta)
    v1 = u_freestream * co
    v2 = u_freestream * si

    prim = SVector(rho_freestream, v1, v2, p_freestream)
    return prim2cons(prim, equations)
end

initial_condition = uniform_flow_state

boundary_condition_uniform_flow = BoundaryConditionDirichlet(uniform_flow_state)
boundary_conditions = Dict(:Bottom => boundary_condition_uniform_flow,
                           :Top => boundary_condition_uniform_flow,
                           :Right => boundary_condition_uniform_flow,
                           :Left => boundary_condition_uniform_flow,
                           :Circle => boundary_condition_slip_wall)

###############################################################################
# Get the DG approximation space

solver = DGSEM(polydeg = 4, surface_flux = flux_hll)

###############################################################################
# Get the curved quad mesh from a file
mesh_file = Trixi.download("https://gist.githubusercontent.com/andrewwinters5000/8b9b11a1eedfa54b215c122c3d17b271/raw/0d2b5d98c87e67a6f384693a8b8e54b4c9fcbf3d/mesh_box_around_circle.mesh",
                           joinpath(@__DIR__, "mesh_box_around_circle.mesh"))

mesh = UnstructuredMesh2D(mesh_file)

###############################################################################
# create the semi discretization object

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 10,
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
