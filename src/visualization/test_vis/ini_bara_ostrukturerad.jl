
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_convergence_test
function sine_IC(xyz,t,equations::CompressibleEulerEquations2D)
    x,y = xyz
    rho = 2 + sin(.1*pi*x)*sin(.1*pi*y)
    u,v = 1.0,1.0
    p = 1.0
    return prim2cons(SVector{4}(rho,u,v,p),equations)
end
initial_condition = sine_IC
source_terms = source_terms_convergence_test

boundary_condition_initial_condition = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = Dict( :Bottom => boundary_condition_initial_condition,
                            :Top    => boundary_condition_initial_condition,
                            :Right  => boundary_condition_initial_condition,
                            :Left   => boundary_condition_initial_condition,
                            :Circle => boundary_condition_initial_condition)

###############################################################################
# Get the DG approximation space

solver = DGSEM(polydeg=4, surface_flux=flux_hll)

###############################################################################
# Get the curved quad mesh from a file

# mesh_file = joinpath(@__DIR__, "mesh_box_around_circle.mesh")
mesh_file = joinpath(@__DIR__, "Trixi_hexe.mesh")

mesh = UnstructuredQuadMesh(mesh_file)

###############################################################################
# create the semi discretization object

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_terms,
                                    boundary_conditions=boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=10,
                                     save_initial_solution=true,
                                     save_final_solution=true)

stepsize_callback = StepsizeCallback(cfl=1.0)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution, stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary


# # visualization 
# using NodesAndModes, GLMakie, Triangulate, GeometryBasics
# include("../makie_visualization.jl")

using GLMakie
trixi_pcolor(sol,1)
trixi_wireframe!(sol,1,color=:white) 
Makie.current_figure()


