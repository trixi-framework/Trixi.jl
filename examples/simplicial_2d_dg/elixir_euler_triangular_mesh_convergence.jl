# !!! warning "Experimental features"

# run using 
# convergence_test(joinpath(examples_dir(), "triangular_mesh_2D", "elixir_euler_triangular_mesh_convergence.jl"), 4)

using StartUpDG
using Trixi, OrdinaryDiffEq

polydeg = 3
rd = RefElemData(Tri(), polydeg)
dg = DG(rd, nothing #= mortar =#, 
        SurfaceIntegralWeakForm(FluxLaxFriedrichs()), VolumeIntegralWeakForm())

equations = CompressibleEulerEquations2D(1.4)        
initial_condition = initial_condition_convergence_test
source_terms = source_terms_convergence_test

# example where we tag two separate boundary segments of the mesh
cells_per_dimension = (8,8) # detected by `extract_initial_resolution` for convergence tests
VX, VY, EToV = StartUpDG.uniform_mesh(Tri(), cells_per_dimension...)
mesh = VertexMappedMesh(VX, VY, EToV, rd)

boundary_condition_convergence_test = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = (; :entire_boundary => boundary_condition_convergence_test)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg,
                                    source_terms = source_terms, 
                                    boundary_conditions = boundary_conditions) 

tspan = (0.0, 0.25)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval=10)
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, uEltype=real(dg))
callbacks = CallbackSet(summary_callback, alive_callback, analysis_callback)

###############################################################################
# run the simulation

dt0 = StartUpDG.estimate_h(rd,mesh.md) / StartUpDG.inverse_trace_constant(rd)
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt = .5*dt0, save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary

l2,linf = analysis_callback(sol)
