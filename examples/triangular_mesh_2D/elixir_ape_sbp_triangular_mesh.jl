# !!! warning "Experimental features"

using StartUpDG, StructArrays
using Trixi, OrdinaryDiffEq

rd = RefElemData(Tri(), SBP(), N=4)
dg = DG(rd, nothing #= mortar =#, 
        SurfaceIntegralWeakForm(FluxLaxFriedrichs()), VolumeIntegralWeakForm())

v_mean_global = (0.25, 0.25)
c_mean_global = 1.0
rho_mean_global = 1.0
equations = AcousticPerturbationEquations2D(v_mean_global, c_mean_global, rho_mean_global)

initial_condition = initial_condition_convergence_test
source_terms = source_terms_convergence_test

# example where we tag two separate boundary segments of the mesh
VX, VY, EToV = StartUpDG.uniform_mesh(Tri(), 8)
mesh = VertexMappedMesh(VX, VY, EToV, rd)

boundary_condition_convergence_test = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = (; :entire_boundary => boundary_condition_convergence_test)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg,
                                    source_terms = source_terms, 
                                    boundary_conditions = boundary_conditions) 

tspan = (0.0, 0.1)
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
