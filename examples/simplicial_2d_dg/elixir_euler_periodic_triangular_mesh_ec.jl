# !!! warning "Experimental features"

using StartUpDG
using Trixi, OrdinaryDiffEq

polydeg = 3
rd = RefElemData(Tri(), polydeg)

dg = DG(rd, nothing #= mortar =#,
        SurfaceIntegralWeakForm(flux_ranocha),
        VolumeIntegralFluxDifferencing(flux_ranocha))

equations = CompressibleEulerEquations2D(1.4)
initial_condition = initial_condition_weak_blast_wave
source_terms = nothing

# example where we tag two separate boundary segments of the mesh
vertex_coordinates_x, vertex_coordinates_y, EToV = StartUpDG.uniform_mesh(Tri(), 4)
mesh = VertexMappedMesh(vertex_coordinates_x, vertex_coordinates_y, EToV, rd, is_periodic=(true,true))
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg,
                                    source_terms = source_terms)

tspan = (0.0, 0.4)
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
            dt = 0.5*dt0, save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
