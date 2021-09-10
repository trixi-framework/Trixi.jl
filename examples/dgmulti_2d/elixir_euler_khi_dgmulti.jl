using Trixi, OrdinaryDiffEq

dg = DGMulti(polydeg = 3, element_type = Quad(), approximation_type = Polynomial(),
             surface_integral = SurfaceIntegralWeakForm(FluxLaxFriedrichs()),
             volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha))

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_kelvin_helmholtz_instability
# initial_condition = initial_condition_weak_blast_wave

num_cells_per_dimension = 32
vertex_coordinates, EToV = StartUpDG.uniform_mesh(dg.basis.elementType, num_cells_per_dimension)
mesh = VertexMappedMesh(vertex_coordinates, EToV, dg, is_periodic=(true, true))
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg)

tspan = (0.0, 10.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval=100)
analysis_interval = 500
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, uEltype=real(dg))
callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback)

# callbacks = CallbackSet(summary_callback)

###############################################################################
# run the simulation

# sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
#             dt = estimate_dt(mesh, dg), save_everystep=false, callback=callbacks);
tsave = LinRange(tspan..., 100)
sol = solve(ode, RDPK3SpFSAL49(), abstol=1.0e-6, reltol=1.0e-6,
            save_everystep=false, saveat=tsave, callback=callbacks);

summary_callback() # print the timer summary

