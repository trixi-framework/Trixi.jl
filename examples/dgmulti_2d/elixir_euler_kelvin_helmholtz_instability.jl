using Trixi, OrdinaryDiffEq

dg = DGMulti(polydeg = 3, element_type = Quad(), approximation_type = SBP(),
             surface_integral = SurfaceIntegralWeakForm(FluxLaxFriedrichs()),
             volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha))

equations = CompressibleEulerEquations2D(1.4)
initial_condition = initial_condition_khi

vertex_coordinates, EToV = StartUpDG.uniform_mesh(dg.basis.elementType, 32)
mesh = VertexMappedMesh(vertex_coordinates, EToV, dg, is_periodic=(true, true))
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg)

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval=10)
callbacks = CallbackSet(summary_callback,
                        alive_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt= estimate_dt(mesh, dg), save_everystep=false, callback=callbacks);

summary_callback() # print the timer summary
