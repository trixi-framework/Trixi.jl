
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible ideal GLM-MHD equations

c_h = 1.0
equations = IdealGlmMhdEquations2D(1.4, c_h)

initial_condition = initial_condition_weak_blast_wave

surface_flux = (flux_hindenlang_gassner, flux_nonconservative_powell)
volume_flux = (flux_hindenlang_gassner, flux_nonconservative_powell)
dg = DGMulti(polydeg=1, element_type = Quad(), approximation_type = Polynomial(),
             surface_integral = SurfaceIntegralWeakForm(surface_flux),
             volume_integral = VolumeIntegralFluxDifferencing(volume_flux))


num_cells_per_dimension = 4
vertex_coordinates, EToV = StartUpDG.uniform_mesh(dg.basis.elementType, num_cells_per_dimension)
vertex_coordinates = map(x -> 2.0 .* x, vertex_coordinates) # map domain to [-2, 2]^2
mesh = VertexMappedMesh(vertex_coordinates, EToV, dg, is_periodic=(true, true))

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.1)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, uEltype=real(dg))
alive_callback = AliveCallback(alive_interval=10, analysis_interval=analysis_interval)

cfl = .50
stepsize_callback = StepsizeCallback(cfl=cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale=0.5, cfl=.50)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        stepsize_callback,
                        alive_callback)


###############################################################################
# run the simulation

tol = 1.0e-8
sol = solve(ode, RDPK3SpFSAL49(), abstol=tol, reltol=tol,
            save_everystep=false, callback=callbacks);

summary_callback() # print the timer summary
