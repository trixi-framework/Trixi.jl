using OrdinaryDiffEqLowStorageRK
using Trixi

surface_flux = FluxLaxFriedrichs(max_abs_speed_naive)
volume_flux = flux_ranocha
dg = DGMulti(polydeg = 3, element_type = Tri(), approximation_type = Polynomial(),
             surface_integral = SurfaceIntegralWeakForm(surface_flux),
             volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

equations = CompressibleEulerEquations2D(1.4)
equations_parabolic = LaplaceDiffusionEntropyVariables2D(0.001, equations)

initial_condition = initial_condition_weak_blast_wave

cells_per_dimension = (16, 16)
mesh = DGMultiMesh(dg, cells_per_dimension,
                   coordinates_min = (-1.0, -1.0), coordinates_max = (1.0, 1.0),
                   periodicity = true)
semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, dg)

tspan = (0.0, 1.1)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval = 10)
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval, uEltype = real(dg))
callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback)

###############################################################################
# run the simulation

alg = RDPK3SpFSAL35()
sol = solve(ode, alg; abstol = 1.0e-6, reltol = 1.0e-6,
            ode_default_options()..., callback = callbacks);
