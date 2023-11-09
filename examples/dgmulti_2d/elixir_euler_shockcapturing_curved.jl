
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_weak_blast_wave

surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha

polydeg = 3
basis = DGMultiBasis(Quad(), polydeg, approximation_type = GaussSBP())

indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)
dg = DGMulti(basis,
             surface_integral = SurfaceIntegralWeakForm(surface_flux),
             volume_integral = volume_integral)

function mapping(xi, eta)
    x = xi + 0.1 * sin(pi * xi) * sin(pi * eta)
    y = eta + 0.1 * sin(pi * xi) * sin(pi * eta)
    return SVector(x, y)
end
cells_per_dimension = (16, 16)
mesh = DGMultiMesh(dg, cells_per_dimension, mapping)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg)

tspan = (0.0, 0.15)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval = 10)
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval, uEltype = real(dg))
callbacks = CallbackSet(summary_callback, alive_callback, analysis_callback)

###############################################################################
# run the simulation

sol = solve(ode, RDPK3SpFSAL49(); abstol = 1.0e-6, reltol = 1.0e-6,
            ode_default_options()..., callback = callbacks);

summary_callback() # print the timer summary
