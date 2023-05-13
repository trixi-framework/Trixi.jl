
# using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_weak_blast_wave

surface_flux = flux_lax_friedrichs
volume_flux  = flux_ranocha

polydeg = 3
basis = DGMultiBasis(Quad(), polydeg, approximation_type=GaussSBP())

indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.5,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         variable=density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)
dg = DGMulti(basis,
             surface_integral = SurfaceIntegralWeakForm(FluxHLL()),
             volume_integral = volume_integral)

cells_per_dimension = (8, 8)
mesh = DGMultiMesh(dg, cells_per_dimension, periodicity=true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg)

tspan = (0.0, 0.4)
ode = semidiscretize(semi, tspan)

u = ode.u0
du = similar(u)
indicator_sc(u, mesh, equations, dg, semi.cache)

Trixi.calc_volume_integral!(du, u, mesh, Trixi.have_nonconservative_terms(equations),
                            equations, volume_integral, dg, semi.cache)
