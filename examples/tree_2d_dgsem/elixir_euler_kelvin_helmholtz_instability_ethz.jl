using OrdinaryDiffEq
using Trixi
using StableRNGs


###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)

"""
    initial_condition_kelvin_helmholtz_instability_ethz(x, t, equations::CompressibleEulerEquations2D)

A version of the classical Kelvin-Helmholtz instability based on
- Ulrik S. Fjordholm, Roger Käppeli, Siddhartha Mishra, Eitan Tadmor (2014)
  Construction of approximate entropy measure valued
  solutions for hyperbolic systems of conservation laws
  [arXiv: 1402.0909](https://arxiv.org/abs/1402.0909)
"""
function initial_condition_kelvin_helmholtz_instability_ethz(x, t, equations::CompressibleEulerEquations2D)
  # typical resolution 128^2, 256^2
  # domain size is [0,+1]^2
  # interface is sharp, but randomly perturbed
  rng = StableRNG(100)
  m = 10
  a1 = rand(rng, m)
  a2 = rand(rng, m)
  a1 .= a1 / sum(a1)
  a2 .= a2 / sum(a2)
  b1 = (rand(rng, m) .- 0.5) .* pi
  b2 = (rand(rng, m) .- 0.5) .* pi
  Y1 = 0.0
  Y2 = 0.0
  for n = 1:m
    Y1 += a1[n] * cos(b1[n] + 2 * n * pi * x[1])
    Y2 += a2[n] * cos(b2[n] + 2 * n * pi * x[1])
  end

  J1 = 0.25
  J2 = 0.75
  epsilon = 0.01
  I1 = J1 + epsilon * Y1
  I2 = J2 + epsilon * Y2

  if (x[2] > I1) && (x[2] < I2)
    rho = 2
    v1 = -0.5
  else
    rho = 1
    v1 = 0.5
  end
  v2 = 0
  p = 2.5

  return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_kelvin_helmholtz_instability_ethz

surface_flux = flux_hllc
volume_flux  = flux_ranocha
polydeg = 3
basis = LobattoLegendreBasis(polydeg)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.001,
                                         alpha_min=0.0001,
                                         alpha_smooth=true,
                                         variable=density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)

solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (0.0, 0.0)
coordinates_max = (1.0, 1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=6,
                n_cells_max=100_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 400
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=400,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution)

###############################################################################
# run the simulation
sol = solve(ode, SSPRK43(),
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
