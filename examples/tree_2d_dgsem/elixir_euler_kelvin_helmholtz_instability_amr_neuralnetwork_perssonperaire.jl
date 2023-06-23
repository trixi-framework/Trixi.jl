using Downloads: download
using Flux
using BSON: load
network = joinpath(@__DIR__, "modelnnpp-0.904-0.0005.bson")
download("https://github.com/trixi-framework/Trixi_IndicatorNeuralNetwork_networks/raw/main/networks/modelnnpp-0.904-0.0005.bson", network)
model2d = load(network, @__MODULE__)[:model2d]

using Random: seed!
seed!(0)

using OrdinaryDiffEq
using Trixi

# This elixir was one of the setups used in the following master thesis:
# - Julia Odenthal (2021)
#   Shock capturing with artificial neural networks
#   University of Cologne, advisors: Gregor Gassner, Michael Schlottke-Lakemper
# This motivates the particular choice of fluxes, mesh resolution etc.

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)

"""
    initial_condition_kelvin_helmholtz_instability(x, t, equations::CompressibleEulerEquations2D)

A version of the classical Kelvin-Helmholtz instability based on
https://rsaa.anu.edu.au/research/established-projects/fyris/2-d-kelvin-helmholtz-test.
"""
function initial_condition_kelvin_helmholtz_instability(x, t, equations::CompressibleEulerEquations2D)
  # change discontinuity to tanh
  # typical resolution 128^2, 256^2
  # domain size is [-0.5,0.5]^2
  dens0 = 1.0 # outside density
  dens1 = 2.0 # inside density
  velx0 = -0.5 # outside velocity
  velx1 = 0.5 # inside velocity
  slope = 50 # used for tanh instead of discontinuous initial condition
  # pressure equilibrium
  p     = 2.5
  #  y velocity v2 is only white noise
  v2  = 0.01*(rand(Float64,1)[1]-0.5)
  # density
  rho = dens0 + (dens1-dens0) * 0.5*(1+(tanh(slope*(x[2]+0.25)) - (tanh(slope*(x[2]-0.25)) + 1)))
  #  x velocity is also augmented with noise
  v1 = velx0 + (velx1-velx0) * 0.5*(1+(tanh(slope*(x[2]+0.25)) - (tanh(slope*(x[2]-0.25)) + 1)))+0.01*(rand(Float64,1)[1]-0.5)
  return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_kelvin_helmholtz_instability

surface_flux = flux_lax_friedrichs
volume_flux  = flux_chandrashekar
polydeg = 3
basis = LobattoLegendreBasis(polydeg)
indicator_sc = IndicatorNeuralNetwork(equations, basis,
                                      indicator_type=NeuralNetworkPerssonPeraire(),
                                      alpha_max=0.002,
                                      alpha_min=0.0001,
                                      alpha_smooth=true,
                                      alpha_continuous=true,
                                      alpha_amr=false,
                                      variable=density_pressure,
                                      network=model2d)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-0.5, -0.5)
coordinates_max = ( 0.5,  0.5)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=5,
                n_cells_max=100_000)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 5.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

amr_indicator = IndicatorNeuralNetwork(semi,
                                       indicator_type=NeuralNetworkPerssonPeraire(),
                                       alpha_max=1.0,
                                       alpha_min=0.0001,
                                       alpha_smooth=false,
                                       alpha_continuous=true,
                                       alpha_amr=true,
                                       variable=density_pressure,
                                       network=model2d)
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level=4,
                                      med_level=6, med_threshold=0.3, # med_level = current level
                                      max_level=7, max_threshold=0.5)
amr_callback = AMRCallback(semi, amr_controller,
                           interval=1,
                           adapt_initial_condition=true,
                           adapt_initial_condition_only_refine=true)

stepsize_callback = StepsizeCallback(cfl=1.3)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        amr_callback, stepsize_callback)


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
