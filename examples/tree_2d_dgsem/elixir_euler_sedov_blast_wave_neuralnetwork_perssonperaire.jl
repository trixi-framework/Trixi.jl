using Downloads: download
using Flux
using BSON: load
network = joinpath(@__DIR__, "modelnnpp-0.904-0.0005.bson")
download("https://github.com/trixi-framework/Trixi_IndicatorNeuralNetwork_networks/raw/main/networks/modelnnpp-0.904-0.0005.bson", network)
model2d = load(network, @__MODULE__)[:model2d]

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
    initial_condition_sedov_blast_wave(x, t, equations::CompressibleEulerEquations2D)

The Sedov blast wave setup based on Flash
- http://flash.uchicago.edu/site/flashcode/user_support/flash_ug_devel/node184.html#SECTION010114000000000000000
"""
function initial_condition_sedov_blast_wave(x, t, equations::CompressibleEulerEquations2D)
  # Set up polar coordinates
  inicenter = SVector(0.0, 0.0)
  x_norm = x[1] - inicenter[1]
  y_norm = x[2] - inicenter[2]
  r = sqrt(x_norm^2 + y_norm^2)

  # Setup based on http://flash.uchicago.edu/site/flashcode/user_support/flash_ug_devel/node184.html#SECTION010114000000000000000
  r0 = 0.21875 # = 3.5 * smallest dx (for domain length=4 and max-ref=6)
  # r0 = 0.5 # = more reasonable setup
  E = 1.0
  p0_inner = 3 * (equations.gamma - 1) * E / (3 * pi * r0^2)
  p0_outer = 1.0e-5 # = true Sedov setup
  # p0_outer = 1.0e-3 # = more reasonable setup

  # Calculate primitive variables
  rho = 1.0
  v1  = 0.0
  v2  = 0.0
  p   = r > r0 ? p0_outer : p0_inner

  return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_sedov_blast_wave

surface_flux = flux_lax_friedrichs
volume_flux  = flux_chandrashekar
basis = LobattoLegendreBasis(3)
indicator_sc = IndicatorNeuralNetwork(equations, basis,
                                      indicator_type=NeuralNetworkPerssonPeraire(),
                                      alpha_max=0.5,
                                      alpha_min=0.001,
                                      alpha_smooth=true,
                                      alpha_continuous=true,
                                      alpha_amr=false,
                                      variable=density_pressure,
                                      network=model2d)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-2.0, -2.0)
coordinates_max = ( 2.0,  2.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=6,
                n_cells_max=100_000)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 12.5)
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
                                       alpha_max=0.5,
                                       alpha_min=0.001,
                                       alpha_smooth=true,
                                       alpha_continuous=true,
                                       alpha_amr=true,
                                       variable=density_pressure,
                                       network=model2d)
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level=4,
                                      max_level =6, max_threshold=0.22)
amr_callback = AMRCallback(semi, amr_controller,
                           interval=5,
                           adapt_initial_condition=true,
                           adapt_initial_condition_only_refine=true)

stepsize_callback = StepsizeCallback(cfl=0.9)

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
