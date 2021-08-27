using Downloads: download
using Flux
using NNlib
using BSON: load
network = joinpath(@__DIR__, "modelnnpp-0.97-0.0001.bson")
download("https://gist.github.com/JuliaOd/97728c2c15d6a7255ced6e46e3a605b6/raw/modelnnpp-0.97-0.0001.bson", network)
model1d = load(network, @__MODULE__)[:model1d]

using OrdinaryDiffEq
using Trixi

# This elixir was one of the setups used in the following master thesis:
# - Julia Odenthal (2021)
#   Shock capturing with artifical neural networks
#   University of Cologne, advisors: Gregor Gassner, Michael Schlottke-Lakemper
# This motivates the particular choice of fluxes, mesh resolution etc.


###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations1D(1.4)

initial_condition = initial_condition_blast_wave

surface_flux = flux_lax_friedrichs
volume_flux  = flux_chandrashekar
basis = LobattoLegendreBasis(3)
indicator_sc = IndicatorNeuralNetwork(equations, basis,
                                      indicator_type=NeuralNetworkPerssonPeraire(),
                                      alpha_max=0.5,
                                      alpha_min=0.001,
                                      alpha_smooth=true,
                                      alpha_continuous=false,
                                      alpha_amr=false,
                                      variable=density_pressure,
                                      network=model1d)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-2.0,)
coordinates_max = ( 2.0,)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=6,
                n_cells_max=10_000)


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

stepsize_callback = StepsizeCallback(cfl=0.5)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks, maxiters=1e5);
summary_callback() # print the timer summary
