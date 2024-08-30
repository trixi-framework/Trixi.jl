# The same setup as tree_2d_dgsem/elixir_advection_basic.jl
# to verify the StructuredMesh implementation against TreeMesh

using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

gamma = 1.0
k0 = 3.2e-5
rho0 = 1000.0
# G = 9.81
G = 0.0

equations = ThreeEquations2D(gamma, k0, rho0, G)

function smootherstep(left, right, x)
  # Scale, and clamp x to 0..1 range.
  x = clamp((x - left) / (right - left))

  return x * x * x * (x * (6.0 * x - 15.0) + 10.0)
end

@inline function clamp(x, lowerlimit = 0.0, upperlimit = 1.0)
  if (x < lowerlimit) return lowerlimit end
  if (x > upperlimit) return upperlimit end
  return x
end

function initial_condition(x, t, equations::ThreeEquations2D)
  
    r = Trixi.norm(x)

    width = 0.1
    radius = 1.0

    s = smootherstep(radius + 0.5*width, radius - 0.5*width, r)

    rho = 1000.0
    v1 = -100.0 * x[1] * s
    v2 =  100.0 * x[2] * s
    alpha = clamp(s, 1e-3, 1.0)

    # # liquid domain
    # if((x[1]^2 + x[2]^2) <= 1)
    #     rho = 1000.0
    #     alpha = 1.0 - 10^(-3)
    #     v1 = -100.0 * x[1]
    #     v2 = 100.0 * x[2]
    # else
    #     rho = 1000.0
    #     v1 = 0.0
    #     v2 = 0.0
    #     alpha = 10^(-3)
    # end
    # phi = x[2]
    phi = 0.0

      # rho = 1000.0
      # v1 = 0.0
      # v2 = 0.0
      # alpha = 1.0
    
    return prim2cons(SVector(rho, v1, v2, alpha, phi, 0.0), equations)
end

# volume_flux = (flux_central, flux_nonconservative_ThreeEquations)
volume_flux = (Trixi.flux_entropy_cons_gamma_one_ThreeEquations, Trixi.flux_non_cons_entropy_cons_gamma_one_ThreeEquations)

surface_flux = (Trixi.flux_entropy_cons_gamma_one_ThreeEquations, Trixi.flux_non_cons_entropy_cons_gamma_one_ThreeEquations)
# surface_flux = (flux_lax_friedrichs, Trixi.flux_non_cons_entropy_cons_gamma_one_ThreeEquations)

polydeg = 3

basis = LobattoLegendreBasis(polydeg)

indicator_sc = IndicatorHennemannGassner(equations, basis,
                                          alpha_max=1.0,
                                          alpha_min=0.001,
                                          alpha_smooth=true,
                                          variable=alpha_rho)

volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                  volume_flux_dg=volume_flux,
                                                  volume_flux_fv=surface_flux)

# volume_integral = VolumeIntegralFluxDifferencing(volume_flux)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
# solver = DGSEM(basis, surface_flux = surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-3.0, -3.0) # minimum coordinates (min(x), min(y))
coordinates_max = ( 3.0,  3.0) # maximum coordinates (max(x), max(y))

trees_per_dimension = (1, 1)

initial_refinement_level = 6

mesh = P4estMesh(trees_per_dimension, polydeg = polydeg ,
                  coordinates_min = coordinates_min, coordinates_max = coordinates_max,
                  initial_refinement_level = initial_refinement_level)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.0076)

# Create ODE problem with time span from 0.0 to 1.0
ode = semidiscretize(semi, tspan)

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval = 20,
                                     solution_variables = cons2cons)

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval = 100)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl = 0.5)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, stepsize_callback, save_solution)

###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);

# Print the timer summary
summary_callback()

# Finalize `T8codeMesh` to make sure MPI related objects in t8code are
# released before `MPI` finalizes.
!isinteractive() && finalize(mesh)
