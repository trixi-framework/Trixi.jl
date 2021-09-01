
using Random: seed!
using OrdinaryDiffEq
using Trixi
using LinearAlgebra

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 1.4
equations = CompressibleEulerEquations3D(gamma)

function initial_condition_convergence_test_sphere(x_, t_, equations::CompressibleEulerEquations3D)
  x = x_ / 6371220
  t = t_ / 6371220
  c = 2
  A = 0.1
  L = 2
  f = 1/L
  ω = 2 * pi * f
  ini = c + A * sin(ω * (x[1] + x[2] + x[3] - t))

  rho = ini
  rho_v1 = ini
  rho_v2 = ini
  rho_v3 = ini
  rho_e = ini^2

  return SVector(rho, rho_v1, rho_v2, rho_v3, rho_e)
end

@inline function source_terms_convergence_test_sphere(u, x_, t_, equations::CompressibleEulerEquations3D)
  x = x_ / 6371220
  t = t_ / 6371220
  # Same settings as in `initial_condition`
  c = 2
  A = 0.1
  L = 2
  f = 1/L
  ω = 2 * pi * f
  γ = equations.gamma

  x1, x2, x3 = x
  si, co = sincos(((x1 + x2 + x3) - t) * ω)
  tmp1 = si * A
  tmp2 = co * A * ω
  tmp3 = ((((((4 * tmp1 * γ - 4 * tmp1) + 4 * c * γ) - 4c) - 3γ) + 7) * tmp2) / 2

  du1 = 2 * tmp2
  du2 = tmp3
  du3 = tmp3
  du4 = tmp3
  du5 = ((((((12 * tmp1 * γ - 4 * tmp1) + 12 * c * γ) - 4c) - 9γ) + 9) * tmp2) / 2

  # Original terms (without performance enhancements)
  # tmp2 = ((((((4 * sin(((x1 + x2 + x3) - t) * ω) * A * γ - 4 * sin(((x1 + x2 + x3) - t) * ω) * A) + 4 * c * γ) - 4c) - 3γ) + 7) * cos(((x1 + x2 + x3) - t) * ω) * A * ω) / 2
  # du1 = 2 * cos(((x1 + x2 + x3) - t) * ω) * A * ω
  # du2 = tmp2
  # du3 = tmp2
  # du4 = tmp2
  # du5 = ((((((12 * sin(((x1 + x2 + x3) - t) * ω) * A * γ - 4 * sin(((x1 + x2 + x3) - t) * ω) * A) + 12 * c * γ) - 4c) - 9γ) + 9) * cos(((x1 + x2 + x3) - t) * ω) * A * ω) / 2

  return SVector(du1, du2, du3, du4, du5) / 6371220
end


function indicator_test(u::AbstractArray{<:Any,5},
                        mesh, equations, dg::DGSEM, cache;
                        kwargs...)
  alpha = zeros(Int, nelements(dg, cache))

  for element in eachelement(dg, cache)
    # Calculate coordinates at Gauss-Lobatto nodes
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
      x = Trixi.get_node_coords(cache.elements.node_coordinates, equations, dg, i, j, k, element)
      lambda, phi, r = cart_to_sphere(x)
      if 0.22 < lambda < 3.3 && 0.45 < phi < 1.3
        alpha[element] = 1
      end
    end
  end

  return alpha
end

function Trixi.get_element_variables!(element_variables, indicator::typeof(indicator_test), ::AMRCallback)
  return nothing
end

initial_condition = initial_condition_convergence_test_sphere

# boundary_condition = BoundaryConditionWall(boundary_state_slip_wall)
boundary_condition = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = Dict(
  :inside  => boundary_condition,
  :outside => boundary_condition,
  # :x_neg => boundary_condition,
  # :x_pos => boundary_condition,
  # :y_neg => boundary_condition,
  # :y_pos => boundary_condition,
  # :z_neg => boundary_condition,
  # :z_pos => boundary_condition,
)

# surface_flux = flux_lax_friedrichs
# surface_flux = FluxRotated(flux_mars)
surface_flux = FluxRotated(flux_hllc)
volume_flux  = flux_kennedy_gruber
solver = DGSEM(polydeg=8, surface_flux=surface_flux, volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

# # One face of the cubed sphere
# mapping(xi, eta, zeta) = Trixi.cubed_sphere_mapping(xi, eta, zeta, 6371220.0, 30000.0, 1)

# trees_per_dimension = (8, 8, 4)
# mesh = P4estMesh(trees_per_dimension, polydeg=5,
#                  mapping=mapping,
#                  initial_refinement_level=0,
#                  periodicity=(false, false, false))

mesh = Trixi.P4estMeshCubedSphere(8, 4, 6371220.0, 30000.0,
                                  polydeg=4, initial_refinement_level=0)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms=source_terms_convergence_test_sphere,
                                    boundary_conditions=boundary_conditions)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 5.0 * 6371220)
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

amr_controller = ControllerThreeLevel(semi, indicator_test,
                                      base_level=0,
                                      max_level=1, max_threshold=0.6)
amr_callback = AMRCallback(semi, amr_controller,
                           interval=typemax(Int),
                           adapt_initial_condition=true,
                           adapt_initial_condition_only_refine=true)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        amr_callback,
                        stepsize_callback)


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);

# use a Runge-Kutta method with automatic (error based) time step size control
# sol = solve(ode, RDPK3SpFSAL49(), abstol=1.0e-6, reltol=1.0e-6,
#             save_everystep=false, callback=callbacks);

summary_callback() # print the timer summary
