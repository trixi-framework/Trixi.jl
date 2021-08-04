
using Random: seed!
using OrdinaryDiffEq
using Trixi
using LinearAlgebra

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 1.4
equations = CompressibleEulerEquations3D(gamma)


function initial_condition_test(x, t, equations::CompressibleEulerEquations3D)
  RadEarth = 6371220.0 # Earth radius
  p = 1e5
  rho = 100.0
  v1 = -10 * x[2] / RadEarth
  v2 = 10 * x[1] / RadEarth
  v3 = 0.0

  return prim2cons(SVector(rho, v1, v2, v3, p), equations)
end

@inline function source_terms_test(u, x, t, equations::CompressibleEulerEquations3D)
  RadEarth = 6371220.0 # Earth radius
  rho = 100.0

  du1 = 0.0
  du2 = -rho * (10 / RadEarth) * (10 * x[1] / RadEarth)
  du3 = -rho * (10 / RadEarth) * (10 * x[2] / RadEarth)
  du4 = 0.0
  du5 = 0.0

  return SVector(du1, du2, du3, du4, du5)
end

function flux_mars(u_ll, u_rr, orientation::Integer, equations::CompressibleEulerEquations3D)
  # This only works in x-direction
  @assert orientation == 1

  cS = 360

  sRho_L = 1 / u_ll[1]
  sRho_R = 1 / u_rr[1]

  Vel_L_1 = u_ll[2] * sRho_L
  Vel_L_2 = u_ll[3] * sRho_L
  Vel_L_3 = u_ll[4] * sRho_L
  Vel_R_1 = u_rr[2] * sRho_R
  Vel_R_2 = u_rr[3] * sRho_R
  Vel_R_3 = u_rr[4] * sRho_R

  p_L = (equations.gamma - 1) * (u_ll[5] - 0.5 * (u_ll[2] * Vel_L_1 + u_ll[3] * Vel_L_2 + u_ll[4] * Vel_L_3))
  p_R = (equations.gamma - 1) * (u_rr[5] - 0.5 * (u_rr[2] * Vel_R_1 + u_rr[3] * Vel_R_2 + u_rr[4] * Vel_R_3))
  rhoM = 0.5 * (u_ll[1] + u_rr[1])
  pM = 0.5*(p_L + p_R) -0.5*cS*rhoM*(Vel_R_1 - Vel_L_1)
  vM = 0.5*(Vel_R_1 + Vel_L_1) -1.0/(2.0*rhoM*cS)*(p_R - p_L)
  if vM >= 0
    f1 = u_ll[1] * vM
    f2 = u_ll[2] * vM
    f3 = u_ll[3] * vM
    f4 = u_ll[4] * vM
    f5 = u_ll[5] * vM

    f2 += pM
    f5 += pM*vM
  else
    f1 = u_rr[1] * vM
    f2 = u_rr[2] * vM
    f3 = u_rr[3] * vM
    f4 = u_rr[4] * vM
    f5 = u_rr[5] * vM

    f2 += pM
    f5 += pM*vM
  end

  return SVector(f1, f2, f3, f4, f5)
end

function indicator_test(u::AbstractArray{<:Any,5},
                        equations, dg::DGSEM, cache;
                        kwargs...)
  alpha = zeros(Int, nelements(dg, cache))

  for element in eachelement(dg, cache)
    # Calculate coordinates at Gauss-Lobatto nodes
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
      x = Trixi.get_node_coords(cache.elements.node_coordinates, equations, dg, i, j, k, element)
      r = norm(x)
      if r > 6371220 + 16000
        alpha[element] = 1
      end
    end
  end

  return alpha
end

function Trixi.get_element_variables!(element_variables, indicator::typeof(indicator_test), ::AMRCallback)
  return nothing
end

initial_condition = initial_condition_test

# boundary_condition = BoundaryConditionWall(boundary_state_slip_wall)
boundary_condition = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = Dict(
  # :inside  => boundary_condition,
  # :outside => boundary_condition,
  :x_neg => boundary_condition,
  :x_pos => boundary_condition,
  :y_neg => boundary_condition,
  :y_pos => boundary_condition,
  :z_neg => boundary_condition,
  :z_pos => boundary_condition,
)

# surface_flux = flux_lax_friedrichs
# surface_flux = FluxRotated(flux_mars)
surface_flux = FluxRotated(flux_hllc)
volume_flux  = flux_kennedy_gruber
solver = DGSEM(polydeg=5, surface_flux=surface_flux, volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

# # Cylinder
# function mapping(xi, eta, zeta)
#   RadEarth = 6371220.0 # Earth radius

#   x = cos((xi + 1) * pi) * (RadEarth + (0.5 * zeta + 0.5) * 30000)
#   y = sin((xi + 1) * pi) * (RadEarth + (0.5 * zeta + 0.5) * 30000)
#   z = eta * 1000000

#   return SVector(x, y, z)
# end

# One face of the cubed sphere
mapping(xi, eta, zeta) = Trixi.cubed_sphere_mapping(xi, eta, zeta, 6371220.0, 30000.0, 1)

trees_per_dimension = (8, 8, 4)
mesh = P4estMesh(trees_per_dimension, polydeg=5,
                 mapping=mapping,
                 initial_refinement_level=1,
                 periodicity=(false, false, false))

# mesh = Trixi.P4estMeshCubedSphere(4, 4, 6371220.0, 30000.0,
#                                   polydeg=5, initial_refinement_level=0)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms=source_terms_test,
                                    boundary_conditions=boundary_conditions)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 60 * 60.0)
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
                                      max_level=2, max_threshold=0.6)
amr_callback = AMRCallback(semi, amr_controller,
                           interval=typemax(Int),
                           adapt_initial_condition=true,
                           adapt_initial_condition_only_refine=true)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
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
