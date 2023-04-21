# A manufactured solution of a circular wind with constant angular velocity
# on a planetary-sized non-conforming mesh
#
# Note that this elixir can take several hours to run.
# Using 24 threads of an AMD Ryzen Threadripper 3990X (more threads don't speed it up further)
# and `check-bounds=no`, this elixirs takes about 20 minutes to run.

using OrdinaryDiffEq
using Trixi
using LinearAlgebra

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 1.4
equations = CompressibleEulerEquations3D(gamma)


function initial_condition_circular_wind(x, t, equations::CompressibleEulerEquations3D)
  radius_earth = 6.371229e6
  p = 1e5
  rho = 1.0
  v1 = -10 * x[2] / radius_earth
  v2 = 10 * x[1] / radius_earth
  v3 = 0.0

  return prim2cons(SVector(rho, v1, v2, v3, p), equations)
end

@inline function source_terms_circular_wind(u, x, t, equations::CompressibleEulerEquations3D)
  radius_earth = 6.371229e6
  rho = 1.0

  du1 = 0.0
  du2 = -rho * (10 / radius_earth) * (10 * x[1] / radius_earth)
  du3 = -rho * (10 / radius_earth) * (10 * x[2] / radius_earth)
  du4 = 0.0
  du5 = 0.0

  return SVector(du1, du2, du3, du4, du5)
end


function indicator_test(u::AbstractArray{<:Any,5},
                        mesh, equations, dg::DGSEM, cache;
                        kwargs...)
  alpha = zeros(Int, nelements(dg, cache))

  for element in eachelement(dg, cache)
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

function cart_to_sphere(x)
  r = norm(x)
  lambda = atan(x[2], x[1])
  if lambda < 0
    lambda += 2 * pi
  end
  phi = asin(x[3] / r)

  return lambda, phi, r
end

function Trixi.get_element_variables!(element_variables, indicator::typeof(indicator_test), ::AMRCallback)
  return nothing
end

initial_condition = initial_condition_circular_wind

boundary_conditions = Dict(
  :inside  => boundary_condition_slip_wall,
  :outside => boundary_condition_slip_wall
)

# The speed of sound in this example is 374 m/s.
surface_flux = FluxLMARS(374)
# Note that a free stream is not preserved if N < 2 * N_geo, where N is the
# polydeg of the solver and N_geo is the polydeg of the mesh.
# However, the FSP error is negligible in this example.
solver = DGSEM(polydeg=4, surface_flux=surface_flux)

# Other mesh configurations to run this simulation on.
# The cylinder allows to use a structured mesh, the face of the cubed sphere
# can be used for performance reasons.

# # Cylinder
# function mapping(xi, eta, zeta)
#   radius_earth = 6.371229e6

#   x = cos((xi + 1) * pi) * (radius_earth + (0.5 * zeta + 0.5) * 30000)
#   y = sin((xi + 1) * pi) * (radius_earth + (0.5 * zeta + 0.5) * 30000)
#   z = eta * 1000000

#   return SVector(x, y, z)
# end

# # One face of the cubed sphere
# mapping(xi, eta, zeta) = Trixi.cubed_sphere_mapping(xi, eta, zeta, 6.371229e6, 30000.0, 1)

# trees_per_dimension = (8, 8, 4)
# mesh = P4estMesh(trees_per_dimension, polydeg=3,
#                  mapping=mapping,
#                  initial_refinement_level=0,
#                  periodicity=false)

trees_per_cube_face = (6, 2)
mesh = Trixi.P4estMeshCubedSphere(trees_per_cube_face..., 6.371229e6, 30000.0,
                                  polydeg=4, initial_refinement_level=0)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms=source_terms_circular_wind,
                                    boundary_conditions=boundary_conditions)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 10 * 24 * 60 * 60.0) # time in seconds for 10 days
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 5000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=5000,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

amr_controller = ControllerThreeLevel(semi, indicator_test,
                                      base_level=0,
                                      max_level=1, max_threshold=0.6)
amr_callback = AMRCallback(semi, amr_controller,
                           interval=0, # Only initial refinement
                           adapt_initial_condition=true,
                           adapt_initial_condition_only_refine=true)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        amr_callback)


###############################################################################
# run the simulation

# Use a Runge-Kutta method with automatic (error based) time step size control
# Enable threading of the RK method for better performance on multiple threads
sol = solve(ode, RDPK3SpFSAL49(thread=OrdinaryDiffEq.True()); abstol=1.0e-6, reltol=1.0e-6,
            ode_default_options()..., callback=callbacks);

summary_callback() # print the timer summary
