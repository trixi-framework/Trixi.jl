# Analogous to elixir_euler_source_terms.jl, but uses a locally refined cubed sphere mesh
# of the size of the Earth's atmosphere (using an atmospheric height of 30km).
# The initial condition and source terms have also been rescaled to planetary size.

using OrdinaryDiffEq
using Trixi
using LinearAlgebra

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 1.4
equations = CompressibleEulerEquations3D(gamma)


function initial_condition_convergence_test_sphere(x, t, equations::CompressibleEulerEquations3D)
  x_scaled = x / 6.371229e6
  t_scaled = t / 6.371229e6

  return initial_condition_convergence_test(x_scaled, t_scaled, equations)
end

@inline function source_terms_convergence_test_sphere(u, x, t, equations::CompressibleEulerEquations3D)
  x_scaled = x / 6.371229e6
  t_scaled = t / 6.371229e6

  return source_terms_convergence_test(u, x_scaled, t_scaled, equations) / 6.371229e6
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

initial_condition = initial_condition_convergence_test_sphere

boundary_condition = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = Dict(
  :inside  => boundary_condition,
  :outside => boundary_condition
)

surface_flux = flux_hll
# Note that a free stream is not preserved if N < 2 * N_geo, where N is the
# polydeg of the solver and N_geo is the polydeg of the mesh.
# However, the FSP error is negligible in this example.
solver = DGSEM(polydeg=4, surface_flux=surface_flux)

# For performance reasons, only one face of the cubed sphere can be used:

# One face of the cubed sphere
# mapping(xi, eta, zeta) = Trixi.cubed_sphere_mapping(xi, eta, zeta, 6.371229e6, 30000.0, 1)

# trees_per_dimension = (8, 8, 4)
# mesh = P4estMesh(trees_per_dimension, polydeg=4,
#                  mapping=mapping,
#                  initial_refinement_level=0,
#                  periodicity=false)

trees_per_cube_face = (6, 2)
mesh = Trixi.P4estMeshCubedSphere(trees_per_cube_face..., 6.371229e6, 30000.0,
                                  polydeg=4, initial_refinement_level=0)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms=source_terms_convergence_test_sphere,
                                    boundary_conditions=boundary_conditions)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1e5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
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
sol = solve(ode, RDPK3SpFSAL49(); abstol=1.0e-6, reltol=1.0e-6,
            ode_default_options()..., callback=callbacks);

summary_callback() # print the timer summary
