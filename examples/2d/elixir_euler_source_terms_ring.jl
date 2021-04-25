
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_convergence_test

source_terms = source_terms_convergence_test

solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

function gnomonic_projection(inner_radius, thickness, offset)
  R1 = inner_radius
  R2 = R1 + thickness

  alpha(xi) = xi * pi/4 + offset
  radius(eta) = R1 + R2 * (0.5 * (eta + 1))

  mapping(eta, xi) = radius(eta) * SVector(cos(alpha(xi)), sin(alpha(xi)))
end

mapping_as_string(offset) = """
  function gnomonic_projection(inner_radius, thickness, offset)
    R1 = inner_radius;
    R2 = R1 + thickness;

    alpha(xi) = xi * pi/4 + offset;
    radius(eta) = R1 + R2 * (0.5 * (eta + 1));

    mapping(eta, xi) = radius(eta) * SVector(cos(alpha(xi)), sin(alpha(xi)));
  end; mapping = gnomonic_projection(1, 1, $offset)
"""


indices_y_neg = (:i, 1)
indices_y_pos = (:i, :end)

mesh1 = CurvedMesh((16, 16), gnomonic_projection(1, 1, 0), 
                   periodicity=false, mapping_as_string=mapping_as_string(0))

semi1 = SemidiscretizationHyperbolic(mesh1, equations, initial_condition, solver,
  source_terms=source_terms, boundary_conditions=(
    x_neg=boundary_condition_convergence_test,
    x_pos=boundary_condition_convergence_test,
    y_neg=Trixi.BoundaryConditionCoupled(4, 2, indices_y_pos, 2, Float64),
    y_pos=Trixi.BoundaryConditionCoupled(2, 2, indices_y_neg, 2, Float64),
  ))

mesh2 = CurvedMesh((16, 16), gnomonic_projection(1, 1, pi/2), 
                   periodicity=false, mapping_as_string=mapping_as_string(pi/2))

semi2 = SemidiscretizationHyperbolic(mesh2, equations, initial_condition, solver,
  source_terms=source_terms, boundary_conditions=(
    x_neg=boundary_condition_convergence_test,
    x_pos=boundary_condition_convergence_test,
    y_neg=Trixi.BoundaryConditionCoupled(1, 2, indices_y_pos, 2, Float64),
    y_pos=Trixi.BoundaryConditionCoupled(3, 2, indices_y_neg, 2, Float64),
  ))

mesh3 = CurvedMesh((16, 16), gnomonic_projection(1, 1, 2 * pi/2), 
                   periodicity=false, mapping_as_string=mapping_as_string(pi))

semi3 = SemidiscretizationHyperbolic(mesh3, equations, initial_condition, solver,
  source_terms=source_terms, boundary_conditions=(
    x_neg=boundary_condition_convergence_test,
    x_pos=boundary_condition_convergence_test,
    y_neg=Trixi.BoundaryConditionCoupled(2, 2, indices_y_pos, 2, Float64),
    y_pos=Trixi.BoundaryConditionCoupled(4, 2, indices_y_neg, 2, Float64),
  ))

mesh4 = CurvedMesh((16, 16), gnomonic_projection(1, 1, 3 * pi/2), 
                   periodicity=false, mapping_as_string=mapping_as_string(3*pi/2))

semi4 = SemidiscretizationHyperbolic(mesh4, equations, initial_condition, solver,
  source_terms=source_terms, boundary_conditions=(
    x_neg=boundary_condition_convergence_test,
    x_pos=boundary_condition_convergence_test,
    y_neg=Trixi.BoundaryConditionCoupled(3, 2, indices_y_pos, 2, Float64),
    y_pos=Trixi.BoundaryConditionCoupled(1, 2, indices_y_neg, 2, Float64),
  ))

semi = SemidiscretizationHyperbolicCoupled((semi1, semi2, semi3, semi4))

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=1.0)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)
###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
