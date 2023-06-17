
using Trixi, OrdinaryDiffEq

dg = DGMulti(polydeg = 3, element_type = Quad(), approximation_type = SBP(),
             surface_integral = SurfaceIntegralWeakForm(FluxHLL()),
             volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha))

equations = CompressibleEulerEquations2D(1.4)
initial_condition = initial_condition_convergence_test
source_terms = source_terms_convergence_test

# example where we tag two separate boundary segments of the mesh
top_boundary(x, tol=50*eps()) = abs(x[2]-1)<tol
rest_of_boundary(x, tol=50*eps()) = !top_boundary(x, tol)
is_on_boundary = Dict(:top => top_boundary, :rest => rest_of_boundary)

function mapping(xi, eta)
  x = xi  + 0.1 * sin(pi * xi) * sin(pi * eta)
  y = eta + 0.1 * sin(pi * xi) * sin(pi * eta)
  return SVector(x, y)
end
cells_per_dimension = (16, 16)
mesh = DGMultiMesh(dg, cells_per_dimension, mapping, is_on_boundary=is_on_boundary)

boundary_condition_convergence_test = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = (; :top => boundary_condition_convergence_test,
                        :rest => boundary_condition_convergence_test)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg,
                                    source_terms = source_terms,
                                    boundary_conditions = boundary_conditions)

tspan = (0.0, 0.4)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval=10)
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, uEltype=real(dg))
callbacks = CallbackSet(summary_callback, alive_callback, analysis_callback)

###############################################################################
# run the simulation

sol = solve(ode, RDPK3SpFSAL49(); abstol=1.0e-6, reltol=1.0e-6,
            ode_default_options()..., callback=callbacks);

summary_callback() # print the timer summary
