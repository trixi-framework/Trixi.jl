#src # Structured mesh with mapping

# Here, we want to introduce another mesh type than
# [Trixi](https://github.com/trixi-framework/Trixi.jl)'s standard mesh [`TreeMesh`](@ref).
# More precisely, this tutorial is about the curved mesh type [`StructuredMesh`](@ref) which supports
# fully curved meshes created for instance by an input mapping.

# # Create a curved mesh


using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

function initial_condition_pressure_perturbation(x, t, equations::CompressibleEulerEquations2D)
  xs = 1.5 # location of the initial disturbance on the x axis
  w = 1/8 # half width
  p = exp(-log(2) * ((x[1]-xs)^2 + x[2]^2)/w^2)
  v1 = 0.0
  v2 = 0.0
  rho = 1.0

  return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_pressure_perturbation

boundary_conditions = (boundary_condition_slip_wall, # wall reflection
                       boundary_condition_slip_wall, # wall reflection
                       boundary_condition_slip_wall, # wall reflection
                       boundary_condition_slip_wall) # TODO

###############################################################################
# Get the DG approximation space

solver = DGSEM(polydeg=4, surface_flux=flux_ranocha)

###############################################################################
# Get the curved quad mesh from a mapping function
#
#
# TODO: Add picture of domain? Like that or LateX picture with github?
#
#    /____________/
# (-5, 0)    (-0.5, 0)


r0 = 0.5 # inner radius
r1 = 5.0 # outer radius
f1(xi)  = SVector( r0 + 0.5 * (r1 - r0) * (xi + 1), 0.0)
f2(xi)  = SVector(-r0 - 0.5 * (r1 - r0) * (xi + 1), 0.0)
f3(eta) = SVector(r0 * cos(0.5 * pi * (eta + 1)), r0 * sin(0.5 * pi * (eta + 1)))
f4(eta) = SVector(r1 * cos(0.5 * pi * (eta + 1)), r1 * sin(0.5 * pi * (eta + 1)))

cells_per_dimension = (16, 16)

# Create curved mesh with 16 x 16 elements
mesh = StructuredMesh(cells_per_dimension, (f1, f2, f3, f4), periodicity=false)

###############################################################################
# create the semi discretization object

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

stepsize_callback = StepsizeCallback(cfl=1.0)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);



# For more curved mesh types, you can have a look in the [tutorial](@ref hohqmesh_tutorial)
# about the Trixi's unstructured mesh type [`UnstructuredMesh2D`] and its use of the
# [High-Order Hex-Quad Mesh (HOHQMesh) generator](https://github.com/trixi-framework/HOHQMesh)
# created and developed by David Kopriva.