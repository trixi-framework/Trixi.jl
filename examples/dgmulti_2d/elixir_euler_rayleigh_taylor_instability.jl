using Trixi, OrdinaryDiffEq

dg = DGMulti(polydeg = 3, element_type = Quad(), approximation_type = Polynomial(),
             surface_integral = SurfaceIntegralWeakForm(flux_hll),
             volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha))

equations = CompressibleEulerEquations2D(1.4)

function create_initial_condition(;slope)
  u0(x,t,equations) = initial_condition_rti(x, t, equations, slope)
  return u0
end
initial_condition = create_initial_condition(slope = 1000)
source_terms = source_terms_rti

num_elements = 32
cells_per_dimension = (num_elements, 4 * num_elements)
vertex_coordinates, EToV = StartUpDG.uniform_mesh(dg.basis.elementType, cells_per_dimension...)

vx, vy = vertex_coordinates
vx = map(x-> .25 * .5*(1 + x), vx) # map to [0, .25]
vy = map(x-> .5*(1+x), vy) # map to [0, 1] for single mode RTI
vertex_coordinates = (vx, vy)

# boundary_conditions = (; :entire_boundary => BoundaryConditionDirichlet(initial_condition))
boundary_conditions = (; :entire_boundary => BoundaryConditionWall(boundary_state_slip_wall))

mesh = VertexMappedMesh(vertex_coordinates, EToV, dg)
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg;
                                    source_terms = source_terms,
                                    boundary_conditions = boundary_conditions)

tspan = (0.0, 1.95)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval=10)
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, uEltype=real(dg))
stepsize_callback = StepsizeCallback(cfl=1.)
callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        stepsize_callback,
                        alive_callback)

###############################################################################
# run the simulation

sol = solve(ode, RDPK3SpFSAL49(), abstol=1.0e-6, reltol=1.0e-6,
            save_everystep=false, callback=callbacks);

summary_callback() # print the timer summary