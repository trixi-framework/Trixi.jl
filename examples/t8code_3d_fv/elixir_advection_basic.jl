using OrdinaryDiffEq
using Trixi
using T8code

advection_velocity = (0.2, -0.7, 0.5)
equations = LinearScalarAdvectionEquation3D(advection_velocity)

initial_condition = initial_condition_convergence_test

solver = FV(order = 2, extended_reconstruction_stencil = false,
            surface_flux = flux_lax_friedrichs)

# Option 1: coordinates
# For all problems see the 2D file...
coordinates_min = (0.0, 0.0, 0.0) # minimum coordinates (min(x), min(y), min(z))
coordinates_max = (8.0, 8.0, 8.0) # maximum coordinates (max(x), max(y), max(z))

mapping_coordinates = Trixi.coordinates2mapping(coordinates_min, coordinates_max)

# Option 3: classic mapping
function mapping(xi, eta, zeta)
    x = 4 * (xi + 1)
    y = 4 * (eta + 1)
    z = 4 * (zeta + 1)

    return SVector(x, y, z)
end

function trixi_t8_mapping(cmesh, gtreeid, ref_coords, num_coords, out_coords,
                          tree_data, user_data)
    ltreeid = t8_cmesh_get_local_id(cmesh, gtreeid)
    eclass = t8_cmesh_get_tree_class(cmesh, ltreeid)
    T8code.t8_geom_compute_linear_geometry(eclass, tree_data,
                                           ref_coords, num_coords, out_coords)

    for i in 1:num_coords
        offset_3d = 3 * (i - 1) + 1

        xi = unsafe_load(out_coords, offset_3d)
        eta = unsafe_load(out_coords, offset_3d + 1)
        zeta = unsafe_load(out_coords, offset_3d + 2)
        # xyz = mapping_coordinates(xi, eta, zeta)
        xyz = mapping(xi, eta, zeta)

        unsafe_store!(out_coords, xyz[1], offset_3d)
        unsafe_store!(out_coords, xyz[2], offset_3d + 1)
        unsafe_store!(out_coords, xyz[3], offset_3d + 2)
    end

    return nothing
end

function trixi_t8_mapping_c()
    @cfunction($trixi_t8_mapping, Cvoid,
               (t8_cmesh_t, t8_gloidx_t, Ptr{Cdouble}, Csize_t,
                Ptr{Cdouble}, Ptr{Cvoid}, Ptr{Cvoid}))
end

trees_per_dimension = (2, 2, 2)

eclass = T8_ECLASS_HEX
mesh = T8codeMesh(trees_per_dimension, eclass;
                  mapping = trixi_t8_mapping_c(),
                  # Plan is to use either
                  # coordinates_max = coordinates_max, coordinates_min = coordinates_min,
                  # or at least
                  # mapping = mapping,
                  initial_refinement_level = 5)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan);

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 10,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 0.9)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        stepsize_callback, save_solution)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback()
