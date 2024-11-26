using OrdinaryDiffEq
using Trixi
using T8code

advection_velocity = (0.2, -0.7)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

initial_condition = initial_condition_convergence_test

solver = FV(order = 2, extended_reconstruction_stencil = false,
            surface_flux = flux_lax_friedrichs)

# Option 1: coordinates
coordinates_min = (0.0, 0.0) # minimum coordinates (min(x), min(y))
coordinates_max = (8.0, 8.0) # maximum coordinates (max(x), max(y))

mapping_coordinates = Trixi.coordinates2mapping(coordinates_min, coordinates_max)

# Option 2: faces
# waving flag
# f1(s) = SVector(-1.0, s - 1.0)
# f2(s) = SVector(1.0, s + 1.0)
# f3(s) = SVector(s, -1.0 + sin(0.5 * pi * s))
# f4(s) = SVector(s, 1.0 + sin(0.5 * pi * s))

# [0,8]^2
f1(s) = SVector(0.0, 4 * (s + 1))
f2(s) = SVector(8.0, 4 * (s + 1))
f3(s) = SVector(4 * (s + 1), 0.0)
f4(s) = SVector(4 * (s + 1), 8.0)
faces = (f1, f2, f3, f4)
Trixi.validate_faces(faces)
mapping_faces = Trixi.transfinite_mapping(faces)

# Option 3: classic mapping
function mapping(xi, eta)
    x = 4 * (xi + 1)
    y = 4 * (eta + 1)

    return SVector(x, y)
end

# Note and TODO:
# Normally, this should be put somewhere else. For now, that doesn't work properly.
# For instance, in `src/auxiliary/t8code.jl`
# Problem: Even when define this routine somewhere else (then by using a closure) and called
# directly within this elixir (e.g. mapping = trixi_t8_mapping_c(mapping)), we get the SegFault error.
# - Is the function called with the correct parameters? Memory location correct? It seems so, yes.
# - Life time issue for the GC tracked Julia object used in C? **Yes, see gc deactivation in elixir.**
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
        # xy = mapping_coordinates(xi, eta)
        # xy = mapping_faces(xi, eta)
        xy = mapping(xi, eta)

        unsafe_store!(out_coords, xy[1], offset_3d)
        unsafe_store!(out_coords, xy[2], offset_3d + 1)
    end

    return nothing
end

function trixi_t8_mapping_c()
    @cfunction($trixi_t8_mapping, Cvoid,
               (t8_cmesh_t, t8_gloidx_t, Ptr{Cdouble}, Csize_t,
                Ptr{Cdouble}, Ptr{Cvoid}, Ptr{Cvoid}))
end

trees_per_dimension = (2, 2)

# Disabling the gc for almost the entire elixir seems to work in order to fix the SegFault errors with trixi_t8_mapping_c and mapping_coordinates
# It's also possible to move this to the constructor.
GC.enable(false)

# Notes:
# Life time issue for the GC tracked Julia object used in C. Only with coordinates_min/max
# - GC enabled:
#   o for mapping, mapping_faces: every three options work
#   o trixi_t8_mapping_c from elixir: `mapping_coordinates` doesn't work. SegFaults when evaluate geometry
#   o trixi_t8_mapping_c() from file: `mapping_coordinates` doesn't work. SegFaults when evaluate geometry
#   o pass mapping_coordinates doesn't work. SegFaults when evaluate geometry
#   o pass coordinates_min/max doesn't work. SegFaults when evaluate geometry
# - GC disabled as in elixir:
#   o Everything seems to work

# NOTE: When I remove `using T8code` at the top of this elixir, things like `t8_cmesh_t` and `T8_ECLASS_QUAD` are unknown.
eclass = T8_ECLASS_QUAD
mesh = T8codeMesh(trees_per_dimension, eclass,
                  # mapping = Trixi.trixi_t8_mapping_c(mapping_coordinates),
                  # Plan is to use either
                  coordinates_max = coordinates_max, coordinates_min = coordinates_min,
                  # or at least
                  # mapping = mapping_coordinates,
                  initial_refinement_level = 6)

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

# Note: Since the mesh must be finalizized by hand in the elixir, it is not defined anymore here.
# Moved allocation test to the elixirs for now.
let
    t = sol.t[end]
    u_ode = sol.u[end]
    du_ode = similar(u_ode)
    @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
end

GC.enable(true)

# Finalize `T8codeMesh` to make sure MPI related objects in t8code are
# released before `MPI` finalizes.
!isinteractive() && finalize(mesh)
