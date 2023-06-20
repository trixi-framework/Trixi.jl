using Downloads: download
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations3D(1.4)

initial_condition = initial_condition_constant

boundary_conditions = Dict(:all => BoundaryConditionDirichlet(initial_condition))

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs,
               volume_integral = VolumeIntegralWeakForm())

# Mapping as described in https://arxiv.org/abs/2012.12040 but reduced to 2D.
# This particular mesh is unstructured in the yz-plane, but extruded in x-direction.
# Apply the warping mapping in the yz-plane to get a curved 2D mesh that is extruded
# in x-direction to ensure free stream preservation on a non-conforming mesh.
# See https://doi.org/10.1007/s10915-018-00897-9, Section 6.
function mapping(xi, eta_, zeta_)
    # Transform input variables between -1 and 1 onto [0,3]
    eta = 1.5 * eta_ + 1.5
    zeta = 1.5 * zeta_ + 1.5

    z = zeta +
        1 / 6 * (cos(1.5 * pi * (2 * eta - 3) / 3) *
                 cos(0.5 * pi * (2 * zeta - 3) / 3))

    y = eta + 1 / 6 * (cos(0.5 * pi * (2 * eta - 3) / 3) *
                       cos(2 * pi * (2 * z - 3) / 3))

    return SVector(xi, y, z)
end

# Unstructured mesh with 48 cells of the cube domain [-1, 1]^3
mesh_file = joinpath(@__DIR__, "cube_unstructured_2.inp")
isfile(mesh_file) ||
    download("https://gist.githubusercontent.com/efaulhaber/b8df0033798e4926dec515fc045e8c2c/raw/b9254cde1d1fb64b6acc8416bc5ccdd77a240227/cube_unstructured_2.inp",
             mesh_file)

# INP mesh files are only support by p4est. Hence, we
# create a p4est connecvity object first from which
# we can create a t8code mesh.
conn = Trixi.read_inp_p4est(mesh_file, Val(3))

mesh = T8codeMesh{3}(conn, polydeg = 3,
                     mapping = mapping,
                     initial_refinement_level = 0)

# Note: This is actually a `p8est_quadrant_t` which is much bigger than the
# following struct. But we only need the first four fields for our purpose.
struct t8_dhex_t
    x::Int32
    y::Int32
    z::Int32
    level::Int8
    # [...] # See `p8est.h` in `p4est` for more info.
end

# Refine quadrants in y-direction of each tree at one edge to level 2
function adapt_callback(forest,
                        forest_from,
                        which_tree,
                        lelement_id,
                        ts,
                        is_family,
                        num_elements,
                        elements_ptr)::Cint
    vertex = Vector{Cdouble}(undef, 3)
    elements = unsafe_wrap(Array, elements_ptr, num_elements)
    Trixi.t8_element_vertex_reference_coords(ts, elements[1], 0, pointer(vertex))

    el = unsafe_load(Ptr{t8_dhex_t}(elements[1]))

    if convert(Int, which_tree) < 4 && el.x == 0 && el.y == 0 && el.level < 2
        # return true (refine)
        return 1
    else
        # return false (don't refine)
        return 0
    end
end

@assert(Trixi.t8_forest_is_committed(mesh.forest)!=0);

# Init new forest.
new_forest_ref = Ref{Trixi.t8_forest_t}()
Trixi.t8_forest_init(new_forest_ref);
new_forest = new_forest_ref[]

let set_from = C_NULL, recursive = 1, set_for_coarsening = 0, no_repartition = 0,
    do_ghost = 1

    Trixi.t8_forest_set_user_data(new_forest, C_NULL)
    Trixi.t8_forest_set_adapt(new_forest, mesh.forest,
                              Trixi.@t8_adapt_callback(adapt_callback), recursive)
    Trixi.t8_forest_set_balance(new_forest, set_from, no_repartition)
    Trixi.t8_forest_set_partition(new_forest, set_from, set_for_coarsening)
    Trixi.t8_forest_set_ghost(new_forest, do_ghost, Trixi.T8_GHOST_FACES)
    Trixi.t8_forest_commit(new_forest)
end

mesh.forest = new_forest

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

# save_restart = SaveRestartCallback(interval=100,
#                                    save_final_restart=true)
# 
# save_solution = SaveSolutionCallback(interval=100,
#                                      save_initial_solution=true,
#                                      save_final_solution=true,
#                                      solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl = 1.2)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        # save_restart, save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false), #maxiters=1,
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
