using Downloads: download
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_convergence_test

source_terms = source_terms_convergence_test

# BCs must be passed as Dict
boundary_condition = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = Dict(:all => boundary_condition)

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

# Deformed rectangle that looks like a waving flag,
# lower and upper faces are sinus curves, left and right are vertical lines.
f1(s) = SVector(-1.0, s - 1.0)
f2(s) = SVector(1.0, s + 1.0)
f3(s) = SVector(s, -1.0 + sin(0.5 * pi * s))
f4(s) = SVector(s, 1.0 + sin(0.5 * pi * s))
faces = (f1, f2, f3, f4)

Trixi.validate_faces(faces)
mapping_flag = Trixi.transfinite_mapping(faces)

# Get the uncurved mesh from a file (downloads the file if not available locally)
# Unstructured mesh with 24 cells of the square domain [-1, 1]^n
mesh_file = joinpath(@__DIR__, "square_unstructured_2.inp")
isfile(mesh_file) ||
    download("https://gist.githubusercontent.com/efaulhaber/63ff2ea224409e55ee8423b3a33e316a/raw/7db58af7446d1479753ae718930741c47a3b79b7/square_unstructured_2.inp",
             mesh_file)

# INP mesh files are only support by p4est. Hence, we
# create a p4est connecvity object first from which
# we can create a t8code mesh.
conn = Trixi.read_inp_p4est(mesh_file, Val(2))

mesh = T8codeMesh{2}(conn, polydeg = 3,
                     mapping = mapping_flag,
                     initial_refinement_level = 1)

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

    level = Trixi.t8_element_level(ts, elements[1])

    # TODO: Make this condition more general.
    if vertex[1] < 1e-8 && vertex[2] < 1e-8 && level < 2
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

# Check out `examples/t8_step4_partition_balance_ghost.jl` in
# https://github.com/DLR-AMR/T8code.jl for detailed explanations.
let set_from = C_NULL, recursive = 1, set_for_coarsening = 0, no_repartition = 0
    Trixi.t8_forest_set_user_data(new_forest, C_NULL)
    Trixi.t8_forest_set_adapt(new_forest, mesh.forest,
                              Trixi.@t8_adapt_callback(adapt_callback), recursive)
    Trixi.t8_forest_set_balance(new_forest, set_from, no_repartition)
    Trixi.t8_forest_set_partition(new_forest, set_from, set_for_coarsening)
    Trixi.t8_forest_set_ghost(new_forest, 1, Trixi.T8_GHOST_FACES)
    Trixi.t8_forest_commit(new_forest)
end

mesh.forest = new_forest

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_terms,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 0.8)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback)
###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
