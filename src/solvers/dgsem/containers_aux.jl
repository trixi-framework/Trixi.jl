# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Container for the values of the auxiliary variables at the volume and surface
# quadrature nodes, see [`n_aux_node_vars`](@ref). It is stored as `cache.aux_vars` and
# only created if an `aux_field` is passed to the semidiscretization.
struct AuxNodeVarsContainer{NDIMS, uEltype <: Real, NDIMSP1, NDIMSP2, NDIMSP3, AuxField}
    aux_node_vars::Array{uEltype, NDIMSP2}          # [var, i, j, k, element]
    aux_surface_node_vars::Array{uEltype, NDIMSP2}  # [leftright, var, i, j, interface]
    aux_boundary_node_vars::Array{uEltype, NDIMSP1} # [var, i, j, boundary]
    aux_mortar_node_vars::Array{uEltype, NDIMSP3}   # [leftright, var, position, i, j, mortar]

    # function `aux_field(x, equations)` used to initialize the auxiliary variables
    aux_field::AuxField
end

# Create and initialize the container of auxiliary variables
function init_aux_vars(mesh, equations, solver, cache, aux_field)
    NDIMS = ndims(mesh)
    n_aux = n_aux_node_vars(equations)
    n_nodes = nnodes(solver)
    uEltype = eltype(cache.elements)
    nan_uEltype = convert(uEltype, NaN)

    check_aux_field(aux_field, mesh, equations, solver, cache)

    # Volume nodes of the elements
    aux_node_vars = fill(nan_uEltype,
                         (n_aux, ntuple(_ -> n_nodes, NDIMS)...,
                          nelements(cache.elements)))
    # Surface nodes on both sides of the interfaces
    aux_surface_node_vars = fill(nan_uEltype,
                                 (2, n_aux, ntuple(_ -> n_nodes, NDIMS - 1)...,
                                  ninterfaces(cache.interfaces)))
    # Surface nodes at the boundaries. Only the values inside of the domain are stored.
    aux_boundary_node_vars = fill(nan_uEltype,
                                  (n_aux, ntuple(_ -> n_nodes, NDIMS - 1)...,
                                   nboundaries(cache.boundaries)))
    # Surface nodes on both sides of each position of the mortars
    aux_mortar_node_vars = fill(nan_uEltype,
                                (2, n_aux, 2^(NDIMS - 1),
                                 ntuple(_ -> n_nodes, NDIMS - 1)...,
                                 nmortars(cache.mortars)))

    aux_vars = AuxNodeVarsContainer{NDIMS, uEltype, NDIMS + 1, NDIMS + 2, NDIMS + 3,
                                    typeof(aux_field)}(aux_node_vars,
                                                       aux_surface_node_vars,
                                                       aux_boundary_node_vars,
                                                       aux_mortar_node_vars,
                                                       aux_field)

    init_aux_node_vars!(aux_vars, mesh, equations, solver, cache)
    init_aux_surface_node_vars!(aux_vars, mesh, equations, solver, cache)
    init_aux_boundary_node_vars!(aux_vars, mesh, equations, solver, cache)
    init_aux_mortar_node_vars!(aux_vars, mesh, equations, solver, cache)

    return aux_vars
end

# Make sure that `aux_field` provides exactly the number of auxiliary variables the
# equations expect. Without this check, too few values result in a confusing `BoundsError`
# and too many are silently ignored.
function check_aux_field(aux_field, mesh, equations, solver, cache)
    # Nothing to check without elements, e.g., on an empty MPI rank
    if nelements(cache.elements) == 0
        return nothing
    end

    x_local = get_node_coords(cache.elements.node_coordinates, equations, solver,
                              ntuple(_ -> 1, ndims(mesh))..., 1)
    n_returned = length(aux_field(x_local, equations))
    if n_returned != n_aux_node_vars(equations)
        throw(ArgumentError("`aux_field` returned $n_returned values but " *
                            "`$(nameof(typeof(equations)))` has " *
                            "$(n_aux_node_vars(equations)) auxiliary variables"))
    end

    return nothing
end

# Evaluate the auxiliary variables at the volume nodes. This works for every mesh type
# providing `cache.elements.node_coordinates`.
function init_aux_node_vars!(aux_vars, mesh, equations, solver, cache)
    @unpack aux_node_vars, aux_field = aux_vars
    @unpack node_coordinates = cache.elements

    # All combinations of node indices, independent of the number of dimensions
    node_indices = CartesianIndices(ntuple(_ -> nnodes(solver), ndims(mesh)))

    @threaded for element in eachelement(solver, cache)
        for node_index in node_indices
            x_local = get_node_coords(node_coordinates, equations, solver,
                                      node_index, element)
            set_aux_node_vars!(aux_node_vars, aux_field(x_local, equations),
                               equations, solver, node_index, element)
        end
    end

    return nothing
end

# The remaining initializations require information about the coupling of the elements
# and are thus implemented for each mesh type separately, currently only for
# `P4estMesh{3}` in `dgsem_p4est/containers_3d.jl`.
function init_aux_surface_node_vars!(aux_vars, mesh, equations, solver, cache)
    throw(ArgumentError(unsupported_aux_vars_message(mesh, solver)))
end

function init_aux_boundary_node_vars!(aux_vars, mesh, equations, solver, cache)
    throw(ArgumentError(unsupported_aux_vars_message(mesh, solver)))
end

function init_aux_mortar_node_vars!(aux_vars, mesh, equations, solver, cache)
    throw(ArgumentError(unsupported_aux_vars_message(mesh, solver)))
end

function unsupported_aux_vars_message(mesh, solver)
    return "auxiliary variables are not implemented for $(typeof(mesh)) with " *
           "$(typeof(solver))"
end
end # @muladd
