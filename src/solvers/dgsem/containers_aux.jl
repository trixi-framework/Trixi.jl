# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Container for the values of the auxiliary variables at the volume and surface
# quadrature nodes, see [`n_aux_node_vars`](@ref). It is stored as `cache.aux_vars` and
# only created for equations with auxiliary variables.
struct AuxNodeVarsContainer{NDIMS, uEltype <: Real,
                            AuxNodeVars <: AbstractArray{uEltype},
                            AuxSurfaceNodeVars <: AbstractArray{uEltype}, AuxField}
    aux_node_vars::AuxNodeVars                # [var, i, j, k, element]
    aux_surface_node_vars::AuxSurfaceNodeVars # [leftright, var, i, j, interface]

    # function `aux_field(x, equations)` used to initialize the auxiliary variables
    aux_field::AuxField
end

# Move the arrays of the auxiliary variables, e.g., to a GPU, see `trixi_adapt`
function Adapt.adapt_structure(to, aux_vars::AuxNodeVarsContainer{NDIMS}) where {NDIMS}
    aux_node_vars = adapt(to, aux_vars.aux_node_vars)
    aux_surface_node_vars = adapt(to, aux_vars.aux_surface_node_vars)
    @unpack aux_field = aux_vars

    return AuxNodeVarsContainer{NDIMS, eltype(aux_node_vars), typeof(aux_node_vars),
                                typeof(aux_surface_node_vars),
                                typeof(aux_field)}(aux_node_vars,
                                                   aux_surface_node_vars, aux_field)
end

# Check whether the arrays of auxiliary variables have the axes we assume in the inner
# loops of Trixi.jl, cf. `check_axes`. Nothing is checked without auxiliary variables.
@inline check_axes_aux_node_vars(::Nothing, equations, solver, cache) = nothing

@inline function check_axes_aux_node_vars(aux_node_vars, equations, solver, cache)
    # [var, i, j, k, element]
    node_axes = ntuple(_ -> eachnode(solver), Val(ndims(aux_node_vars) - 2))
    return check_axes(aux_node_vars,
                      (eachauxvariable(equations), node_axes...,
                       eachelement(solver, cache)))
end

@inline check_axes_aux_surface_node_vars(::Nothing, equations, solver, cache) = nothing

@inline function check_axes_aux_surface_node_vars(aux_surface_node_vars, equations,
                                                  solver, cache)
    # [leftright, var, i, j, interface]
    surface_node_axes = ntuple(_ -> eachnode(solver),
                               Val(ndims(aux_surface_node_vars) - 3))
    return check_axes(aux_surface_node_vars,
                      (Base.OneTo(2), eachauxvariable(equations), surface_node_axes...,
                       eachinterface(solver, cache)))
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

    aux_vars = AuxNodeVarsContainer{NDIMS, uEltype, typeof(aux_node_vars),
                                    typeof(aux_surface_node_vars),
                                    typeof(aux_field)}(aux_node_vars,
                                                       aux_surface_node_vars,
                                                       aux_field)

    init_aux_node_vars!(aux_vars, mesh, equations, solver, cache)
    init_aux_surface_node_vars!(aux_vars, mesh, equations, solver, cache)

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

# Filling the interface values requires information about the coupling of the elements
# and is thus implemented for each mesh type separately, currently only for
# `P4estMesh{3}` in `dgsem_p4est/containers_3d.jl`.
function init_aux_surface_node_vars!(aux_vars, mesh, equations, solver, cache)
    throw(ArgumentError("auxiliary variables are not implemented for $(typeof(mesh)) " *
                        "with $(typeof(solver))"))
end
end # @muladd
