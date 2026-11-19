# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    AssemblyCGSEM(mesh::P4estMesh{2}, equations, solver::CGSEM, cache)

Gather-scatter data of a continuous Galerkin spectral element method (CGSEM) on a
conforming `mesh` with element-local storage of the solution, built from the DGSEM
`cache` of the wrapped solver `solver.dg`.

Every node on an element face has one copy in each element sharing it.
`AssemblyCGSEM` numbers the distinct (global) nodes, stores the inverse of the
resulting local-to-global index map in compressed form (see below), and precomputes
the inverse of the summed Jacobian ``1 / \sum_k J_k`` over the copies of every global
node, i.e., the inverse of the assembled lumped mass matrix divided by the reference
quadrature weight, which is the same for all copies of a node. It is used by
[`assemble_and_apply_jacobian!`](@ref) and [`l2_projection!`](@ref).

The global numbering is obtained by union-find over the node pairs of all
interfaces of `cache`, which also connects nodes that are shared only through a
vertex. Interior nodes form singleton rows. The mesh must be conforming (no mortars).
The basis is a [`LobattoLegendreBasis`](@ref), which is checked by [`CGSEM`](@ref),
since Gauss-Legendre nodes do not lie on element faces.

## Local-to-global index map and its inverse

For simplicity, consider a 1D mesh of two elements with `polydeg = 2` (three nodes per
element) sharing one node. An element-local node is identified by its `CartesianIndex`
`(i, element)` into the element-local arrays (`(i, j, element)` in 2D), and the
local-to-global index map assigns one global node to each of them:
```
element-local nodes (i, element)          global nodes

(1,1)--(2,1)--(3,1) (1,2)--(2,2)--(3,2)   1-----2-----3-----4-----5
     element 1           element 2          element 1   element 2

local node    (1,1) (2,1) (3,1) (1,2) (2,2) (3,2)
global node     1     2     3     3     4     5
```
The global nodes are numbered by first appearance in memory order of the local nodes
(`i` fastest, then `element`). The inverse map from each global node to all of its
copies is a ragged list:
```
global node 1: (1,1)
global node 2: (2,1)
global node 3: (3,1), (1,2)
global node 4: (2,2)
global node 5: (3,2)
```

## Compressed representation of the global-to-local index map

The ragged list is stored in compressed row format in the fields `copies` and
`offsets`: `copies` concatenates all rows, and `offsets[n]` is the position of the first
copy of global node `n` in `copies`, with `offsets[end] == length(copies) + 1`, i.e.,
```
copies  = [(1,1), (2,1), (3,1), (1,2), (2,2), (3,2)]
offsets = [1, 2, 3, 5, 6, 7]
```
The copies of global node `n` are `copies[offsets[n]:(offsets[n + 1] - 1)]`, returned
by [`local_nodes_of`](@ref), e.g., `(3,1)` and `(1,2)` for `n = 3`. Within each row,
the copies are sorted by their linear index, i.e., by element id. In this 1D example,
`copies` coincides with the memory order of the local nodes because the two copies of
the shared node are adjacent in memory; in 2D, where a vertex node is shared by up to
four elements, and with periodic boundaries, the copies of a global node are scattered
in memory. Since the rows partition the element-local nodes, a loop over the global
nodes that reads and writes only the copies of its own row is free of write conflicts
and can be parallelized without atomic operations.

This representation and the gather-scatter kernels follow the `ogs` library of
libParanumal (Chalmers et al.), which stores the inverse map as a CSR matrix with at
most one nonzero per column (`rowStarts` and `colIds`) and whose gather-scatter
kernel reduces each row into a single value that is then written back to all entries
of the row. Here, the global numbering is derived from the interfaces of the DGSEM
`cache` by union-find, the summation, the application of the inverse assembled lumped
mass matrix, and the scatter are fused into one pass, and the kernels operate on the
`SVector`s of all variables of a node.

## References
- N. Chalmers, A. Karakus, A. P. Austin, K. Swirydowicz, T. Warburton (2022)
  libParanumal: a performance portable high-order finite element library (Release 0.5.0)
  [DOI: 10.5281/zenodo.4004744](https://doi.org/10.5281/zenodo.4004744),
  [GitHub](https://github.com/paranumal/libparanumal)
"""
struct AssemblyCGSEM{NDIMSP1, RealT <: Real}
    # Size `(nnodes, ..., nelements)` of the node index space of the element-local arrays
    dims::NTuple{NDIMSP1, Int}
    # Compressed row representation of the global-to-local index map, i.e., of the
    # inverse of the local-to-global map (see the docstring). Read the copies of a global
    # node with `local_nodes_of`, which returns `copies[offsets[n]:(offsets[n + 1] - 1)]`.
    # Within a row, the copies are sorted by their linear index, i.e., by element id.
    # This order is fixed so that the sums in `assemble_and_apply_jacobian!` and
    # `l2_projection!` are deterministic.
    offsets::Vector{Int}
    copies::Vector{CartesianIndex{NDIMSP1}}
    # 1 / sum_k J_k over the copies of each global node. For singleton rows, this is
    # exactly `inverse_jacobian` of the DGSEM cache.
    inverse_sum_jacobian::Vector{RealT}
end

# @eval due to @muladd
@eval Adapt.@adapt_structure(AssemblyCGSEM)

function AssemblyCGSEM(mesh::P4estMesh{2}, equations, solver::CGSEM, cache)
    (; dg) = solver
    n_mortars = nmortars(cache.mortars)
    if n_mortars > 0
        throw(ArgumentError("CGSEM requires a conforming mesh, " *
                            "but the mesh has $n_mortars mortars"))
    end

    dims = (nnodes(dg), nnodes(dg), nelements(dg, cache))
    linear_indices = LinearIndices(dims)
    n_local = length(linear_indices)

    # Union-find over the node pairs of all interfaces. The root of each set is always
    # its smallest linear index (see `union_sets!`).
    parent = collect(1:n_local)
    (; neighbor_ids, node_indices) = cache.interfaces
    index_range = eachnode(dg)
    for interface in eachinterface(dg, cache)
        primary_element = neighbor_ids[1, interface]
        secondary_element = neighbor_ids[2, interface]
        primary_indices = node_indices[1, interface]
        secondary_indices = node_indices[2, interface]

        i_primary, i_primary_step = index_to_start_step_2d(primary_indices[1],
                                                           index_range)
        j_primary, j_primary_step = index_to_start_step_2d(primary_indices[2],
                                                           index_range)
        i_secondary, i_secondary_step = index_to_start_step_2d(secondary_indices[1],
                                                               index_range)
        j_secondary, j_secondary_step = index_to_start_step_2d(secondary_indices[2],
                                                               index_range)

        for i in index_range
            union_sets!(parent,
                        linear_indices[i_primary, j_primary, primary_element],
                        linear_indices[i_secondary, j_secondary, secondary_element])
            i_primary += i_primary_step
            j_primary += j_primary_step
            i_secondary += i_secondary_step
            j_secondary += j_secondary_step
        end
    end

    # Local-to-global index map: number the global nodes by first appearance in memory
    # order. Since the root of a set is its smallest member, it has already been
    # numbered when a non-root node of the set is visited.
    global_ids = zeros(Int, n_local)
    n_global = 0
    for node in 1:n_local
        root = find_root!(parent, node)
        if root == node
            n_global += 1
            global_ids[node] = n_global
        else
            global_ids[node] = global_ids[root]
        end
    end

    # Invert the map into compressed row storage of the copies of each global node
    offsets = zeros(Int, n_global + 1)
    for node in 1:n_local
        offsets[global_ids[node] + 1] += 1
    end
    offsets[1] = 1
    for n in 1:n_global
        offsets[n + 1] += offsets[n]
    end
    copies = Vector{CartesianIndex{3}}(undef, n_local)
    next_position = copy(offsets)
    cartesian_indices = CartesianIndices(dims)
    for node in 1:n_local
        n = global_ids[node]
        copies[next_position[n]] = cartesian_indices[node]
        next_position[n] += 1
    end

    # Inverse of the summed Jacobian of the copies of each global node
    (; inverse_jacobian) = cache.elements
    RealT = eltype(inverse_jacobian)
    inverse_sum_jacobian = Vector{RealT}(undef, n_global)
    assembly = AssemblyCGSEM{3, RealT}(dims, offsets, copies, inverse_sum_jacobian)
    for global_index in 1:n_global
        local_nodes = local_nodes_of(global_index, assembly)
        if length(local_nodes) == 1
            # Keep the DGSEM value exactly for interior nodes
            inverse_sum_jacobian[global_index] = inverse_jacobian[local_nodes[1]]
        else
            sum_jacobian = zero(RealT)
            for local_index in local_nodes
                sum_jacobian += inv(inverse_jacobian[local_index])
            end
            inverse_sum_jacobian[global_index] = inv(sum_jacobian)
        end
    end

    return assembly
end

# Find the root of `node` with path compression
function find_root!(parent, node)
    root = node
    while parent[root] != root
        root = parent[root]
    end
    while parent[node] != root
        parent[node], node = root, parent[node]
    end
    return root
end

# Merge the sets of `node1` and `node2`, keeping the smaller root as the root
function union_sets!(parent, node1, node2)
    root1 = find_root!(parent, node1)
    root2 = find_root!(parent, node2)
    if root1 != root2
        parent[max(root1, root2)] = min(root1, root2)
    end
    return nothing
end

"""
    nnodes_unique(assembly::AssemblyCGSEM)

Number of distinct (global) nodes of the continuous space described by `assembly`.
"""
@inline nnodes_unique(assembly::AssemblyCGSEM) = length(assembly.inverse_sum_jacobian)

"""
    local_nodes_of(global_index, assembly::AssemblyCGSEM)

Element-local `CartesianIndex`es of all copies of the global node `global_index`, i.e.,
one row of the compressed global-to-local index map of `assembly` (see
[`AssemblyCGSEM`](@ref)), as a view into `assembly.copies` in ascending element-id
order. Interior nodes have exactly one copy.
"""
@inline function local_nodes_of(global_index, assembly::AssemblyCGSEM)
    (; offsets, copies) = assembly
    @boundscheck checkbounds(offsets, global_index:(global_index + 1))
    return @inbounds view(copies, offsets[global_index]:(offsets[global_index + 1] - 1))
end

function Base.show(io::IO, assembly::AssemblyCGSEM)
    @nospecialize assembly # reduce precompilation time

    print(io, "AssemblyCGSEM(", nnodes_unique(assembly), " unique nodes, ",
          length(assembly.copies), " element-local nodes)")
    return nothing
end

# Check that `u` has the layout `[variable, node_i, node_j, element]` of the DGSEM cache
# the `assembly` was created from.
function check_axes(u, assembly::AssemblyCGSEM, equations)
    check_axes(u, (eachvariable(equations), map(Base.OneTo, assembly.dims)...))
    return nothing
end

@doc raw"""
    assemble_and_apply_jacobian!(du, assembly::AssemblyCGSEM,
                                 mesh, equations, solver::CGSEM, cache)

Gather-scatter step of CGSEM. The value at every copy of a global node is replaced by
```math
\dot u_n = -\frac{\sum_k du_k}{\sum_k J_k},
```
i.e., the element-local residuals are summed and the inverse of the assembled lumped
mass matrix is applied (the reference quadrature weight is the same for all copies
and has already been divided out by the DGSEM kernels). This replaces
`apply_jacobian!` of DGSEM, including its sign, and must be called after all
element-local contributions, in particular the source terms, have been added to `du`.
The copies are summed in a fixed order, so all copies receive bitwise identical values
and the result does not depend on the number of threads.

See also [`assemble_and_apply_jacobian_parabolic!`](@ref).
"""
function assemble_and_apply_jacobian!(du, assembly::AssemblyCGSEM,
                                      mesh, equations, solver::CGSEM, cache)
    # The DGSEM kernels compute the residual of `du/dt + div(f) = 0`, cf. `apply_jacobian!`
    assemble_and_scale!(du, assembly, -1, equations, solver)
    return nothing
end

@doc raw"""
    assemble_and_apply_jacobian_parabolic!(du, assembly::AssemblyCGSEM,
                                           mesh, equations_parabolic, solver::CGSEM,
                                           cache)

Gather-scatter step of the parabolic right-hand side of CGSEM. As in
[`assemble_and_apply_jacobian!`](@ref), the value at every copy of a global node is
replaced by the sum of all copies divided by the summed Jacobian, but without the sign
flip,
```math
\dot u_n = \frac{\sum_k du_k}{\sum_k J_k},
```
since the DGSEM kernels compute the parabolic terms as a right-hand side. This replaces
`apply_jacobian_parabolic!` of DGSEM.
"""
function assemble_and_apply_jacobian_parabolic!(du, assembly::AssemblyCGSEM,
                                                mesh, equations_parabolic,
                                                solver::CGSEM, cache)
    assemble_and_scale!(du, assembly, 1, equations_parabolic, solver)
    return nothing
end

# Kernel of the gather-scatter steps above: every copy of a global node is replaced by
# `jacobian_sign` times the sum of all copies divided by the summed Jacobian
function assemble_and_scale!(du, assembly::AssemblyCGSEM, jacobian_sign, equations,
                             solver::CGSEM)
    (; dg) = solver
    (; inverse_sum_jacobian) = assembly

    # Explicit bounds check, which allows us to assume inbounds access below
    @boundscheck check_axes(du, assembly, equations)

    # The rows of the global-to-local map partition the element-local nodes, so the
    # loop is free of write conflicts.
    @threaded for global_index in eachindex(inverse_sum_jacobian)
        @inbounds begin
            local_nodes = local_nodes_of(global_index, assembly)

            # Gather: sum the residuals of all copies in their fixed (ascending element
            # id) order so that the result is deterministic
            du_sum = zero(SVector{nvariables(equations), eltype(du)})
            for local_index in local_nodes
                du_sum = du_sum + get_node_vars(du, equations, dg, local_index)
            end
            du_assembled = (jacobian_sign * inverse_sum_jacobian[global_index]) * du_sum

            # Scatter: all copies receive the same value
            for local_index in local_nodes
                set_node_vars!(du, du_assembled, equations, dg, local_index)
            end
        end
    end

    return nothing
end

@doc raw"""
    l2_projection!(u, assembly::AssemblyCGSEM, mesh, equations, solver::CGSEM, cache)

L2 projection of the element-local nodal values `u` onto the continuous (C0) space
described by `assembly`, with respect to the lumped (collocated) Gauss-Lobatto-Legendre
mass matrix. The value at every copy of a global node is replaced by the
Jacobian-weighted average
```math
v_n = \frac{\sum_k J_k u_k}{\sum_k J_k}
```
of its copies; interior nodes are not modified. The projection is idempotent (up to
roundoff errors) and preserves the discrete integral of every variable. It is used for
the initial condition of CGSEM and can be applied to any DGSEM solution on a conforming
mesh by wrapping the DGSEM solver `dg` of its semidiscretization `semi`, e.g.,
```julia
u = Trixi.wrap_array(u_ode, semi)
solver = CGSEM(dg)
assembly = Trixi.AssemblyCGSEM(mesh, equations, solver, semi.cache)
Trixi.l2_projection!(u, assembly, mesh, equations, solver, semi.cache)
```
"""
function l2_projection!(u, assembly::AssemblyCGSEM, mesh, equations, solver::CGSEM,
                        cache)
    (; dg) = solver
    (; inverse_sum_jacobian) = assembly
    (; inverse_jacobian) = cache.elements

    # Explicit bounds check, which allows us to assume inbounds access below
    @boundscheck begin
        check_axes(u, assembly, equations)
        check_axes(inverse_jacobian, map(Base.OneTo, assembly.dims))
    end

    # The rows of the global-to-local map partition the element-local nodes, so the
    # loop is free of write conflicts.
    @threaded for global_index in eachindex(inverse_sum_jacobian)
        @inbounds begin
            local_nodes = local_nodes_of(global_index, assembly)

            # Interior nodes have a single copy and are left unchanged
            if length(local_nodes) > 1
                # Gather: Jacobian-weighted sum of all copies in their fixed (ascending
                # element id) order so that the result is deterministic
                u_sum = zero(SVector{nvariables(equations), eltype(u)})
                for local_index in local_nodes
                    u_sum = u_sum +
                            inv(inverse_jacobian[local_index]) *
                            get_node_vars(u, equations, dg, local_index)
                end
                u_projected = inverse_sum_jacobian[global_index] * u_sum

                # Scatter: all copies receive the same value
                for local_index in local_nodes
                    set_node_vars!(u, u_projected, equations, dg, local_index)
                end
            end
        end
    end

    return nothing
end
end # @muladd
