# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function rhs_hyperbolic!(backend::Backend,
                         du, u, t,
                         mesh::Union{P4estMesh{2}, P4estMeshView{2}, T8codeMesh{2},
                                     P4estMesh{3}, T8codeMesh{3}},
                         equations,
                         boundary_conditions, source_terms::Source,
                         dg::DG, cache) where {Source}

    # Calculate volume integral
    @trixi_timeit_ext backend timer() "volume integral" begin
        calc_volume_integral!(backend, du, u, mesh,
                              have_nonconservative_terms(equations), equations,
                              dg.volume_integral, dg, cache)
    end

    # Calculate interface fluxes
    @trixi_timeit_ext backend timer() "prolong2interfaces + flux" begin
        prolong2interfaces_and_calc_interface_flux!(backend,
                                                    cache.elements.surface_flux_values,
                                                    u, mesh,
                                                    have_nonconservative_terms(equations),
                                                    equations,
                                                    dg.surface_integral, dg, cache)
    end

    # Prolong solution to boundaries
    @trixi_timeit_ext backend timer() "prolong2boundaries" begin
        prolong2boundaries!(backend, cache, u, mesh, equations, dg)
    end

    # Calculate boundary fluxes
    @trixi_timeit_ext backend timer() "boundary flux" begin
        calc_boundary_flux!(backend, cache, t, boundary_conditions, mesh, equations,
                            dg.surface_integral, dg)
    end

    # Prolong solution to mortars
    @trixi_timeit_ext backend timer() "prolong2mortars" begin
        prolong2mortars!(backend, cache, u, mesh, equations,
                         dg.mortar, dg)
    end

    # Calculate mortar fluxes
    @trixi_timeit_ext backend timer() "mortar flux" begin
        calc_mortar_flux!(backend, cache.elements.surface_flux_values, mesh,
                          have_nonconservative_terms(equations), equations,
                          dg.mortar, dg.surface_integral, dg, cache)
    end

    # Calculate surface integrals, apply Jacobian from mapping to reference element
    # and calculate source terms
    @trixi_timeit_ext backend timer() "surface, Jacobian + source terms" begin
        calc_surface_integral_and_apply_jacobian_and_calc_sources!(backend, du, u, t,
                                                                   source_terms, mesh,
                                                                   equations,
                                                                   dg.surface_integral,
                                                                   dg, cache)
    end

    return nothing
end

# Weak form volume integral parallelized over the individual solution nodes. Each
# thread computes the contravariant fluxes of its node and stores them in local memory,
# from where all threads of the element apply the derivative matrix. A workgroup contains
# several elements so that it is not too small. This overwrites `du`, so `du` does not
# need to be reset before.
@inline function calc_volume_integral!(backend::Backend, du, u,
                                       mesh::Union{P4estMesh{2}, T8codeMesh{2}},
                                       have_nonconservative_terms::False, equations,
                                       volume_integral::VolumeIntegralWeakForm,
                                       dg::DGSEM, cache)
    nelements(dg, cache) == 0 && return nothing
    # Explicit bounds check, which allows us to assume inbounds access in the kernel
    @boundscheck begin
        check_axes(u, mesh, equations, dg, cache)
        check_axes(du, mesh, equations, dg, cache)
        # Required, e.g., for the `contravariant_vectors` of curvilinear meshes
        check_axes(cache.elements, equations, dg, cache)
    end
    @unpack derivative_hat = dg.basis
    @unpack contravariant_vectors = cache.elements
    NNODES = nnodes(dg)
    NELEMENTS_WG = max(1, div(GPU_HALFSWEEP_WORKGROUP_SIZE, NNODES^2))
    kernel! = weak_form_nodes_2d_KAkernel!(backend, (NNODES, NNODES, NELEMENTS_WG))
    kernel!(du, u, equations, dg, Val(NNODES), Val(nvariables(equations)),
            Val(NELEMENTS_WG), derivative_hat, contravariant_vectors;
            ndrange = (NNODES, NNODES, nelements(dg, cache)))
    return nothing
end

@kernel inbounds=true function weak_form_nodes_2d_KAkernel!(du, u, equations, dg::DGSEM,
                                                            ::Val{NNODES},
                                                            ::Val{NVARIABLES},
                                                            ::Val{NELEMENTS_WG},
                                                            derivative_hat,
                                                            contravariant_vectors) where {
                                                                                          NNODES,
                                                                                          NVARIABLES,
                                                                                          NELEMENTS_WG
                                                                                          }
    i, j, element = @index(Global, NTuple)
    _, _, local_element = @index(Local, NTuple)

    # The variable index is last, see `get_local_node_vars`
    flux1_local = @localmem eltype(du) (NNODES, NNODES, NELEMENTS_WG, NVARIABLES)
    flux2_local = @localmem eltype(du) (NNODES, NNODES, NELEMENTS_WG, NVARIABLES)

    u_node = get_node_vars(u, equations, dg, i, j, element)
    flux1 = flux(u_node, 1, equations)
    flux2 = flux(u_node, 2, equations)
    Ja11, Ja12 = get_contravariant_vector(1, contravariant_vectors, i, j, element)
    Ja21, Ja22 = get_contravariant_vector(2, contravariant_vectors, i, j, element)
    contravariant_flux1 = Ja11 * flux1 + Ja12 * flux2
    contravariant_flux2 = Ja21 * flux1 + Ja22 * flux2
    for v in 1:NVARIABLES
        flux1_local[i, j, local_element, v] = contravariant_flux1[v]
        flux2_local[i, j, local_element, v] = contravariant_flux2[v]
    end
    @synchronize

    # Use `get_local_node_vars` instead of a closure, see the half sweep kernel above
    du_local = zero(SVector{NVARIABLES, eltype(du)})
    for l in 1:NNODES
        du_local = du_local +
                   derivative_hat[i, l] *
                   get_local_node_vars(flux1_local, Val(NVARIABLES), l, j,
                                       local_element)
    end
    for l in 1:NNODES
        du_local = du_local +
                   derivative_hat[j, l] *
                   get_local_node_vars(flux2_local, Val(NVARIABLES), i, l,
                                       local_element)
    end
    set_node_vars!(du, du_local, equations, dg, i, j, element)
end

# Flux differencing volume integral parallelized over the individual solution nodes.
# The generic fallback in `src/solvers/dg_gpu.jl` uses one thread per element, i.e.,
# only `nelements` threads, which leaves most of the GPU idle for typical 2D meshes,
# and it accumulates into `du` in global memory. The kernels here use one thread per
# node, sum the contributions in registers, and overwrite `du`, so `du` does not need
# to be reset before. See [`HalfSweep`](@ref), [`FullSweep`](@ref), and
# [`FullSweepGlobal`](@ref) for the kernel types.
@inline function calc_volume_integral!(backend::Backend, du, u,
                                       mesh::Union{P4estMesh{2}, T8codeMesh{2}},
                                       have_nonconservative_terms::False, equations,
                                       volume_integral::VolumeIntegralFluxDifferencing,
                                       dg::DGSEM, cache)
    nelements(dg, cache) == 0 && return nothing
    # Explicit bounds check, which allows us to assume inbounds access in the kernel
    @boundscheck begin
        check_axes(u, mesh, equations, dg, cache)
        check_axes(du, mesh, equations, dg, cache)
        # Required, e.g., for the `contravariant_vectors` of curvilinear meshes
        check_axes(cache.elements, equations, dg, cache)
    end
    kernel_type = flux_differencing_kernel(backend,
                                           get(cache, :flux_differencing_kernel,
                                               FullSweepGlobal()))
    calc_volume_integral_flux_differencing_2d!(backend, kernel_type, du, u, equations,
                                               volume_integral.volume_flux, dg, cache)
    return nothing
end

# Each thread computes the full sweep of its node, i.e., two-point fluxes are evaluated
# twice, once by each partner, instead of using their symmetry.
function calc_volume_integral_flux_differencing_2d!(backend,
                                                    ::Union{FullSweep,
                                                            FullSweepGlobal},
                                                    du, u, equations, volume_flux,
                                                    dg, cache)
    @unpack derivative_split = dg.basis
    @unpack contravariant_vectors = cache.elements
    NNODES = nnodes(dg)
    kernel! = flux_differencing_nodes_2d_KAkernel!(backend)
    kernel!(du, u, equations, dg, volume_flux, Val(NNODES),
            Val(nvariables(equations)), derivative_split, contravariant_vectors,
            ndrange = (NNODES, NNODES, nelements(dg, cache)))
    return nothing
end

# Half sweep using the symmetry of the volume flux: each two-point flux is computed once
# and shared with the partner node via local memory, as in the 3D [`HalfSweep`](@ref)
# kernel. A workgroup contains several elements so that it is not too small.
function calc_volume_integral_flux_differencing_2d!(backend, ::HalfSweep,
                                                    du, u, equations, volume_flux,
                                                    dg, cache)
    @unpack derivative_split = dg.basis
    @unpack contravariant_vectors = cache.elements
    NNODES = nnodes(dg)
    NELEMENTS_WG = max(1, div(GPU_HALFSWEEP_WORKGROUP_SIZE, NNODES^2))
    kernel! = flux_differencing_halfsweep_2d_KAkernel!(backend,
                                                       (NNODES, NNODES, NELEMENTS_WG))
    kernel!(du, u, equations, dg, volume_flux, Val(NNODES),
            Val(nvariables(equations)), Val(NELEMENTS_WG), derivative_split,
            contravariant_vectors;
            ndrange = (NNODES, NNODES, nelements(dg, cache)))
    return nothing
end

# Number of threads per workgroup of the 2D half sweep kernel (rounded down to full elements)
const GPU_HALFSWEEP_WORKGROUP_SIZE = 128

# For the cyclic distribution of the half sweep and the weighting of the antipodal
# pair for an even number of nodes, see the 3D `HalfSweep` kernel. In contrast to the
# 3D kernel, all two-point fluxes of a node are computed and stored in local memory
# first, followed by a single synchronization, after which each thread sums the fluxes
# of its own node and the ones computed by its partners. Thus, no variable needs to be
# kept across a synchronization.
@kernel inbounds=true function flux_differencing_halfsweep_2d_KAkernel!(du, u,
                                                                        equations,
                                                                        dg::DGSEM,
                                                                        volume_flux,
                                                                        ::Val{NNODES},
                                                                        ::Val{NVARIABLES},
                                                                        ::Val{NELEMENTS_WG},
                                                                        derivative_split,
                                                                        contravariant_vectors) where {
                                                                                                      NNODES,
                                                                                                      NVARIABLES,
                                                                                                      NELEMENTS_WG
                                                                                                      }
    i, j, element = @index(Global, NTuple)
    _, _, local_element = @index(Local, NTuple)

    # fluxes[i, j, local_element, offset, direction, v] between the node (i, j) and its
    # partner at `offset` in `direction`. The variable index is last, see
    # `get_local_node_vars`.
    fluxes = @localmem eltype(du) (NNODES, NNODES, NELEMENTS_WG, NNODES ÷ 2, 2,
                                   NVARIABLES)

    u_node = get_node_vars(u, equations, dg, i, j, element)
    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, element)
    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, element)
    for offset in 1:(NNODES ÷ 2)
        ii = mod(i - 1 + offset, NNODES) + 1
        u_node_ii = get_node_vars(u, equations, dg, ii, j, element)
        Ja1_avg = 0.5f0 * (Ja1_node +
                   get_contravariant_vector(1, contravariant_vectors, ii, j, element))
        fluxtilde1 = volume_flux(u_node, u_node_ii, Ja1_avg, equations)

        jj = mod(j - 1 + offset, NNODES) + 1
        u_node_jj = get_node_vars(u, equations, dg, i, jj, element)
        Ja2_avg = 0.5f0 * (Ja2_node +
                   get_contravariant_vector(2, contravariant_vectors, i, jj, element))
        fluxtilde2 = volume_flux(u_node, u_node_jj, Ja2_avg, equations)

        for v in 1:NVARIABLES
            fluxes[i, j, local_element, offset, 1, v] = fluxtilde1[v]
            fluxes[i, j, local_element, offset, 2, v] = fluxtilde2[v]
        end
    end

    @synchronize

    du_local = zero(SVector{NVARIABLES, eltype(du)})
    for offset in 1:(NNODES ÷ 2)
        # weight the antipodal pair by 1/2 only when the number of nodes is even
        weight = (iseven(NNODES) && offset == NNODES ÷ 2) ? 0.5f0 : 1.0f0
        ii = mod(i - 1 + offset, NNODES) + 1
        iib = mod(i - 1 - offset, NNODES) + 1
        jj = mod(j - 1 + offset, NNODES) + 1
        jjb = mod(j - 1 - offset, NNODES) + 1
        w1 = weight * derivative_split[i, ii]
        w1b = weight * derivative_split[i, iib]
        w2 = weight * derivative_split[j, jj]
        w2b = weight * derivative_split[j, jjb]
        # Use `get_local_node_vars` instead of a closure since the indices `i, j` are
        # recomputed after `@synchronize` by KernelAbstractions.jl, so that a closure
        # capturing them would box them.
        du_local = du_local +
                   w1 *
                   get_local_node_vars(fluxes, Val(NVARIABLES), i, j, local_element,
                                       offset, 1) +
                   w1b * get_local_node_vars(fluxes, Val(NVARIABLES), iib, j,
                                       local_element, offset, 1) +
                   w2 *
                   get_local_node_vars(fluxes, Val(NVARIABLES), i, j, local_element,
                                       offset, 2) +
                   w2b * get_local_node_vars(fluxes, Val(NVARIABLES), i, jjb,
                                       local_element, offset, 2)
    end

    set_node_vars!(du, du_local, equations, dg, i, j, element)
end

@kernel inbounds=true function flux_differencing_nodes_2d_KAkernel!(du, u, equations,
                                                                    dg::DGSEM,
                                                                    volume_flux,
                                                                    ::Val{NNODES},
                                                                    ::Val{NVARIABLES},
                                                                    derivative_split,
                                                                    contravariant_vectors) where {
                                                                                                  NNODES,
                                                                                                  NVARIABLES
                                                                                                  }
    i, j, element = @index(Global, NTuple)

    u_node = get_node_vars(u, equations, dg, i, j, element)
    du_local = zero(SVector{NVARIABLES, eltype(du)})

    # x direction; the diagonal entries of `derivative_split` are zero
    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, element)
    for ii in 1:NNODES
        if ii != i
            Ja1_avg = 0.5f0 * (Ja1_node +
                       get_contravariant_vector(1, contravariant_vectors,
                                                ii, j, element))
            fluxtilde1 = volume_flux(u_node,
                                     get_node_vars(u, equations, dg, ii, j, element),
                                     Ja1_avg, equations)
            du_local = du_local + derivative_split[i, ii] * fluxtilde1
        end
    end

    # y direction
    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, element)
    for jj in 1:NNODES
        if jj != j
            Ja2_avg = 0.5f0 * (Ja2_node +
                       get_contravariant_vector(2, contravariant_vectors,
                                                i, jj, element))
            fluxtilde2 = volume_flux(u_node,
                                     get_node_vars(u, equations, dg, i, jj, element),
                                     Ja2_avg, equations)
            du_local = du_local + derivative_split[j, jj] * fluxtilde2
        end
    end

    set_node_vars!(du, du_local, equations, dg, i, j, element)
end

function prolong2interfaces_and_calc_interface_flux!(backend::Backend,
                                                     surface_flux_values, u,
                                                     mesh::Union{P4estMesh{2},
                                                                 P4estMeshView{2},
                                                                 T8codeMesh{2}},
                                                     have_nonconservative_terms,
                                                     equations,
                                                     surface_integral,
                                                     dg::DGSEM{<:LobattoLegendreBasis},
                                                     cache)
    @unpack neighbor_ids, node_indices = cache.interfaces
    @unpack contravariant_vectors = cache.elements
    ninterfaces(cache.interfaces) == 0 && return nothing
    # Explicit bounds check, which allows us to assume inbounds access in the kernel
    @boundscheck begin
        check_axes(u, mesh, equations, dg, cache)
        check_axes(cache.interfaces, equations, dg, cache)
        check_axes(cache.elements, equations, dg, cache)
        check_axes_surface_flux_values(surface_flux_values, mesh, equations, dg, cache)
    end
    index_range = eachnode(dg)
    kernel! = prolong2interfaces_and_calc_interface_flux_KAkernel!(backend)
    kernel!(surface_flux_values, u, typeof(mesh), have_nonconservative_terms, equations,
            surface_integral, dg, neighbor_ids, node_indices, contravariant_vectors,
            index_range,
            ndrange = (nnodes(dg), ninterfaces(cache.interfaces)))

    return nothing
end

@kernel inbounds=true function prolong2interfaces_and_calc_interface_flux_KAkernel!(surface_flux_values,
                                                                                    u,
                                                                                    MeshT::Type{<:Union{P4estMesh{2},
                                                                                                        P4estMeshView{2},
                                                                                                        T8codeMesh{2}}},
                                                                                    have_nonconservative_terms,
                                                                                    equations,
                                                                                    surface_integral,
                                                                                    dg,
                                                                                    neighbor_ids,
                                                                                    node_indices,
                                                                                    contravariant_vectors,
                                                                                    index_range)
    i, interface = @index(Global, NTuple)
    prolong2interfaces_and_calc_interface_flux_per_node!(surface_flux_values, u, MeshT,
                                                         have_nonconservative_terms,
                                                         equations,
                                                         surface_integral, dg,
                                                         neighbor_ids,
                                                         node_indices,
                                                         contravariant_vectors,
                                                         index_range, i,
                                                         interface)
end

@inline function delayed_index_2d(start, step, i)
    return start + (i - 1) * step
end

Base.@propagate_inbounds function get_interface_values(u, equations, dg, neighbor_ids,
                                                       node_indices,
                                                       contravariant_vectors,
                                                       index_range, i, interface)
    primary_element = neighbor_ids[1, interface]
    primary_indices = node_indices[1, interface]
    primary_direction = indices2direction(primary_indices)

    i_primary_start, i_primary_step = index_to_start_step_2d(primary_indices[1],
                                                             index_range)
    j_primary_start, j_primary_step = index_to_start_step_2d(primary_indices[2],
                                                             index_range)

    i_primary = delayed_index_2d(i_primary_start, i_primary_step, i)
    j_primary = delayed_index_2d(j_primary_start, j_primary_step, i)

    secondary_element = neighbor_ids[2, interface]
    secondary_indices = node_indices[2, interface]
    secondary_direction = indices2direction(secondary_indices)

    i_secondary_start, i_secondary_step = index_to_start_step_2d(secondary_indices[1],
                                                                 index_range)
    j_secondary_start, j_secondary_step = index_to_start_step_2d(secondary_indices[2],
                                                                 index_range)

    i_secondary_node = delayed_index_2d(i_secondary_start, i_secondary_step, i)
    j_secondary_node = delayed_index_2d(j_secondary_start, j_secondary_step, i)

    if i_secondary_step == 0
        i_secondary = j_secondary_node
    else
        i_secondary = i_secondary_node
    end

    u_ll = get_node_vars(u, equations, dg, i_primary, j_primary, primary_element)
    u_rr = get_node_vars(u, equations, dg, i_secondary_node, j_secondary_node,
                         secondary_element)

    normal_direction = get_normal_direction(primary_direction, contravariant_vectors,
                                            i_primary, j_primary, primary_element)

    return (u_ll, u_rr, normal_direction, primary_direction, primary_element,
            i_secondary, secondary_direction, secondary_element)
end

Base.@propagate_inbounds function prolong2interfaces_and_calc_interface_flux_per_node!(surface_flux_values,
                                                                                       u,
                                                                                       MeshT::Type{<:Union{P4estMesh{2},
                                                                                                           P4estMeshView{2},
                                                                                                           T8codeMesh{2}}},
                                                                                       have_nonconservative_terms::False,
                                                                                       equations,
                                                                                       surface_integral,
                                                                                       dg,
                                                                                       neighbor_ids,
                                                                                       node_indices,
                                                                                       contravariant_vectors,
                                                                                       index_range,
                                                                                       i,
                                                                                       interface)
    @unpack surface_flux = surface_integral

    u_ll, u_rr, normal_direction, primary_direction, primary_element,
    i_secondary, secondary_direction, secondary_element = get_interface_values(u,
                                                                               equations,
                                                                               dg,
                                                                               neighbor_ids,
                                                                               node_indices,
                                                                               contravariant_vectors,
                                                                               index_range,
                                                                               i,
                                                                               interface)

    flux_ = surface_flux(u_ll, u_rr, normal_direction, equations)
    for v in eachvariable(equations)
        surface_flux_values[v, i, primary_direction, primary_element] = flux_[v]
        surface_flux_values[v, i_secondary, secondary_direction,
        secondary_element] = -flux_[v]
    end
    return nothing
end

Base.@propagate_inbounds function prolong2interfaces_and_calc_interface_flux_per_node!(surface_flux_values,
                                                                                       u,
                                                                                       MeshT::Type{<:Union{P4estMesh{2},
                                                                                                           P4estMeshView{2},
                                                                                                           T8codeMesh{2}}},
                                                                                       have_nonconservative_terms::True,
                                                                                       equations,
                                                                                       surface_integral,
                                                                                       dg,
                                                                                       neighbor_ids,
                                                                                       node_indices,
                                                                                       contravariant_vectors,
                                                                                       index_range,
                                                                                       i,
                                                                                       interface)
    prolong2interfaces_and_calc_interface_flux_per_node!(surface_flux_values, u, MeshT,
                                                         have_nonconservative_terms,
                                                         combine_conservative_and_nonconservative_fluxes(surface_integral.surface_flux,
                                                                                                         equations),
                                                         equations, surface_integral,
                                                         dg,
                                                         neighbor_ids,
                                                         node_indices,
                                                         contravariant_vectors,
                                                         index_range, i, interface)
    return nothing
end

Base.@propagate_inbounds function prolong2interfaces_and_calc_interface_flux_per_node!(surface_flux_values,
                                                                                       u,
                                                                                       MeshT::Type{<:Union{P4estMesh{2},
                                                                                                           P4estMeshView{2},
                                                                                                           T8codeMesh{2}}},
                                                                                       have_nonconservative_terms::True,
                                                                                       combine_conservative_and_nonconservative_fluxes::True,
                                                                                       equations,
                                                                                       surface_integral,
                                                                                       dg,
                                                                                       neighbor_ids,
                                                                                       node_indices,
                                                                                       contravariant_vectors,
                                                                                       index_range,
                                                                                       i,
                                                                                       interface)
    @unpack surface_flux = surface_integral

    u_ll, u_rr, normal_direction, primary_direction, primary_element,
    i_secondary, secondary_direction, secondary_element = get_interface_values(u,
                                                                               equations,
                                                                               dg,
                                                                               neighbor_ids,
                                                                               node_indices,
                                                                               contravariant_vectors,
                                                                               index_range,
                                                                               i,
                                                                               interface)

    flux_left, flux_right = surface_flux(u_ll, u_rr, normal_direction, equations)
    for v in eachvariable(equations)
        surface_flux_values[v, i, primary_direction, primary_element] = flux_left[v]
        surface_flux_values[v, i_secondary, secondary_direction,
        secondary_element] = -flux_right[v]
    end
    return nothing
end

@kernel inbounds=true function prolong2boundaries_kernel!(u,
                                                          MeshT::Type{<:Union{P4estMesh{2},
                                                                              P4estMeshView{2},
                                                                              T8codeMesh{2}}},
                                                          equations, dg, index_range,
                                                          u_boundaries, neighbor_ids,
                                                          node_indices)
    i, boundary = @index(Global, NTuple)
    prolong2boundaries_per_node!(u, MeshT, equations, dg, index_range, u_boundaries,
                                 neighbor_ids, node_indices, i, boundary)
end

@inline function boundary_node_ndrange(mesh::Union{P4estMesh{2}, P4estMeshView{2},
                                                   T8codeMesh{2}}, dg)
    return (nnodes(dg),)
end

Base.@propagate_inbounds function prolong2boundaries_per_node!(u,
                                                               MeshT::Type{<:Union{P4estMesh{2},
                                                                                   P4estMeshView{2},
                                                                                   T8codeMesh{2}}},
                                                               equations, dg::DG,
                                                               index_range,
                                                               u_boundaries,
                                                               neighbor_ids,
                                                               node_indices, i,
                                                               boundary)
    # Copy solution data from the element using "delayed indexing" with
    # a start value and a step size to get the correct face and orientation.
    element = neighbor_ids[boundary]
    node_index = node_indices[boundary]

    i_node_start, i_node_step = index_to_start_step_2d(node_index[1], index_range)
    j_node_start, j_node_step = index_to_start_step_2d(node_index[2], index_range)

    i_node = delayed_index_2d(i_node_start, i_node_step, i)
    j_node = delayed_index_2d(j_node_start, j_node_step, i)

    u_node = get_node_vars(u, equations, dg, i_node, j_node, element)
    set_node_vars!(u_boundaries, u_node, equations, dg, i, boundary)

    return nothing
end

@kernel inbounds=true function calc_boundary_flux_kernel!(u,
                                                          surface_flux_values,
                                                          boundary_condition_indices,
                                                          neighbor_ids,
                                                          node_indices_arr,
                                                          t,
                                                          boundary_condition::BC,
                                                          index_range,
                                                          MeshT::Type{<:Union{P4estMesh{2},
                                                                              P4estMeshView{2},
                                                                              T8codeMesh{2}}},
                                                          equations,
                                                          surface_integral,
                                                          dg,
                                                          cache, node_coordinates,
                                                          contravariant_vectors) where {BC}
    i, local_index = @index(Global, NTuple)

    if local_index <= length(boundary_condition_indices)
        boundary = boundary_condition_indices[local_index]

        calc_boundary_flux_per_node!(u,
                                     surface_flux_values, t, boundary_condition,
                                     MeshT, equations, surface_integral, dg, cache,
                                     boundary, neighbor_ids, node_indices_arr,
                                     index_range, node_coordinates,
                                     contravariant_vectors, i)
    end
end

Base.@propagate_inbounds function calc_boundary_flux_per_node!(u,
                                                               surface_flux_values, t,
                                                               boundary_condition,
                                                               MeshT::Type{<:Union{P4estMesh{2},
                                                                                   P4estMeshView{2},
                                                                                   T8codeMesh{2}}},
                                                               equations,
                                                               surface_integral, dg,
                                                               cache,
                                                               boundary, neighbor_ids,
                                                               node_indices_arr,
                                                               index_range,
                                                               node_coordinates,
                                                               contravariant_vectors, i)

    # Get information on the adjacent element, compute the surface fluxes,
    # and store them
    element = neighbor_ids[boundary]
    node_indices = node_indices_arr[boundary]
    direction = indices2direction(node_indices)

    i_node_start, i_node_step = index_to_start_step_2d(node_indices[1], index_range)
    j_node_start, j_node_step = index_to_start_step_2d(node_indices[2], index_range)

    i_node = delayed_index_2d(i_node_start, i_node_step, i)
    j_node = delayed_index_2d(j_node_start, j_node_step, i)

    calc_boundary_flux!(u, surface_flux_values, t, boundary_condition, MeshT,
                        have_nonconservative_terms(equations), equations,
                        surface_integral, dg, cache, i_node, j_node,
                        i, direction, element, boundary, node_coordinates,
                        contravariant_vectors)
end

# inlined version of the boundary flux calculation along a physical interface
Base.@propagate_inbounds function calc_boundary_flux!(u, surface_flux_values, t,
                                                      boundary_condition,
                                                      MeshT::Type{<:Union{P4estMesh{2},
                                                                          P4estMeshView{2},
                                                                          T8codeMesh{2}}},
                                                      have_nonconservative_terms::False,
                                                      equations,
                                                      surface_integral, dg, cache,
                                                      i_index, j_index, node_index,
                                                      direction_index, element_index,
                                                      boundary_index, node_coordinates,
                                                      contravariant_vectors)
    @unpack surface_flux = surface_integral

    # Extract solution data from boundary container
    u_inner = get_node_vars(u, equations, dg, node_index, boundary_index)

    # Outward-pointing normal direction (not normalized)
    normal_direction = get_normal_direction(direction_index, contravariant_vectors,
                                            i_index, j_index, element_index)

    # Coordinates at boundary node
    x = get_node_coords(node_coordinates, equations, dg,
                        i_index, j_index, element_index)

    flux_ = boundary_condition(u_inner, normal_direction, x, t, surface_flux, equations)

    # Copy flux to element storage in the correct orientation
    for v in eachvariable(equations)
        surface_flux_values[v, node_index, direction_index, element_index] = flux_[v]
    end
end

Base.@propagate_inbounds function calc_boundary_flux!(u, surface_flux_values, t,
                                                      boundary_condition,
                                                      MeshT::Type{<:Union{P4estMesh{2},
                                                                          P4estMeshView{2},
                                                                          T8codeMesh{2}}},
                                                      have_nonconservative_terms::True,
                                                      equations,
                                                      surface_integral, dg, cache,
                                                      i_index, j_index, node_index,
                                                      direction_index, element_index,
                                                      boundary_index, node_coordinates,
                                                      contravariant_vectors)
    calc_boundary_flux!(u, surface_flux_values, t, boundary_condition, MeshT,
                        have_nonconservative_terms,
                        combine_conservative_and_nonconservative_fluxes(surface_integral.surface_flux,
                                                                        equations),
                        equations,
                        surface_integral, dg, cache,
                        i_index, j_index, node_index,
                        direction_index, element_index, boundary_index,
                        node_coordinates, contravariant_vectors)
    return nothing
end

Base.@propagate_inbounds function calc_boundary_flux!(u, surface_flux_values, t,
                                                      boundary_condition,
                                                      MeshT::Type{<:Union{P4estMesh{2},
                                                                          P4estMeshView{2},
                                                                          T8codeMesh{2}}},
                                                      have_nonconservative_terms::True,
                                                      combine_conservative_and_nonconservative_fluxes::True,
                                                      equations,
                                                      surface_integral, dg::DG, cache,
                                                      i_index, j_index, node_index,
                                                      direction_index, element_index,
                                                      boundary_index, node_coordinates,
                                                      contravariant_vectors)
    @unpack surface_flux = surface_integral

    # Extract solution data from boundary container
    u_inner = get_node_vars(u, equations, dg, node_index, boundary_index)

    # Outward-pointing normal direction (not normalized)
    normal_direction = get_normal_direction(direction_index, contravariant_vectors,
                                            i_index, j_index, element_index)

    # Coordinates at boundary node
    x = get_node_coords(node_coordinates, equations, dg,
                        i_index, j_index, element_index)

    # Call pointwise numerical flux functions for the conservative and nonconservative part
    # in the normal direction on the boundary
    flux = boundary_condition(u_inner, normal_direction, x, t,
                              surface_flux, equations)

    # Copy flux to element storage in the correct orientation
    for v in eachvariable(equations)
        surface_flux_values[v, node_index,
        direction_index, element_index] = flux[v]
    end

    return nothing
end

# For GPU backends mortars are not yet implemented
function prolong2mortars!(backend::Backend, cache, u, mesh, equations, mortar, dg)
    @assert isempty(eachmortar(dg, cache))
    return nothing
end

# For GPU backends mortars are not yet implemented
function calc_mortar_flux!(backend::Backend, surface_flux_values, mesh,
                           have_nonconservative_terms, equations, mortar,
                           surface_integral, dg, cache)
    @assert isempty(eachmortar(dg, cache))
end

function calc_surface_integral_and_apply_jacobian_and_calc_sources!(backend::Backend,
                                                                    du, u, t,
                                                                    source_terms,
                                                                    mesh::Union{P4estMesh{2},
                                                                                T8codeMesh{2},
                                                                                P4estMeshView{2}},
                                                                    equations,
                                                                    surface_integral::SurfaceIntegralWeakForm,
                                                                    dg::DGSEM{<:LobattoLegendreBasis},
                                                                    cache)
    nelements(dg, cache) == 0 && return nothing
    @unpack inverse_weights = dg.basis
    @unpack surface_flux_values, inverse_jacobian, node_coordinates = cache.elements
    # Explicit bounds check, which allows us to assume inbounds access in the kernel
    @boundscheck begin
        check_axes(du, mesh, equations, dg, cache)
        check_axes(u, mesh, equations, dg, cache)
        check_axes(cache.elements, equations, dg, cache)
        check_axes_surface_flux_values(surface_flux_values, mesh, equations, dg, cache)
    end
    kernel_cache = kernel_filter_cache(cache)
    NNODES = nnodes(dg)
    kernel! = calc_surface_integral_and_apply_jacobian_and_calc_sources_KAkernel!(backend)
    kernel!(du, u, t, source_terms, node_coordinates, typeof(mesh), equations,
            inverse_weights[1], Val(NNODES), surface_flux_values, dg, inverse_jacobian,
            kernel_cache,
            ndrange = (NNODES, NNODES, nelements(dg, cache)))

    return nothing
end

@kernel inbounds=true function calc_surface_integral_and_apply_jacobian_and_calc_sources_KAkernel!(du,
                                                                                                   u,
                                                                                                   t,
                                                                                                   source_terms::Source,
                                                                                                   node_coordinates,
                                                                                                   MeshT::Type{<:Union{P4estMesh{2},
                                                                                                                       P4estMeshView{2},
                                                                                                                       T8codeMesh{2}}},
                                                                                                   equations::AbstractEquations{2},
                                                                                                   factor,
                                                                                                   ::Val{NNODES},
                                                                                                   surface_flux_values,
                                                                                                   dg::DGSEM,
                                                                                                   inverse_jacobian,
                                                                                                   cache) where {
                                                                                                                 NNODES,
                                                                                                                 Source
                                                                                                                 }
    i, j, element = @index(Global, NTuple)
    # Note that all fluxes have been computed with outward-pointing normal vectors.
    # This computes the **negative** surface integral contribution,
    # i.e., M^{-1} * boundary_interpolation^T (which is for Gauss-Lobatto DGSEM just M^{-1} * B)
    # and the missing "-" is taken care of by the Jacobian factor below.
    #
    # We also use explicit assignments instead of `+=` to let `@muladd` turn these
    # into FMAs (see comment at the top of the file).
    #
    # factor = inverse_weights[1]
    # For LGL basis: Identical to weighted boundary interpolation at x = ±1
    x_node_interface = (i == 1) | (i == NNODES)
    y_node_interface = (j == 1) | (j == NNODES)
    x_face = ifelse(i == 1, 1, 2)
    y_face = ifelse(j == 1, 3, 4)
    _zero = zero(eltype(du))
    surface_node = SVector(ntuple(@inline(v->ifelse(x_node_interface,
                                                    @inbounds(surface_flux_values[v, j,
                                                                                  x_face,
                                                                                  element]),
                                                    _zero) +
                                             ifelse(y_node_interface,
                                                    @inbounds(surface_flux_values[v, i,
                                                                                  y_face,
                                                                                  element]),
                                                    _zero)),
                                  Val(nvariables(equations))))
    source_node = calc_source_terms_node(u, t, source_terms, node_coordinates,
                                         equations, dg, i, j, element)
    jacobian_factor = inverse_jacobian[i, j, element]
    du_local = get_node_vars(du, equations, dg, i, j, element) + factor * surface_node
    du_node = source_node - jacobian_factor * du_local
    set_node_vars!(du, du_node, equations, dg, i, j, element)
end

Base.@propagate_inbounds function calc_source_terms_node(u, t, source_terms,
                                                         node_coordinates,
                                                         equations, dg::DG, indices...)
    u_local = get_node_vars(u, equations, dg, indices...)
    x_local = get_node_coords(node_coordinates, equations, dg, indices...)

    return source_terms(u_local, x_local, t, equations)
end

@inline function calc_source_terms_node(u, t, source_terms::Nothing,
                                        node_coordinates,
                                        equations, dg::DG, indices...)
    return zero(SVector{nvariables(equations), eltype(u)})
end
end #muladd
