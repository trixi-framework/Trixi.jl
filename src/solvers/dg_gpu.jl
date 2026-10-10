@inline function get_node_turbo(turbo_local, ::Val{NAUX},
                                indices...) where {NAUX}
    return ntuple(v -> (@inbounds turbo_local[v, indices...]), Val(NAUX))
end

# The volume integral kernels shared with the CPU code (`volume_integral_kernel!`) handle
# one element per work-item and update `du` with many read-modify-write accesses per node.
# On GPUs, these global memory accesses are not coalesced across work-items and they make
# the kernels slow. Thus, on GPUs, the general fallback below accumulates the volume terms
# of an element in work-item local storage and writes `du` once. On the CPU backend of
# KernelAbstractions.jl, the kernels shared with the CPU code are faster.
@inline use_gpu_volume_kernels(backend::Backend) = !(backend isa KernelAbstractions.CPU)

# The work-item local storage only supports indexing `du[v, node..., element]`, which is
# sufficient for these volume integrals. Others, e.g., `VolumeIntegralEntropyCorrection`,
# also use `du[.., element]` as an array.
@inline function accumulate_volume_integral_locally(backend::Backend,
                                                    ::Union{VolumeIntegralWeakForm,
                                                            VolumeIntegralFluxDifferencing})
    return use_gpu_volume_kernels(backend)
end
@inline accumulate_volume_integral_locally(backend::Backend, volume_integral) = false

# This is a general fallback for volume integral kernels, parallelizing across
# elements on GPUs in the same way as we do on CPUs. Optimized kernels, e.g.,
# for flux differencing, parallelize across the individual solution nodes
# and are contained in the files src/solvers/dgsem_p4est/dg_2d_gpu.jl and
# src/solvers/dgsem_p4est/dg_3d_gpu.jl.
function calc_volume_integral!(backend::Backend, du, u, mesh,
                               have_nonconservative_terms, equations,
                               volume_integral, dg::DGSEM, cache)
    nelements(dg, cache) == 0 && return nothing
    # Explicit bounds check, which allows us to assume inbounds access in the kernel
    @boundscheck begin
        check_axes(u, mesh, equations, dg, cache)
        check_axes(du, mesh, equations, dg, cache)
        # Required, e.g., for the `contravariant_vectors` of curvilinear meshes
        check_axes(cache.elements, equations, dg, cache)
    end

    accumulate_locally = accumulate_volume_integral_locally(backend, volume_integral)
    if !accumulate_locally
        # Reset du
        # In the usual (CPU) code, this is called at the beginning of rhs_hyperbolic!
        # However, we can significantly improve the performance on GPUs by avoiding
        # launching an additional kernel for this memory reset. Thus, GPU volume
        # kernels write directly into the existing `du` array, and we reset it here
        # if the kernel adds to `du`.
        @trixi_timeit_ext backend timer() "reset ∂u/∂t" begin
            set_zero!(du, dg, cache)
        end
    end

    kernel! = volume_integral_KAkernel!(backend)
    kernel_cache = kernel_filter_cache(cache)
    kernel!(du, u, typeof(mesh), have_nonconservative_terms, equations,
            volume_integral, dg, kernel_cache, Val(accumulate_locally),
            ndrange = nelements(dg, cache))
    return nothing
end

# Julia does not specialize on arguments of type `Type` or `Function` that are only
# passed through to other functions but not used directly, see
# https://docs.julialang.org/en/v1/manual/performance-tips/#Be-aware-of-when-Julia-avoids-specializing
# With the KernelAbstractions.jl v0.9 CPU backend, kernels are ordinary Julia functions, so
# this leads to dynamic dispatch (and allocations) for every element or node.
# Thus, kernel arguments such as `MeshT`, `source_terms`, or `boundary_condition` need a
# type parameter (e.g., `::Type{MeshT}` or `source_terms::Source` with
# `where {MeshT}` or `where {Source}`) or a type annotation matching all methods of the
# called functions (e.g., `MeshT::Type{<:Union{P4estMesh{3}, T8codeMesh{3}}}`) to
# avoid this. GPU backends always specialize fully.
@kernel inbounds=true function volume_integral_KAkernel!(du, u, ::Type{MeshT},
                                                         have_nonconservative_terms,
                                                         equations,
                                                         volume_integral, dg::DGSEM,
                                                         cache,
                                                         ::Val{ACCUMULATE_LOCALLY}) where {
                                                                                           MeshT,
                                                                                           ACCUMULATE_LOCALLY
                                                                                           }
    element = @index(Global)
    if ACCUMULATE_LOCALLY
        # Accumulate the volume integral of this element in work-item local storage
        # and overwrite `du` once at the end, see `use_gpu_volume_kernels`.
        du_element = zero_element_local(du, equations, dg)
        volume_integral_kernel!(ElementLocal(du_element), u, element, MeshT,
                                have_nonconservative_terms, equations, volume_integral,
                                dg, cache)
        store_element_local!(du, du_element, element)
    else
        volume_integral_kernel!(du, u, element, MeshT, have_nonconservative_terms,
                                equations, volume_integral, dg, cache)
    end
end

# Work-item local storage for the values of `du` in one element, i.e., `du[:, .., element]`.
@inline function zero_element_local(du::AbstractArray{T, N}, equations,
                                    dg::DG) where {T, N}
    S = Tuple{nvariables(equations), ntuple(_ -> nnodes(dg), Val(N - 2))...}
    return zero(MArray{S, T})
end

# Wrapper to pass element local storage `data::MArray` to the CPU kernels: index
# `[v, node..., element]` maps to `data[v, node...]`.
# We access `data` with the natural alignment of its elements. `getindex` and `setindex!`
# of an `MArray` use `unsafe_load`/`unsafe_store!` with an alignment of 1 byte, which
# the NVPTX back-end lowers to byte-wise loads and stores of local memory.
struct ElementLocal{A <: MArray}
    data::A
end

@inline function element_local_pointer(a::ElementLocal)
    return Base.unsafe_convert(Ptr{eltype(a.data)}, pointer_from_objref(a.data))
end

Base.@propagate_inbounds function Base.getindex(a::ElementLocal, I::Vararg{Integer})
    data = a.data
    T = eltype(data)
    i = LinearIndices(data)[Base.front(I)...]
    GC.@preserve data begin
        return Core.Intrinsics.pointerref(element_local_pointer(a), i,
                                          Base.datatype_alignment(T))
    end
end

Base.@propagate_inbounds function Base.setindex!(a::ElementLocal, value,
                                                 I::Vararg{Integer})
    data = a.data
    T = eltype(data)
    i = LinearIndices(data)[Base.front(I)...]
    GC.@preserve data begin
        Core.Intrinsics.pointerset(element_local_pointer(a), convert(T, value), i,
                                   Base.datatype_alignment(T))
    end
    return a
end

Base.@propagate_inbounds function store_element_local!(du, du_element, element)
    a = ElementLocal(du_element)
    for I in CartesianIndices(du_element)
        du[Tuple(I)..., element] = a[Tuple(I)..., element]
    end
    return nothing
end

# The half sweep and full sweep kernels use local share data, which is limited to
# 48 KiB per workgroup on NVIDIA GPUs and 64 KiB on AMD GPUs (as of 2026).
function check_flux_differencing_shared_memory(kernel::Union{HalfSweep, FullSweep}, semi)
    if semi.mesh isa Union{P4estMesh{2}, T8codeMesh{2}}
        # Equations with nonconservative terms use the generic kernel without local memory
        if have_nonconservative_terms(semi.equations) === True()
            return nothing
        end
        return check_flux_differencing_shared_memory_2d(kernel, semi)
    end

    dg = semi.solver
    equations = semi.equations
    volume_integral = semi.solver.volume_integral

    nshared = nvariables(equations)

    if volume_integral isa VolumeIntegralFluxDifferencing
        volume_flux = volume_integral.volume_flux
        if volume_flux isa FluxTurbo
            nturbo = typeof(nturbovars(volume_flux.numerical_flux, equations)).parameters[1]
            nshared = kernel isa HalfSweep ? nvariables(equations) + nturbo : nturbo
        end
    end

    # One workgroup handles all nodes of an element
    nnodes_element = nnodes(dg)^ndims(semi)
    shared_memory = nshared * nnodes_element * sizeof(real(dg))

    if shared_memory > 48 * 1024
        @warn "The shared memory required by the selected flux differencing kernel may exceed the limit of the GPU.
        In case, consider using `flux_differencing_kernel = FullSweepGlobal()`." flux_differencing_kernel=kernel nvariables=nvariables(semi.equations) polydeg=polydeg(dg) shared_memory=Base.format_bytes(shared_memory)
    end

    workgroup_size = nnodes_element

    if workgroup_size > 1024
        @warn "The workgroup size required by the selected flux differencing kernel likely exceeds the device limit.
        Please, consider reducing the polynomial degree or using `flux_differencing_kernel = FullSweepGlobal()`." flux_differencing_kernel=kernel nvariables=nvariables(semi.equations) polydeg=polydeg(dg) workgroup_size
    end

    return nothing
end

# This kernel does not use any shared memory
check_flux_differencing_shared_memory(::FullSweepGlobal, semi) = nothing

# The 2D kernels for the `P4estMesh{2}` and `T8codeMesh{2}` in
# `src/solvers/dgsem_p4est/dg_2d_gpu.jl` use local memory only for the half sweep,
# where each workgroup contains several elements. The full sweep kernel does not use
# any local memory.
check_flux_differencing_shared_memory_2d(::FullSweep, semi) = nothing

function check_flux_differencing_shared_memory_2d(kernel::HalfSweep, semi)
    dg = semi.solver
    n_nodes = nnodes(dg)
    nelements_workgroup = max(1, div(GPU_HALFSWEEP_WORKGROUP_SIZE, n_nodes^2))
    # Two-point fluxes of all nodes with half of their partners in both directions
    shared_memory = nvariables(semi.equations) * (n_nodes ÷ 2) * 2 * n_nodes^2 *
                    nelements_workgroup * sizeof(real(dg))

    if shared_memory > 48 * 1024
        @warn "The shared memory required by the selected flux differencing kernel may exceed the limit of the GPU.
        In case, consider using `flux_differencing_kernel = FullSweep()`." flux_differencing_kernel=kernel nvariables=nvariables(semi.equations) polydeg=polydeg(dg) shared_memory=Base.format_bytes(shared_memory)
    end

    workgroup_size = n_nodes^2 * nelements_workgroup
    if workgroup_size > 1024
        @warn "The workgroup size required by the selected flux differencing kernel likely exceeds the device limit.
        Please, consider reducing the polynomial degree or using `flux_differencing_kernel = FullSweep()`." flux_differencing_kernel=kernel nvariables=nvariables(semi.equations) polydeg=polydeg(dg) workgroup_size
    end

    return nothing
end
