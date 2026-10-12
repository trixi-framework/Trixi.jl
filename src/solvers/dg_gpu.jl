@inline function get_node_turbo(turbo_local, ::Val{NAUX},
                                indices...) where {NAUX}
    return ntuple(v -> (@inbounds turbo_local[v, indices...]), Val(NAUX))
end

# This is a general fallback for volume integral kernels, parallelizing across
# elements on GPUs in the same way as we do on CPUs. Optimized kernels, e.g.,
# for flux differencing, parallelize across the individual solution nodes
# and are contained in the files src/solvers/dgsem_p4est/dg_2d_gpu.jl and
# src/solvers/dgsem_p4est/dg_3d_gpu.jl.
function calc_volume_integral!(backend::Backend, du, u, mesh,
                               have_nonconservative_terms, equations,
                               volume_integral, dg::DGSEM, cache)
    return calc_volume_integral_per_element!(backend, du, u, mesh,
                                             have_nonconservative_terms, equations,
                                             volume_integral, dg, cache)
end

# Also used by specialized kernels for cases they do not support, e.g., nonconservative
# terms without combined fluxes in `src/solvers/dgsem_p4est/dg_2d_gpu.jl`.
function calc_volume_integral_per_element!(backend::Backend, du, u, mesh,
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

    # Reset du
    # In the usual (CPU) code, this is called at the beginning of rhs_hyperbolic!
    # However, we can significantly improve the performance on GPUs by avoiding
    # launching an additional kernel for this memory reset. Thus, specialized
    # GPU volume kernels write directly into the existing `du` array, and we reset
    # it here for the general (fallback) case.
    @trixi_timeit_ext backend timer() "reset ∂u/∂t" begin
        set_zero!(du, dg, cache)
    end

    kernel! = volume_integral_KAkernel!(backend)
    kernel_cache = kernel_filter_cache(cache)
    kernel!(du, u, typeof(mesh), have_nonconservative_terms, equations,
            volume_integral, dg, kernel_cache,
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
                                                         cache) where {MeshT}
    element = @index(Global)
    volume_integral_kernel!(du, u, element, MeshT, have_nonconservative_terms,
                            equations, volume_integral, dg, cache)
end

# The half sweep and full sweep kernels use local share data, which is limited to
# 48 KiB per workgroup on NVIDIA GPUs and 64 KiB on AMD GPUs (as of 2026).
function check_flux_differencing_shared_memory(kernel::Union{HalfSweep, FullSweep}, semi)
    dg = semi.solver
    shared_memory, workgroup_size = flux_differencing_local_memory(kernel, semi.mesh,
                                                                   semi.equations, dg)

    if shared_memory > 48 * 1024
        @warn "The shared memory required by the selected flux differencing kernel may exceed the limit of the GPU.
        In case, consider using `tuning = Tuning(flux_differencing_kernel = FullSweepGlobal())` in `trixi_adapt`." flux_differencing_kernel=kernel nvariables=nvariables(semi.equations) polydeg=polydeg(dg) shared_memory=Base.format_bytes(shared_memory)
    end

    if workgroup_size > 1024
        @warn "The workgroup size required by the selected flux differencing kernel likely exceeds the device limit.
        Please, consider reducing the polynomial degree or using `tuning = Tuning(flux_differencing_kernel = FullSweepGlobal())` in `trixi_adapt`." flux_differencing_kernel=kernel nvariables=nvariables(semi.equations) polydeg=polydeg(dg) workgroup_size
    end

    return nothing
end

# Local memory in bytes and number of threads of a workgroup of the flux differencing
# kernels, where one workgroup handles all nodes of an element
function flux_differencing_local_memory(kernel, mesh, equations, dg)
    nshared = nvariables(equations)

    volume_integral = dg.volume_integral
    if volume_integral isa VolumeIntegralFluxDifferencing
        volume_flux = volume_integral.volume_flux
        if volume_flux isa FluxTurbo
            nturbo = typeof(nturbovars(volume_flux.numerical_flux, equations)).parameters[1]
            nshared = kernel isa HalfSweep ? nvariables(equations) + nturbo : nturbo
        end
    end

    nnodes_element = nnodes(dg)^ndims(mesh)
    return nshared * nnodes_element * sizeof(real(dg)), nnodes_element
end

# This kernel does not use any shared memory
check_flux_differencing_shared_memory(::FullSweepGlobal, semi) = nothing

# The 2D kernels for the `P4estMesh{2}` and `T8codeMesh{2}` in
# `src/solvers/dgsem_p4est/dg_2d_gpu.jl` use workgroups containing several elements.
function flux_differencing_local_memory(kernel, mesh::Union{P4estMesh{2}, T8codeMesh{2}},
                                        equations, dg)
    # Equations with nonconservative terms use the kernels parallelized over the nodes
    # only for combined fluxes, otherwise the generic kernel without local memory
    volume_integral = dg.volume_integral
    if volume_integral isa VolumeIntegralFluxDifferencing &&
       have_nonconservative_terms(equations) === True() &&
       combine_conservative_and_nonconservative_fluxes(volume_integral.volume_flux,
                                                       equations) === False()
        return 0, 0
    end

    n_nodes = nnodes(dg)
    if kernel isa HalfSweep
        # Two-point fluxes of all nodes with half of their partners in both directions
        nshared = nvariables(equations) * div(n_nodes, 2) * 2
    else
        # Solution at all nodes
        nshared = nvariables(equations)
    end
    workgroup_size = n_nodes^2 * nelements_workgroup_2d(n_nodes)
    return nshared * workgroup_size * sizeof(real(dg)), workgroup_size
end
