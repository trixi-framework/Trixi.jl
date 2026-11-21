@inline function get_node_turbo(turbo_local, ::Val{NTURBO},
                                indices...) where {NTURBO}
    return ntuple(v -> (@inbounds turbo_local[v, indices...]), Val(NTURBO))
end

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
        check_axes_volume_integrals(mesh, equations, volume_integral, dg, cache)
        check_axes_aux(get_aux_vars(cache), equations, dg, cache)
    end

    # As on CPUs, the volume integral kernels overwrite `du` (`set_not_add = True()`),
    # so that we do not need to launch an additional kernel to reset `du` to zero.
    # This is also what the specialized GPU volume kernels do.
    kernel! = volume_integral_KAkernel!(backend)
    kernel_cache = kernel_filter_cache(cache)
    kernel!(du, u, typeof(mesh), have_nonconservative_terms, True(), equations,
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
                                                         set_not_add, equations,
                                                         volume_integral, dg::DGSEM,
                                                         cache) where {MeshT}
    element = @index(Global)
    volume_integral_kernel!(du, u, element, MeshT, have_nonconservative_terms,
                            set_not_add, equations, volume_integral, dg, cache)
end

# The half sweep and full sweep kernels use local share data, which is limited to
# 48 KiB per workgroup on NVIDIA GPUs and 64 KiB on AMD GPUs (as of 2026).
function check_flux_differencing_shared_memory(kernel::Union{HalfSweep, FullSweep}, semi)
    dg = semi.solver
    equations = semi.equations
    volume_integral = semi.solver.volume_integral

    # `FullSweep` stages the auxiliary variables in shared memory as well
    nshared = kernel isa HalfSweep ? nvariables(equations) :
              nvariables(equations) + nauxvariables(equations)

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
        In case, consider using `tuning = Tuning(flux_differencing_kernel = FullSweepGlobal())` in `trixi_adapt`." flux_differencing_kernel=kernel nvariables=nvariables(semi.equations) nauxvariables=nauxvariables(semi.equations) polydeg=polydeg(dg) shared_memory=Base.format_bytes(shared_memory)
    end

    workgroup_size = nnodes_element

    if workgroup_size > 1024
        @warn "The workgroup size required by the selected flux differencing kernel likely exceeds the device limit.
        Please, consider reducing the polynomial degree or using `tuning = Tuning(flux_differencing_kernel = FullSweepGlobal())` in `trixi_adapt`." flux_differencing_kernel=kernel nvariables=nvariables(semi.equations) polydeg=polydeg(dg) workgroup_size
    end

    return nothing
end

# This kernel does not use any shared memory
check_flux_differencing_shared_memory(::FullSweepGlobal, semi) = nothing
