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
    nelements(dg, cache) == 0 && return nothing
    # Reset du
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

@kernel function volume_integral_KAkernel!(du, u, MeshT,
                                           have_nonconservative_terms, equations,
                                           volume_integral, dg::DGSEM, cache)
    element = @index(Global)
    volume_integral_kernel!(du, u, element, MeshT, have_nonconservative_terms,
                            equations, volume_integral, dg, cache)
end

# The half sweep and full sweep kernels use local share data, which is limited to
# 48 KiB per workgroup on NVIDIA GPUs and 64 KiB on AMD GPUs (as of 2026).
function check_flux_differencing_shared_memory(kernel::Union{HalfSweep, FullSweep}, semi)
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

    shared_memory = nshared * nnodes(dg)^3 * sizeof(real(dg))

    if shared_memory > 48 * 1024
        @warn "The shared memory required by the selected flux differencing kernel may exceed the limit of the GPU.
        In case, consider using `flux_differencing_kernel = FullSweepGlobal()`." flux_differencing_kernel=kernel nvariables=nvariables(semi.equations) polydeg=polydeg(dg) shared_memory=Base.format_bytes(shared_memory)
    end

    workgroup_size = nnodes(dg)^3

    if workgroup_size > 1024
        @warn "The workgroup size required by the selected flux differencing kernel likely exceeds the device limit.
        Please, consider reducing the polynomial degree or using `flux_differencing_kernel = FullSweepGlobal()`." flux_differencing_kernel=kernel nvariables=nvariables(semi.equations) polydeg=polydeg(dg) workgroup_size
    end

    return nothing
end

# This kernel does not use any shared memory
check_flux_differencing_shared_memory(::FullSweepGlobal, semi) = nothing
