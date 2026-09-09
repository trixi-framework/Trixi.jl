@inline function get_node_turbo(turbo_local, ::Val{NAUX},
                                indices...) where {NAUX}
    return ntuple(v -> (@inbounds turbo_local[v, indices...]), Val(NAUX))
end

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
# 48 KiB per workgroup on NVIDIA GPUs and 64 KiB on AMD GPUs.
function check_flux_differencing_shared_memory(kernel::Union{HalfSweep, FullSweep}, semi)
    dg = semi.solver
    shared_memory = nvariables(semi.equations) * nnodes(dg)^3 * sizeof(real(dg))

    if shared_memory > 48 * 1024
        @warn "The shared memory required by the selected flux differencing kernel may exceed the limit of the GPU. In case, consider using `flux_differencing_kernel = FullSweepGlobal()`." kernel nvariables=nvariables(semi.equations) polydeg=polydeg(dg) shared_memory=Base.format_bytes(shared_memory)
    end

    return nothing
end

# This kernel does not use any shared memory
check_flux_differencing_shared_memory(::FullSweepGlobal, semi) = nothing
