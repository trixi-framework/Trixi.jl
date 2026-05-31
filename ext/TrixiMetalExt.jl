# Package extension for adding Metal-based features to Trixi.jl
module TrixiMetalExt

using Metal: Metal, MtlArray, MtlDeviceArray, MetalBackend, Adaptor, @device_override
import Trixi

function Trixi.storage_type(::Type{<:MtlArray})
    return MtlArray
end

function Trixi.unsafe_wrap_or_alloc(::Adaptor, vec, size)
    return Trixi.unsafe_wrap_or_alloc(MtlDeviceArray, vec, size)
end

function Trixi.unsafe_wrap_or_alloc(::Type{<:MtlDeviceArray}, vec::MtlDeviceArray, size)
    return reshape(vec, size)
end

function Trixi.unsafe_wrap_or_alloc(::Type{<:MtlArray}, vec::MtlArray, size)
    return reshape(vec, size)
end

@inline function Trixi.wrap_array(u_ode::MtlArray{<:Any, 1}, mesh::Trixi.AbstractMesh,
                                  equations, dg::Trixi.DGSEM, cache)
    @boundscheck begin
        @assert length(u_ode) ==
                Trixi.nvariables(equations) * Trixi.nnodes(dg)^Trixi.ndims(mesh) *
                Trixi.nelements(dg, cache)
    end
    return reshape(u_ode,
                   (Trixi.nvariables(equations),
                    ntuple(_ -> Trixi.nnodes(dg), Trixi.ndims(mesh))...,
                    Trixi.nelements(dg, cache)))
end

@static if Trixi._PREFERENCE_LOG == "log_Trixi_NaN"
    # Metal only supports single and half-precision floating-point types
    @device_override Trixi.log(x::Float32) = ccall("extern air.log.f32", llvmcall, Cfloat,
                                                   (Cfloat,), x)
    @device_override Trixi.log(x::Float16) = ccall("extern air.log.f16", llvmcall, Float16,
                                                   (Float16,), x)
end

function Trixi.trixi_backend_info!(setup, ::MetalBackend)
    push!(setup, "Backend" => "KernelAbstractions Metal")
    # Reimplementation of Metal.versioninfo() to fit with Trixi's summary box format
    # see https://github.com/JuliaGPU/Metal.jl/blob/main/src/utilities.jl
    if !Metal.functional()
        push!(setup, "Metal" => "Metal not functional")
        return nothing
    end

    push!(setup, "Metal toolchain:" => "")
    push!(setup, "  - macOS" => string(Metal.macos_version()))
    push!(setup, "  - Darwin" => string(Metal.darwin_version()))

    devs = Metal.MTL.devices()
    if isempty(devs)
        push!(setup, "Metal devices" => "none")
    else
        push!(setup, "Metal devices:" => "")
    end
    for (i, dev) in enumerate(devs)
        cores = Metal.num_gpu_cores()
        allocated = Base.format_bytes(dev.currentAllocatedSize)
        recommended = Base.format_bytes(dev.recommendedMaxWorkingSetSize)
        push!(setup,
              "  $i" => "$(dev.name) ($cores GPU cores, $allocated / $recommended available)")
    end
    return nothing
end

function Trixi.trixi_device_memory_use(::MetalBackend)
    return Metal.device().currentAllocatedSize
end

end
