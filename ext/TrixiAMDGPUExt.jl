# Package extension for adding AMDGPU-based features to Trixi.jl
module TrixiAMDGPUExt

using AMDGPU: AMDGPU, ROCBackend, ROCArray, ROCDeviceArray
import AMDGPU.Device: @device_override
import AMDGPU.Runtime: Adaptor
import Trixi

function Trixi.storage_type(::Type{<:ROCArray})
    return ROCArray
end

function Trixi.unsafe_wrap_or_alloc(::Adaptor, vec, size)
    return Trixi.unsafe_wrap_or_alloc(ROCDeviceArray, vec, size)
end

function Trixi.unsafe_wrap_or_alloc(::Type{<:ROCDeviceArray}, vec::ROCDeviceArray, size)
    return reshape(vec, size)
end

@static if Trixi._PREFERENCE_LOG == "log_Trixi_NaN"
    @device_override Trixi.log(x::Float64) = ccall("extern __ocml_log_f64", llvmcall,
                                                   Cdouble,
                                                   (Cdouble,), x)
    @device_override Trixi.log(x::Float32) = ccall("extern __ocml_log_f32", llvmcall,
                                                   Cfloat,
                                                   (Cfloat,), x)
    # TODO: Trixi.log(x::Float16)
end

function Trixi.trixi_backend_info!(setup, ::ROCBackend)
    push!(setup, "Backend" => "KernelAbstractions AMDGPU")
    # Reimplementation of AMDGPU.versioninfo() to fit with Trixi's summary box format
    # see https://github.com/JuliaGPU/AMDGPU.jl/blob/e62d814e88a67ea58005802ee83632a68ef629ba/src/utils.jl
    if !AMDGPU.functional()
        push!(setup, "AMDGPU" => "AMDGPU not functional")
        return nothing
    end

    devs = AMDGPU.devices()
    if isempty(devs)
        push!(setup, "AMDGPU devices" => "none")
    else
        push!(setup, "AMDGPU devices:" => "")
    end
    for (i, dev) in enumerate(devs)
        push!(setup, "  - device $i" => string(dev))
    end
    return nothing
end

function Trixi.trixi_device_memory_use(backend::ROCBackend)
    # This is raw memory used not the pool.
    return AMDGPU.Mem.used()
end

end
