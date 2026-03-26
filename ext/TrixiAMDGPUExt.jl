# Package extension for adding AMDGPU-based features to Trixi.jl
module TrixiAMDGPUExt

using AMDGPU: AMDGPU, ROCArray, ROCDeviceArray
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
    @device_override Trixi.log(x::Float64) = ccall("extern __ocml_log_f64", llvmcall, Cdouble,
                                                   (Cdouble,), x)
    @device_override Trixi.log(x::Float32) = ccall("extern __ocml_log_f32", llvmcall, Cfloat,
                                                   (Cfloat,), x)
    # TODO: Trixi.log(x::Float16)
end

end
