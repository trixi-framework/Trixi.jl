# Package extension for adding CUDA-based features to Trixi.jl
module TrixiCUDAExt

import CUDA: CuArray, CuDeviceArray, KernelAdaptor, @device_override
import Trixi

function Trixi.storage_type(::Type{<:CuArray})
    return CuArray
end

function Trixi.unsafe_wrap_or_alloc(::KernelAdaptor, vec, size)
    return Trixi.unsafe_wrap_or_alloc(CuDeviceArray, vec, size)
end

function Trixi.unsafe_wrap_or_alloc(::Type{<:CuDeviceArray}, vec::CuDeviceArray, size)
    return reshape(vec, size)
end

@static if TRIXI._PREFERENCE_LOG == "log_Trixi_NaN"
    @device_override Trixi.log(x::Float64) = ccall("extern __nv_log", llvmcall, Cdouble,
                                                   (Cdouble,), x)
    @device_override Trixi.log(x::Float32) = ccall("extern __nv_logf", llvmcall, Cfloat,
                                                   (Cfloat,), x)
    # TODO: Trixi.log(x::Float16)
end

end
