# Package extension for adding CUDA-based features to Trixi.jl
module TrixiCUDAExt

import CUDA: CuArray, CuDeviceArray, KernelAdaptor
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

end
