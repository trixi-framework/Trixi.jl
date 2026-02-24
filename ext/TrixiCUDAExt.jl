# Package extension for adding CUDA-based features to Trixi.jl
module TrixiCUDAExt

import CUDA: CuArray
import Trixi

function Trixi.storage_type(::Type{<:CuArray})
    return CuArray
end

function Trixi.unsafe_wrap_or_alloc(::CUDA.KernelAdaptor, vec, size)
    return Trixi.unsafe_wrap_or_alloc(CuDeviceArray, vec, size)
end

end
