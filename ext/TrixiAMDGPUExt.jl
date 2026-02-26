# Package extension for adding AMDGPU-based features to Trixi.jl
module TrixiAMDGPUExt

import AMDGPU: ROCArray, ROCDeviceArray
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

end
