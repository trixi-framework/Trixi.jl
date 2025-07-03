# Package extension for adding AMDGPU-based features to Trixi.jl
module TrixiAMDGPUExt

import AMDGPU: ROCArray
import Trixi

function Trixi.storage_type(::Type{<:ROCArray})
    return ROCArray
end

function Trixi.unsafe_wrap_or_alloc(to::Type{<:ROCArray}, vector, size)
    if length(vector) == 0
        return similar(vector, size)
    else
        return unsafe_wrap(to, pointer(vector), size, lock=false)
    end
end

end
