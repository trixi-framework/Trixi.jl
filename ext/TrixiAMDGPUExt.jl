# Package extension for adding AMDGPU-based features to Trixi.jl
module TrixiAMDGPUExt

import AMDGPU: ROCArray
import Trixi

function Trixi.storage_type(::Type{<:ROCArray})
    return ROCArray
end

end
