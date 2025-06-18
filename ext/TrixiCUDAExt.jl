# Package extension for adding CUDA-based features to Trixi.jl
module TrixiCUDAExt

import CUDA: CuArray
import Trixi

function Trixi.storage_type(::Type{<:CuArray})
    return CuArray
end

end
