# Package extension for adding Metal-based features to Trixi.jl
module TrixiMetalExt

import Metal: MtlArray, MtlDeviceArray, Adaptor
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

end
