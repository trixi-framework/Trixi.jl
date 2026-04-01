module TrixiNVTXExt

using NVTX
using CUDA: CUDABackend
import Trixi: trixi_range_active, trixi_range_start, trixi_range_end

# One can also use Nsight Systems and thus NVTX for CPU code

const domain = NVTX.Domain("Trixi")
const color = 0xff40e0d0 # turquoise

function trixi_range_active(::CUDABackend)
    return NVTX.isactive()
end

function trixi_range_start(::CUDABackend, label)
    return NVTX.range_start(NVTX.init!(domain); message = label, color = color)
end

function trixi_range_end(::CUDABackend, id)
    NVTX.range_end(id)
    return nothing
end

end # module
