module TrixiNVTXExt

# This extension provides tracing profiler integration for NVIDIA Nsight Systems via NVTX.jl.

using NVTX
using CUDA: CUDABackend
import Trixi: profiling_range_active, profiling_range_start, profiling_range_end

# One can also use Nsight Systems and thus NVTX for CPU code

const domain = NVTX.Domain("Trixi")
const color = 0xff40e0d0 # turquoise

function profiling_range_active(::CUDABackend)
    return NVTX.isactive()
end

function profiling_range_start(::CUDABackend, label)
    return NVTX.range_start(NVTX.init!(domain); message = label, color = color)
end

function profiling_range_end(::CUDABackend, id)
    NVTX.range_end(id)
    return nothing
end

end # module
