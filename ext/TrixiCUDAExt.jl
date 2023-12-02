# Package extension for some GPGPU API calls missing in KernelAbstractions

module TrixiCUDAExt

using Trixi
if isdefined(Base, :get_extension)
    using CUDA: CuArray
    using CUDA.CUDAKernels: CUDABackend
else
  # Until Julia v1.9 is the minimum required version for Trixi.jl, we still support Requires.jl
    using ..CUDA: CuArray
    using ..CUDA.CUDAKernels: CUDABackend
end

function Trixi.get_array_type(backend::CUDABackend)
    return CuArray
end

end