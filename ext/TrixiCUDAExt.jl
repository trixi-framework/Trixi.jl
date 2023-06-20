# Package extension for some GPGPU API calls missing in KernelAbstractions

module TrixiCUDAExt

using Trixi
if isdefined(Base, :get_extension)
    using CUDA.CUDAKernels
else
    # Until Julia v1.9 is the minimum required version for Trixi.jl, we still support Requires.jl
    using ..CUDA.CUDAKernels
end

function Trixi.get_array_type(backend::CUDABackend)
  return CuArray
end

end