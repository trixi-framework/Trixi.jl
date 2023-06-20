# Package extension for some GPGPU API calls missing in KernelAbstractions

module TrixiAMDGPUExt

using Trixi
if isdefined(Base, :get_extension)
  using Metal
  using Metal.MetalKernels
else
  # Until Julia v1.9 is the minimum required version for Trixi.jl, we still support Requires.jl
  using ..Metal
  using ..Metal.MetalKernels
end

function Trixi.get_array_type(backend::MetalBackend)
  return MtlArray
end

end