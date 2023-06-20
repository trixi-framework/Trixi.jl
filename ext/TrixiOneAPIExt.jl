# Package extension for some GPGPU API calls missing in KernelAbstractions

module TrixiOneAPIExt

using Trixi
if isdefined(Base, :get_extension)
  using oneAPI
  using oneAPI.oneAPIKernels
else
  # Until Julia v1.9 is the minimum required version for Trixi.jl, we still support Requires.jl
  using ..oneAPI
  using ..oneAPI.oneAPIKernels
end

function Trixi.get_array_type(backend::oneAPIBackend)
  return oneArray
end

end