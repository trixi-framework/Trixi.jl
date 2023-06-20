# Package extension for some GPGPU API calls missing in KernelAbstractions

module TrixiAMDGPUExt

using Trixi
if isdefined(Base, :get_extension)
    using AMDGPU.ROCKernels
else
    # Until Julia v1.9 is the minimum required version for Trixi.jl, we still support Requires.jl
    using ..AMDGPU.ROCKernels
end

function Trixi.get_array_type(backend::ROCBackend)
  return ROCArray
end

end