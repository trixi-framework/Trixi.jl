# Package extension for some GPGPU API calls missing in KernelAbstractions

module TrixiAMDGPUExt

using Trixi
if isdefined(Base, :get_extension)
    using AMDGPU: ROCArray
    using AMDGPU.ROCKernels: ROCBackend
else
  # Until Julia v1.9 is the minimum required version for Trixi.jl, we still support Requires.jl
    using ..AMDGPU: ROCArray
    using ..AMDGPU.ROCKernels: ROCBackend
end

function Trixi.get_array_type(backend::ROCBackend)
    return ROCArray
end

end