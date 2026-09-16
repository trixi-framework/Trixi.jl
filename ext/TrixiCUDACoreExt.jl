# Package extension for adding CUDA-based features to Trixi.jl
module TrixiCUDACoreExt

using CUDACore: CUDACore, CuArray, CuDeviceArray, CUDABackend, KernelAdaptor,
                @device_override
import Trixi

function Trixi.storage_type(::Type{<:CuArray})
    return CuArray
end

function Trixi.unsafe_wrap_or_alloc(::KernelAdaptor, vec, size)
    return Trixi.unsafe_wrap_or_alloc(CuDeviceArray, vec, size)
end

function Trixi.unsafe_wrap_or_alloc(::Type{<:CuDeviceArray}, vec::CuDeviceArray, size)
    return reshape(vec, size)
end

@static if Trixi._PREFERENCE_LOG == "log_Trixi_NaN"
    @device_override Trixi.log(x::Float64) = ccall("extern __nv_log", llvmcall, Cdouble,
                                                   (Cdouble,), x)
    @device_override Trixi.log(x::Float32) = ccall("extern __nv_logf", llvmcall, Cfloat,
                                                   (Cfloat,), x)
    # TODO: Trixi.log(x::Float16)
end

function Trixi.trixi_backend_info!(setup, ::CUDABackend)
    push!(setup, "Backend" => "KernelAbstractions CUDA")
    # Reimplementation of CUDA.versioninfo() to fit with Trixi's summary box format
    # see https://github.com/JuliaGPU/CUDA.jl/blob/9f56ee20afef4a770a028066d2fa9e7825d258da/src/utilities.jl#L41
    if !CUDACore.functional()
        push!(setup, "CUDA" => "CUDA not functional")
        return nothing
    end

    push!(setup, "CUDA toolchain:" => "")
    push!(setup, "  - runtime" => string(CUDACore.runtime_version()))
    push!(setup,
          "  - toolkit" => CUDACore.local_toolkit ? "local installation" :
                           "artifact installation")
    push!(setup, "  - driver" => string(CUDACore.driver_version()))
    push!(setup, "  - compiler" => string(CUDACore.compiler_version()))

    # Skip CUDA libraries
    # Skip Julia packages
    # Skip Toolchain
    # Skip Environment
    # Skip Preferences

    devs = CUDACore.devices()
    if isempty(devs)
        push!(setup, "CUDA devices" => "none")
    else
        push!(setup, "CUDA devices" => "")
    end
    for (i, dev) in enumerate(devs)
        function query_cuda()
            str = CUDACore.name(dev)
            cap = CUDACore.capability(dev)
            mem = CUDACore.device!(dev) do
                (free = CUDACore.free_memory(), total = CUDACore.total_memory())
            end
            (; str, cap, mem)
        end

        str, cap, mem = query_cuda()

        push!(setup,
              "  $i (model)" => "$str (sm_$(cap.major)$(cap.minor))")
        push!(setup,
              "  $i (memory)" => "$(Base.format_bytes(mem.free)) / $(Base.format_bytes(mem.total)) available")
    end
end

function Trixi.trixi_device_memory_use(::CUDABackend)
    info = CUDACore.MemoryInfo()
    used_bytes = info.total_bytes - info.free_bytes
    return used_bytes
end

end
