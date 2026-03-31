# Package extension for adding CUDA-based features to Trixi.jl
module TrixiCUDAExt

using CUDA: CUDA, CuArray, CuDeviceArray, CUDABackend, KernelAdaptor, @device_override
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
    if !CUDA.functional()
        push!(setup, "CUDA" => "CUDA not functional")
        return nothing
    end

    push!(setup, "CUDA toolchain:" => "")
    push!(setup, "  - runtime" => string(CUDA.runtime_version()))
    push!(setup,
          "  - toolkit" => CUDA.local_toolkit ? "local installation" :
                           "artifact installation")

    if CUDA.has_nvml()
        driver_str = string(CUDA.NVML.driver_version())
    else
        driver_str = "unknown"
    end
    driver_str *= " for $(CUDA.driver_version())"
    push!(setup, "  - driver" => driver_str)
    push!(setup, "  - compiler" => string(CUDA.compiler_version()))

    # Skip CUDA libraries
    # Skip Julia packages
    # Skip Toolchain
    # Skip Environment
    # Skip Preferences

    devs = CUDA.devices()
    if isempty(devs)
        push!(setup, "CUDA devices" => "none")
    else
        push!(setup, "CUDA devices:" => "")
    end
    for (i, dev) in enumerate(devs)
        function query_nvml()
            mig = CUDA.uuid(dev) != CUDA.parent_uuid(dev)
            nvml_gpu = CUDA.NVML.Device(CUDA.parent_uuid(dev))
            nvml_dev = CUDA.NVML.Device(CUDA.uuid(dev); mig)

            str = CUDA.NVML.name(nvml_dev)
            cap = CUDA.NVML.compute_capability(nvml_gpu)
            mem = CUDA.NVML.memory_info(nvml_dev)

            (; str, cap, mem)
        end

        function query_cuda()
            str = CUDA.name(dev)
            cap = CUDA.capability(dev)
            mem = CUDA.device!(dev) do
                # this requires a device context, so we prefer NVML
                (free = CUDA.free_memory(), total = CUDA.total_memory())
            end
            (; str, cap, mem)
        end

        str, cap, mem = if CUDA.has_nvml()
            try
                query_nvml()
            catch err
                if !isa(err, CUDA.NVML.NVMLError) ||
                   !in(err.code,
                       [CUDA.NVML.ERROR_NOT_SUPPORTED, CUDA.NVML.ERROR_NO_PERMISSION])
                    rethrow()
                end
                query_cuda()
            end
        else
            query_cuda()
        end
        push!(setup,
              "  - device $i" => "$str (sm_$(cap.major)$(cap.minor), $(Base.format_bytes(mem.free)) / $(Base.format_bytes(mem.total)) available)")
    end
end

end
