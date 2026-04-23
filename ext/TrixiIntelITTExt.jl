module TrixiIntelITTExt

# This extension provides tracing profiler integration for Intel VTune via IntelITT.jl.

using Trixi: CPU
import Trixi: profiling_range_active, profiling_range_start, profiling_range_end

import IntelITT

const domain = Ref{IntelITT.Domain}()
function __init__()
    domain[] = IntelITT.Domain("Trixi")
end

function profiling_range_active(::Union{Nothing, CPU})
    return IntelITT.isactive()
end

function profiling_range_start(::Union{Nothing, CPU}, label)
    task = IntelITT.Task(domain[], label)
    IntelITT.start(task)
    return task
end

function profiling_range_end(::Union{Nothing, CPU}, id)
    IntelITT.stop(id)
    return nothing
end

end
