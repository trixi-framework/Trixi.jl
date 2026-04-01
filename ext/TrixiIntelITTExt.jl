module TrixiIntelITTExt

using Trixi: CPU
import Trixi: trixi_range_active, trixi_range_start, trixi_range_end

import IntelITT

const domain = Ref{IntelITT.Domain}()
function __init__()
    domain[] = IntelITT.Domain("Trixi")
end

function trixi_range_active(::Union{Nothing, CPU})
    return IntelITT.isactive()
end

function trixi_range_start(::Union{Nothing, CPU}, label)
    task = IntelITT.Task(domain[], label)
    IntelITT.start(task)
    return task
end

function trixi_range_end(::Union{Nothing, CPU}, id)
    IntelITT.stop(id)
    return nothing
end

end # module
