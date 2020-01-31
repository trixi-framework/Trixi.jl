module Jul1dge

const ndim = 1
export ndim

include("equation/equation.jl")
include("mesh/mesh.jl")
include("dg/dg.jl")
# include("io/io.jl")
include("timedisc/timedisc.jl")

end
