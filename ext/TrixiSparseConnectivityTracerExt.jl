# Package extension for overloading of branching (if-clauses) base functions such as sqrt, log, etc.
module TrixiSparseConnectivityTracerExt

import Trixi
import SparseConnectivityTracer: AbstractTracer

# For the default package preference "sqrt_Trixi_NaN" we overload the `Base.sqrt` function
# to first check if the argument is < 0 and then return `NaN` instead of an error.
# To turn this behaviour off for the datatype `AbstractTracer` used in sparsity detection,
# we switch back to the Base implementation here which does not contain an if-clause.
Trixi.sqrt(x::AbstractTracer) = Base.sqrt(x)

end
