# Package extension for overloading of branching (if-clauses) base functions such as sqrt, log, etc.
module TrixiSparseConnectivityTracerExt

import Trixi
import SparseConnectivityTracer: AbstractTracer

# For the default package preference "sqrt_Trixi_NaN" we overload the `Base.sqrt` function
# to first check if the argument is < 0 and then return `NaN` instead of an error.
# To turn this behaviour off for the datatype `AbstractTracer` used in sparsity detection,
# we switch back to the Base implementation here which does not contain an if-clause.
Trixi.sqrt(x::AbstractTracer) = Base.sqrt(x)

# if-clause free (i.e., non-optimized) implementations of some helper functions
# that compute specialized mean values used in advanced flux functions

@inline function Trixi.ln_mean(x::AbstractTracer, y::AbstractTracer)
    return (y - x) / log(y / x)
end

@inline function Trixi.inv_ln_mean(x::AbstractTracer, y::AbstractTracer)
    return log(y / x) / (y - x)
end

@inline function Trixi.stolarsky_mean(x::AbstractTracer, y::AbstractTracer, gamma::Real)
    yg = exp((gamma - 1) * log(y)) # equivalent to y^(gamma - 1) but faster for non-integers
    xg = exp((gamma - 1) * log(x)) # equivalent to x^(gamma - 1) but faster for non-integers
    return (gamma - 1) * (yg * y - xg * x) / (gamma * (yg - xg))
end

end
