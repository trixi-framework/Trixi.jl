# Package extension for using SparseDiffTools with Trixi for implicit solvers
module TrixiSparseDiffToolsExt

using Trixi

import Base: *, zero, one # For overloading with type `Real`

###############################################################################
### Hacks ###

# Required for setting up the Lobatto Legendre basis for abstract `Real` type
function Trixi.eps(::Type{Real}, RealT = Float64)
    return eps(RealT)
end

# There are several places in trixi where they do one(RealT) or zero(uEltype) where RealT or uEltype is Real
# this just returns an Int64 1 or 0 respectively. We don't want to use ints so we override this behavior
# Real(x::Real) = Float64(x)
Base.one(::Type{Real}) = 1.0
Base.zero(::Type{Real}) = 0.0

# Multiplying two Matrix{Real}s gives a Matrix{Any}.
# This causes problems when instantiating the Legendre basis, which calls
# `calc_{forward,reverse}_{upper, lower}` which in turn uses the matrix multiplication
# which is overloaded here in construction of the interpolation/projection operators 
# required for mortars.
function *(A::Matrix{Real}, B::Matrix{Real})::Matrix{Real}
    m, n = size(A, 1), size(B, 2)
    kA = size(A, 2)
    kB = size(B, 1)
    @assert kA == kB "Matrix dimensions must match for multiplication"
    
    C = Matrix{Real}(undef, m, n)
    for i in 1:m, j in 1:n
        #acc::Real = zero(promote_type(typeof(A[i,1]), typeof(B[1,j])))
        acc = zero(Real)
        for k in 1:kA
            acc += A[i,k] * B[k,j]
        end
        C[i,j] = acc
    end
    return C
end

end # module TrixiNLsolveExt