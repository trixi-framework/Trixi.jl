# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Used by both `dg::DGSEM` and `dg::FDSBP`
function set_zero!(du, dg, cache)
    # du .= zero(eltype(du)) doesn't scale when using multiple threads.
    # See https://github.com/trixi-framework/Trixi.jl/pull/924 for a performance comparison.
    @threaded for element in eachelement(dg, cache)
        du[.., element] .= zero(eltype(du))
    end

    return nothing
end

# define types for parabolic solvers
include("solvers_parabolic.jl")

include("dg.jl")
include("dgmulti.jl")
end # @muladd
