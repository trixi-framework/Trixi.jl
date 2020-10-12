
# abstract supertype of specific semidiscretizations such as
# - SemidiscretizationHyperbolic for hyperbolic conservation laws
# - SemidiscretizationEulerGravity for Euler with self-gravity
abstract type AbstractSemidiscretization end


# abstract supertype of specific equations such as the compressible Euler equations
abstract type AbstractEquations{NDIMS, NVARS} end


# abstract supertype of specific SBP bases such as a Lobatto-Legendre nodal basis
abstract type AbstractBasisSBP{RealT<:Real} end


# abstract supertype of mortar methods, e.g. using L² projections
abstract type AbstractMortar{RealT<:Real} end

# abstract supertype of mortar methods using L² projection
# which will be specialized for different SBP bases
abstract type MortarL2{RealT<:Real} <: AbstractMortar{RealT} end


# abstract supertype of functionality related to the analysis of
# numerical solutions, e.g. the calculation of errors
abstract type SolutionAnalyzer{RealT<:Real} end


# abstract supertype of grid-transfer methods used for AMR,
# e.g. refinement and coarsening based on L² projections
abstract type AdaptorAMR{RealT<:Real} end

# abstract supertype of AMR grid-transfer operations using L² projections
# which will be specialized for different SBP bases
abstract type AdaptorL2{RealT<:Real} <: AdaptorAMR{RealT} end


# TODO: Taal decide, which abstract types shall be defined here?
