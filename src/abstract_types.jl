
# abstract supertype of specific semidiscretizations such as
# - SemidiscretizationHyperbolic for hyperbolic conservation laws
# - SemidiscretizationEulerGravity for Euler with self-gravity
abstract type AbstractSemidiscretization end


# abstract supertype of specific equations such as the compressible Euler equations
abstract type AbstractEquations{NDIMS, NVARS} end


# TODO: Taal decide, which abstract types shall be defined here?
