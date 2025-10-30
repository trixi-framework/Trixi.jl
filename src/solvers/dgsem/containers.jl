# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

abstract type AbstractElementContainer <: AbstractContainer end
function nelements end

abstract type AbstractInterfaceContainer <: AbstractContainer end
function ninterfaces end
abstract type AbstractMPIInterfaceContainer <: AbstractContainer end
function nmpiinterfaces end

abstract type AbstractBoundaryContainer <: AbstractContainer end
function nboundaries end

abstract type AbstractMortarContainer <: AbstractContainer end
function nmortars end
abstract type AbstractMPIMortarContainer <: AbstractContainer end
function nmpimortars end

end # @muladd