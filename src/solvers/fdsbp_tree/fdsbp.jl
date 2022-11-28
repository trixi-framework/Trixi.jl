# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


# For dispatch
const FDSBP = DG{Basis} where {Basis<:AbstractDerivativeOperator}


# General interface methods for SummationByPartsOperators.jl and Trixi.jl
nnodes(D::AbstractDerivativeOperator) = size(D, 1)
eachnode(D::AbstractDerivativeOperator) = Base.OneTo(nnodes(D))
get_nodes(D::AbstractDerivativeOperator) = grid(D)

# TODO: This is hack to enable the FDSBP solver to use the
#       `SaveSolutionCallback`.
polydeg(D::AbstractDerivativeOperator) = size(D, 1) - 1
polydeg(fdsbp::FDSBP) = polydeg(fdsbp.basis)


# TODO: FD. No mortars supported at the moment
init_mortars(cell_ids, mesh, elements, mortar::Nothing) = nothing
create_cache(mesh, equations, mortar::Nothing, uEltype) = NamedTuple()
nmortars(mortar::Nothing) = 0

function prolong2mortars!(cache, u, mesh, equations, mortar::Nothing,
                          surface_integral, dg::DG)
  @assert isempty(eachmortar(dg, cache))
end

function calc_mortar_flux!(surface_flux_values, mesh,
                           nonconservative_terms, equations,
                           mortar::Nothing,
                           surface_integral, dg::DG, cache)
  @assert isempty(eachmortar(dg, cache))
end


# We do not use a specialized setup to analyze solutions
SolutionAnalyzer(D::AbstractDerivativeOperator) = D


# dimension-specific implementations
include("fdsbp_1d.jl")
include("fdsbp_2d.jl")
include("fdsbp_3d.jl")


end
