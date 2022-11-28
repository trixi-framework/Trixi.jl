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


# 2D containers
init_mortars(cell_ids, mesh, elements, mortar) = nothing


create_cache(mesh, equations, mortar, uEltype) = NamedTuple()
nmortars(mortar) = 0


function prolong2mortars!(cache, u, mesh, equations, mortar,
        surface_integral, dg::DG)
@assert isempty(eachmortar(dg, cache))
end


function calc_mortar_flux!(surface_flux_values, mesh,
         nonconservative_terms, equations,
         mortar,
         surface_integral, dg::DG, cache)
@assert isempty(eachmortar(dg, cache))
end


SolutionAnalyzer(D::AbstractDerivativeOperator) = D

end
