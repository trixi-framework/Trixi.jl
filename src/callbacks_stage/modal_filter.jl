# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    ModalFilter(; polydeg_cutoff::Integer,
                  cons2filter = cons2cons,
                  filter2cons = cons2cons)

A modal filter that will cut-off all modes of the solution `u` higher than `polydeg_cutoff`.
The filtering will be done in the filter variables, for which a forward (`cons2filter`) and
reverse (`filter2cons`) transformation is required. By default, the identity transformation
(`cons2cons`) is used.
"""
struct ModalFilter{RealT, Cons2Filter, Filter2Cons}
    cons2filter::Cons2Filter
    filter2cons::Filter2Cons
    filter_matrix::Matrix{RealT}
end

function ModalFilter(dg; polydeg_cutoff::Integer, cons2filter = cons2cons,
                         filter2cons = cons2cons)
    RealT = real(dg)
    if dg.basis isa LobattoLegendreBasis
        nodes_cutoff, _ = gauss_lobatto_nodes_weights(polydeg_cutoff)
    elseif dg.basis isa GaussLegendreBasis
        nodes_cutoff, _ = gauss_nodes_weights(polydeg_cutoff)
    else
        throw(ArgumentError("Only LobattoLegendreBasis and GaussLegendreBasis are supported."))
    end
    _, filter_matrix_ = calc_projection_matrix(dg.basis.nodes, nodes_cutoff)
    filter_matrix = Matrix{RealT}(filter_matrix_)

    modal_filter = ModalFilter{RealT,
                               typeof(cons2filter),
                               typeof(filter2cons)}(cons2filter, filter2cons, filter_matrix)
end

# Main function that applies the actual, mesh- and solver-specific filter
function (modal_filter::ModalFilter)(u_ode, semi::AbstractSemidiscretization)
    u = wrap_array(u_ode, semi)
    @unpack cons2filter, filter2cons, filter_matrix = modal_filter

    @trixi_timeit timer() "modal filter" begin
        # println("Applying modal filter")
        apply_modal_filter!(u, u, cons2filter, filter2cons, filter_matrix,
                            mesh_equations_solver_cache(semi)...)  
    end

    return nothing
end

# This version is called as the stage limiter version of the filter
function (modal_filter::ModalFilter)(u_ode, integrator, semi::AbstractSemidiscretization, t)
    modal_filter(u_ode, semi)
end

include("modal_filter_dg2d.jl")
end # @muladd
