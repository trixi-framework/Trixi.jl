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
struct ModalFilter{RealT <: Real, Cons2Filter, Filter2Cons, Cache}
    cons2filter::Cons2Filter
    filter2cons::Filter2Cons
    filter_matrix::Matrix{RealT}
    cache::Cache
end

function ModalFilter(dg, equations; filter_coefficients = nothing,
                     polydeg_cutoff = nothing,
                     cons2filter = cons2cons, filter2cons = cons2cons)
    # Sanity checks for the input arguments
    if filter_coefficients !== nothing && polydeg_cutoff !== nothing
        throw(ArgumentError("Only one of `filter_coefficients` and `polydeg_cutoff` can be specified."))
    elseif filter_coefficients == nothing && polydeg_cutoff == nothing
        throw(ArgumentError("Either `filter_coefficients` or `polydeg_cutoff` must be specified."))
    end

    # Compute the filter matrix
    if !isnothing(filter_coefficients)
        filter_matrix_ = calc_modal_filter_matrix(dg.basis.nodes, filter_coefficients)
    else
        filter_matrix_ = calc_modal_filter_matrix(dg.basis.nodes, polydeg_cutoff)
    end
    RealT = real(dg)
    filter_matrix = Matrix{RealT}(filter_matrix_)

    # Create cache for temporary arrays
    nnodes_ = nnodes(dg)
    nvars = nvariables(equations)
    uEltype = RealT # TODO: This should probably be the element type of the solution

    A3 = Array{uEltype, 3}
    u_element_filtered_threaded = A3[A3(undef, nvars, nnodes_, nnodes_)
                                     for _ in 1:Threads.nthreads()]
    tmp_threaded = A3[A3(undef, nvars, nnodes_, nnodes_) for _ in 1:Threads.nthreads()]
    cache = (; u_element_filtered_threaded, tmp_threaded)

    ModalFilter{RealT,
                typeof(cons2filter),
                typeof(filter2cons),
                typeof(cache)}(cons2filter, filter2cons, filter_matrix, cache)
end

# Main function that applies the actual, mesh- and solver-specific filter
function (modal_filter::ModalFilter)(u_ode, semi::AbstractSemidiscretization)
    u = wrap_array(u_ode, semi)
    @unpack cons2filter, filter2cons, filter_matrix, cache = modal_filter
    @unpack u_element_filtered_threaded, tmp_threaded = cache

    @trixi_timeit timer() "modal filter" begin
        # println("Applying modal filter")
        apply_modal_filter!(u, cons2filter, filter2cons, filter_matrix,
                            mesh_equations_solver_cache(semi)...,
                            u_element_filtered_threaded, tmp_threaded)
    end

    return nothing
end

# This version is called as the stage limiter version of the filter
function (modal_filter::ModalFilter)(u_ode, integrator,
                                     semi::AbstractSemidiscretization, t)
    modal_filter(u_ode, semi)
end

include("modal_filter_dg2d.jl")
end # @muladd
