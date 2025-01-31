# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# this method is used when the indicator is constructed as for AMR
function create_cache(typ::Type{IndicatorMax}, mesh, equations,
                      solver::FV, cache)
    alpha = Vector{real(mesh)}()

    indicator_threaded = [zero(real(mesh)) for _ in 1:Threads.nthreads()]

    return (; alpha, indicator_threaded)
end

function (indicator_max::IndicatorMax)(u, mesh, equations, solver::FV, cache;
                                       kwargs...)
    @unpack alpha, indicator_threaded = indicator_max.cache
    resize!(alpha, nelements(solver, cache))
    indicator_variable = indicator_max.variable

    @threaded for element in eachelement(solver, cache)
        indicator = indicator_threaded[Threads.threadid()]

        u_local = get_node_vars(u, equations, solver, element)
        indicator = indicator_variable(u_local, equations)

        alpha[element] = indicator
    end

    return alpha
end
end # @muladd
