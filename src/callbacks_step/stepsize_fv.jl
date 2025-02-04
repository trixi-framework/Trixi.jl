# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function max_dt(u, t, mesh,
                constant_speed::False, equations, solver::FV, cache)
    dt = typemax(eltype(u))

    for element in eachelement(solver, cache)
        u_node = get_node_vars(u, equations, solver, element)
        lambda1, lambda2 = max_abs_speeds(u_node, equations)
        dx = cache.elements.dx[element]
        dt = min(dt, dx / (lambda1 + lambda2))
    end

    if mpi_isparallel()
        dt = MPI.Allreduce!(Ref(dt), min, mpi_comm())[]
    end

    return dt
end

function max_dt(u, t, mesh,
                constant_speed::True, equations, solver::FV, cache)
    dx_min = typemax(eltype(u))

    max_lambda1, max_lambda2 = max_abs_speeds(equations)
    for element in eachelement(solver, cache)
        dx = cache.elements.dx[element]
        dx_min = min(dx_min, dx)
    end
    dt = dx_min / (max_lambda1 + max_lambda2)

    if mpi_isparallel()
        dt = MPI.Allreduce!(Ref(dt), min, mpi_comm())[]
    end

    return dt
end
end # @muladd
