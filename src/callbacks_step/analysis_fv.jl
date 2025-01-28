# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function calc_error_norms(func, u, t, analyzer,
                          mesh::T8codeMesh, equations,
                          initial_condition, solver::FV, cache, cache_analysis)
    # Set up data structures
    l2_error = zero(func(get_node_vars(u, equations, solver, 1), equations))
    linf_error = copy(l2_error)
    total_volume = zero(real(mesh))

    # Iterate over all elements for error calculations
    for element in eachelement(solver, cache)
        midpoint = get_node_coords(cache.elements.midpoint, equations, solver, element)
        volume = cache.elements.volume[element]

        u_exact = initial_condition(midpoint, t, equations)
        diff = func(u_exact, equations) -
               func(get_node_vars(u, equations, solver, element), equations)
        l2_error += diff .^ 2 * volume
        linf_error = @. max(linf_error, abs(diff))
        total_volume += volume
    end

    # Accumulate local results on root process
    if mpi_isparallel()
        global_l2_error = Vector(l2_error)
        global_linf_error = Vector(linf_error)
        MPI.Reduce!(global_l2_error, +, mpi_root(), mpi_comm())
        MPI.Reduce!(global_linf_error, max, mpi_root(), mpi_comm())
        total_volume_ = MPI.Reduce(total_volume, +, mpi_root(), mpi_comm())
        if mpi_isroot()
            l2_error = convert(typeof(l2_error), global_l2_error)
            linf_error = convert(typeof(linf_error), global_linf_error)
            # For L2 error, divide by total volume
            l2_error = @. sqrt(l2_error / total_volume_)
        else
            l2_error = convert(typeof(l2_error), NaN * global_l2_error)
            linf_error = convert(typeof(linf_error), NaN * global_linf_error)
        end
    else
        # For L2 error, divide by total volume
        l2_error = @. sqrt(l2_error / total_volume)
    end

    return l2_error, linf_error
end

function integrate_via_indices(func::Func, u,
                               mesh::T8codeMesh, equations,
                               solver::FV, cache, args...;
                               normalize = true) where {Func}
    # Initialize integral with zeros of the right shape
    integral = zero(func(u, 1, equations, solver, args...))
    total_volume = zero(real(mesh))

    # Use quadrature to numerically integrate over entire domain
    for element in eachelement(solver, cache)
        volume = cache.elements.volume[element]
        integral += volume * func(u, element, equations, solver, args...)
        total_volume += volume
    end

    if mpi_isparallel()
        global_integral = MPI.Reduce!(Ref(integral), +, mpi_root(), mpi_comm())
        total_volume_ = MPI.Reduce(total_volume, +, mpi_root(), mpi_comm())
        if mpi_isroot()
            integral = convert(typeof(integral), global_integral[])
            # Normalize with total volume
            if normalize
                integral = integral / total_volume_
            end
        else
            integral = convert(typeof(integral), NaN * integral)
            total_volume_ = total_volume # non-root processes receive nothing from reduce -> overwrite
        end
    else
        # Normalize with total volume
        if normalize
            integral = integral / total_volume
        end
    end

    return integral
end

function integrate(func::Func, u,
                   mesh,
                   equations, solver::FV, cache; normalize = true) where {Func}
    integrate_via_indices(u, mesh, equations, solver, cache;
                          normalize = normalize) do u, element, equations, solver
        u_local = get_node_vars(u, equations, solver, element)
        return func(u_local, equations)
    end
end

function analyze(::typeof(entropy_timederivative), du, u, t,
                 mesh::T8codeMesh,
                 equations, solver::FV, cache)
    # Calculate ∫(∂S/∂u ⋅ ∂u/∂t)dΩ
    integrate_via_indices(u, mesh, equations, solver, cache,
                          du) do u, element, equations, solver, du
        u_node = get_node_vars(u, equations, solver, element)
        du_node = get_node_vars(du, equations, solver, element)
        dot(cons2entropy(u_node, equations), du_node)
    end
end
end # @muladd
