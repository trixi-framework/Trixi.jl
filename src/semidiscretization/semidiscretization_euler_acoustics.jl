# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    SemidiscretizationEulerAcoustics(semi_acoustics::SemiAcoustics, semi_euler::SemiEuler;
                                     source_region=x->true, weights=x->1.0)

!!! warning "Experimental code"
    This semidiscretization is experimental and may change in any future release.

Construct a semidiscretization of the acoustic perturbation equations that is coupled with
the compressible Euler equations via source terms for the perturbed velocity. Both
semidiscretizations have to use the same mesh and solvers with a shared basis. The coupling region
is described by a function `source_region` that maps the coordinates of a single node to `true` or
`false` depending on whether the point lies within the coupling region or not. A weighting function
`weights` that maps coordinates to weights is applied to the acoustic source terms.
Note that this semidiscretization should be used in conjunction with
[`EulerAcousticsCouplingCallback`](@ref) and only works in two dimensions.
"""
struct SemidiscretizationEulerAcoustics{SemiAcoustics, SemiEuler, Cache} <:
       AbstractSemidiscretization
    semi_acoustics::SemiAcoustics
    semi_euler::SemiEuler
    performance_counter::PerformanceCounter
    cache::Cache

    function SemidiscretizationEulerAcoustics{SemiAcoustics, SemiEuler, Cache}(semi_acoustics,
                                                                               semi_euler,
                                                                               cache) where {
                                                                                             SemiAcoustics,
                                                                                             SemiEuler,
                                                                                             Cache
                                                                                             }

        # Currently both semidiscretizations need to use a shared mesh
        @assert semi_acoustics.mesh == semi_euler.mesh

        # Check if both solvers use the same polynomial basis
        @assert semi_acoustics.solver.basis == semi_euler.solver.basis

        performance_counter = PerformanceCounter()
        new(semi_acoustics, semi_euler, performance_counter, cache)
    end
end

function SemidiscretizationEulerAcoustics(semi_acoustics::SemiAcoustics,
                                          semi_euler::SemiEuler;
                                          source_region = x -> true,
                                          weights = x -> 1.0) where
         {Mesh,
          SemiAcoustics <:
          SemidiscretizationHyperbolic{Mesh, <:AbstractAcousticPerturbationEquations},
          SemiEuler <:
          SemidiscretizationHyperbolic{Mesh, <:AbstractCompressibleEulerEquations}}
    cache = create_cache(SemidiscretizationEulerAcoustics, source_region, weights,
                         mesh_equations_solver_cache(semi_acoustics)...)

    return SemidiscretizationEulerAcoustics{typeof(semi_acoustics), typeof(semi_euler),
                                            typeof(cache)}(semi_acoustics, semi_euler,
                                                           cache)
end

function create_cache(::Type{SemidiscretizationEulerAcoustics}, source_region, weights,
                      mesh, equations::AcousticPerturbationEquations2D, dg::DGSEM,
                      cache)
    coupled_element_ids = get_coupled_element_ids(source_region, equations, dg, cache)

    acoustic_source_terms = zeros(eltype(cache.elements),
                                  (ndims(equations), nnodes(dg), nnodes(dg),
                                   length(coupled_element_ids)))

    acoustic_source_weights = precompute_weights(source_region, weights,
                                                 coupled_element_ids,
                                                 equations, dg, cache)

    return (; acoustic_source_terms, acoustic_source_weights, coupled_element_ids)
end

function get_coupled_element_ids(source_region, equations, dg::DGSEM, cache)
    coupled_element_ids = Vector{Int}(undef, 0)

    for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            x = get_node_coords(cache.elements.node_coordinates, equations, dg, i, j,
                                element)
            if source_region(x)
                push!(coupled_element_ids, element)
                break
            end
        end
    end

    return coupled_element_ids
end

function precompute_weights(source_region, weights, coupled_element_ids, equations,
                            dg::DGSEM, cache)
    acoustic_source_weights = zeros(eltype(cache.elements),
                                    (nnodes(dg), nnodes(dg),
                                     length(coupled_element_ids)))

    @threaded for k in eachindex(coupled_element_ids)
        element = coupled_element_ids[k]
        for j in eachnode(dg), i in eachnode(dg)
            x = get_node_coords(cache.elements.node_coordinates, equations, dg, i, j,
                                element)
            acoustic_source_weights[i, j, k] = source_region(x) ? weights(x) :
                                               zero(weights(x))
        end
    end

    return acoustic_source_weights
end

function Base.show(io::IO, semi::SemidiscretizationEulerAcoustics)
    @nospecialize semi # reduce precompilation time

    print(io, "SemidiscretizationEulerAcoustics(")
    print(io, semi.semi_acoustics)
    print(io, ", ", semi.semi_euler)
    print(io, ", cache(")
    for (idx, key) in enumerate(keys(semi.cache))
        idx > 1 && print(io, " ")
        print(io, key)
    end
    print(io, "))")
end

function Base.show(io::IO, mime::MIME"text/plain",
                   semi::SemidiscretizationEulerAcoustics)
    @nospecialize semi # reduce precompilation time

    if get(io, :compact, false)
        show(io, semi)
    else
        summary_header(io, "SemidiscretizationEulerAcoustics")
        summary_line(io, "semidiscretization Euler",
                     semi.semi_euler |> typeof |> nameof)
        show(increment_indent(io), mime, semi.semi_euler)
        summary_line(io, "semidiscretization acoustics",
                     semi.semi_acoustics |> typeof |> nameof)
        show(increment_indent(io), mime, semi.semi_acoustics)
        summary_footer(io)
    end
end

# The acoustics semidiscretization is the main semidiscretization.
@inline function mesh_equations_solver_cache(semi::SemidiscretizationEulerAcoustics)
    return mesh_equations_solver_cache(semi.semi_acoustics)
end

@inline Base.ndims(semi::SemidiscretizationEulerAcoustics) = ndims(semi.semi_acoustics)
@inline Base.real(semi::SemidiscretizationEulerAcoustics) = real(semi.semi_acoustics)

# Computes the coefficients of the initial condition
@inline function compute_coefficients(t, semi::SemidiscretizationEulerAcoustics)
    compute_coefficients(t, semi.semi_acoustics)
end

@inline function compute_coefficients!(u_ode, t, semi::SemidiscretizationEulerAcoustics)
    compute_coefficients!(u_ode, t, semi.semi_acoustics)
end

@inline function calc_error_norms(func, u, t, analyzer,
                                  semi::SemidiscretizationEulerAcoustics,
                                  cache_analysis)
    calc_error_norms(func, u, t, analyzer, semi.semi_acoustics, cache_analysis)
end

function rhs!(du_ode, u_ode, semi::SemidiscretizationEulerAcoustics, t)
    @unpack semi_acoustics, cache = semi
    @unpack acoustic_source_terms, acoustic_source_weights, coupled_element_ids = cache

    du_acoustics = wrap_array(du_ode, semi_acoustics)

    time_start = time_ns()

    @trixi_timeit timer() "acoustics rhs!" rhs!(du_ode, u_ode, semi_acoustics, t)

    @trixi_timeit timer() "add acoustic source terms" begin
        add_acoustic_source_terms!(du_acoustics, acoustic_source_terms,
                                   acoustic_source_weights, coupled_element_ids,
                                   mesh_equations_solver_cache(semi_acoustics)...)
    end

    runtime = time_ns() - time_start
    put!(semi.performance_counter, runtime)

    return nothing
end

function add_acoustic_source_terms!(du_acoustics, acoustic_source_terms, source_weights,
                                    coupled_element_ids, mesh::TreeMesh{2}, equations,
                                    dg::DGSEM,
                                    cache)
    @threaded for k in eachindex(coupled_element_ids)
        element = coupled_element_ids[k]

        for j in eachnode(dg), i in eachnode(dg)
            du_acoustics[1, i, j, element] += source_weights[i, j, k] *
                                              acoustic_source_terms[1, i, j, k]
            du_acoustics[2, i, j, element] += source_weights[i, j, k] *
                                              acoustic_source_terms[2, i, j, k]
        end
    end

    return nothing
end
end # @muladd
