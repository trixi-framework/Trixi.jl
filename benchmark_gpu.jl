@muladd begin
#! format: noindent

@inline function get_node_turbo(turbo_local, ::Val{NAUX},
                                indices...) where {NAUX}
    return ntuple(v -> (@inbounds turbo_local[v, indices...]), Val(NAUX))
end

@inline function calc_volume_integral!(backend::Backend, du, u,
                                       mesh::Union{P4estMesh{3}, T8codeMesh{3}},
                                       have_nonconservative_terms, equations,
                                       volume_integral::VolumeIntegralFluxDifferencing{<:FluxTurbo},
                                       dg::DGSEM, cache, ::Val{1}, ::True)
    @unpack derivative_split = dg.basis
    @unpack contravariant_vectors = cache.elements
    @unpack numerical_flux = volume_integral.volume_flux
    NNODES = nnodes(dg)
    kernel! = version_turbo_1!(backend, (NNODES, NNODES, NNODES, 1))
    kernel!(du, u, equations,
            typeof(mesh),
            have_nonconservative_terms,
            dg,
            numerical_flux,
            Val(NNODES), Val(nvariables(equations)),
            nturbovars(numerical_flux, equations),
            derivative_split,
            contravariant_vectors,
            ndrange = (NNODES, NNODES, NNODES, nelements(dg, cache)))
    return nothing
end

## Version 1: three separate not unrolled loops
@kernel function version_turbo_1!(du, u, equations,
                                  MeshT::Type{<:Union{P4estMesh{3},
                                                      T8codeMesh{3}}},
                                  have_nonconservative_terms::False,
                                  dg::DGSEM,
                                  numerical_flux::NumericalFlux,
                                  ::Val{NNODES},
                                  ::Val{NVARIABLES},
                                  ::Val{NAUX},
                                  derivative_split,
                                  contravariant_vectors,
                                  alpha = true) where {NNODES,
                                                       NVARIABLES,
                                                       NAUX, NumericalFlux}
    i, j, k, element = @index(Global, NTuple)

    turbo_local = @localmem eltype(du) (NAUX, NNODES, NNODES, NNODES)

    u_node = get_node_vars(u, equations, dg, i, j, k, element)
    turbo_node = cons2turbo(numerical_flux, u_node..., equations)
    @inbounds for v in 1:NAUX
        turbo_local[v, i, j, k] = turbo_node[v]
    end
    @synchronize

    du_local = zero(SVector{NVARIABLES, eltype(du)})

    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
    for ii in 1:NNODES
        Ja1_avg = 0.5f0 * (Ja1_node +
                   get_contravariant_vector(1, contravariant_vectors,
                                            ii, j, k, element))
        fluxtilde1 = flux_turbo(numerical_flux,
                                turbo_node...,
                                get_node_turbo(turbo_local, Val(NAUX), ii, j, k)...,
                                Ja1_avg[1], Ja1_avg[2], Ja1_avg[3],
                                equations)
        du_local = du_local +
                   (alpha * derivative_split[i, ii]) * SVector{NVARIABLES}(fluxtilde1)
    end

    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
    for jj in 1:NNODES
        Ja2_avg = 0.5f0 * (Ja2_node +
                   get_contravariant_vector(2, contravariant_vectors,
                                            i, jj, k, element))
        fluxtilde2 = flux_turbo(numerical_flux,
                                turbo_node...,
                                get_node_turbo(turbo_local, Val(NAUX), i, jj, k)...,
                                Ja2_avg[1], Ja2_avg[2], Ja2_avg[3],
                                equations)
        du_local = du_local +
                   (alpha * derivative_split[j, jj]) * SVector{NVARIABLES}(fluxtilde2)
    end

    Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)
    for kk in 1:NNODES
        Ja3_avg = 0.5f0 * (Ja3_node +
                   get_contravariant_vector(3, contravariant_vectors,
                                            i, j, kk, element))
        fluxtilde3 = flux_turbo(numerical_flux,
                                turbo_node...,
                                get_node_turbo(turbo_local, Val(NAUX), i, j, kk)...,
                                Ja3_avg[1], Ja3_avg[2], Ja3_avg[3],
                                equations)
        du_local = du_local +
                   (alpha * derivative_split[k, kk]) * SVector{NVARIABLES}(fluxtilde3)
    end

    add_to_node_vars!(du, du_local, equations, dg, i, j, k, element)
end

@kernel function version_turbo_1!(du, u, equations,
                                  MeshT::Type{<:Union{P4estMesh{3},
                                                      T8codeMesh{3}}},
                                  have_nonconservative_terms::True,
                                  dg::DGSEM,
                                  numerical_flux::NumericalFlux,
                                  ::Val{NNODES},
                                  ::Val{NVARIABLES},
                                  ::Val{NAUX},
                                  derivative_split,
                                  contravariant_vectors,
                                  alpha = true) where {NNODES,
                                                       NVARIABLES,
                                                       NAUX, NumericalFlux}
    i, j, k, element = @index(Global, NTuple)

    turbo_local = @localmem eltype(du) (NAUX, NNODES, NNODES, NNODES)

    u_node = get_node_vars(u, equations, dg, i, j, k, element)
    turbo_node = cons2turbo(numerical_flux, u_node..., equations)
    @inbounds for v in 1:NAUX
        turbo_local[v, i, j, k] = turbo_node[v]
    end
    @synchronize

    du_local = zero(SVector{NVARIABLES, eltype(du)})

    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
    for ii in 1:NNODES
        Ja1_avg = 0.5f0 * (Ja1_node +
                   get_contravariant_vector(1, contravariant_vectors,
                                            ii, j, k, element))
        fluxtilde1_left, _ = flux_turbo(numerical_flux,
                                        turbo_node...,
                                        get_node_turbo(turbo_local, Val(NAUX),
                                                       ii, j, k)...,
                                        Ja1_avg[1], Ja1_avg[2], Ja1_avg[3],
                                        equations)
        du_local = du_local +
                   (alpha * derivative_split[i, ii]) *
                   SVector{NVARIABLES}(fluxtilde1_left)
    end

    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
    for jj in 1:NNODES
        Ja2_avg = 0.5f0 * (Ja2_node +
                   get_contravariant_vector(2, contravariant_vectors,
                                            i, jj, k, element))
        fluxtilde2_left, _ = flux_turbo(numerical_flux,
                                        turbo_node...,
                                        get_node_turbo(turbo_local, Val(NAUX),
                                                       i, jj, k)...,
                                        Ja2_avg[1], Ja2_avg[2], Ja2_avg[3],
                                        equations)
        du_local = du_local +
                   (alpha * derivative_split[j, jj]) *
                   SVector{NVARIABLES}(fluxtilde2_left)
    end

    Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)
    for kk in 1:NNODES
        Ja3_avg = 0.5f0 * (Ja3_node +
                   get_contravariant_vector(3, contravariant_vectors,
                                            i, j, kk, element))
        fluxtilde3_left, _ = flux_turbo(numerical_flux,
                                        turbo_node...,
                                        get_node_turbo(turbo_local, Val(NAUX),
                                                       i, j, kk)...,
                                        Ja3_avg[1], Ja3_avg[2], Ja3_avg[3],
                                        equations)
        du_local = du_local +
                   (alpha * derivative_split[k, kk]) *
                   SVector{NVARIABLES}(fluxtilde3_left)
    end

    add_to_node_vars!(du, du_local, equations, dg, i, j, k, element)
end

@inline function calc_volume_integral!(backend::Backend, du, u,
                                       mesh::Union{P4estMesh{3}, T8codeMesh{3}},
                                       have_nonconservative_terms, equations,
                                       volume_integral::VolumeIntegralFluxDifferencing,
                                       dg::DGSEM, cache, ::Val{1}, ::False)
    @unpack derivative_split = dg.basis
    @unpack contravariant_vectors = cache.elements
    NNODES = nnodes(dg)
    kernel! = version_1!(backend, (NNODES, NNODES, NNODES, 1))
    kernel!(du, u, equations,
            typeof(mesh),
            have_nonconservative_terms,
            combine_conservative_and_nonconservative_fluxes(volume_integral.volume_flux,
                                                            equations),
            dg,
            volume_integral.volume_flux,
            Val(NNODES), Val(nvariables(equations)),
            derivative_split,
            contravariant_vectors,
            ndrange = (NNODES, NNODES, NNODES, nelements(dg, cache)))
    return nothing
end

## Version 1 without turbo, conservative systems
@kernel function version_1!(du, u, equations,
                            MeshT::Type{<:Union{P4estMesh{3}, T8codeMesh{3}}},
                            have_nonconservative_terms::False,
                            combine_conservative_and_nonconservative_fluxes::False,
                            dg::DGSEM,
                            volume_flux,
                            ::Val{NNODES},
                            ::Val{NVARIABLES},
                            derivative_split,
                            contravariant_vectors,
                            alpha = true) where {NNODES, NVARIABLES}
    i, j, k, element = @index(Global, NTuple)

    u_local = @localmem eltype(du) (NVARIABLES, NNODES, NNODES, NNODES)

    u_node = get_node_vars(u, equations, dg, i, j, k, element)
    @inbounds for v in 1:NVARIABLES
        u_local[v, i, j, k] = u_node[v]
    end
    @synchronize

    du_local = zero(SVector{NVARIABLES, eltype(du)})

    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
    for ii in 1:NNODES
        Ja1_avg = 0.5f0 * (Ja1_node +
                   get_contravariant_vector(1, contravariant_vectors,
                                            ii, j, k, element))
        fluxtilde1 = volume_flux(u_node,
                                 get_node_flux(u_local, Val(NVARIABLES), ii, j, k),
                                 Ja1_avg, equations)
        du_local = du_local + (alpha * derivative_split[i, ii]) * fluxtilde1
    end

    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
    for jj in 1:NNODES
        Ja2_avg = 0.5f0 * (Ja2_node +
                   get_contravariant_vector(2, contravariant_vectors,
                                            i, jj, k, element))
        fluxtilde2 = volume_flux(u_node,
                                 get_node_flux(u_local, Val(NVARIABLES), i, jj, k),
                                 Ja2_avg, equations)
        du_local = du_local + (alpha * derivative_split[j, jj]) * fluxtilde2
    end

    Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)
    for kk in 1:NNODES
        Ja3_avg = 0.5f0 * (Ja3_node +
                   get_contravariant_vector(3, contravariant_vectors,
                                            i, j, kk, element))
        fluxtilde3 = volume_flux(u_node,
                                 get_node_flux(u_local, Val(NVARIABLES), i, j, kk),
                                 Ja3_avg, equations)
        du_local = du_local + (alpha * derivative_split[k, kk]) * fluxtilde3
    end

    add_to_node_vars!(du, du_local, equations, dg, i, j, k, element)
end

@kernel function version_1!(du, u, equations,
                            MeshT::Type{<:Union{P4estMesh{3}, T8codeMesh{3}}},
                            have_nonconservative_terms::True,
                            combine_conservative_and_nonconservative_fluxes::True,
                            dg::DGSEM,
                            volume_flux,
                            ::Val{NNODES},
                            ::Val{NVARIABLES},
                            derivative_split,
                            contravariant_vectors,
                            alpha = true) where {NNODES, NVARIABLES}
    i, j, k, element = @index(Global, NTuple)

    u_local = @localmem eltype(du) (NVARIABLES, NNODES, NNODES, NNODES)

    u_node = get_node_vars(u, equations, dg, i, j, k, element)
    @inbounds for v in 1:NVARIABLES
        u_local[v, i, j, k] = u_node[v]
    end
    @synchronize

    du_local = zero(SVector{NVARIABLES, eltype(du)})

    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
    for ii in 1:NNODES
        Ja1_avg = 0.5f0 * (Ja1_node +
                   get_contravariant_vector(1, contravariant_vectors,
                                            ii, j, k, element))
        fluxtilde1_left, _ = volume_flux(u_node,
                                         get_node_flux(u_local, Val(NVARIABLES),
                                                       ii, j, k),
                                         Ja1_avg, equations)
        du_local = du_local + (alpha * derivative_split[i, ii]) * fluxtilde1_left
    end

    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
    for jj in 1:NNODES
        Ja2_avg = 0.5f0 * (Ja2_node +
                   get_contravariant_vector(2, contravariant_vectors,
                                            i, jj, k, element))
        fluxtilde2_left, _ = volume_flux(u_node,
                                         get_node_flux(u_local, Val(NVARIABLES),
                                                       i, jj, k),
                                         Ja2_avg, equations)
        du_local = du_local + (alpha * derivative_split[j, jj]) * fluxtilde2_left
    end

    Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)
    for kk in 1:NNODES
        Ja3_avg = 0.5f0 * (Ja3_node +
                   get_contravariant_vector(3, contravariant_vectors,
                                            i, j, kk, element))
        fluxtilde3_left, _ = volume_flux(u_node,
                                         get_node_flux(u_local, Val(NVARIABLES),
                                                       i, j, kk),
                                         Ja3_avg, equations)
        du_local = du_local + (alpha * derivative_split[k, kk]) * fluxtilde3_left
    end

    add_to_node_vars!(du, du_local, equations, dg, i, j, k, element)
end

## Version 2: three separate unrolled loops
@inline function calc_volume_integral!(backend::Backend, du, u,
                                       mesh::Union{P4estMesh{3}, T8codeMesh{3}},
                                       have_nonconservative_terms, equations,
                                       volume_integral::VolumeIntegralFluxDifferencing{<:FluxTurbo},
                                       dg::DGSEM, cache, ::Val{2}, ::True)
    @unpack derivative_split = dg.basis
    @unpack contravariant_vectors = cache.elements
    @unpack numerical_flux = volume_integral.volume_flux
    NNODES = nnodes(dg)
    kernel! = version_turbo_2!(backend, (NNODES, NNODES, NNODES, 1))
    kernel!(du, u, equations,
            typeof(mesh),
            have_nonconservative_terms,
            dg,
            numerical_flux,
            Val(NNODES), Val(nvariables(equations)),
            nturbovars(numerical_flux, equations),
            derivative_split,
            contravariant_vectors,
            ndrange = (NNODES, NNODES, NNODES, nelements(dg, cache)))
    return nothing
end

@kernel function version_turbo_2!(du, u, equations,
                                  MeshT::Type{<:Union{P4estMesh{3},
                                                      T8codeMesh{3}}},
                                  have_nonconservative_terms::False,
                                  dg::DGSEM,
                                  numerical_flux::NumericalFlux,
                                  ::Val{NNODES},
                                  ::Val{NVARIABLES},
                                  ::Val{NAUX},
                                  derivative_split,
                                  contravariant_vectors,
                                  alpha = true) where {NNODES, NVARIABLES,
                                                       NAUX, NumericalFlux}
    i, j, k, element = @index(Global, NTuple)

    turbo_local = @localmem eltype(du) (NAUX, NNODES, NNODES, NNODES)

    u_node = get_node_vars(u, equations, dg, i, j, k, element)
    turbo_node = cons2turbo(numerical_flux, u_node..., equations)
    @inbounds for v in 1:NAUX
        turbo_local[v, i, j, k] = turbo_node[v]
    end
    @synchronize

    du_local = zero(SVector{NVARIABLES, eltype(du)})

    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
    KernelAbstractions.Extras.@unroll for ii in 1:NNODES
        Ja1_avg = 0.5f0 * (Ja1_node +
                   get_contravariant_vector(1, contravariant_vectors,
                                            ii, j, k, element))
        fluxtilde1 = flux_turbo(numerical_flux,
                                turbo_node...,
                                get_node_turbo(turbo_local, Val(NAUX),
                                               ii, j, k)...,
                                Ja1_avg[1], Ja1_avg[2], Ja1_avg[3],
                                equations)
        du_local = du_local +
                   (alpha * derivative_split[i, ii]) *
                   SVector{NVARIABLES}(fluxtilde1)
    end

    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
    KernelAbstractions.Extras.@unroll for jj in 1:NNODES
        Ja2_avg = 0.5f0 * (Ja2_node +
                   get_contravariant_vector(2, contravariant_vectors,
                                            i, jj, k, element))
        fluxtilde2 = flux_turbo(numerical_flux,
                                turbo_node...,
                                get_node_turbo(turbo_local, Val(NAUX),
                                               i, jj, k)...,
                                Ja2_avg[1], Ja2_avg[2], Ja2_avg[3],
                                equations)
        du_local = du_local +
                   (alpha * derivative_split[j, jj]) *
                   SVector{NVARIABLES}(fluxtilde2)
    end

    Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)
    KernelAbstractions.Extras.@unroll for kk in 1:NNODES
        Ja3_avg = 0.5f0 * (Ja3_node +
                   get_contravariant_vector(3, contravariant_vectors,
                                            i, j, kk, element))
        fluxtilde3 = flux_turbo(numerical_flux,
                                turbo_node...,
                                get_node_turbo(turbo_local, Val(NAUX),
                                               i, j, kk)...,
                                Ja3_avg[1], Ja3_avg[2], Ja3_avg[3],
                                equations)
        du_local = du_local +
                   (alpha * derivative_split[k, kk]) *
                   SVector{NVARIABLES}(fluxtilde3)
    end

    add_to_node_vars!(du, du_local, equations, dg, i, j, k, element)
end

@kernel function version_turbo_2!(du, u, equations,
                                  MeshT::Type{<:Union{P4estMesh{3},
                                                      T8codeMesh{3}}},
                                  have_nonconservative_terms::True,
                                  dg::DGSEM,
                                  numerical_flux::NumericalFlux,
                                  ::Val{NNODES},
                                  ::Val{NVARIABLES},
                                  ::Val{NAUX},
                                  derivative_split,
                                  contravariant_vectors,
                                  alpha = true) where {NNODES, NVARIABLES,
                                                       NAUX, NumericalFlux}
    i, j, k, element = @index(Global, NTuple)

    turbo_local = @localmem eltype(du) (NAUX, NNODES, NNODES, NNODES)

    u_node = get_node_vars(u, equations, dg, i, j, k, element)
    turbo_node = cons2turbo(numerical_flux, u_node..., equations)
    @inbounds for v in 1:NAUX
        turbo_local[v, i, j, k] = turbo_node[v]
    end
    @synchronize

    du_local = zero(SVector{NVARIABLES, eltype(du)})

    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
    KernelAbstractions.Extras.@unroll for ii in 1:NNODES
        Ja1_avg = 0.5f0 * (Ja1_node +
                   get_contravariant_vector(1, contravariant_vectors,
                                            ii, j, k, element))
        fluxtilde1_left, _ = flux_turbo(numerical_flux,
                                        turbo_node...,
                                        get_node_turbo(turbo_local, Val(NAUX),
                                                       ii, j, k)...,
                                        Ja1_avg[1], Ja1_avg[2], Ja1_avg[3],
                                        equations)
        du_local = du_local +
                   (alpha * derivative_split[i, ii]) *
                   SVector{NVARIABLES}(fluxtilde1_left)
    end

    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
    KernelAbstractions.Extras.@unroll for jj in 1:NNODES
        Ja2_avg = 0.5f0 * (Ja2_node +
                   get_contravariant_vector(2, contravariant_vectors,
                                            i, jj, k, element))
        fluxtilde2_left, _ = flux_turbo(numerical_flux,
                                        turbo_node...,
                                        get_node_turbo(turbo_local, Val(NAUX),
                                                       i, jj, k)...,
                                        Ja2_avg[1], Ja2_avg[2], Ja2_avg[3],
                                        equations)
        du_local = du_local +
                   (alpha * derivative_split[j, jj]) *
                   SVector{NVARIABLES}(fluxtilde2_left)
    end

    Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)
    KernelAbstractions.Extras.@unroll for kk in 1:NNODES
        Ja3_avg = 0.5f0 * (Ja3_node +
                   get_contravariant_vector(3, contravariant_vectors,
                                            i, j, kk, element))
        fluxtilde3_left, _ = flux_turbo(numerical_flux,
                                        turbo_node...,
                                        get_node_turbo(turbo_local, Val(NAUX),
                                                       i, j, kk)...,
                                        Ja3_avg[1], Ja3_avg[2], Ja3_avg[3],
                                        equations)
        du_local = du_local +
                   (alpha * derivative_split[k, kk]) *
                   SVector{NVARIABLES}(fluxtilde3_left)
    end

    add_to_node_vars!(du, du_local, equations, dg, i, j, k, element)
end

## Version 2 without turbo: three separate unrolled loops
@inline function calc_volume_integral!(backend::Backend, du, u,
                                       mesh::Union{P4estMesh{3}, T8codeMesh{3}},
                                       have_nonconservative_terms, equations,
                                       volume_integral::VolumeIntegralFluxDifferencing,
                                       dg::DGSEM, cache, ::Val{2}, ::False)
    @unpack derivative_split = dg.basis
    @unpack contravariant_vectors = cache.elements
    NNODES = nnodes(dg)
    kernel! = version_2!(backend, (NNODES, NNODES, NNODES, 1))
    kernel!(du, u, equations,
            typeof(mesh),
            have_nonconservative_terms,
            combine_conservative_and_nonconservative_fluxes(volume_integral.volume_flux,
                                                            equations),
            dg,
            volume_integral.volume_flux,
            Val(NNODES), Val(nvariables(equations)),
            derivative_split,
            contravariant_vectors,
            ndrange = (NNODES, NNODES, NNODES, nelements(dg, cache)))
    return nothing
end

@kernel function version_2!(du, u, equations,
                            MeshT::Type{<:Union{P4estMesh{3},
                                                T8codeMesh{3}}},
                            have_nonconservative_terms::False,
                            combine_conservative_and_nonconservative_fluxes::False,
                            dg::DGSEM,
                            volume_flux,
                            ::Val{NNODES},
                            ::Val{NVARIABLES},
                            derivative_split,
                            contravariant_vectors,
                            alpha = true) where {NNODES, NVARIABLES}
    i, j, k, element = @index(Global, NTuple)

    u_local = @localmem eltype(du) (NVARIABLES, NNODES, NNODES, NNODES)

    u_node = get_node_vars(u, equations, dg, i, j, k, element)
    @inbounds for v in 1:NVARIABLES
        u_local[v, i, j, k] = u_node[v]
    end
    @synchronize

    du_local = zero(SVector{NVARIABLES, eltype(du)})

    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
    KernelAbstractions.Extras.@unroll for ii in 1:NNODES
        Ja1_avg = 0.5f0 * (Ja1_node +
                   get_contravariant_vector(1, contravariant_vectors,
                                            ii, j, k, element))
        fluxtilde1 = volume_flux(u_node,
                                 get_node_flux(u_local, Val(NVARIABLES),
                                               ii, j, k),
                                 Ja1_avg, equations)
        du_local = du_local + (alpha * derivative_split[i, ii]) * fluxtilde1
    end

    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
    KernelAbstractions.Extras.@unroll for jj in 1:NNODES
        Ja2_avg = 0.5f0 * (Ja2_node +
                   get_contravariant_vector(2, contravariant_vectors,
                                            i, jj, k, element))
        fluxtilde2 = volume_flux(u_node,
                                 get_node_flux(u_local, Val(NVARIABLES),
                                               i, jj, k),
                                 Ja2_avg, equations)
        du_local = du_local + (alpha * derivative_split[j, jj]) * fluxtilde2
    end

    Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)
    KernelAbstractions.Extras.@unroll for kk in 1:NNODES
        Ja3_avg = 0.5f0 * (Ja3_node +
                   get_contravariant_vector(3, contravariant_vectors,
                                            i, j, kk, element))
        fluxtilde3 = volume_flux(u_node,
                                 get_node_flux(u_local, Val(NVARIABLES),
                                               i, j, kk),
                                 Ja3_avg, equations)
        du_local = du_local + (alpha * derivative_split[k, kk]) * fluxtilde3
    end

    add_to_node_vars!(du, du_local, equations, dg, i, j, k, element)
end

@kernel function version_2!(du, u, equations,
                            MeshT::Type{<:Union{P4estMesh{3},
                                                T8codeMesh{3}}},
                            have_nonconservative_terms::True,
                            combine_conservative_and_nonconservative_fluxes::True,
                            dg::DGSEM,
                            volume_flux,
                            ::Val{NNODES},
                            ::Val{NVARIABLES},
                            derivative_split,
                            contravariant_vectors,
                            alpha = true) where {NNODES, NVARIABLES}
    i, j, k, element = @index(Global, NTuple)

    u_local = @localmem eltype(du) (NVARIABLES, NNODES, NNODES, NNODES)

    u_node = get_node_vars(u, equations, dg, i, j, k, element)
    @inbounds for v in 1:NVARIABLES
        u_local[v, i, j, k] = u_node[v]
    end
    @synchronize

    du_local = zero(SVector{NVARIABLES, eltype(du)})

    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
    KernelAbstractions.Extras.@unroll for ii in 1:NNODES
        Ja1_avg = 0.5f0 * (Ja1_node +
                   get_contravariant_vector(1, contravariant_vectors,
                                            ii, j, k, element))
        fluxtilde1_left, _ = volume_flux(u_node,
                                         get_node_flux(u_local, Val(NVARIABLES),
                                                       ii, j, k),
                                         Ja1_avg, equations)
        du_local = du_local + (alpha * derivative_split[i, ii]) * fluxtilde1_left
    end

    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
    KernelAbstractions.Extras.@unroll for jj in 1:NNODES
        Ja2_avg = 0.5f0 * (Ja2_node +
                   get_contravariant_vector(2, contravariant_vectors,
                                            i, jj, k, element))
        fluxtilde2_left, _ = volume_flux(u_node,
                                         get_node_flux(u_local, Val(NVARIABLES),
                                                       i, jj, k),
                                         Ja2_avg, equations)
        du_local = du_local + (alpha * derivative_split[j, jj]) * fluxtilde2_left
    end

    Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)
    KernelAbstractions.Extras.@unroll for kk in 1:NNODES
        Ja3_avg = 0.5f0 * (Ja3_node +
                   get_contravariant_vector(3, contravariant_vectors,
                                            i, j, kk, element))
        fluxtilde3_left, _ = volume_flux(u_node,
                                         get_node_flux(u_local, Val(NVARIABLES),
                                                       i, j, kk),
                                         Ja3_avg, equations)
        du_local = du_local + (alpha * derivative_split[k, kk]) * fluxtilde3_left
    end

    add_to_node_vars!(du, du_local, equations, dg, i, j, k, element)
end

## Version 3: one fused not unrolled loop
@inline function calc_volume_integral!(backend::Backend, du, u,
                                       mesh::Union{P4estMesh{3}, T8codeMesh{3}},
                                       have_nonconservative_terms, equations,
                                       volume_integral::VolumeIntegralFluxDifferencing{<:FluxTurbo},
                                       dg::DGSEM, cache, ::Val{3}, ::True)
    @unpack derivative_split = dg.basis
    @unpack contravariant_vectors = cache.elements
    @unpack numerical_flux = volume_integral.volume_flux
    NNODES = nnodes(dg)
    kernel! = version_turbo_3!(backend, (NNODES, NNODES, NNODES, 1))
    kernel!(du, u, equations,
            typeof(mesh),
            have_nonconservative_terms,
            dg,
            numerical_flux,
            Val(NNODES), Val(nvariables(equations)),
            nturbovars(numerical_flux, equations),
            derivative_split,
            contravariant_vectors,
            ndrange = (NNODES, NNODES, NNODES, nelements(dg, cache)))
    return nothing
end

@kernel function version_turbo_3!(du, u, equations,
                                  MeshT::Type{<:Union{P4estMesh{3},
                                                      T8codeMesh{3}}},
                                  have_nonconservative_terms::False,
                                  dg::DGSEM,
                                  numerical_flux,
                                  ::Val{NNODES},
                                  ::Val{NVARIABLES},
                                  ::Val{NAUX},
                                  derivative_split,
                                  contravariant_vectors,
                                  alpha = true) where {NNODES, NVARIABLES,
                                                       NAUX}
    i, j, k, element = @index(Global, NTuple)

    turbo_local = @localmem eltype(du) (NAUX, NNODES, NNODES, NNODES)

    u_node = get_node_vars(u, equations, dg, i, j, k, element)
    turbo_node = cons2turbo(numerical_flux, u_node..., equations)
    @inbounds for v in 1:NAUX
        turbo_local[v, i, j, k] = turbo_node[v]
    end
    @synchronize

    du_local = zero(SVector{NVARIABLES, eltype(du)})

    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
    Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)
    for m in 1:NNODES
        Ja1_avg = 0.5f0 * (Ja1_node +
                   get_contravariant_vector(1, contravariant_vectors,
                                            m, j, k, element))
        fluxtilde1 = flux_turbo(numerical_flux,
                                turbo_node...,
                                get_node_turbo(turbo_local, Val(NAUX),
                                               m, j, k)...,
                                Ja1_avg[1], Ja1_avg[2], Ja1_avg[3],
                                equations)
        du_local = du_local +
                   (alpha * derivative_split[i, m]) *
                   SVector{NVARIABLES}(fluxtilde1)

        Ja2_avg = 0.5f0 * (Ja2_node +
                   get_contravariant_vector(2, contravariant_vectors,
                                            i, m, k, element))
        fluxtilde2 = flux_turbo(numerical_flux,
                                turbo_node...,
                                get_node_turbo(turbo_local, Val(NAUX),
                                               i, m, k)...,
                                Ja2_avg[1], Ja2_avg[2], Ja2_avg[3],
                                equations)
        du_local = du_local +
                   (alpha * derivative_split[j, m]) *
                   SVector{NVARIABLES}(fluxtilde2)

        Ja3_avg = 0.5f0 * (Ja3_node +
                   get_contravariant_vector(3, contravariant_vectors,
                                            i, j, m, element))
        fluxtilde3 = flux_turbo(numerical_flux,
                                turbo_node...,
                                get_node_turbo(turbo_local, Val(NAUX),
                                               i, j, m)...,
                                Ja3_avg[1], Ja3_avg[2], Ja3_avg[3],
                                equations)
        du_local = du_local +
                   (alpha * derivative_split[k, m]) *
                   SVector{NVARIABLES}(fluxtilde3)
    end

    add_to_node_vars!(du, du_local, equations, dg, i, j, k, element)
end

@kernel function version_turbo_3!(du, u, equations,
                                  MeshT::Type{<:Union{P4estMesh{3},
                                                      T8codeMesh{3}}},
                                  have_nonconservative_terms::True,
                                  dg::DGSEM,
                                  numerical_flux,
                                  ::Val{NNODES},
                                  ::Val{NVARIABLES},
                                  ::Val{NAUX},
                                  derivative_split,
                                  contravariant_vectors,
                                  alpha = true) where {NNODES, NVARIABLES,
                                                       NAUX}
    i, j, k, element = @index(Global, NTuple)

    turbo_local = @localmem eltype(du) (NAUX, NNODES, NNODES, NNODES)

    u_node = get_node_vars(u, equations, dg, i, j, k, element)
    turbo_node = cons2turbo(numerical_flux, u_node..., equations)
    @inbounds for v in 1:NAUX
        turbo_local[v, i, j, k] = turbo_node[v]
    end
    @synchronize

    du_local = zero(SVector{NVARIABLES, eltype(du)})

    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
    Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)
    for m in 1:NNODES
        Ja1_avg = 0.5f0 * (Ja1_node +
                   get_contravariant_vector(1, contravariant_vectors,
                                            m, j, k, element))
        fluxtilde1_left, _ = flux_turbo(numerical_flux,
                                        turbo_node...,
                                        get_node_turbo(turbo_local, Val(NAUX),
                                                       m, j, k)...,
                                        Ja1_avg[1], Ja1_avg[2], Ja1_avg[3],
                                        equations)
        du_local = du_local +
                   (alpha * derivative_split[i, m]) *
                   SVector{NVARIABLES}(fluxtilde1_left)

        Ja2_avg = 0.5f0 * (Ja2_node +
                   get_contravariant_vector(2, contravariant_vectors,
                                            i, m, k, element))
        fluxtilde2_left, _ = flux_turbo(numerical_flux,
                                        turbo_node...,
                                        get_node_turbo(turbo_local, Val(NAUX),
                                                       i, m, k)...,
                                        Ja2_avg[1], Ja2_avg[2], Ja2_avg[3],
                                        equations)
        du_local = du_local +
                   (alpha * derivative_split[j, m]) *
                   SVector{NVARIABLES}(fluxtilde2_left)

        Ja3_avg = 0.5f0 * (Ja3_node +
                   get_contravariant_vector(3, contravariant_vectors,
                                            i, j, m, element))
        fluxtilde3_left, _ = flux_turbo(numerical_flux,
                                        turbo_node...,
                                        get_node_turbo(turbo_local, Val(NAUX),
                                                       i, j, m)...,
                                        Ja3_avg[1], Ja3_avg[2], Ja3_avg[3],
                                        equations)
        du_local = du_local +
                   (alpha * derivative_split[k, m]) *
                   SVector{NVARIABLES}(fluxtilde3_left)
    end

    add_to_node_vars!(du, du_local, equations, dg, i, j, k, element)
end

## Version 3 without turbo: one fused not unrolled loop
@inline function calc_volume_integral!(backend::Backend, du, u,
                                       mesh::Union{P4estMesh{3}, T8codeMesh{3}},
                                       have_nonconservative_terms, equations,
                                       volume_integral::VolumeIntegralFluxDifferencing,
                                       dg::DGSEM, cache, ::Val{3}, ::False)
    @unpack derivative_split = dg.basis
    @unpack contravariant_vectors = cache.elements
    NNODES = nnodes(dg)
    kernel! = version_3!(backend, (NNODES, NNODES, NNODES, 1))
    kernel!(du, u, equations,
            typeof(mesh),
            have_nonconservative_terms,
            combine_conservative_and_nonconservative_fluxes(volume_integral.volume_flux,
                                                            equations),
            dg,
            volume_integral.volume_flux,
            Val(NNODES), Val(nvariables(equations)),
            derivative_split,
            contravariant_vectors,
            ndrange = (NNODES, NNODES, NNODES, nelements(dg, cache)))
    return nothing
end

@kernel function version_3!(du, u, equations,
                            MeshT::Type{<:Union{P4estMesh{3},
                                                T8codeMesh{3}}},
                            have_nonconservative_terms::False,
                            combine_conservative_and_nonconservative_fluxes::False,
                            dg::DGSEM,
                            volume_flux,
                            ::Val{NNODES},
                            ::Val{NVARIABLES},
                            derivative_split,
                            contravariant_vectors,
                            alpha = true) where {NNODES, NVARIABLES}
    i, j, k, element = @index(Global, NTuple)

    u_local = @localmem eltype(du) (NVARIABLES, NNODES, NNODES, NNODES)

    u_node = get_node_vars(u, equations, dg, i, j, k, element)
    @inbounds for v in 1:NVARIABLES
        u_local[v, i, j, k] = u_node[v]
    end
    @synchronize

    du_local = zero(SVector{NVARIABLES, eltype(du)})

    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
    Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)
    for m in 1:NNODES
        Ja1_avg = 0.5f0 * (Ja1_node +
                   get_contravariant_vector(1, contravariant_vectors,
                                            m, j, k, element))
        fluxtilde1 = volume_flux(u_node,
                                 get_node_flux(u_local, Val(NVARIABLES),
                                               m, j, k),
                                 Ja1_avg, equations)
        du_local = du_local + (alpha * derivative_split[i, m]) * fluxtilde1

        Ja2_avg = 0.5f0 * (Ja2_node +
                   get_contravariant_vector(2, contravariant_vectors,
                                            i, m, k, element))
        fluxtilde2 = volume_flux(u_node,
                                 get_node_flux(u_local, Val(NVARIABLES),
                                               i, m, k),
                                 Ja2_avg, equations)
        du_local = du_local + (alpha * derivative_split[j, m]) * fluxtilde2

        Ja3_avg = 0.5f0 * (Ja3_node +
                   get_contravariant_vector(3, contravariant_vectors,
                                            i, j, m, element))
        fluxtilde3 = volume_flux(u_node,
                                 get_node_flux(u_local, Val(NVARIABLES),
                                               i, j, m),
                                 Ja3_avg, equations)
        du_local = du_local + (alpha * derivative_split[k, m]) * fluxtilde3
    end

    add_to_node_vars!(du, du_local, equations, dg, i, j, k, element)
end

@kernel function version_3!(du, u, equations,
                            MeshT::Type{<:Union{P4estMesh{3},
                                                T8codeMesh{3}}},
                            have_nonconservative_terms::True,
                            combine_conservative_and_nonconservative_fluxes::True,
                            dg::DGSEM,
                            volume_flux,
                            ::Val{NNODES},
                            ::Val{NVARIABLES},
                            derivative_split,
                            contravariant_vectors,
                            alpha = true) where {NNODES, NVARIABLES}
    i, j, k, element = @index(Global, NTuple)

    u_local = @localmem eltype(du) (NVARIABLES, NNODES, NNODES, NNODES)

    u_node = get_node_vars(u, equations, dg, i, j, k, element)
    @inbounds for v in 1:NVARIABLES
        u_local[v, i, j, k] = u_node[v]
    end
    @synchronize

    du_local = zero(SVector{NVARIABLES, eltype(du)})

    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
    Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)
    for m in 1:NNODES
        Ja1_avg = 0.5f0 * (Ja1_node +
                   get_contravariant_vector(1, contravariant_vectors,
                                            m, j, k, element))
        fluxtilde1_left, _ = volume_flux(u_node,
                                         get_node_flux(u_local, Val(NVARIABLES),
                                                       m, j, k),
                                         Ja1_avg, equations)
        du_local = du_local + (alpha * derivative_split[i, m]) * fluxtilde1_left

        Ja2_avg = 0.5f0 * (Ja2_node +
                   get_contravariant_vector(2, contravariant_vectors,
                                            i, m, k, element))
        fluxtilde2_left, _ = volume_flux(u_node,
                                         get_node_flux(u_local, Val(NVARIABLES),
                                                       i, m, k),
                                         Ja2_avg, equations)
        du_local = du_local + (alpha * derivative_split[j, m]) * fluxtilde2_left

        Ja3_avg = 0.5f0 * (Ja3_node +
                   get_contravariant_vector(3, contravariant_vectors,
                                            i, j, m, element))
        fluxtilde3_left, _ = volume_flux(u_node,
                                         get_node_flux(u_local, Val(NVARIABLES),
                                                       i, j, m),
                                         Ja3_avg, equations)
        du_local = du_local + (alpha * derivative_split[k, m]) * fluxtilde3_left
    end

    add_to_node_vars!(du, du_local, equations, dg, i, j, k, element)
end

## Version 4: one fused unrolled loop
@inline function calc_volume_integral!(backend::Backend, du, u,
                                       mesh::Union{P4estMesh{3}, T8codeMesh{3}},
                                       have_nonconservative_terms, equations,
                                       volume_integral::VolumeIntegralFluxDifferencing{<:FluxTurbo},
                                       dg::DGSEM, cache, ::Val{4}, ::True)
    @unpack derivative_split = dg.basis
    @unpack contravariant_vectors = cache.elements
    @unpack numerical_flux = volume_integral.volume_flux
    NNODES = nnodes(dg)
    kernel! = version_turbo_4!(backend, (NNODES, NNODES, NNODES, 1))
    kernel!(du, u, equations,
            typeof(mesh),
            have_nonconservative_terms,
            dg,
            numerical_flux,
            Val(NNODES), Val(nvariables(equations)),
            nturbovars(numerical_flux, equations),
            derivative_split,
            contravariant_vectors,
            ndrange = (NNODES, NNODES, NNODES, nelements(dg, cache)))
    return nothing
end

@kernel function version_turbo_4!(du, u, equations,
                                  MeshT::Type{<:Union{P4estMesh{3},
                                                      T8codeMesh{3}}},
                                  have_nonconservative_terms::False,
                                  dg::DGSEM,
                                  numerical_flux,
                                  ::Val{NNODES},
                                  ::Val{NVARIABLES},
                                  ::Val{NAUX},
                                  derivative_split,
                                  contravariant_vectors,
                                  alpha = true) where {NNODES, NVARIABLES,
                                                       NAUX}
    i, j, k, element = @index(Global, NTuple)

    turbo_local = @localmem eltype(du) (NAUX, NNODES, NNODES, NNODES)

    u_node = get_node_vars(u, equations, dg, i, j, k, element)
    turbo_node = cons2turbo(numerical_flux, u_node..., equations)
    @inbounds for v in 1:NAUX
        turbo_local[v, i, j, k] = turbo_node[v]
    end
    @synchronize

    du_local = zero(SVector{NVARIABLES, eltype(du)})

    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
    Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)
    KernelAbstractions.Extras.@unroll for m in 1:NNODES
        Ja1_avg = 0.5f0 * (Ja1_node +
                   get_contravariant_vector(1, contravariant_vectors,
                                            m, j, k, element))
        fluxtilde1 = flux_turbo(numerical_flux,
                                turbo_node...,
                                get_node_turbo(turbo_local, Val(NAUX),
                                               m, j, k)...,
                                Ja1_avg[1], Ja1_avg[2], Ja1_avg[3],
                                equations)
        du_local = du_local +
                   (alpha * derivative_split[i, m]) *
                   SVector{NVARIABLES}(fluxtilde1)

        Ja2_avg = 0.5f0 * (Ja2_node +
                   get_contravariant_vector(2, contravariant_vectors,
                                            i, m, k, element))
        fluxtilde2 = flux_turbo(numerical_flux,
                                turbo_node...,
                                get_node_turbo(turbo_local, Val(NAUX),
                                               i, m, k)...,
                                Ja2_avg[1], Ja2_avg[2], Ja2_avg[3],
                                equations)
        du_local = du_local +
                   (alpha * derivative_split[j, m]) *
                   SVector{NVARIABLES}(fluxtilde2)

        Ja3_avg = 0.5f0 * (Ja3_node +
                   get_contravariant_vector(3, contravariant_vectors,
                                            i, j, m, element))
        fluxtilde3 = flux_turbo(numerical_flux,
                                turbo_node...,
                                get_node_turbo(turbo_local, Val(NAUX),
                                               i, j, m)...,
                                Ja3_avg[1], Ja3_avg[2], Ja3_avg[3],
                                equations)
        du_local = du_local +
                   (alpha * derivative_split[k, m]) *
                   SVector{NVARIABLES}(fluxtilde3)
    end

    add_to_node_vars!(du, du_local, equations, dg, i, j, k, element)
end

@kernel function version_turbo_4!(du, u, equations,
                                  MeshT::Type{<:Union{P4estMesh{3},
                                                      T8codeMesh{3}}},
                                  have_nonconservative_terms::True,
                                  dg::DGSEM,
                                  numerical_flux,
                                  ::Val{NNODES},
                                  ::Val{NVARIABLES},
                                  ::Val{NAUX},
                                  derivative_split,
                                  contravariant_vectors,
                                  alpha = true) where {NNODES, NVARIABLES,
                                                       NAUX}
    i, j, k, element = @index(Global, NTuple)

    turbo_local = @localmem eltype(du) (NAUX, NNODES, NNODES, NNODES)

    u_node = get_node_vars(u, equations, dg, i, j, k, element)
    turbo_node = cons2turbo(numerical_flux, u_node..., equations)
    @inbounds for v in 1:NAUX
        turbo_local[v, i, j, k] = turbo_node[v]
    end
    @synchronize

    du_local = zero(SVector{NVARIABLES, eltype(du)})

    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
    Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)
    KernelAbstractions.Extras.@unroll for m in 1:NNODES
        Ja1_avg = 0.5f0 * (Ja1_node +
                   get_contravariant_vector(1, contravariant_vectors,
                                            m, j, k, element))
        fluxtilde1_left, _ = flux_turbo(numerical_flux,
                                        turbo_node...,
                                        get_node_turbo(turbo_local, Val(NAUX),
                                                       m, j, k)...,
                                        Ja1_avg[1], Ja1_avg[2], Ja1_avg[3],
                                        equations)
        du_local = du_local +
                   (alpha * derivative_split[i, m]) *
                   SVector{NVARIABLES}(fluxtilde1_left)

        Ja2_avg = 0.5f0 * (Ja2_node +
                   get_contravariant_vector(2, contravariant_vectors,
                                            i, m, k, element))
        fluxtilde2_left, _ = flux_turbo(numerical_flux,
                                        turbo_node...,
                                        get_node_turbo(turbo_local, Val(NAUX),
                                                       i, m, k)...,
                                        Ja2_avg[1], Ja2_avg[2], Ja2_avg[3],
                                        equations)
        du_local = du_local +
                   (alpha * derivative_split[j, m]) *
                   SVector{NVARIABLES}(fluxtilde2_left)

        Ja3_avg = 0.5f0 * (Ja3_node +
                   get_contravariant_vector(3, contravariant_vectors,
                                            i, j, m, element))
        fluxtilde3_left, _ = flux_turbo(numerical_flux,
                                        turbo_node...,
                                        get_node_turbo(turbo_local, Val(NAUX),
                                                       i, j, m)...,
                                        Ja3_avg[1], Ja3_avg[2], Ja3_avg[3],
                                        equations)
        du_local = du_local +
                   (alpha * derivative_split[k, m]) *
                   SVector{NVARIABLES}(fluxtilde3_left)
    end

    add_to_node_vars!(du, du_local, equations, dg, i, j, k, element)
end

## Version 4 without turbo: one fused unrolled loop
@inline function calc_volume_integral!(backend::Backend, du, u,
                                       mesh::Union{P4estMesh{3}, T8codeMesh{3}},
                                       have_nonconservative_terms, equations,
                                       volume_integral::VolumeIntegralFluxDifferencing,
                                       dg::DGSEM, cache, ::Val{4}, ::False)
    @unpack derivative_split = dg.basis
    @unpack contravariant_vectors = cache.elements
    NNODES = nnodes(dg)
    kernel! = version_4!(backend, (NNODES, NNODES, NNODES, 1))
    kernel!(du, u, equations,
            typeof(mesh),
            have_nonconservative_terms,
            combine_conservative_and_nonconservative_fluxes(volume_integral.volume_flux,
                                                            equations),
            dg,
            volume_integral.volume_flux,
            Val(NNODES), Val(nvariables(equations)),
            derivative_split,
            contravariant_vectors,
            ndrange = (NNODES, NNODES, NNODES, nelements(dg, cache)))
    return nothing
end

@kernel function version_4!(du, u, equations,
                            MeshT::Type{<:Union{P4estMesh{3},
                                                T8codeMesh{3}}},
                            have_nonconservative_terms::False,
                            combine_conservative_and_nonconservative_fluxes::False,
                            dg::DGSEM,
                            volume_flux,
                            ::Val{NNODES},
                            ::Val{NVARIABLES},
                            derivative_split,
                            contravariant_vectors,
                            alpha = true) where {NNODES, NVARIABLES}
    i, j, k, element = @index(Global, NTuple)

    u_local = @localmem eltype(du) (NVARIABLES, NNODES, NNODES, NNODES)

    u_node = get_node_vars(u, equations, dg, i, j, k, element)
    @inbounds for v in 1:NVARIABLES
        u_local[v, i, j, k] = u_node[v]
    end
    @synchronize

    du_local = zero(SVector{NVARIABLES, eltype(du)})

    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
    Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)
    KernelAbstractions.Extras.@unroll for m in 1:NNODES
        Ja1_avg = 0.5f0 * (Ja1_node +
                   get_contravariant_vector(1, contravariant_vectors,
                                            m, j, k, element))
        fluxtilde1 = volume_flux(u_node,
                                 get_node_flux(u_local, Val(NVARIABLES),
                                               m, j, k),
                                 Ja1_avg, equations)
        du_local = du_local + (alpha * derivative_split[i, m]) * fluxtilde1

        Ja2_avg = 0.5f0 * (Ja2_node +
                   get_contravariant_vector(2, contravariant_vectors,
                                            i, m, k, element))
        fluxtilde2 = volume_flux(u_node,
                                 get_node_flux(u_local, Val(NVARIABLES),
                                               i, m, k),
                                 Ja2_avg, equations)
        du_local = du_local + (alpha * derivative_split[j, m]) * fluxtilde2

        Ja3_avg = 0.5f0 * (Ja3_node +
                   get_contravariant_vector(3, contravariant_vectors,
                                            i, j, m, element))
        fluxtilde3 = volume_flux(u_node,
                                 get_node_flux(u_local, Val(NVARIABLES),
                                               i, j, m),
                                 Ja3_avg, equations)
        du_local = du_local + (alpha * derivative_split[k, m]) * fluxtilde3
    end

    add_to_node_vars!(du, du_local, equations, dg, i, j, k, element)
end

@kernel function version_4!(du, u, equations,
                            MeshT::Type{<:Union{P4estMesh{3},
                                                T8codeMesh{3}}},
                            have_nonconservative_terms::True,
                            combine_conservative_and_nonconservative_fluxes::True,
                            dg::DGSEM,
                            volume_flux,
                            ::Val{NNODES},
                            ::Val{NVARIABLES},
                            derivative_split,
                            contravariant_vectors,
                            alpha = true) where {NNODES, NVARIABLES}
    i, j, k, element = @index(Global, NTuple)

    u_local = @localmem eltype(du) (NVARIABLES, NNODES, NNODES, NNODES)

    u_node = get_node_vars(u, equations, dg, i, j, k, element)
    @inbounds for v in 1:NVARIABLES
        u_local[v, i, j, k] = u_node[v]
    end
    @synchronize

    du_local = zero(SVector{NVARIABLES, eltype(du)})

    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
    Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)
    KernelAbstractions.Extras.@unroll for m in 1:NNODES
        Ja1_avg = 0.5f0 * (Ja1_node +
                   get_contravariant_vector(1, contravariant_vectors,
                                            m, j, k, element))
        fluxtilde1_left, _ = volume_flux(u_node,
                                         get_node_flux(u_local, Val(NVARIABLES),
                                                       m, j, k),
                                         Ja1_avg, equations)
        du_local = du_local + (alpha * derivative_split[i, m]) * fluxtilde1_left

        Ja2_avg = 0.5f0 * (Ja2_node +
                   get_contravariant_vector(2, contravariant_vectors,
                                            i, m, k, element))
        fluxtilde2_left, _ = volume_flux(u_node,
                                         get_node_flux(u_local, Val(NVARIABLES),
                                                       i, m, k),
                                         Ja2_avg, equations)
        du_local = du_local + (alpha * derivative_split[j, m]) * fluxtilde2_left

        Ja3_avg = 0.5f0 * (Ja3_node +
                   get_contravariant_vector(3, contravariant_vectors,
                                            i, j, m, element))
        fluxtilde3_left, _ = volume_flux(u_node,
                                         get_node_flux(u_local, Val(NVARIABLES),
                                                       i, j, m),
                                         Ja3_avg, equations)
        du_local = du_local + (alpha * derivative_split[k, m]) * fluxtilde3_left
    end

    add_to_node_vars!(du, du_local, equations, dg, i, j, k, element)
end

@inline function calc_volume_integral!(backend::Backend, du, u,
                                       mesh::Union{P4estMesh{3}, T8codeMesh{3}},
                                       have_nonconservative_terms, equations,
                                       volume_integral::VolumeIntegralFluxDifferencing{<:FluxTurbo},
                                       dg::DGSEM, cache, ::Val{0}, ::True)
    @unpack derivative_split = dg.basis
    @unpack contravariant_vectors = cache.elements
    @unpack numerical_flux = volume_integral.volume_flux
    NNODES = nnodes(dg)
    kernel! = version_turbo_0!(backend, (NNODES, NNODES, NNODES, 1))
    kernel!(du, u, equations,
            typeof(mesh),
            have_nonconservative_terms,
            dg,
            numerical_flux,
            Val(NNODES), Val(nvariables(equations)),
            nturbovars(numerical_flux, equations),
            derivative_split,
            contravariant_vectors,
            ndrange = (NNODES, NNODES, NNODES, nelements(dg, cache)))
    return nothing
end

@kernel function version_turbo_0!(du, u, equations,
                                  MeshT::Type{<:Union{P4estMesh{3},
                                                      T8codeMesh{3}}},
                                  have_nonconservative_terms::False,
                                  dg::DGSEM,
                                  numerical_flux::NumericalFlux,
                                  ::Val{NNODES},
                                  ::Val{NVARIABLES},
                                  ::Val{NAUX},
                                  derivative_split,
                                  contravariant_vectors,
                                  alpha = true) where {NNODES,
                                                       NVARIABLES,
                                                       NAUX, NumericalFlux}
    i, j, k, element = @index(Global, NTuple)

    flux_local = @localmem eltype(du) (NVARIABLES, NNODES, NNODES, NNODES)
    turbo_local = @localmem eltype(du) (NAUX, NNODES, NNODES, NNODES)

    u_node = get_node_vars(u, equations, dg, i, j, k, element)
    turbo_node = cons2turbo(numerical_flux, u_node..., equations)
    @inbounds for v in 1:NAUX
        turbo_local[v, i, j, k] = turbo_node[v]
    end
    @synchronize

    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
    Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)

    @uniform half_nnodes = div(NNODES, 2)
    @uniform even_nodes = iseven(NNODES)

    du_local = zero(SVector{NVARIABLES, eltype(du)})

    KernelAbstractions.Extras.@unroll for offset in 1:half_nnodes
        # weight the antipodal pair by 1/2 only when the number of nodes is even
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        # first coordinate direction: rotate the partner index along `i`
        ii = mod(i - 1 + offset, NNODES) + 1
        # pull the contravariant vectors and compute the average
        Ja1_node_ii = get_contravariant_vector(1, contravariant_vectors,
                                               ii, j, k, element)
        Ja1_avg = 0.5f0 * (Ja1_node + Ja1_node_ii)
        # compute the contravariant volume flux in the direction of the averaged
        # contravariant vector, using the precomputed variables of both nodes
        fluxtilde1 = flux_turbo(numerical_flux,
                                get_node_turbo(turbo_local,
                                               Val(NAUX),
                                               i, j, k)...,
                                get_node_turbo(turbo_local,
                                               Val(NAUX),
                                               ii, j, k)...,
                                Ja1_avg[1], Ja1_avg[2],
                                Ja1_avg[3],
                                equations)

        @inbounds for v in 1:NVARIABLES
            flux_local[v, i, j, k] = fluxtilde1[v]
        end
        @synchronize
        iib = mod(i - 1 - offset, NNODES) + 1
        du_local = du_local +
                   (weight * alpha * derivative_split[i, ii]) *
                   SVector{NVARIABLES}(fluxtilde1) +
                   (weight * alpha * derivative_split[i, iib]) *
                   get_node_flux(flux_local, Val(NVARIABLES), iib, j, k)
        @synchronize
    end

    KernelAbstractions.Extras.@unroll for offset in 1:half_nnodes
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        # second coordinate direction: rotate the partner index along `j`
        jj = mod(j - 1 + offset, NNODES) + 1
        Ja2_node_jj = get_contravariant_vector(2, contravariant_vectors,
                                               i, jj, k, element)
        Ja2_avg = 0.5f0 * (Ja2_node + Ja2_node_jj)
        fluxtilde2 = flux_turbo(numerical_flux,
                                get_node_turbo(turbo_local,
                                               Val(NAUX),
                                               i, j, k)...,
                                get_node_turbo(turbo_local,
                                               Val(NAUX),
                                               i, jj, k)...,
                                Ja2_avg[1], Ja2_avg[2],
                                Ja2_avg[3],
                                equations)

        @inbounds for v in 1:NVARIABLES
            flux_local[v, i, j, k] = fluxtilde2[v]
        end
        @synchronize
        jjb = mod(j - 1 - offset, NNODES) + 1
        du_local = du_local +
                   (weight * alpha * derivative_split[j, jj]) *
                   SVector{NVARIABLES}(fluxtilde2) +
                   (weight * alpha * derivative_split[j, jjb]) *
                   get_node_flux(flux_local, Val(NVARIABLES), i, jjb, k)
        @synchronize
    end

    KernelAbstractions.Extras.@unroll for offset in 1:half_nnodes
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        # third coordinate direction: rotate the partner index along `k`
        kk = mod(k - 1 + offset, NNODES) + 1
        Ja3_node_kk = get_contravariant_vector(3, contravariant_vectors,
                                               i, j, kk, element)
        Ja3_avg = 0.5f0 * (Ja3_node + Ja3_node_kk)
        fluxtilde3 = flux_turbo(numerical_flux,
                                get_node_turbo(turbo_local,
                                               Val(NAUX),
                                               i, j, k)...,
                                get_node_turbo(turbo_local,
                                               Val(NAUX),
                                               i, j, kk)...,
                                Ja3_avg[1], Ja3_avg[2],
                                Ja3_avg[3],
                                equations)

        @inbounds for v in 1:NVARIABLES
            flux_local[v, i, j, k] = fluxtilde3[v]
        end
        @synchronize
        kkb = mod(k - 1 - offset, NNODES) + 1
        du_local = du_local +
                   (weight * alpha * derivative_split[k, kk]) *
                   SVector{NVARIABLES}(fluxtilde3) +
                   (weight * alpha * derivative_split[k, kkb]) *
                   get_node_flux(flux_local, Val(NVARIABLES), i, j, kkb)
        @synchronize
    end

    add_to_node_vars!(du, du_local, equations, dg, i, j, k, element)
end

@kernel function version_turbo_0!(du, u, equations,
                                  MeshT::Type{<:Union{P4estMesh{3},
                                                      T8codeMesh{3}}},
                                  have_nonconservative_terms::True,
                                  dg::DGSEM,
                                  numerical_flux::NumericalFlux,
                                  ::Val{NNODES},
                                  ::Val{NVARIABLES},
                                  ::Val{NAUX},
                                  derivative_split,
                                  contravariant_vectors,
                                  alpha = true) where {NNODES,
                                                       NVARIABLES,
                                                       NAUX, NumericalFlux}
    i, j, k, element = @index(Global, NTuple)

    flux_local = @localmem eltype(du) (NVARIABLES, NNODES, NNODES, NNODES)
    turbo_local = @localmem eltype(du) (NAUX, NNODES, NNODES, NNODES)

    u_node = get_node_vars(u, equations, dg, i, j, k, element)
    turbo_node = cons2turbo(numerical_flux, u_node..., equations)
    @inbounds for v in 1:NAUX
        turbo_local[v, i, j, k] = turbo_node[v]
    end
    @synchronize

    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
    Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)

    @uniform half_nnodes = div(NNODES, 2)
    @uniform even_nodes = iseven(NNODES)

    du_local = zero(SVector{NVARIABLES, eltype(du)})

    KernelAbstractions.Extras.@unroll for offset in 1:half_nnodes
        # weight the antipodal pair by 1/2 only when the number of nodes is even
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        # first coordinate direction: rotate the partner index along `i`
        ii = mod(i - 1 + offset, NNODES) + 1
        # pull the contravariant vectors and compute the average
        Ja1_node_ii = get_contravariant_vector(1, contravariant_vectors,
                                               ii, j, k, element)
        Ja1_avg = 0.5f0 * (Ja1_node + Ja1_node_ii)
        # compute the contravariant volume flux in the direction of the averaged
        # contravariant vector, using the precomputed variables of both nodes
        fluxtilde1_left, fluxtilde1_right = flux_turbo(numerical_flux,
                                                       get_node_turbo(turbo_local,
                                                                      Val(NAUX),
                                                                      i, j, k)...,
                                                       get_node_turbo(turbo_local,
                                                                      Val(NAUX),
                                                                      ii, j, k)...,
                                                       Ja1_avg[1], Ja1_avg[2],
                                                       Ja1_avg[3],
                                                       equations)

        @inbounds for v in 1:NVARIABLES
            flux_local[v, i, j, k] = fluxtilde1_right[v]
        end
        @synchronize
        iib = mod(i - 1 - offset, NNODES) + 1
        du_local = du_local +
                   (weight * alpha * derivative_split[i, ii]) *
                   SVector{NVARIABLES}(fluxtilde1_left) +
                   (weight * alpha * derivative_split[i, iib]) *
                   get_node_flux(flux_local, Val(NVARIABLES), iib, j, k)
        @synchronize
    end

    KernelAbstractions.Extras.@unroll for offset in 1:half_nnodes
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        # second coordinate direction: rotate the partner index along `j`
        jj = mod(j - 1 + offset, NNODES) + 1
        Ja2_node_jj = get_contravariant_vector(2, contravariant_vectors,
                                               i, jj, k, element)
        Ja2_avg = 0.5f0 * (Ja2_node + Ja2_node_jj)
        fluxtilde2_left, fluxtilde2_right = flux_turbo(numerical_flux,
                                                       get_node_turbo(turbo_local,
                                                                      Val(NAUX),
                                                                      i, j, k)...,
                                                       get_node_turbo(turbo_local,
                                                                      Val(NAUX),
                                                                      i, jj, k)...,
                                                       Ja2_avg[1], Ja2_avg[2],
                                                       Ja2_avg[3],
                                                       equations)

        @inbounds for v in 1:NVARIABLES
            flux_local[v, i, j, k] = fluxtilde2_right[v]
        end
        @synchronize
        jjb = mod(j - 1 - offset, NNODES) + 1
        du_local = du_local +
                   (weight * alpha * derivative_split[j, jj]) *
                   SVector{NVARIABLES}(fluxtilde2_left) +
                   (weight * alpha * derivative_split[j, jjb]) *
                   get_node_flux(flux_local, Val(NVARIABLES), i, jjb, k)
        @synchronize
    end

    KernelAbstractions.Extras.@unroll for offset in 1:half_nnodes
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        # third coordinate direction: rotate the partner index along `k`
        kk = mod(k - 1 + offset, NNODES) + 1
        Ja3_node_kk = get_contravariant_vector(3, contravariant_vectors,
                                               i, j, kk, element)
        Ja3_avg = 0.5f0 * (Ja3_node + Ja3_node_kk)
        fluxtilde3_left, fluxtilde3_right = flux_turbo(numerical_flux,
                                                       get_node_turbo(turbo_local,
                                                                      Val(NAUX),
                                                                      i, j, k)...,
                                                       get_node_turbo(turbo_local,
                                                                      Val(NAUX),
                                                                      i, j, kk)...,
                                                       Ja3_avg[1], Ja3_avg[2],
                                                       Ja3_avg[3],
                                                       equations)

        @inbounds for v in 1:NVARIABLES
            flux_local[v, i, j, k] = fluxtilde3_right[v]
        end
        @synchronize
        kkb = mod(k - 1 - offset, NNODES) + 1
        du_local = du_local +
                   (weight * alpha * derivative_split[k, kk]) *
                   SVector{NVARIABLES}(fluxtilde3_left) +
                   (weight * alpha * derivative_split[k, kkb]) *
                   get_node_flux(flux_local, Val(NVARIABLES), i, j, kkb)
        @synchronize
    end

    add_to_node_vars!(du, du_local, equations, dg, i, j, k, element)
end

@inline function calc_volume_integral!(backend::Backend, du, u,
                                       mesh::Union{P4estMesh{3}, T8codeMesh{3}},
                                       have_nonconservative_terms, equations,
                                       volume_integral::VolumeIntegralFluxDifferencing,
                                       dg::DGSEM, cache, ::Val{0}, ::False)
    @unpack derivative_split = dg.basis
    @unpack contravariant_vectors = cache.elements
    NNODES = nnodes(dg)
    kernel! = version_0!(backend, (NNODES, NNODES, NNODES, 1))
    kernel!(du, u, equations,
            typeof(mesh),
            have_nonconservative_terms,
            combine_conservative_and_nonconservative_fluxes(volume_integral.volume_flux,
                                                            equations),
            dg,
            volume_integral.volume_flux,
            Val(NNODES), Val(nvariables(equations)),
            derivative_split,
            contravariant_vectors,
            ndrange = (NNODES, NNODES, NNODES, nelements(dg, cache)))
    return nothing
end

@kernel function version_0!(du, u, equations,
                            MeshT::Type{<:Union{P4estMesh{3},
                                                T8codeMesh{3}}},
                            have_nonconservative_terms::False,
                            combine_conservative_and_nonconservative_fluxes::False,
                            dg::DGSEM,
                            volume_flux,
                            ::Val{NNODES},
                            ::Val{NVARIABLES},
                            derivative_split,
                            contravariant_vectors,
                            alpha = true) where {NNODES, NVARIABLES}
    i, j, k, element = @index(Global, NTuple)

    flux_local = @localmem eltype(du) (NVARIABLES, NNODES, NNODES,
                                       NNODES)

    u_node = get_node_vars(u, equations, dg, i, j, k, element)

    # pull the contravariant vectors in each coordinate direction
    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k,
                                        element)
    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k,
                                        element)
    Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k,
                                        element)

    # All diagonal entries of `derivative_split` are zero. Thus, we can skip
    # the computation of the diagonal terms. In addition, we use the symmetry
    # of the `volume_flux` to save half of the possible two-point flux
    # computations.
    #
    # Instead of assigning thread i the partners i+1, …, N,
    # we distribute the half-sweep cyclically: each thread visits
    # half = div(N,2) partners at a fixed rotating offset.
    # Every unordered pair is still covered exactly
    # once, but now every thread performs the same number of loop iterations.
    # When N is even (odd polynomial degree) the antipodal pair at
    # offset half is shared by two threads, so its contribution is weighted by
    # 1/2 to avoid double counting.
    #
    # See Section 4.1 (Eq. 6) of
    # - Waterhouse, Waruszewski, Wilcox, Giraldo (2026)
    #   GPU Performance of an Entropy-Stable Discontinuous Galerkin Euler Solver
    #   with Non-Conservative Terms.
    #   arXiv (pre-print): https://arxiv.org/abs/2605.16684

    @uniform half_nnodes = div(NNODES, 2)
    @uniform even_nodes = iseven(NNODES)

    du_local = zero(SVector{NVARIABLES, eltype(du)})

    KernelAbstractions.Extras.@unroll for offset in 1:half_nnodes
        # weight the antipodal pair by 1/2 only when the number of nodes is even
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        # first coordinate direction: rotate the partner index along `i`
        ii = mod(i - 1 + offset, NNODES) + 1
        u_node_ii = get_node_vars(u, equations, dg, ii, j, k, element)
        # pull the contravariant vectors and compute the average
        Ja1_node_ii = get_contravariant_vector(1, contravariant_vectors,
                                               ii, j, k, element)
        Ja1_avg = 0.5f0 * (Ja1_node + Ja1_node_ii)
        # compute the contravariant volume flux in the direction of the
        # averaged contravariant vector
        fluxtilde1 = volume_flux(u_node, u_node_ii, Ja1_avg, equations)

        @inbounds for v in 1:NVARIABLES
            flux_local[v, i, j, k] = fluxtilde1[v]
        end

        @synchronize
        iib = mod(i - 1 - offset, NNODES) + 1
        du_local = du_local + (weight * alpha * derivative_split[i, ii]) * fluxtilde1 +
                   (weight * alpha * derivative_split[i, iib]) *
                   get_node_flux(flux_local, Val(NVARIABLES), iib, j, k)
        @synchronize
    end

    KernelAbstractions.Extras.@unroll for offset in 1:half_nnodes
        # weight the antipodal pair by 1/2 only when the number of nodes is even
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        # second coordinate direction: rotate the partner index along `j`
        jj = mod(j - 1 + offset, NNODES) + 1
        u_node_jj = get_node_vars(u, equations, dg, i, jj, k, element)
        # pull the contravariant vectors and compute the average
        Ja2_node_jj = get_contravariant_vector(2, contravariant_vectors,
                                               i, jj, k, element)
        Ja2_avg = 0.5f0 * (Ja2_node + Ja2_node_jj)
        # compute the contravariant volume flux in the direction of the
        # averaged contravariant vector
        fluxtilde2 = volume_flux(u_node, u_node_jj, Ja2_avg, equations)
        @inbounds for v in 1:NVARIABLES
            flux_local[v, i, j, k] = fluxtilde2[v]
        end
        @synchronize
        jjb = mod(j - 1 - offset, NNODES) + 1
        du_local = du_local + (weight * alpha * derivative_split[j, jj]) * fluxtilde2 +
                   (weight * alpha * derivative_split[j, jjb]) *
                   get_node_flux(flux_local, Val(NVARIABLES), i, jjb, k)
        @synchronize
    end

    KernelAbstractions.Extras.@unroll for offset in 1:half_nnodes
        # weight the antipodal pair by 1/2 only when the number of nodes is even
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        # third coordinate direction: rotate the partner index along `k`
        kk = mod(k - 1 + offset, NNODES) + 1
        u_node_kk = get_node_vars(u, equations, dg, i, j, kk, element)
        # pull the contravariant vectors and compute the average
        Ja3_node_kk = get_contravariant_vector(3, contravariant_vectors,
                                               i, j, kk, element)
        Ja3_avg = 0.5f0 * (Ja3_node + Ja3_node_kk)
        # compute the contravariant volume flux in the direction of the
        # averaged contravariant vector
        fluxtilde3 = volume_flux(u_node, u_node_kk, Ja3_avg, equations)
        @inbounds for v in 1:NVARIABLES
            flux_local[v, i, j, k] = fluxtilde3[v]
        end
        @synchronize
        kkb = mod(k - 1 - offset, NNODES) + 1
        du_local = du_local + (weight * alpha * derivative_split[k, kk]) * fluxtilde3 +
                   (weight * alpha * derivative_split[k, kkb]) *
                   get_node_flux(flux_local, Val(NVARIABLES), i, j, kkb)
        @synchronize
    end

    add_to_node_vars!(du, du_local, equations, dg, i, j, k, element)
end

@kernel function version_0!(du, u, equations,
                            MeshT::Type{<:Union{P4estMesh{3},
                                                T8codeMesh{3}}},
                            have_nonconservative_terms::True,
                            combine_conservative_and_nonconservative_fluxes::True,
                            dg::DGSEM,
                            volume_flux,
                            ::Val{NNODES},
                            ::Val{NVARIABLES},
                            derivative_split,
                            contravariant_vectors,
                            alpha = true) where {NNODES, NVARIABLES}
    i, j, k, element = @index(Global, NTuple)

    flux_local = @localmem eltype(du) (NVARIABLES, NNODES, NNODES,
                                       NNODES)

    u_node = get_node_vars(u, equations, dg, i, j, k, element)

    # pull the contravariant vectors in each coordinate direction
    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k,
                                        element)
    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k,
                                        element)
    Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k,
                                        element)

    # All diagonal entries of `derivative_split` are zero. Thus, we can skip
    # the computation of the diagonal terms. In addition, we use the symmetry
    # of the `volume_flux` to save half of the possible two-point flux
    # computations.
    #
    # Instead of assigning thread i the partners i+1, …, N,
    # we distribute the half-sweep cyclically: each thread visits
    # half = div(N,2) partners at a fixed rotating offset.
    # Every unordered pair is still covered exactly
    # once, but now every thread performs the same number of loop iterations.
    # When N is even (odd polynomial degree) the antipodal pair at
    # offset half is shared by two threads, so its contribution is weighted by
    # 1/2 to avoid double counting.
    #
    # See Section 4.1 (Eq. 6) of
    # - Waterhouse, Waruszewski, Wilcox, Giraldo (2026)
    #   GPU Performance of an Entropy-Stable Discontinuous Galerkin Euler Solver
    #   with Non-Conservative Terms.
    #   arXiv (pre-print): https://arxiv.org/abs/2605.16684

    @uniform half_nnodes = div(NNODES, 2)
    @uniform even_nodes = iseven(NNODES)

    du_local = zero(SVector{NVARIABLES, eltype(du)})

    KernelAbstractions.Extras.@unroll for offset in 1:half_nnodes
        # weight the antipodal pair by 1/2 only when the number of nodes is even
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        # first coordinate direction: rotate the partner index along `i`
        ii = mod(i - 1 + offset, NNODES) + 1
        u_node_ii = get_node_vars(u, equations, dg, ii, j, k, element)
        # pull the contravariant vectors and compute the average
        Ja1_node_ii = get_contravariant_vector(1, contravariant_vectors,
                                               ii, j, k, element)

        Ja1_avg = 0.5f0 * (Ja1_node + Ja1_node_ii)
        # compute the contravariant volume flux in the direction of the
        # averaged contravariant vector
        fluxtilde1_left, fluxtilde1_right = volume_flux(u_node, u_node_ii, Ja1_avg,
                                                        equations)

        @inbounds for v in 1:NVARIABLES
            flux_local[v, i, j, k] = fluxtilde1_right[v]
        end
        @synchronize
        iib = mod(i - 1 - offset, NNODES) + 1
        du_local = du_local +
                   (weight * alpha * derivative_split[i, ii]) * fluxtilde1_left +
                   (weight * alpha * derivative_split[i, iib]) *
                   get_node_flux(flux_local, Val(NVARIABLES), iib, j, k)
        @synchronize
    end

    KernelAbstractions.Extras.@unroll for offset in 1:half_nnodes
        # weight the antipodal pair by 1/2 only when the number of nodes is even
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        # second coordinate direction: rotate the partner index along `j`
        jj = mod(j - 1 + offset, NNODES) + 1
        u_node_jj = get_node_vars(u, equations, dg, i, jj, k, element)
        # pull the contravariant vectors and compute the average
        Ja2_node_jj = get_contravariant_vector(2, contravariant_vectors,
                                               i, jj, k, element)
        Ja2_avg = 0.5f0 * (Ja2_node + Ja2_node_jj)
        # compute the contravariant volume flux in the direction of the
        # averaged contravariant vector
        fluxtilde2_left, fluxtilde2_right = volume_flux(u_node, u_node_jj, Ja2_avg,
                                                        equations)
        @inbounds for v in 1:NVARIABLES
            flux_local[v, i, j, k] = fluxtilde2_right[v]
        end
        @synchronize
        jjb = mod(j - 1 - offset, NNODES) + 1
        du_local = du_local +
                   (weight * alpha * derivative_split[j, jj]) * fluxtilde2_left +
                   (weight * alpha * derivative_split[j, jjb]) *
                   get_node_flux(flux_local, Val(NVARIABLES), i, jjb, k)
        @synchronize
    end

    KernelAbstractions.Extras.@unroll for offset in 1:half_nnodes
        # weight the antipodal pair by 1/2 only when the number of nodes is even
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        # third coordinate direction: rotate the partner index along `k`
        kk = mod(k - 1 + offset, NNODES) + 1
        u_node_kk = get_node_vars(u, equations, dg, i, j, kk, element)
        # pull the contravariant vectors and compute the average
        Ja3_node_kk = get_contravariant_vector(3, contravariant_vectors,
                                               i, j, kk, element)
        Ja3_avg = 0.5f0 * (Ja3_node + Ja3_node_kk)
        # compute the contravariant volume flux in the direction of the
        # averaged contravariant vector
        fluxtilde3_left, fluxtilde3_right = volume_flux(u_node, u_node_kk, Ja3_avg,
                                                        equations)
        @inbounds for v in 1:NVARIABLES
            flux_local[v, i, j, k] = fluxtilde3_right[v]
        end
        @synchronize
        kkb = mod(k - 1 - offset, NNODES) + 1
        du_local = du_local +
                   (weight * alpha * derivative_split[k, kk]) * fluxtilde3_left +
                   (weight * alpha * derivative_split[k, kkb]) *
                   get_node_flux(flux_local, Val(NVARIABLES), i, j, kkb)
        @synchronize
    end

    add_to_node_vars!(du, du_local, equations, dg, i, j, k, element)
end

@kernel function version_turbo_5!(du, u, equations,
                                  MeshT::Type{<:Union{P4estMesh{3},
                                                      T8codeMesh{3}}},
                                  have_nonconservative_terms::False,
                                  dg::DGSEM,
                                  numerical_flux::NumericalFlux,
                                  ::Val{NNODES},
                                  ::Val{NVARIABLES},
                                  ::Val{NAUX},
                                  derivative_split,
                                  contravariant_vectors,
                                  alpha = true) where {NNODES,
                                                       NVARIABLES,
                                                       NAUX, NumericalFlux}
    i, j, k, element = @index(Global, NTuple)

    flux_local = @localmem eltype(du) (NVARIABLES, NNODES, NNODES, NNODES)
    turbo_local = @localmem eltype(du) (NAUX, NNODES, NNODES, NNODES)

    # Convert the conserved variables of this node once into the precomputed variables and
    # share them with the whole workgroup, which handles exactly one element.
    u_node = get_node_vars(u, equations, dg, i, j, k, element)
    turbo_node = cons2turbo(numerical_flux, u_node..., equations)
    @inbounds for v in 1:NAUX
        turbo_local[v, i, j, k] = turbo_node[v]
    end
    @synchronize

    # pull the contravariant vectors in each coordinate direction
    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
    Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)

    # The half-sweep over the two-point fluxes is distributed cyclically exactly as in
    # `flux_differencing_KAkernel!`, see the comment there.
    @uniform half_nnodes = div(NNODES, 2)
    @uniform even_nodes = iseven(NNODES)

    du_local = zero(SVector{NVARIABLES, eltype(du)})

    for offset in 1:half_nnodes
        # weight the antipodal pair by 1/2 only when the number of nodes is even
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        # first coordinate direction: rotate the partner index along `i`
        ii = mod(i - 1 + offset, NNODES) + 1
        # pull the contravariant vectors and compute the average
        Ja1_node_ii = get_contravariant_vector(1, contravariant_vectors,
                                               ii, j, k, element)
        Ja1_avg = 0.5f0 * (Ja1_node + Ja1_node_ii)
        # compute the contravariant volume flux in the direction of the averaged
        # contravariant vector, using the precomputed variables of both nodes
        fluxtilde1 = flux_turbo(numerical_flux,
                                get_node_turbo(turbo_local,
                                               Val(NAUX),
                                               i, j, k)...,
                                get_node_turbo(turbo_local,
                                               Val(NAUX),
                                               ii, j, k)...,
                                Ja1_avg[1], Ja1_avg[2],
                                Ja1_avg[3],
                                equations)

        @inbounds for v in 1:NVARIABLES
            flux_local[v, i, j, k] = fluxtilde1[v]
        end
        du_local = du_local +
                   (weight * alpha * derivative_split[i, ii]) *
                   SVector{NVARIABLES}(fluxtilde1)
        @synchronize
        iib = mod(i - 1 - offset, NNODES) + 1
        du_local = du_local +
                   (weight * alpha * derivative_split[i, iib]) *
                   get_node_flux(flux_local, Val(NVARIABLES), iib, j, k)
        @synchronize
    end

    for offset in 1:half_nnodes
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        # second coordinate direction: rotate the partner index along `j`
        jj = mod(j - 1 + offset, NNODES) + 1
        Ja2_node_jj = get_contravariant_vector(2, contravariant_vectors,
                                               i, jj, k, element)
        Ja2_avg = 0.5f0 * (Ja2_node + Ja2_node_jj)
        fluxtilde2 = flux_turbo(numerical_flux,
                                get_node_turbo(turbo_local,
                                               Val(NAUX),
                                               i, j, k)...,
                                get_node_turbo(turbo_local,
                                               Val(NAUX),
                                               i, jj, k)...,
                                Ja2_avg[1], Ja2_avg[2],
                                Ja2_avg[3],
                                equations)

        @inbounds for v in 1:NVARIABLES
            flux_local[v, i, j, k] = fluxtilde2[v]
        end
        du_local = du_local +
                   (weight * alpha * derivative_split[j, jj]) *
                   SVector{NVARIABLES}(fluxtilde2)
        @synchronize
        jjb = mod(j - 1 - offset, NNODES) + 1
        du_local = du_local +
                   (weight * alpha * derivative_split[j, jjb]) *
                   get_node_flux(flux_local, Val(NVARIABLES), i, jjb, k)
        @synchronize
    end

    for offset in 1:half_nnodes
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        # third coordinate direction: rotate the partner index along `k`
        kk = mod(k - 1 + offset, NNODES) + 1
        Ja3_node_kk = get_contravariant_vector(3, contravariant_vectors,
                                               i, j, kk, element)
        Ja3_avg = 0.5f0 * (Ja3_node + Ja3_node_kk)
        fluxtilde3 = flux_turbo(numerical_flux,
                                get_node_turbo(turbo_local,
                                               Val(NAUX),
                                               i, j, k)...,
                                get_node_turbo(turbo_local,
                                               Val(NAUX),
                                               i, j, kk)...,
                                Ja3_avg[1], Ja3_avg[2],
                                Ja3_avg[3],
                                equations)

        @inbounds for v in 1:NVARIABLES
            flux_local[v, i, j, k] = fluxtilde3[v]
        end
        du_local = du_local +
                   (weight * alpha * derivative_split[k, kk]) *
                   SVector{NVARIABLES}(fluxtilde3)
        @synchronize
        kkb = mod(k - 1 - offset, NNODES) + 1
        du_local = du_local +
                   (weight * alpha * derivative_split[k, kkb]) *
                   get_node_flux(flux_local, Val(NVARIABLES), i, j, kkb)
        @synchronize
    end

    add_to_node_vars!(du, du_local, equations, dg, i, j, k, element)
end

@kernel function version_turbo_5!(du, u, equations,
                                  MeshT::Type{<:Union{P4estMesh{3},
                                                      T8codeMesh{3}}},
                                  have_nonconservative_terms::True,
                                  dg::DGSEM,
                                  numerical_flux::NumericalFlux,
                                  ::Val{NNODES},
                                  ::Val{NVARIABLES},
                                  ::Val{NAUX},
                                  derivative_split,
                                  contravariant_vectors,
                                  alpha = true) where {NNODES,
                                                       NVARIABLES,
                                                       NAUX, NumericalFlux}
    i, j, k, element = @index(Global, NTuple)

    flux_local = @localmem eltype(du) (NVARIABLES, NNODES, NNODES, NNODES)
    turbo_local = @localmem eltype(du) (NAUX, NNODES, NNODES, NNODES)

    u_node = get_node_vars(u, equations, dg, i, j, k, element)
    turbo_node = cons2turbo(numerical_flux, u_node..., equations)
    @inbounds for v in 1:NAUX
        turbo_local[v, i, j, k] = turbo_node[v]
    end
    @synchronize

    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
    Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)

    @uniform half_nnodes = div(NNODES, 2)
    @uniform even_nodes = iseven(NNODES)

    du_local = zero(SVector{NVARIABLES, eltype(du)})

    for offset in 1:half_nnodes
        # weight the antipodal pair by 1/2 only when the number of nodes is even
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        # first coordinate direction: rotate the partner index along `i`
        ii = mod(i - 1 + offset, NNODES) + 1
        # pull the contravariant vectors and compute the average
        Ja1_node_ii = get_contravariant_vector(1, contravariant_vectors,
                                               ii, j, k, element)
        Ja1_avg = 0.5f0 * (Ja1_node + Ja1_node_ii)
        # compute the contravariant volume flux in the direction of the averaged
        # contravariant vector, using the precomputed variables of both nodes
        fluxtilde1_left, fluxtilde1_right = flux_turbo(numerical_flux,
                                                       get_node_turbo(turbo_local,
                                                                      Val(NAUX),
                                                                      i, j, k)...,
                                                       get_node_turbo(turbo_local,
                                                                      Val(NAUX),
                                                                      ii, j, k)...,
                                                       Ja1_avg[1], Ja1_avg[2],
                                                       Ja1_avg[3],
                                                       equations)

        @inbounds for v in 1:NVARIABLES
            flux_local[v, i, j, k] = fluxtilde1_right[v]
        end
        du_local = du_local +
                   (weight * alpha * derivative_split[i, ii]) *
                   SVector{NVARIABLES}(fluxtilde1_left)
        @synchronize
        iib = mod(i - 1 - offset, NNODES) + 1
        du_local = du_local +
                   (weight * alpha * derivative_split[i, iib]) *
                   get_node_flux(flux_local, Val(NVARIABLES), iib, j, k)
        @synchronize
    end

    for offset in 1:half_nnodes
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        # second coordinate direction: rotate the partner index along `j`
        jj = mod(j - 1 + offset, NNODES) + 1
        Ja2_node_jj = get_contravariant_vector(2, contravariant_vectors,
                                               i, jj, k, element)
        Ja2_avg = 0.5f0 * (Ja2_node + Ja2_node_jj)
        fluxtilde2_left, fluxtilde2_right = flux_turbo(numerical_flux,
                                                       get_node_turbo(turbo_local,
                                                                      Val(NAUX),
                                                                      i, j, k)...,
                                                       get_node_turbo(turbo_local,
                                                                      Val(NAUX),
                                                                      i, jj, k)...,
                                                       Ja2_avg[1], Ja2_avg[2],
                                                       Ja2_avg[3],
                                                       equations)

        @inbounds for v in 1:NVARIABLES
            flux_local[v, i, j, k] = fluxtilde2_right[v]
        end
        du_local = du_local +
                   (weight * alpha * derivative_split[j, jj]) *
                   SVector{NVARIABLES}(fluxtilde2_left)
        @synchronize
        jjb = mod(j - 1 - offset, NNODES) + 1
        du_local = du_local +
                   (weight * alpha * derivative_split[j, jjb]) *
                   get_node_flux(flux_local, Val(NVARIABLES), i, jjb, k)
        @synchronize
    end

    for offset in 1:half_nnodes
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        # third coordinate direction: rotate the partner index along `k`
        kk = mod(k - 1 + offset, NNODES) + 1
        Ja3_node_kk = get_contravariant_vector(3, contravariant_vectors,
                                               i, j, kk, element)
        Ja3_avg = 0.5f0 * (Ja3_node + Ja3_node_kk)
        fluxtilde3_left, fluxtilde3_right = flux_turbo(numerical_flux,
                                                       get_node_turbo(turbo_local,
                                                                      Val(NAUX),
                                                                      i, j, k)...,
                                                       get_node_turbo(turbo_local,
                                                                      Val(NAUX),
                                                                      i, j, kk)...,
                                                       Ja3_avg[1], Ja3_avg[2],
                                                       Ja3_avg[3],
                                                       equations)

        @inbounds for v in 1:NVARIABLES
            flux_local[v, i, j, k] = fluxtilde3_right[v]
        end
        du_local = du_local +
                   (weight * alpha * derivative_split[k, kk]) *
                   SVector{NVARIABLES}(fluxtilde3_left)
        @synchronize
        kkb = mod(k - 1 - offset, NNODES) + 1
        du_local = du_local +
                   (weight * alpha * derivative_split[k, kkb]) *
                   get_node_flux(flux_local, Val(NVARIABLES), i, j, kkb)
        @synchronize
    end

    add_to_node_vars!(du, du_local, equations, dg, i, j, k, element)
end

@kernel function version_5!(du, u, equations,
                            MeshT::Type{<:Union{P4estMesh{3},
                                                T8codeMesh{3}}},
                            have_nonconservative_terms::False,
                            combine_conservative_and_nonconservative_fluxes::False,
                            dg::DGSEM,
                            volume_flux,
                            ::Val{NNODES},
                            ::Val{NVARIABLES},
                            derivative_split,
                            contravariant_vectors,
                            alpha = true) where {NNODES, NVARIABLES}
    # `true * [some floating point value] == [exactly the same floating point value]`
    # This can (hopefully) be optimized away due to constant propagation.
    i, j, k, element = @index(Global, NTuple)

    flux_local = @localmem eltype(du) (NVARIABLES, NNODES, NNODES,
                                       NNODES)

    u_node = get_node_vars(u, equations, dg, i, j, k, element)

    # pull the contravariant vectors in each coordinate direction
    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k,
                                        element)
    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k,
                                        element)
    Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k,
                                        element)

    # All diagonal entries of `derivative_split` are zero. Thus, we can skip
    # the computation of the diagonal terms. In addition, we use the symmetry
    # of the `volume_flux` to save half of the possible two-point flux
    # computations.
    #
    # Instead of assigning thread i the partners i+1, …, N,
    # we distribute the half-sweep cyclically: each thread visits
    # half = div(N,2) partners at a fixed rotating offset.
    # Every unordered pair is still covered exactly
    # once, but now every thread performs the same number of loop iterations.
    # When N is even (odd polynomial degree) the antipodal pair at
    # offset half is shared by two threads, so its contribution is weighted by
    # 1/2 to avoid double counting.
    #
    # See Section 4.1 (Eq. 6) of
    # - Waterhouse, Waruszewski, Wilcox, Giraldo (2026)
    #   GPU Performance of an Entropy-Stable Discontinuous Galerkin Euler Solver
    #   with Non-Conservative Terms.
    #   arXiv (pre-print): https://arxiv.org/abs/2605.16684

    @uniform half_nnodes = div(NNODES, 2)
    @uniform even_nodes = iseven(NNODES)

    du_local = zero(SVector{NVARIABLES, eltype(du)})

    for offset in 1:half_nnodes
        # weight the antipodal pair by 1/2 only when the number of nodes is even
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        # first coordinate direction: rotate the partner index along `i`
        ii = mod(i - 1 + offset, NNODES) + 1
        u_node_ii = get_node_vars(u, equations, dg, ii, j, k, element)
        # pull the contravariant vectors and compute the average
        Ja1_node_ii = get_contravariant_vector(1, contravariant_vectors,
                                               ii, j, k, element)
        Ja1_avg = 0.5f0 * (Ja1_node + Ja1_node_ii)
        # compute the contravariant volume flux in the direction of the
        # averaged contravariant vector
        fluxtilde1 = volume_flux(u_node, u_node_ii, Ja1_avg, equations)

        @inbounds for v in 1:NVARIABLES
            flux_local[v, i, j, k] = fluxtilde1[v]
        end

        du_local = du_local +
                   (weight * alpha * derivative_split[i, ii]) * fluxtilde1
        @synchronize
        iib = mod(i - 1 - offset, NNODES) + 1
        du_local = du_local +
                   (weight * alpha * derivative_split[i, iib]) *
                   get_node_flux(flux_local, Val(NVARIABLES), iib, j, k)
        @synchronize
    end

    for offset in 1:half_nnodes
        # weight the antipodal pair by 1/2 only when the number of nodes is even
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        # second coordinate direction: rotate the partner index along `j`
        jj = mod(j - 1 + offset, NNODES) + 1
        u_node_jj = get_node_vars(u, equations, dg, i, jj, k, element)
        # pull the contravariant vectors and compute the average
        Ja2_node_jj = get_contravariant_vector(2, contravariant_vectors,
                                               i, jj, k, element)
        Ja2_avg = 0.5f0 * (Ja2_node + Ja2_node_jj)
        # compute the contravariant volume flux in the direction of the
        # averaged contravariant vector
        fluxtilde2 = volume_flux(u_node, u_node_jj, Ja2_avg, equations)
        @inbounds for v in 1:NVARIABLES
            flux_local[v, i, j, k] = fluxtilde2[v]
        end
        du_local = du_local +
                   (weight * alpha * derivative_split[j, jj]) * fluxtilde2
        @synchronize
        jjb = mod(j - 1 - offset, NNODES) + 1
        du_local = du_local +
                   (weight * alpha * derivative_split[j, jjb]) *
                   get_node_flux(flux_local, Val(NVARIABLES), i, jjb, k)
        @synchronize
    end

    for offset in 1:half_nnodes
        # weight the antipodal pair by 1/2 only when the number of nodes is even
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        # third coordinate direction: rotate the partner index along `k`
        kk = mod(k - 1 + offset, NNODES) + 1
        u_node_kk = get_node_vars(u, equations, dg, i, j, kk, element)
        # pull the contravariant vectors and compute the average
        Ja3_node_kk = get_contravariant_vector(3, contravariant_vectors,
                                               i, j, kk, element)
        Ja3_avg = 0.5f0 * (Ja3_node + Ja3_node_kk)
        # compute the contravariant volume flux in the direction of the
        # averaged contravariant vector
        fluxtilde3 = volume_flux(u_node, u_node_kk, Ja3_avg, equations)
        @inbounds for v in 1:NVARIABLES
            flux_local[v, i, j, k] = fluxtilde3[v]
        end
        du_local = du_local +
                   (weight * alpha * derivative_split[k, kk]) * fluxtilde3
        @synchronize
        kkb = mod(k - 1 - offset, NNODES) + 1
        du_local = du_local +
                   (weight * alpha * derivative_split[k, kkb]) *
                   get_node_flux(flux_local, Val(NVARIABLES), i, j, kkb)
        @synchronize
    end

    add_to_node_vars!(du, du_local, equations, dg, i, j, k, element)
end

@kernel function version_5!(du, u, equations,
                            MeshT::Type{<:Union{P4estMesh{3},
                                                T8codeMesh{3}}},
                            have_nonconservative_terms::True,
                            combine_conservative_and_nonconservative_fluxes::True,
                            dg::DGSEM,
                            volume_flux,
                            ::Val{NNODES},
                            ::Val{NVARIABLES},
                            derivative_split,
                            contravariant_vectors,
                            alpha = true) where {NNODES, NVARIABLES}
    i, j, k, element = @index(Global, NTuple)

    flux_local = @localmem eltype(du) (NVARIABLES, NNODES, NNODES,
                                       NNODES)

    u_node = get_node_vars(u, equations, dg, i, j, k, element)

    # pull the contravariant vectors in each coordinate direction
    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k,
                                        element)
    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k,
                                        element)
    Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k,
                                        element)

    # All diagonal entries of `derivative_split` are zero. Thus, we can skip
    # the computation of the diagonal terms. In addition, we use the symmetry
    # of the `volume_flux` to save half of the possible two-point flux
    # computations.
    #
    # Instead of assigning thread i the partners i+1, …, N,
    # we distribute the half-sweep cyclically: each thread visits
    # half = div(N,2) partners at a fixed rotating offset.
    # Every unordered pair is still covered exactly
    # once, but now every thread performs the same number of loop iterations.
    # When N is even (odd polynomial degree) the antipodal pair at
    # offset half is shared by two threads, so its contribution is weighted by
    # 1/2 to avoid double counting.
    #
    # See Section 4.1 (Eq. 6) of
    # - Waterhouse, Waruszewski, Wilcox, Giraldo (2026)
    #   GPU Performance of an Entropy-Stable Discontinuous Galerkin Euler Solver
    #   with Non-Conservative Terms.
    #   arXiv (pre-print): https://arxiv.org/abs/2605.16684

    @uniform half_nnodes = div(NNODES, 2)
    @uniform even_nodes = iseven(NNODES)

    du_local = zero(SVector{NVARIABLES, eltype(du)})

    for offset in 1:half_nnodes
        # weight the antipodal pair by 1/2 only when the number of nodes is even
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        # first coordinate direction: rotate the partner index along `i`
        ii = mod(i - 1 + offset, NNODES) + 1
        u_node_ii = get_node_vars(u, equations, dg, ii, j, k, element)
        # pull the contravariant vectors and compute the average
        Ja1_node_ii = get_contravariant_vector(1, contravariant_vectors,
                                               ii, j, k, element)

        Ja1_avg = 0.5f0 * (Ja1_node + Ja1_node_ii)
        # compute the contravariant volume flux in the direction of the
        # averaged contravariant vector
        fluxtilde1_left, fluxtilde1_right = volume_flux(u_node, u_node_ii, Ja1_avg,
                                                        equations)

        @inbounds for v in 1:NVARIABLES
            flux_local[v, i, j, k] = fluxtilde1_right[v]
        end
        du_local = du_local +
                   (weight * alpha * derivative_split[i, ii]) * fluxtilde1_left
        @synchronize
        iib = mod(i - 1 - offset, NNODES) + 1
        du_local = du_local +
                   (weight * alpha * derivative_split[i, iib]) *
                   get_node_flux(flux_local, Val(NVARIABLES), iib, j, k)
        @synchronize
    end

    for offset in 1:half_nnodes
        # weight the antipodal pair by 1/2 only when the number of nodes is even
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        # second coordinate direction: rotate the partner index along `j`
        jj = mod(j - 1 + offset, NNODES) + 1
        u_node_jj = get_node_vars(u, equations, dg, i, jj, k, element)
        # pull the contravariant vectors and compute the average
        Ja2_node_jj = get_contravariant_vector(2, contravariant_vectors,
                                               i, jj, k, element)
        Ja2_avg = 0.5f0 * (Ja2_node + Ja2_node_jj)
        # compute the contravariant volume flux in the direction of the
        # averaged contravariant vector
        fluxtilde2_left, fluxtilde2_right = volume_flux(u_node, u_node_jj, Ja2_avg,
                                                        equations)
        @inbounds for v in 1:NVARIABLES
            flux_local[v, i, j, k] = fluxtilde2_right[v]
        end
        du_local = du_local +
                   (weight * alpha * derivative_split[j, jj]) * fluxtilde2_left
        @synchronize
        jjb = mod(j - 1 - offset, NNODES) + 1
        du_local = du_local +
                   (weight * alpha * derivative_split[j, jjb]) *
                   get_node_flux(flux_local, Val(NVARIABLES), i, jjb, k)
        @synchronize
    end

    for offset in 1:half_nnodes
        # weight the antipodal pair by 1/2 only when the number of nodes is even
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        # third coordinate direction: rotate the partner index along `k`
        kk = mod(k - 1 + offset, NNODES) + 1
        u_node_kk = get_node_vars(u, equations, dg, i, j, kk, element)
        # pull the contravariant vectors and compute the average
        Ja3_node_kk = get_contravariant_vector(3, contravariant_vectors,
                                               i, j, kk, element)
        Ja3_avg = 0.5f0 * (Ja3_node + Ja3_node_kk)
        # compute the contravariant volume flux in the direction of the
        # averaged contravariant vector
        fluxtilde3_left, fluxtilde3_right = volume_flux(u_node, u_node_kk, Ja3_avg,
                                                        equations)
        @inbounds for v in 1:NVARIABLES
            flux_local[v, i, j, k] = fluxtilde3_right[v]
        end
        du_local = du_local +
                   (weight * alpha * derivative_split[k, kk]) * fluxtilde3_left
        @synchronize
        kkb = mod(k - 1 - offset, NNODES) + 1
        du_local = du_local +
                   (weight * alpha * derivative_split[k, kkb]) *
                   get_node_flux(flux_local, Val(NVARIABLES), i, j, kkb)
        @synchronize
    end

    add_to_node_vars!(du, du_local, equations, dg, i, j, k, element)
end

@inline function calc_volume_integral!(backend::Backend, du, u,
                                       mesh::Union{P4estMesh{3}, T8codeMesh{3}},
                                       have_nonconservative_terms, equations,
                                       volume_integral::VolumeIntegralFluxDifferencing{<:FluxTurbo},
                                       dg::DGSEM, cache, ::Val{5}, ::True)
    @unpack derivative_split = dg.basis
    @unpack contravariant_vectors = cache.elements
    @unpack numerical_flux = volume_integral.volume_flux
    NNODES = nnodes(dg)
    kernel! = version_turbo_5!(backend, (NNODES, NNODES, NNODES, 1))
    kernel!(du, u, equations,
            typeof(mesh),
            have_nonconservative_terms,
            dg,
            numerical_flux,
            Val(NNODES), Val(nvariables(equations)),
            nturbovars(numerical_flux, equations),
            derivative_split,
            contravariant_vectors,
            ndrange = (NNODES, NNODES, NNODES, nelements(dg, cache)))
    return nothing
end

@inline function calc_volume_integral!(backend::Backend, du, u,
                                       mesh::Union{P4estMesh{3}, T8codeMesh{3}},
                                       have_nonconservative_terms, equations,
                                       volume_integral::VolumeIntegralFluxDifferencing,
                                       dg::DGSEM, cache, ::Val{5}, ::False)
    @unpack derivative_split = dg.basis
    @unpack contravariant_vectors = cache.elements
    NNODES = nnodes(dg)
    kernel! = version_5!(backend, (NNODES, NNODES, NNODES, 1))
    kernel!(du, u, equations,
            typeof(mesh),
            have_nonconservative_terms,
            combine_conservative_and_nonconservative_fluxes(volume_integral.volume_flux,
                                                            equations),
            dg,
            volume_integral.volume_flux,
            Val(NNODES), Val(nvariables(equations)),
            derivative_split,
            contravariant_vectors,
            ndrange = (NNODES, NNODES, NNODES, nelements(dg, cache)))
    return nothing
end
end # @muladd

using KernelAbstractions: @atomic

@inline function multiply_add_to_first_axis_atomic!(u, factor, u_node::SVector{N},
                                                    indices...) where {N}
    for v in Base.OneTo(N)
        @atomic u[v, indices...] += factor * u_node[v]
    end
    return nothing
end

@inline function calc_volume_integral!(backend::Backend, du, u,
                                       mesh::Union{P4estMesh{3}, T8codeMesh{3}},
                                       have_nonconservative_terms, equations,
                                       volume_integral::VolumeIntegralFluxDifferencing{<:FluxTurbo},
                                       dg::DGSEM, cache, ::Val{6}, ::True)
    @unpack derivative_split = dg.basis
    @unpack contravariant_vectors = cache.elements
    @unpack numerical_flux = volume_integral.volume_flux
    NNODES = nnodes(dg)
    kernel! = version_turbo_6!(backend, (NNODES, NNODES, NNODES, 1))
    kernel!(du, u, equations,
            typeof(mesh),
            have_nonconservative_terms,
            dg,
            numerical_flux,
            Val(NNODES), Val(nvariables(equations)),
            nturbovars(numerical_flux, equations),
            derivative_split,
            contravariant_vectors,
            ndrange = (NNODES, NNODES, NNODES, nelements(dg, cache)))
    return nothing
end

@kernel function version_turbo_6!(du, u, equations,
                                  MeshT::Type{<:Union{P4estMesh{3},
                                                      T8codeMesh{3}}},
                                  have_nonconservative_terms::False,
                                  dg::DGSEM,
                                  numerical_flux,
                                  ::Val{NNODES},
                                  ::Val{NVARIABLES},
                                  ::Val{NAUX},
                                  derivative_split,
                                  contravariant_vectors,
                                  alpha = true) where {NNODES, NVARIABLES,
                                                       NAUX}
    i, j, k, element = @index(Global, NTuple)

    turbo_local = @localmem eltype(du) (NAUX, NNODES, NNODES, NNODES)

    u_node = get_node_vars(u, equations, dg, i, j, k, element)
    turbo_node = cons2turbo(numerical_flux, u_node..., equations)
    @inbounds for v in 1:NAUX
        turbo_local[v, i, j, k] = turbo_node[v]
    end
    @synchronize            # the only barrier; nothing else is shared

    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
    Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)

    @uniform half_nnodes = div(NNODES, 2)
    @uniform even_nodes = iseven(NNODES)

    for offset in 1:half_nnodes
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        ii = mod(i - 1 + offset, NNODES) + 1
        Ja1_avg = 0.5f0 * (Ja1_node +
                   get_contravariant_vector(1, contravariant_vectors,
                                            ii, j, k, element))
        fluxtilde1 = flux_turbo(numerical_flux,
                                turbo_node...,
                                get_node_turbo(turbo_local, Val(NAUX),
                                               ii, j, k)...,
                                Ja1_avg[1], Ja1_avg[2], Ja1_avg[3],
                                equations)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[i, ii],
                                           SVector{NVARIABLES}(fluxtilde1),
                                           i, j, k, element)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[ii, i],
                                           SVector{NVARIABLES}(fluxtilde1),
                                           ii, j, k, element)
    end

    for offset in 1:half_nnodes
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        jj = mod(j - 1 + offset, NNODES) + 1
        Ja2_avg = 0.5f0 * (Ja2_node +
                   get_contravariant_vector(2, contravariant_vectors,
                                            i, jj, k, element))
        fluxtilde2 = flux_turbo(numerical_flux,
                                turbo_node...,
                                get_node_turbo(turbo_local, Val(NAUX),
                                               i, jj, k)...,
                                Ja2_avg[1], Ja2_avg[2], Ja2_avg[3],
                                equations)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[j, jj],
                                           SVector{NVARIABLES}(fluxtilde2),
                                           i, j, k, element)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[jj, j],
                                           SVector{NVARIABLES}(fluxtilde2),
                                           i, jj, k, element)
    end

    for offset in 1:half_nnodes
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        kk = mod(k - 1 + offset, NNODES) + 1
        Ja3_avg = 0.5f0 * (Ja3_node +
                   get_contravariant_vector(3, contravariant_vectors,
                                            i, j, kk, element))
        fluxtilde3 = flux_turbo(numerical_flux,
                                turbo_node...,
                                get_node_turbo(turbo_local, Val(NAUX),
                                               i, j, kk)...,
                                Ja3_avg[1], Ja3_avg[2], Ja3_avg[3],
                                equations)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[k, kk],
                                           SVector{NVARIABLES}(fluxtilde3),
                                           i, j, k, element)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[kk, k],
                                           SVector{NVARIABLES}(fluxtilde3),
                                           i, j, kk, element)
    end
end

@kernel function version_turbo_6!(du, u, equations,
                                  MeshT::Type{<:Union{P4estMesh{3},
                                                      T8codeMesh{3}}},
                                  have_nonconservative_terms::True,
                                  dg::DGSEM,
                                  numerical_flux,
                                  ::Val{NNODES},
                                  ::Val{NVARIABLES},
                                  ::Val{NAUX},
                                  derivative_split,
                                  contravariant_vectors,
                                  alpha = true) where {NNODES, NVARIABLES,
                                                       NAUX}
    i, j, k, element = @index(Global, NTuple)

    turbo_local = @localmem eltype(du) (NAUX, NNODES, NNODES, NNODES)

    u_node = get_node_vars(u, equations, dg, i, j, k, element)
    turbo_node = cons2turbo(numerical_flux, u_node..., equations)
    @inbounds for v in 1:NAUX
        turbo_local[v, i, j, k] = turbo_node[v]
    end
    @synchronize

    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
    Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)

    @uniform half_nnodes = div(NNODES, 2)
    @uniform even_nodes = iseven(NNODES)

    for offset in 1:half_nnodes
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        ii = mod(i - 1 + offset, NNODES) + 1
        Ja1_avg = 0.5f0 * (Ja1_node +
                   get_contravariant_vector(1, contravariant_vectors,
                                            ii, j, k, element))
        fluxtilde1_left, fluxtilde1_right = flux_turbo(numerical_flux,
                                                       turbo_node...,
                                                       get_node_turbo(turbo_local,
                                                                      Val(NAUX),
                                                                      ii, j, k)...,
                                                       Ja1_avg[1], Ja1_avg[2], Ja1_avg[3],
                                                       equations)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[i, ii],
                                           SVector{NVARIABLES}(fluxtilde1_left),
                                           i, j, k, element)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[ii, i],
                                           SVector{NVARIABLES}(fluxtilde1_right),
                                           ii, j, k, element)
    end

    for offset in 1:half_nnodes
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        jj = mod(j - 1 + offset, NNODES) + 1
        Ja2_avg = 0.5f0 * (Ja2_node +
                   get_contravariant_vector(2, contravariant_vectors,
                                            i, jj, k, element))
        fluxtilde2_left, fluxtilde2_right = flux_turbo(numerical_flux,
                                                       turbo_node...,
                                                       get_node_turbo(turbo_local,
                                                                      Val(NAUX),
                                                                      i, jj, k)...,
                                                       Ja2_avg[1], Ja2_avg[2], Ja2_avg[3],
                                                       equations)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[j, jj],
                                           SVector{NVARIABLES}(fluxtilde2_left),
                                           i, j, k, element)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[jj, j],
                                           SVector{NVARIABLES}(fluxtilde2_right),
                                           i, jj, k, element)
    end

    for offset in 1:half_nnodes
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        kk = mod(k - 1 + offset, NNODES) + 1
        Ja3_avg = 0.5f0 * (Ja3_node +
                   get_contravariant_vector(3, contravariant_vectors,
                                            i, j, kk, element))
        fluxtilde3_left, fluxtilde3_right = flux_turbo(numerical_flux,
                                                       turbo_node...,
                                                       get_node_turbo(turbo_local,
                                                                      Val(NAUX),
                                                                      i, j, kk)...,
                                                       Ja3_avg[1], Ja3_avg[2], Ja3_avg[3],
                                                       equations)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[k, kk],
                                           SVector{NVARIABLES}(fluxtilde3_left),
                                           i, j, k, element)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[kk, k],
                                           SVector{NVARIABLES}(fluxtilde3_right),
                                           i, j, kk, element)
    end
end

@inline function calc_volume_integral!(backend::Backend, du, u,
                                       mesh::Union{P4estMesh{3}, T8codeMesh{3}},
                                       have_nonconservative_terms, equations,
                                       volume_integral::VolumeIntegralFluxDifferencing,
                                       dg::DGSEM, cache, ::Val{6}, ::False)
    @unpack derivative_split = dg.basis
    @unpack contravariant_vectors = cache.elements
    NNODES = nnodes(dg)
    kernel! = version_6!(backend, (NNODES, NNODES, NNODES, 1))
    kernel!(du, u, equations,
            typeof(mesh),
            have_nonconservative_terms,
            combine_conservative_and_nonconservative_fluxes(volume_integral.volume_flux,
                                                            equations),
            dg,
            volume_integral.volume_flux,
            Val(NNODES), Val(nvariables(equations)),
            derivative_split,
            contravariant_vectors,
            ndrange = (NNODES, NNODES, NNODES, nelements(dg, cache)))
    return nothing
end

@kernel function version_6!(du, u, equations,
                            MeshT::Type{<:Union{P4estMesh{3},
                                                T8codeMesh{3}}},
                            have_nonconservative_terms::False,
                            combine_conservative_and_nonconservative_fluxes::False,
                            dg::DGSEM,
                            volume_flux,
                            ::Val{NNODES},
                            ::Val{NVARIABLES},
                            derivative_split,
                            contravariant_vectors,
                            alpha = true) where {NNODES, NVARIABLES}
    i, j, k, element = @index(Global, NTuple)

    u_node = get_node_vars(u, equations, dg, i, j, k, element)

    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
    Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)

    @uniform half_nnodes = div(NNODES, 2)
    @uniform even_nodes = iseven(NNODES)

    for offset in 1:half_nnodes
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        ii = mod(i - 1 + offset, NNODES) + 1
        Ja1_avg = 0.5f0 * (Ja1_node +
                   get_contravariant_vector(1, contravariant_vectors,
                                            ii, j, k, element))
        u_node_ii = get_node_vars(u, equations, dg, ii, j, k, element)
        fluxtilde1 = volume_flux(u_node,
                                 u_node_ii,
                                 Ja1_avg, equations)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[i, ii],
                                           SVector{NVARIABLES}(fluxtilde1),
                                           i, j, k, element)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[ii, i],
                                           SVector{NVARIABLES}(fluxtilde1),
                                           ii, j, k, element)
    end

    for offset in 1:half_nnodes
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        jj = mod(j - 1 + offset, NNODES) + 1
        Ja2_avg = 0.5f0 * (Ja2_node +
                   get_contravariant_vector(2, contravariant_vectors,
                                            i, jj, k, element))
        u_node_jj = get_node_vars(u, equations, dg, i, jj, k, element)
        fluxtilde2 = volume_flux(u_node,
                                 u_node_jj,
                                 Ja2_avg, equations)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[j, jj],
                                           SVector{NVARIABLES}(fluxtilde2),
                                           i, j, k, element)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[jj, j],
                                           SVector{NVARIABLES}(fluxtilde2),
                                           i, jj, k, element)
    end

    for offset in 1:half_nnodes
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        kk = mod(k - 1 + offset, NNODES) + 1
        Ja3_avg = 0.5f0 * (Ja3_node +
                   get_contravariant_vector(3, contravariant_vectors,
                                            i, j, kk, element))
        u_node_kk = get_node_vars(u, equations, dg, i, j, kk, element)
        fluxtilde3 = volume_flux(u_node,
                                 u_node_kk,
                                 Ja3_avg, equations)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[k, kk],
                                           SVector{NVARIABLES}(fluxtilde3),
                                           i, j, k, element)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[kk, k],
                                           SVector{NVARIABLES}(fluxtilde3),
                                           i, j, kk, element)
    end
end

@kernel function version_6!(du, u, equations,
                            MeshT::Type{<:Union{P4estMesh{3},
                                                T8codeMesh{3}}},
                            have_nonconservative_terms::True,
                            combine_conservative_and_nonconservative_fluxes::True,
                            dg::DGSEM,
                            volume_flux,
                            ::Val{NNODES},
                            ::Val{NVARIABLES},
                            derivative_split,
                            contravariant_vectors,
                            alpha = true) where {NNODES, NVARIABLES}
    i, j, k, element = @index(Global, NTuple)

    u_node = get_node_vars(u, equations, dg, i, j, k, element)

    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
    Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)

    @uniform half_nnodes = div(NNODES, 2)
    @uniform even_nodes = iseven(NNODES)

    for offset in 1:half_nnodes
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        ii = mod(i - 1 + offset, NNODES) + 1
        Ja1_avg = 0.5f0 * (Ja1_node +
                   get_contravariant_vector(1, contravariant_vectors,
                                            ii, j, k, element))
        u_node_ii = get_node_vars(u, equations, dg, ii, j, k, element)
        fluxtilde1_left, fluxtilde1_right = volume_flux(u_node,
                                                        u_node_ii,
                                                        Ja1_avg, equations)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[i, ii],
                                           SVector{NVARIABLES}(fluxtilde1_left),
                                           i, j, k, element)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[ii, i],
                                           SVector{NVARIABLES}(fluxtilde1_right),
                                           ii, j, k, element)
    end

    for offset in 1:half_nnodes
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        jj = mod(j - 1 + offset, NNODES) + 1
        Ja2_avg = 0.5f0 * (Ja2_node +
                   get_contravariant_vector(2, contravariant_vectors,
                                            i, jj, k, element))
        u_node_jj = get_node_vars(u, equations, dg, i, jj, k, element)
        fluxtilde2_left, fluxtilde2_right = volume_flux(u_node,
                                                        u_node_jj,
                                                        Ja2_avg, equations)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[j, jj],
                                           SVector{NVARIABLES}(fluxtilde2_left),
                                           i, j, k, element)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[jj, j],
                                           SVector{NVARIABLES}(fluxtilde2_right),
                                           i, jj, k, element)
    end

    for offset in 1:half_nnodes
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        kk = mod(k - 1 + offset, NNODES) + 1
        Ja3_avg = 0.5f0 * (Ja3_node +
                   get_contravariant_vector(3, contravariant_vectors,
                                            i, j, kk, element))
        u_node_kk = get_node_vars(u, equations, dg, i, j, kk, element)
        fluxtilde3_left, fluxtilde3_right = volume_flux(u_node,
                                                        u_node_kk,
                                                        Ja3_avg, equations)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[k, kk],
                                           SVector{NVARIABLES}(fluxtilde3_left),
                                           i, j, k, element)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[kk, k],
                                           SVector{NVARIABLES}(fluxtilde3_right),
                                           i, j, kk, element)
    end
end

@inline function calc_volume_integral!(backend::Backend, du, u,
                                       mesh::Union{P4estMesh{3}, T8codeMesh{3}},
                                       have_nonconservative_terms, equations,
                                       volume_integral::VolumeIntegralFluxDifferencing{<:FluxTurbo},
                                       dg::DGSEM, cache, ::Val{7}, ::True)
    @unpack derivative_split = dg.basis
    @unpack contravariant_vectors = cache.elements
    @unpack numerical_flux = volume_integral.volume_flux
    NNODES = nnodes(dg)
    kernel! = version_turbo_7!(backend, (NNODES, NNODES, NNODES, 1))
    kernel!(du, u, equations,
            typeof(mesh),
            have_nonconservative_terms,
            dg,
            numerical_flux,
            Val(NNODES), Val(nvariables(equations)),
            nturbovars(numerical_flux, equations),
            derivative_split,
            contravariant_vectors,
            ndrange = (NNODES, NNODES, NNODES, nelements(dg, cache)))
    return nothing
end

@kernel function version_turbo_7!(du, u, equations,
                                  MeshT::Type{<:Union{P4estMesh{3},
                                                      T8codeMesh{3}}},
                                  have_nonconservative_terms::False,
                                  dg::DGSEM,
                                  numerical_flux,
                                  ::Val{NNODES},
                                  ::Val{NVARIABLES},
                                  ::Val{NAUX},
                                  derivative_split,
                                  contravariant_vectors,
                                  alpha = true) where {NNODES, NVARIABLES,
                                                       NAUX}
    i, j, k, element = @index(Global, NTuple)

    turbo_local = @localmem eltype(du) (NAUX, NNODES, NNODES, NNODES)

    u_node = get_node_vars(u, equations, dg, i, j, k, element)
    turbo_node = cons2turbo(numerical_flux, u_node..., equations)
    @inbounds for v in 1:NAUX
        turbo_local[v, i, j, k] = turbo_node[v]
    end
    @synchronize

    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
    Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)

    @uniform half_nnodes = div(NNODES, 2)
    @uniform even_nodes = iseven(NNODES)

    for offset in 1:half_nnodes
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        ii = mod(i - 1 + offset, NNODES) + 1
        Ja1_avg = 0.5f0 * (Ja1_node +
                   get_contravariant_vector(1, contravariant_vectors,
                                            ii, j, k, element))
        fluxtilde1 = flux_turbo(numerical_flux,
                                turbo_node...,
                                get_node_turbo(turbo_local, Val(NAUX),
                                               ii, j, k)...,
                                Ja1_avg[1], Ja1_avg[2], Ja1_avg[3],
                                equations)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[i, ii],
                                           SVector{NVARIABLES}(fluxtilde1),
                                           i, j, k, element)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[ii, i],
                                           SVector{NVARIABLES}(fluxtilde1),
                                           ii, j, k, element)

        jj = mod(j - 1 + offset, NNODES) + 1
        Ja2_avg = 0.5f0 * (Ja2_node +
                   get_contravariant_vector(2, contravariant_vectors,
                                            i, jj, k, element))
        fluxtilde2 = flux_turbo(numerical_flux,
                                turbo_node...,
                                get_node_turbo(turbo_local, Val(NAUX),
                                               i, jj, k)...,
                                Ja2_avg[1], Ja2_avg[2], Ja2_avg[3],
                                equations)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[j, jj],
                                           SVector{NVARIABLES}(fluxtilde2),
                                           i, j, k, element)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[jj, j],
                                           SVector{NVARIABLES}(fluxtilde2),
                                           i, jj, k, element)

        kk = mod(k - 1 + offset, NNODES) + 1
        Ja3_avg = 0.5f0 * (Ja3_node +
                   get_contravariant_vector(3, contravariant_vectors,
                                            i, j, kk, element))
        fluxtilde3 = flux_turbo(numerical_flux,
                                turbo_node...,
                                get_node_turbo(turbo_local, Val(NAUX),
                                               i, j, kk)...,
                                Ja3_avg[1], Ja3_avg[2], Ja3_avg[3],
                                equations)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[k, kk],
                                           SVector{NVARIABLES}(fluxtilde3),
                                           i, j, k, element)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[kk, k],
                                           SVector{NVARIABLES}(fluxtilde3),
                                           i, j, kk, element)
    end
end

@kernel function version_turbo_7!(du, u, equations,
                                  MeshT::Type{<:Union{P4estMesh{3},
                                                      T8codeMesh{3}}},
                                  have_nonconservative_terms::True,
                                  dg::DGSEM,
                                  numerical_flux,
                                  ::Val{NNODES},
                                  ::Val{NVARIABLES},
                                  ::Val{NAUX},
                                  derivative_split,
                                  contravariant_vectors,
                                  alpha = true) where {NNODES, NVARIABLES,
                                                       NAUX}
    i, j, k, element = @index(Global, NTuple)

    turbo_local = @localmem eltype(du) (NAUX, NNODES, NNODES, NNODES)

    u_node = get_node_vars(u, equations, dg, i, j, k, element)
    turbo_node = cons2turbo(numerical_flux, u_node..., equations)
    @inbounds for v in 1:NAUX
        turbo_local[v, i, j, k] = turbo_node[v]
    end
    @synchronize

    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
    Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)

    @uniform half_nnodes = div(NNODES, 2)
    @uniform even_nodes = iseven(NNODES)

    for offset in 1:half_nnodes
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        ii = mod(i - 1 + offset, NNODES) + 1
        Ja1_avg = 0.5f0 * (Ja1_node +
                   get_contravariant_vector(1, contravariant_vectors,
                                            ii, j, k, element))
        fluxtilde1_left, fluxtilde1_right = flux_turbo(numerical_flux,
                                                       turbo_node...,
                                                       get_node_turbo(turbo_local,
                                                                      Val(NAUX),
                                                                      ii, j, k)...,
                                                       Ja1_avg[1], Ja1_avg[2], Ja1_avg[3],
                                                       equations)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[i, ii],
                                           SVector{NVARIABLES}(fluxtilde1_left),
                                           i, j, k, element)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[ii, i],
                                           SVector{NVARIABLES}(fluxtilde1_right),
                                           ii, j, k, element)

        jj = mod(j - 1 + offset, NNODES) + 1
        Ja2_avg = 0.5f0 * (Ja2_node +
                   get_contravariant_vector(2, contravariant_vectors,
                                            i, jj, k, element))
        fluxtilde2_left, fluxtilde2_right = flux_turbo(numerical_flux,
                                                       turbo_node...,
                                                       get_node_turbo(turbo_local,
                                                                      Val(NAUX),
                                                                      i, jj, k)...,
                                                       Ja2_avg[1], Ja2_avg[2], Ja2_avg[3],
                                                       equations)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[j, jj],
                                           SVector{NVARIABLES}(fluxtilde2_left),
                                           i, j, k, element)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[jj, j],
                                           SVector{NVARIABLES}(fluxtilde2_right),
                                           i, jj, k, element)

        kk = mod(k - 1 + offset, NNODES) + 1
        Ja3_avg = 0.5f0 * (Ja3_node +
                   get_contravariant_vector(3, contravariant_vectors,
                                            i, j, kk, element))
        fluxtilde3_left, fluxtilde3_right = flux_turbo(numerical_flux,
                                                       turbo_node...,
                                                       get_node_turbo(turbo_local,
                                                                      Val(NAUX),
                                                                      i, j, kk)...,
                                                       Ja3_avg[1], Ja3_avg[2], Ja3_avg[3],
                                                       equations)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[k, kk],
                                           SVector{NVARIABLES}(fluxtilde3_left),
                                           i, j, k, element)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[kk, k],
                                           SVector{NVARIABLES}(fluxtilde3_right),
                                           i, j, kk, element)
    end
end

@inline function calc_volume_integral!(backend::Backend, du, u,
                                       mesh::Union{P4estMesh{3}, T8codeMesh{3}},
                                       have_nonconservative_terms, equations,
                                       volume_integral::VolumeIntegralFluxDifferencing,
                                       dg::DGSEM, cache, ::Val{7}, ::False)
    @unpack derivative_split = dg.basis
    @unpack contravariant_vectors = cache.elements
    NNODES = nnodes(dg)
    kernel! = version_7!(backend, (NNODES, NNODES, NNODES, 1))
    kernel!(du, u, equations,
            typeof(mesh),
            have_nonconservative_terms,
            combine_conservative_and_nonconservative_fluxes(volume_integral.volume_flux,
                                                            equations),
            dg,
            volume_integral.volume_flux,
            Val(NNODES), Val(nvariables(equations)),
            derivative_split,
            contravariant_vectors,
            ndrange = (NNODES, NNODES, NNODES, nelements(dg, cache)))
    return nothing
end

@kernel function version_7!(du, u, equations,
                            MeshT::Type{<:Union{P4estMesh{3},
                                                T8codeMesh{3}}},
                            have_nonconservative_terms::False,
                            combine_conservative_and_nonconservative_fluxes::False,
                            dg::DGSEM,
                            volume_flux,
                            ::Val{NNODES},
                            ::Val{NVARIABLES},
                            derivative_split,
                            contravariant_vectors,
                            alpha = true) where {NNODES, NVARIABLES}
    i, j, k, element = @index(Global, NTuple)

    u_node = get_node_vars(u, equations, dg, i, j, k, element)

    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
    Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)

    @uniform half_nnodes = div(NNODES, 2)
    @uniform even_nodes = iseven(NNODES)

    for offset in 1:half_nnodes
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        ii = mod(i - 1 + offset, NNODES) + 1
        Ja1_avg = 0.5f0 * (Ja1_node +
                   get_contravariant_vector(1, contravariant_vectors,
                                            ii, j, k, element))
        u_node_ii = get_node_vars(u, equations, dg, ii, j, k, element)
        fluxtilde1 = volume_flux(u_node,
                                 u_node_ii,
                                 Ja1_avg, equations)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[i, ii],
                                           SVector{NVARIABLES}(fluxtilde1),
                                           i, j, k, element)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[ii, i],
                                           SVector{NVARIABLES}(fluxtilde1),
                                           ii, j, k, element)

        jj = mod(j - 1 + offset, NNODES) + 1
        Ja2_avg = 0.5f0 * (Ja2_node +
                   get_contravariant_vector(2, contravariant_vectors,
                                            i, jj, k, element))
        u_node_jj = get_node_vars(u, equations, dg, i, jj, k, element)
        fluxtilde2 = volume_flux(u_node,
                                 u_node_jj,
                                 Ja2_avg, equations)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[j, jj],
                                           SVector{NVARIABLES}(fluxtilde2),
                                           i, j, k, element)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[jj, j],
                                           SVector{NVARIABLES}(fluxtilde2),
                                           i, jj, k, element)

        kk = mod(k - 1 + offset, NNODES) + 1
        Ja3_avg = 0.5f0 * (Ja3_node +
                   get_contravariant_vector(3, contravariant_vectors,
                                            i, j, kk, element))
        u_node_kk = get_node_vars(u, equations, dg, i, j, kk, element)
        fluxtilde3 = volume_flux(u_node,
                                 u_node_kk,
                                 Ja3_avg, equations)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[k, kk],
                                           SVector{NVARIABLES}(fluxtilde3),
                                           i, j, k, element)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[kk, k],
                                           SVector{NVARIABLES}(fluxtilde3),
                                           i, j, kk, element)
    end
end

@kernel function version_7!(du, u, equations,
                            MeshT::Type{<:Union{P4estMesh{3},
                                                T8codeMesh{3}}},
                            have_nonconservative_terms::True,
                            combine_conservative_and_nonconservative_fluxes::True,
                            dg::DGSEM,
                            volume_flux,
                            ::Val{NNODES},
                            ::Val{NVARIABLES},
                            derivative_split,
                            contravariant_vectors,
                            alpha = true) where {NNODES, NVARIABLES}
    i, j, k, element = @index(Global, NTuple)

    u_node = get_node_vars(u, equations, dg, i, j, k, element)

    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
    Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)

    @uniform half_nnodes = div(NNODES, 2)
    @uniform even_nodes = iseven(NNODES)

    for offset in 1:half_nnodes
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        ii = mod(i - 1 + offset, NNODES) + 1
        Ja1_avg = 0.5f0 * (Ja1_node +
                   get_contravariant_vector(1, contravariant_vectors,
                                            ii, j, k, element))
        u_node_ii = get_node_vars(u, equations, dg, ii, j, k, element)
        fluxtilde1_left, fluxtilde1_right = volume_flux(u_node,
                                                        u_node_ii,
                                                        Ja1_avg, equations)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[i, ii],
                                           SVector{NVARIABLES}(fluxtilde1_left),
                                           i, j, k, element)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[ii, i],
                                           SVector{NVARIABLES}(fluxtilde1_right),
                                           ii, j, k, element)

        jj = mod(j - 1 + offset, NNODES) + 1
        Ja2_avg = 0.5f0 * (Ja2_node +
                   get_contravariant_vector(2, contravariant_vectors,
                                            i, jj, k, element))
        u_node_jj = get_node_vars(u, equations, dg, i, jj, k, element)
        fluxtilde2_left, fluxtilde2_right = volume_flux(u_node,
                                                        u_node_jj,
                                                        Ja2_avg, equations)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[j, jj],
                                           SVector{NVARIABLES}(fluxtilde2_left),
                                           i, j, k, element)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[jj, j],
                                           SVector{NVARIABLES}(fluxtilde2_right),
                                           i, jj, k, element)

        kk = mod(k - 1 + offset, NNODES) + 1
        Ja3_avg = 0.5f0 * (Ja3_node +
                   get_contravariant_vector(3, contravariant_vectors,
                                            i, j, kk, element))
        u_node_kk = get_node_vars(u, equations, dg, i, j, kk, element)
        fluxtilde3_left, fluxtilde3_right = volume_flux(u_node,
                                                        u_node_kk,
                                                        Ja3_avg, equations)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[k, kk],
                                           SVector{NVARIABLES}(fluxtilde3_left),
                                           i, j, k, element)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[kk, k],
                                           SVector{NVARIABLES}(fluxtilde3_right),
                                           i, j, kk, element)
    end
end

@inline function calc_volume_integral!(backend::Backend, du, u,
                                       mesh::Union{P4estMesh{3}, T8codeMesh{3}},
                                       have_nonconservative_terms, equations,
                                       volume_integral::VolumeIntegralFluxDifferencing{<:FluxTurbo},
                                       dg::DGSEM, cache, ::Val{8}, ::True)
    @unpack derivative_split = dg.basis
    @unpack contravariant_vectors = cache.elements
    @unpack numerical_flux = volume_integral.volume_flux
    NNODES = nnodes(dg)
    kernel! = version_turbo_8!(backend, (NNODES, NNODES, NNODES, 1))
    kernel!(du, u, equations,
            typeof(mesh),
            have_nonconservative_terms,
            dg,
            numerical_flux,
            Val(NNODES), Val(nvariables(equations)),
            nturbovars(numerical_flux, equations),
            derivative_split,
            contravariant_vectors,
            ndrange = (NNODES, NNODES, NNODES, nelements(dg, cache)))
    return nothing
end

@kernel function version_turbo_8!(du, u, equations,
                                  MeshT::Type{<:Union{P4estMesh{3},
                                                      T8codeMesh{3}}},
                                  have_nonconservative_terms::False,
                                  dg::DGSEM,
                                  numerical_flux,
                                  ::Val{NNODES},
                                  ::Val{NVARIABLES},
                                  ::Val{NAUX},
                                  derivative_split,
                                  contravariant_vectors,
                                  alpha = true) where {NNODES, NVARIABLES,
                                                       NAUX}
    i, j, k, element = @index(Global, NTuple)

    turbo_local = @localmem eltype(du) (NAUX, NNODES, NNODES, NNODES)

    u_node = get_node_vars(u, equations, dg, i, j, k, element)
    turbo_node = cons2turbo(numerical_flux, u_node..., equations)
    @inbounds for v in 1:NAUX
        turbo_local[v, i, j, k] = turbo_node[v]
    end
    @synchronize            # the only barrier; nothing else is shared

    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
    Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)

    @uniform half_nnodes = div(NNODES, 2)
    @uniform even_nodes = iseven(NNODES)

    KernelAbstractions.Extras.@unroll for offset in 1:half_nnodes
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        ii = mod(i - 1 + offset, NNODES) + 1
        Ja1_avg = 0.5f0 * (Ja1_node +
                   get_contravariant_vector(1, contravariant_vectors,
                                            ii, j, k, element))
        fluxtilde1 = flux_turbo(numerical_flux,
                                turbo_node...,
                                get_node_turbo(turbo_local, Val(NAUX),
                                               ii, j, k)...,
                                Ja1_avg[1], Ja1_avg[2], Ja1_avg[3],
                                equations)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[i, ii],
                                           SVector{NVARIABLES}(fluxtilde1),
                                           i, j, k, element)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[ii, i],
                                           SVector{NVARIABLES}(fluxtilde1),
                                           ii, j, k, element)

        jj = mod(j - 1 + offset, NNODES) + 1
        Ja2_avg = 0.5f0 * (Ja2_node +
                   get_contravariant_vector(2, contravariant_vectors,
                                            i, jj, k, element))
        fluxtilde2 = flux_turbo(numerical_flux,
                                turbo_node...,
                                get_node_turbo(turbo_local, Val(NAUX),
                                               i, jj, k)...,
                                Ja2_avg[1], Ja2_avg[2], Ja2_avg[3],
                                equations)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[j, jj],
                                           SVector{NVARIABLES}(fluxtilde2),
                                           i, j, k, element)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[jj, j],
                                           SVector{NVARIABLES}(fluxtilde2),
                                           i, jj, k, element)

        kk = mod(k - 1 + offset, NNODES) + 1
        Ja3_avg = 0.5f0 * (Ja3_node +
                   get_contravariant_vector(3, contravariant_vectors,
                                            i, j, kk, element))
        fluxtilde3 = flux_turbo(numerical_flux,
                                turbo_node...,
                                get_node_turbo(turbo_local, Val(NAUX),
                                               i, j, kk)...,
                                Ja3_avg[1], Ja3_avg[2], Ja3_avg[3],
                                equations)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[k, kk],
                                           SVector{NVARIABLES}(fluxtilde3),
                                           i, j, k, element)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[kk, k],
                                           SVector{NVARIABLES}(fluxtilde3),
                                           i, j, kk, element)
    end
end

@kernel function version_turbo_8!(du, u, equations,
                                  MeshT::Type{<:Union{P4estMesh{3},
                                                      T8codeMesh{3}}},
                                  have_nonconservative_terms::True,
                                  dg::DGSEM,
                                  numerical_flux,
                                  ::Val{NNODES},
                                  ::Val{NVARIABLES},
                                  ::Val{NAUX},
                                  derivative_split,
                                  contravariant_vectors,
                                  alpha = true) where {NNODES, NVARIABLES,
                                                       NAUX}
    i, j, k, element = @index(Global, NTuple)

    turbo_local = @localmem eltype(du) (NAUX, NNODES, NNODES, NNODES)

    u_node = get_node_vars(u, equations, dg, i, j, k, element)
    turbo_node = cons2turbo(numerical_flux, u_node..., equations)
    @inbounds for v in 1:NAUX
        turbo_local[v, i, j, k] = turbo_node[v]
    end
    @synchronize

    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
    Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)

    @uniform half_nnodes = div(NNODES, 2)
    @uniform even_nodes = iseven(NNODES)

    KernelAbstractions.Extras.@unroll for offset in 1:half_nnodes
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        ii = mod(i - 1 + offset, NNODES) + 1
        Ja1_avg = 0.5f0 * (Ja1_node +
                   get_contravariant_vector(1, contravariant_vectors,
                                            ii, j, k, element))
        fluxtilde1_left, fluxtilde1_right = flux_turbo(numerical_flux,
                                                       turbo_node...,
                                                       get_node_turbo(turbo_local,
                                                                      Val(NAUX),
                                                                      ii, j, k)...,
                                                       Ja1_avg[1], Ja1_avg[2], Ja1_avg[3],
                                                       equations)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[i, ii],
                                           SVector{NVARIABLES}(fluxtilde1_left),
                                           i, j, k, element)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[ii, i],
                                           SVector{NVARIABLES}(fluxtilde1_right),
                                           ii, j, k, element)

        jj = mod(j - 1 + offset, NNODES) + 1
        Ja2_avg = 0.5f0 * (Ja2_node +
                   get_contravariant_vector(2, contravariant_vectors,
                                            i, jj, k, element))
        fluxtilde2_left, fluxtilde2_right = flux_turbo(numerical_flux,
                                                       turbo_node...,
                                                       get_node_turbo(turbo_local,
                                                                      Val(NAUX),
                                                                      i, jj, k)...,
                                                       Ja2_avg[1], Ja2_avg[2], Ja2_avg[3],
                                                       equations)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[j, jj],
                                           SVector{NVARIABLES}(fluxtilde2_left),
                                           i, j, k, element)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[jj, j],
                                           SVector{NVARIABLES}(fluxtilde2_right),
                                           i, jj, k, element)

        kk = mod(k - 1 + offset, NNODES) + 1
        Ja3_avg = 0.5f0 * (Ja3_node +
                   get_contravariant_vector(3, contravariant_vectors,
                                            i, j, kk, element))
        fluxtilde3_left, fluxtilde3_right = flux_turbo(numerical_flux,
                                                       turbo_node...,
                                                       get_node_turbo(turbo_local,
                                                                      Val(NAUX),
                                                                      i, j, kk)...,
                                                       Ja3_avg[1], Ja3_avg[2], Ja3_avg[3],
                                                       equations)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[k, kk],
                                           SVector{NVARIABLES}(fluxtilde3_left),
                                           i, j, k, element)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[kk, k],
                                           SVector{NVARIABLES}(fluxtilde3_right),
                                           i, j, kk, element)
    end
end

@kernel function version_8!(du, u, equations,
                            MeshT::Type{<:Union{P4estMesh{3},
                                                T8codeMesh{3}}},
                            have_nonconservative_terms::False,
                            combine_conservative_and_nonconservative_fluxes::False,
                            dg::DGSEM,
                            volume_flux,
                            ::Val{NNODES},
                            ::Val{NVARIABLES},
                            derivative_split,
                            contravariant_vectors,
                            alpha = true) where {NNODES, NVARIABLES}
    i, j, k, element = @index(Global, NTuple)

    u_node = get_node_vars(u, equations, dg, i, j, k, element)

    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
    Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)

    @uniform half_nnodes = div(NNODES, 2)
    @uniform even_nodes = iseven(NNODES)

    KernelAbstractions.Extras.@unroll for offset in 1:half_nnodes
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        ii = mod(i - 1 + offset, NNODES) + 1
        Ja1_avg = 0.5f0 * (Ja1_node +
                   get_contravariant_vector(1, contravariant_vectors,
                                            ii, j, k, element))
        u_node_ii = get_node_vars(u, equations, dg, ii, j, k, element)
        fluxtilde1 = volume_flux(u_node,
                                 u_node_ii,
                                 Ja1_avg, equations)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[i, ii],
                                           SVector{NVARIABLES}(fluxtilde1),
                                           i, j, k, element)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[ii, i],
                                           SVector{NVARIABLES}(fluxtilde1),
                                           ii, j, k, element)

        jj = mod(j - 1 + offset, NNODES) + 1
        Ja2_avg = 0.5f0 * (Ja2_node +
                   get_contravariant_vector(2, contravariant_vectors,
                                            i, jj, k, element))
        u_node_jj = get_node_vars(u, equations, dg, i, jj, k, element)
        fluxtilde2 = volume_flux(u_node,
                                 u_node_jj,
                                 Ja2_avg, equations)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[j, jj],
                                           SVector{NVARIABLES}(fluxtilde2),
                                           i, j, k, element)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[jj, j],
                                           SVector{NVARIABLES}(fluxtilde2),
                                           i, jj, k, element)

        kk = mod(k - 1 + offset, NNODES) + 1
        Ja3_avg = 0.5f0 * (Ja3_node +
                   get_contravariant_vector(3, contravariant_vectors,
                                            i, j, kk, element))
        u_node_kk = get_node_vars(u, equations, dg, i, j, kk, element)
        fluxtilde3 = volume_flux(u_node,
                                 u_node_kk,
                                 Ja3_avg, equations)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[k, kk],
                                           SVector{NVARIABLES}(fluxtilde3),
                                           i, j, k, element)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[kk, k],
                                           SVector{NVARIABLES}(fluxtilde3),
                                           i, j, kk, element)
    end
end

@kernel function version_8!(du, u, equations,
                            MeshT::Type{<:Union{P4estMesh{3},
                                                T8codeMesh{3}}},
                            have_nonconservative_terms::True,
                            combine_conservative_and_nonconservative_fluxes::True,
                            dg::DGSEM,
                            volume_flux,
                            ::Val{NNODES},
                            ::Val{NVARIABLES},
                            derivative_split,
                            contravariant_vectors,
                            alpha = true) where {NNODES, NVARIABLES}
    i, j, k, element = @index(Global, NTuple)

    u_node = get_node_vars(u, equations, dg, i, j, k, element)

    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
    Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)

    @uniform half_nnodes = div(NNODES, 2)
    @uniform even_nodes = iseven(NNODES)

    KernelAbstractions.Extras.@unroll for offset in 1:half_nnodes
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        ii = mod(i - 1 + offset, NNODES) + 1
        Ja1_avg = 0.5f0 * (Ja1_node +
                   get_contravariant_vector(1, contravariant_vectors,
                                            ii, j, k, element))
        u_node_ii = get_node_vars(u, equations, dg, ii, j, k, element)
        fluxtilde1_left, fluxtilde1_right = volume_flux(u_node,
                                                        u_node_ii,
                                                        Ja1_avg, equations)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[i, ii],
                                           SVector{NVARIABLES}(fluxtilde1_left),
                                           i, j, k, element)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[ii, i],
                                           SVector{NVARIABLES}(fluxtilde1_right),
                                           ii, j, k, element)

        jj = mod(j - 1 + offset, NNODES) + 1
        Ja2_avg = 0.5f0 * (Ja2_node +
                   get_contravariant_vector(2, contravariant_vectors,
                                            i, jj, k, element))
        u_node_jj = get_node_vars(u, equations, dg, i, jj, k, element)
        fluxtilde2_left, fluxtilde2_right = volume_flux(u_node,
                                                        u_node_jj,
                                                        Ja2_avg, equations)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[j, jj],
                                           SVector{NVARIABLES}(fluxtilde2_left),
                                           i, j, k, element)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[jj, j],
                                           SVector{NVARIABLES}(fluxtilde2_right),
                                           i, jj, k, element)

        kk = mod(k - 1 + offset, NNODES) + 1
        Ja3_avg = 0.5f0 * (Ja3_node +
                   get_contravariant_vector(3, contravariant_vectors,
                                            i, j, kk, element))
        u_node_kk = get_node_vars(u, equations, dg, i, j, kk, element)
        fluxtilde3_left, fluxtilde3_right = volume_flux(u_node,
                                                        u_node_kk,
                                                        Ja3_avg, equations)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[k, kk],
                                           SVector{NVARIABLES}(fluxtilde3_left),
                                           i, j, k, element)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[kk, k],
                                           SVector{NVARIABLES}(fluxtilde3_right),
                                           i, j, kk, element)
    end
end

@inline function calc_volume_integral!(backend::Backend, du, u,
                                       mesh::Union{P4estMesh{3}, T8codeMesh{3}},
                                       have_nonconservative_terms, equations,
                                       volume_integral::VolumeIntegralFluxDifferencing,
                                       dg::DGSEM, cache, ::Val{8}, ::False)
    @unpack derivative_split = dg.basis
    @unpack contravariant_vectors = cache.elements
    NNODES = nnodes(dg)
    kernel! = version_8!(backend, (NNODES, NNODES, NNODES, 1))
    kernel!(du, u, equations,
            typeof(mesh),
            have_nonconservative_terms,
            combine_conservative_and_nonconservative_fluxes(volume_integral.volume_flux,
                                                            equations),
            dg,
            volume_integral.volume_flux,
            Val(NNODES), Val(nvariables(equations)),
            derivative_split,
            contravariant_vectors,
            ndrange = (NNODES, NNODES, NNODES, nelements(dg, cache)))
    return nothing
end

@muladd begin
#! format: noindent

@muladd @inline function flux_hindenlang_gassner_nonconservative_powell(u_ll, u_rr,
                                                                        normal_direction::AbstractVector,
                                                                        equations::IdealGlmMhdEquations3D)
    # Unpack left and right states
    rho_ll, v1_ll, v2_ll, v3_ll, p_ll, B1_ll, B2_ll, B3_ll, psi_ll = cons2prim(u_ll,
                                                                               equations)
    rho_rr, v1_rr, v2_rr, v3_rr, p_rr, B1_rr, B2_rr, B3_rr, psi_rr = cons2prim(u_rr,
                                                                               equations)
    v_dot_n_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2] +
                 v3_ll * normal_direction[3]
    v_dot_n_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2] +
                 v3_rr * normal_direction[3]
    B_dot_n_ll = B1_ll * normal_direction[1] + B2_ll * normal_direction[2] +
                 B3_ll * normal_direction[3]
    B_dot_n_rr = B1_rr * normal_direction[1] + B2_rr * normal_direction[2] +
                 B3_rr * normal_direction[3]

    # Compute the necessary mean values needed for either direction
    rho_mean = ln_mean(rho_ll, rho_rr)
    inv_rho_p_mean = p_ll * p_rr * inv_ln_mean(rho_ll * p_rr, rho_rr * p_ll)
    v1_avg = 0.5f0 * (v1_ll + v1_rr)
    v2_avg = 0.5f0 * (v2_ll + v2_rr)
    v3_avg = 0.5f0 * (v3_ll + v3_rr)
    p_avg = 0.5f0 * (p_ll + p_rr)
    psi_avg = 0.5f0 * (psi_ll + psi_rr)
    velocity_square_avg = 0.5f0 * (v1_ll * v1_rr + v2_ll * v2_rr + v3_ll * v3_rr)
    magnetic_square_avg = 0.5f0 * (B1_ll * B1_rr + B2_ll * B2_rr + B3_ll * B3_rr)

    # Calculate fluxes depending on normal_direction
    f1 = rho_mean * 0.5f0 * (v_dot_n_ll + v_dot_n_rr)
    f2 = (f1 * v1_avg + (p_avg + magnetic_square_avg) * normal_direction[1]
          -
          0.5f0 * (B_dot_n_ll * B1_rr + B_dot_n_rr * B1_ll))
    f3 = (f1 * v2_avg + (p_avg + magnetic_square_avg) * normal_direction[2]
          -
          0.5f0 * (B_dot_n_ll * B2_rr + B_dot_n_rr * B2_ll))
    f4 = (f1 * v3_avg + (p_avg + magnetic_square_avg) * normal_direction[3]
          -
          0.5f0 * (B_dot_n_ll * B3_rr + B_dot_n_rr * B3_ll))
    #f5 below
    f6 = (equations.c_h * psi_avg * normal_direction[1]
          +
          0.5f0 * (v_dot_n_ll * B1_ll - v1_ll * B_dot_n_ll +
           v_dot_n_rr * B1_rr - v1_rr * B_dot_n_rr))
    f7 = (equations.c_h * psi_avg * normal_direction[2]
          +
          0.5f0 * (v_dot_n_ll * B2_ll - v2_ll * B_dot_n_ll +
           v_dot_n_rr * B2_rr - v2_rr * B_dot_n_rr))
    f8 = (equations.c_h * psi_avg * normal_direction[3]
          +
          0.5f0 * (v_dot_n_ll * B3_ll - v3_ll * B_dot_n_ll +
           v_dot_n_rr * B3_rr - v3_rr * B_dot_n_rr))
    f9 = equations.c_h * 0.5f0 * (B_dot_n_ll + B_dot_n_rr)
    # total energy flux is complicated and involves the previous components
    f5 = (f1 *
          (velocity_square_avg + inv_rho_p_mean * equations.inv_gamma_minus_one)
          +
          0.5f0 * (+p_ll * v_dot_n_rr + p_rr * v_dot_n_ll
           + (v_dot_n_ll * B1_ll * B1_rr + v_dot_n_rr * B1_rr * B1_ll)
           + (v_dot_n_ll * B2_ll * B2_rr + v_dot_n_rr * B2_rr * B2_ll)
           + (v_dot_n_ll * B3_ll * B3_rr + v_dot_n_rr * B3_rr * B3_ll)
           -
           (v1_ll * B_dot_n_ll * B1_rr + v1_rr * B_dot_n_rr * B1_ll)
           -
           (v2_ll * B_dot_n_ll * B2_rr + v2_rr * B_dot_n_rr * B2_ll)
           -
           (v3_ll * B_dot_n_ll * B3_rr + v3_rr * B_dot_n_rr * B3_ll)
           +
           equations.c_h * (B_dot_n_ll * psi_rr + B_dot_n_rr * psi_ll)))

    v_dot_B_ll = v1_ll * B1_ll + v2_ll * B2_ll + v3_ll * B3_ll

    v_dot_B_rr = v1_rr * B1_rr + v2_rr * B2_rr + v3_rr * B3_rr
    f = SVector(f1, f2, f3, f4, f5, f6, f7, f8, f9)
    # Powell nonconservative term:   (0, B_1, B_2, B_3, v⋅B, v_1, v_2, v_3, 0)
    # Galilean nonconservative term: (0, 0, 0, 0, ψ v_{1,2,3}, 0, 0, 0, v_{1,2,3})
    g_left = SVector(0,
                     B1_ll * B_dot_n_rr,
                     B2_ll * B_dot_n_rr,
                     B3_ll * B_dot_n_rr,
                     v_dot_B_ll * B_dot_n_rr + v_dot_n_ll * psi_ll * psi_rr,
                     v1_ll * B_dot_n_rr,
                     v2_ll * B_dot_n_rr,
                     v3_ll * B_dot_n_rr,
                     v_dot_n_ll * psi_rr)

    g_right = SVector(0,
                      B1_rr * B_dot_n_ll,
                      B2_rr * B_dot_n_ll,
                      B3_rr * B_dot_n_ll,
                      v_dot_B_rr * B_dot_n_ll + v_dot_n_rr * psi_rr * psi_ll,
                      v1_rr * B_dot_n_ll,
                      v2_rr * B_dot_n_ll,
                      v3_rr * B_dot_n_ll,
                      v_dot_n_rr * psi_ll)
    flux_left = f + 0.5f0 * g_left
    flux_right = f + 0.5f0 * g_right
    return flux_left, flux_right
end

@inline combine_conservative_and_nonconservative_fluxes(::typeof(flux_hindenlang_gassner_nonconservative_powell),
equations::IdealGlmMhdEquations3D) = True()
end # @muladd

@muladd begin
    @inline nturbovars(::typeof(flux_hindenlang_gassner_nonconservative_powell), ::IdealGlmMhdEquations3D) = Val(11)

    @inline function cons2turbo(::typeof(flux_hindenlang_gassner_nonconservative_powell),
                                rho, rho_v1, rho_v2, rho_v3, rho_e,
                                B1, B2, B3, psi, equations::IdealGlmMhdEquations3D)
        rho_inv = inv(rho)
        v1 = rho_v1 * rho_inv
        v2 = rho_v2 * rho_inv
        v3 = rho_v3 * rho_inv
        p = (equations.gamma - 1) * (rho_e -
             0.5f0 * (rho_v1 * v1 + rho_v2 * v2 + rho_v3 * v3
              + B1 * B1 + B2 * B2 + B3 * B3
              + psi * psi))
        return (rho, v1, v2, v3, p, B1, B2, B3, psi, log(rho), log(p))
    end

    @inline function ln_mean_pre(x::RealT, y::RealT, log_x::RealT,
                                 log_y::RealT) where {RealT}
        epsilon_f2 = convert(RealT, 1.0e-4)
        f2 = (x * (x - 2 * y) + y * y) / (x * (x + 2 * y) + y * y)
        if f2 < epsilon_f2
            return (x + y) / @evalpoly(f2, 2, convert(RealT, 2 / 3), convert(RealT, 2 / 5),
                             convert(RealT, 2 / 7))
        else
            return (y - x) / (log_y - log_x)
        end
    end

    @inline function inv_ln_mean_pre(x::RealT, y::RealT, log_x::RealT,
                                     log_y::RealT) where {RealT}
        epsilon_f2 = convert(RealT, 1.0e-4)
        f2 = (x * (x - 2 * y) + y * y) / (x * (x + 2 * y) + y * y)
        if f2 < epsilon_f2
            return @evalpoly(f2, 2, convert(RealT, 2 / 3), convert(RealT, 2 / 5),
                             convert(RealT, 2 / 7)) / (x + y)
        else
            return (log_y - log_x) / (y - x)
        end
    end

    @inline function flux_turbo(::typeof(flux_hindenlang_gassner_nonconservative_powell),
                                rho_ll, v1_ll, v2_ll, v3_ll, p_ll, B1_ll, B2_ll,
                                B3_ll,
                                psi_ll, log_rho_ll, log_p_ll,
                                rho_rr, v1_rr, v2_rr, v3_rr, p_rr, B1_rr, B2_rr,
                                B3_rr,
                                psi_rr, log_rho_rr, log_p_rr,
                                n1, n2, n3, equations::IdealGlmMhdEquations3D)
        v_dot_n_ll = v1_ll * n1 + v2_ll * n2 + v3_ll * n3
        v_dot_n_rr = v1_rr * n1 + v2_rr * n2 + v3_rr * n3
        B_dot_n_ll = B1_ll * n1 + B2_ll * n2 + B3_ll * n3
        B_dot_n_rr = B1_rr * n1 + B2_rr * n2 + B3_rr * n3

        # Both logarithmic means reuse the per-node logarithms:
        #   log(rho_ll * p_rr) = log_rho_ll + log_p_rr, and likewise for the other one.
        rho_mean = ln_mean_pre(rho_ll, rho_rr, log_rho_ll, log_rho_rr)
        a = rho_ll * p_rr
        b = rho_rr * p_ll
        inv_rho_p_mean = p_ll * p_rr *
                         inv_ln_mean_pre(a, b, log_rho_ll + log_p_rr, log_rho_rr + log_p_ll)

        v1_avg = 0.5f0 * (v1_ll + v1_rr)
        v2_avg = 0.5f0 * (v2_ll + v2_rr)
        v3_avg = 0.5f0 * (v3_ll + v3_rr)
        p_avg = 0.5f0 * (p_ll + p_rr)
        psi_avg = 0.5f0 * (psi_ll + psi_rr)
        velocity_square_avg = 0.5f0 * (v1_ll * v1_rr + v2_ll * v2_rr + v3_ll * v3_rr)
        magnetic_square_avg = 0.5f0 * (B1_ll * B1_rr + B2_ll * B2_rr + B3_ll * B3_rr)

        f1 = rho_mean * 0.5f0 * (v_dot_n_ll + v_dot_n_rr)
        f2 = (f1 * v1_avg + (p_avg + magnetic_square_avg) * n1
              -
              0.5f0 * (B_dot_n_ll * B1_rr + B_dot_n_rr * B1_ll))
        f3 = (f1 * v2_avg + (p_avg + magnetic_square_avg) * n2
              -
              0.5f0 * (B_dot_n_ll * B2_rr + B_dot_n_rr * B2_ll))
        f4 = (f1 * v3_avg + (p_avg + magnetic_square_avg) * n3
              -
              0.5f0 * (B_dot_n_ll * B3_rr + B_dot_n_rr * B3_ll))
        f6 = (equations.c_h * psi_avg * n1
              +
              0.5f0 * (v_dot_n_ll * B1_ll - v1_ll * B_dot_n_ll +
               v_dot_n_rr * B1_rr - v1_rr * B_dot_n_rr))
        f7 = (equations.c_h * psi_avg * n2
              +
              0.5f0 * (v_dot_n_ll * B2_ll - v2_ll * B_dot_n_ll +
               v_dot_n_rr * B2_rr - v2_rr * B_dot_n_rr))
        f8 = (equations.c_h * psi_avg * n3
              +
              0.5f0 * (v_dot_n_ll * B3_ll - v3_ll * B_dot_n_ll +
               v_dot_n_rr * B3_rr - v3_rr * B_dot_n_rr))
        f9 = equations.c_h * 0.5f0 * (B_dot_n_ll + B_dot_n_rr)
        f5 = (f1 * (velocity_square_avg + inv_rho_p_mean * equations.inv_gamma_minus_one)
              +
              0.5f0 * (+p_ll * v_dot_n_rr + p_rr * v_dot_n_ll
               + (v_dot_n_ll * B1_ll * B1_rr + v_dot_n_rr * B1_rr * B1_ll)
               + (v_dot_n_ll * B2_ll * B2_rr + v_dot_n_rr * B2_rr * B2_ll)
               + (v_dot_n_ll * B3_ll * B3_rr + v_dot_n_rr * B3_rr * B3_ll)
               -
               (v1_ll * B_dot_n_ll * B1_rr + v1_rr * B_dot_n_rr * B1_ll)
               -
               (v2_ll * B_dot_n_ll * B2_rr + v2_rr * B_dot_n_rr * B2_ll)
               -
               (v3_ll * B_dot_n_ll * B3_rr + v3_rr * B_dot_n_rr * B3_ll)
               +
               equations.c_h * (B_dot_n_ll * psi_rr + B_dot_n_rr * psi_ll)))

        v_dot_B_ll = v1_ll * B1_ll + v2_ll * B2_ll + v3_ll * B3_ll
        v_dot_B_rr = v1_rr * B1_rr + v2_rr * B2_rr + v3_rr * B3_rr

        # Powell + Galilean nonconservative terms. The factor 2 in the nonconservative
        # flux and the factor 1/2 of the combined-flux contract cancel exactly, so the
        # terms are added with no prefactor beyond the 1/2 of the two-point average.
        flux_left = SVector(f1,
                            f2 + 0.5f0 * (B1_ll * B_dot_n_rr),
                            f3 + 0.5f0 * (B2_ll * B_dot_n_rr),
                            f4 + 0.5f0 * (B3_ll * B_dot_n_rr),
                            f5 +
                            0.5f0 *
                            (v_dot_B_ll * B_dot_n_rr + v_dot_n_ll * psi_ll * psi_rr),
                            f6 + 0.5f0 * (v1_ll * B_dot_n_rr),
                            f7 + 0.5f0 * (v2_ll * B_dot_n_rr),
                            f8 + 0.5f0 * (v3_ll * B_dot_n_rr),
                            f9 + 0.5f0 * (v_dot_n_ll * psi_rr))
        flux_right = SVector(f1,
                             f2 + 0.5f0 * (B1_rr * B_dot_n_ll),
                             f3 + 0.5f0 * (B2_rr * B_dot_n_ll),
                             f4 + 0.5f0 * (B3_rr * B_dot_n_ll),
                             f5 +
                             0.5f0 *
                             (v_dot_B_rr * B_dot_n_ll + v_dot_n_rr * psi_rr * psi_ll),
                             f6 + 0.5f0 * (v1_rr * B_dot_n_ll),
                             f7 + 0.5f0 * (v2_rr * B_dot_n_ll),
                             f8 + 0.5f0 * (v3_rr * B_dot_n_ll),
                             f9 + 0.5f0 * (v_dot_n_rr * psi_ll))
        return flux_left, flux_right
    end
end # @muladd
