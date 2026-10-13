# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function calc_dt_for_cleaning_speed(cfl::Real,
                                    mesh::TreeMesh,
                                    equations::Union{AbstractIdealGlmMhdEquations,
                                                     AbstractIdealGlmMhdMulticomponentEquations,
                                                     AbstractIdealGlmMhdMultiIonEquations},
                                    dg::DG, cache)
    # compute time step for GLM linear advection equation with c_h=1 for the DG discretization on
    # Cartesian meshes
    max_scaled_speed_for_c_h = maximum(cache.elements.inverse_jacobian) *
                               ndims(equations)

    if mpi_isparallel()
        # Base.max instead of max needed, see comment in src/auxiliary/math.jl
        max_scaled_speed_for_c_h = MPI.Allreduce!(Ref(max_scaled_speed_for_c_h),
                                                  Base.max,
                                                  mpi_comm())[]
    end

    # Note: This depends on the implementation details of the StepsizeCallback and needs to be adapted
    #      as well if that callback changes.
    return cfl * 2 / (nnodes(dg) * max_scaled_speed_for_c_h)
end

# On curvilinear meshes, `inverse_jacobian` is the inverse of the Jacobian determinant,
# which scales like `(2 / Δx)^ndims` instead of `2 / Δx`. Thus, we need to transform the
# speed of the GLM waves to the reference element using the contravariant vectors as in
# `max_dt`. Since the GLM waves propagate with speed `c_h = 1` in all directions, the
# maximal transformed speed in reference direction `i` is the norm of the contravariant
# vector `Ja^i` times the inverse Jacobian. On Cartesian elements, this reduces to the
# value `ndims * 2 / Δx` used for the `TreeMesh` above.
# See also the implementation in FLUXO:
# https://github.com/project-fluxo/fluxo/blob/c7e0cc9b7fd4569dcab67bbb6e5a25c0a84859f1/src/equation/mhd/calctimestep.f90#L203-L218
function calc_dt_for_cleaning_speed(cfl::Real,
                                    mesh::Union{StructuredMesh{2}, UnstructuredMesh2D,
                                                P4estMesh{2}, P4estMeshView{2},
                                                T8codeMesh{2}, StructuredMeshView{2},
                                                StructuredMesh{3}, P4estMesh{3},
                                                T8codeMesh{3}},
                                    equations::Union{AbstractIdealGlmMhdEquations,
                                                     AbstractIdealGlmMhdMulticomponentEquations,
                                                     AbstractIdealGlmMhdMultiIonEquations},
                                    dg::DG, cache)
    backend = trixi_backend(cache.elements.inverse_jacobian)
    max_scaled_speed_for_c_h = calc_max_scaled_speed_for_c_h(backend, mesh, dg, cache)

    if mpi_isparallel()
        # Base.max instead of max needed, see comment in src/auxiliary/math.jl
        max_scaled_speed_for_c_h = MPI.Allreduce!(Ref(max_scaled_speed_for_c_h),
                                                  Base.max,
                                                  mpi_comm())[]
    end

    # Note: This depends on the implementation details of the StepsizeCallback and needs to be adapted
    #      as well if that callback changes.
    return cfl * 2 / (nnodes(dg) * max_scaled_speed_for_c_h)
end

function calc_max_scaled_speed_for_c_h(backend::Nothing, mesh, dg, cache)
    @unpack contravariant_vectors, inverse_jacobian = cache.elements

    max_scaled_speed = zero(eltype(inverse_jacobian))
    @batch reduction=(max, max_scaled_speed) for element in eachelement(dg, cache)
        max_lambda = max_scaled_speed_for_c_h_per_element(typeof(mesh), dg,
                                                          contravariant_vectors,
                                                          inverse_jacobian,
                                                          element)
        # Use `Base.max` to prevent silent failures, as `max` from `@fastmath` doesn't propagate
        # `NaN`s properly. See https://github.com/trixi-framework/Trixi.jl/pull/2445#discussion_r2336812323
        max_scaled_speed = Base.max(max_scaled_speed, max_lambda)
    end
    return max_scaled_speed
end

function calc_max_scaled_speed_for_c_h(backend::Backend, ::MeshT, dg,
                                       cache) where {MeshT}
    @unpack contravariant_vectors, inverse_jacobian = cache.elements

    num_elements = nelements(dg, cache)
    init = neutral = AcceleratedKernels.neutral_element(Base.max,
                                                        eltype(inverse_jacobian))

    # Provide a custom neutral and init element since we "reduce" over 1:num_elements
    max_scaled_speed = AcceleratedKernels.mapreduce(Base.max, 1:num_elements, backend;
                                                    init, neutral) do element
        max_scaled_speed_for_c_h_per_element(MeshT, dg,
                                             contravariant_vectors,
                                             inverse_jacobian,
                                             element)
    end

    return max_scaled_speed
end

@inline function max_scaled_speed_for_c_h_per_element(::Type{<:AbstractMesh{2}}, dg,
                                                      contravariant_vectors,
                                                      inverse_jacobian, element)
    max_lambda1 = max_lambda2 = zero(eltype(inverse_jacobian))
    for j in eachnode(dg), i in eachnode(dg)
        Ja1 = get_contravariant_vector(1, contravariant_vectors, i, j, element)
        Ja2 = get_contravariant_vector(2, contravariant_vectors, i, j, element)
        inv_jacobian = abs(inverse_jacobian[i, j, element])

        max_lambda1 = Base.max(max_lambda1, inv_jacobian * norm(Ja1))
        max_lambda2 = Base.max(max_lambda2, inv_jacobian * norm(Ja2))
    end

    return max_lambda1 + max_lambda2
end

@inline function max_scaled_speed_for_c_h_per_element(::Type{<:AbstractMesh{3}}, dg,
                                                      contravariant_vectors,
                                                      inverse_jacobian, element)
    max_lambda1 = max_lambda2 = max_lambda3 = zero(eltype(inverse_jacobian))
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        Ja1 = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
        Ja2 = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
        Ja3 = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)
        inv_jacobian = abs(inverse_jacobian[i, j, k, element])

        max_lambda1 = Base.max(max_lambda1, inv_jacobian * norm(Ja1))
        max_lambda2 = Base.max(max_lambda2, inv_jacobian * norm(Ja2))
        max_lambda3 = Base.max(max_lambda3, inv_jacobian * norm(Ja3))
    end

    return max_lambda1 + max_lambda2 + max_lambda3
end

function calc_dt_for_cleaning_speed(cfl::Real, mesh::DGMultiMesh,
                                    equations::Union{AbstractIdealGlmMhdEquations,
                                                     AbstractIdealGlmMhdMulticomponentEquations,
                                                     AbstractIdealGlmMhdMultiIonEquations},
                                    dg::DGMulti, cache)
    rd = dg.basis
    md = mesh.md

    # Compute time step for GLM linear advection equation with c_h=1 for a DGMulti discretization.
    # This mimics `max_dt` for `DGMulti` solvers with the speed 1 in each coordinate direction.
    # Note that the Jacobian determinant `md.J` scales like `h^ndims`, so it cannot be used
    # directly to estimate the element size `h` in multiple space dimensions.
    h_min = minimum(element -> StartUpDG.estimate_h(element, rd, md),
                    eachelement(mesh, dg, cache))
    max_scaled_speed_for_c_h = ndims(equations) / h_min

    if mpi_isparallel()
        # Base.max instead of max needed, see comment in src/auxiliary/math.jl
        max_scaled_speed_for_c_h = MPI.Allreduce!(Ref(max_scaled_speed_for_c_h),
                                                  Base.max,
                                                  mpi_comm())[]
    end

    return cfl * 2 * dt_polydeg_scaling(dg) / max_scaled_speed_for_c_h
end
end # @muladd
