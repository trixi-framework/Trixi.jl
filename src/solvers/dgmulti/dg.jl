# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# out <- A*x
mul_by!(A) = @inline (out, x) -> matmul!(out, A, x)
mul_by!(A::T) where {T <: SimpleKronecker} = @inline (out, x) -> mul!(out, A, x)
mul_by!(A::AbstractSparseMatrix) = @inline (out, x) -> mul!(out, A, x)
function mul_by!(A::LinearAlgebra.AdjOrTrans{T, S}) where {T, S <: AbstractSparseMatrix}
    @inline (out, x) -> mul!(out, A, x)
end

#  out <- out + α * A * x
mul_by_accum!(A, α) = @inline (out, x) -> matmul!(out, A, x, α, One())
function mul_by_accum!(A::AbstractSparseMatrix, α)
    @inline (out, x) -> mul!(out, A, x, α, One())
end

# out <- out + A * x
mul_by_accum!(A) = mul_by_accum!(A, One())

# specialize for SBP operators since `matmul!` doesn't work for `UniformScaling` types.
struct MulByUniformScaling end
struct MulByAccumUniformScaling end
mul_by!(A::UniformScaling) = MulByUniformScaling()
mul_by_accum!(A::UniformScaling) = MulByAccumUniformScaling()

# StructArray fallback
@inline function apply_to_each_field(f::F, args::Vararg{Any, N}) where {F, N}
    StructArrays.foreachfield(f, args...)
end

# specialize for UniformScaling types: works for either StructArray{SVector} or Matrix{SVector}
# solution storage formats.
@inline apply_to_each_field(f::MulByUniformScaling, out, x, args...) = copy!(out, x)
@inline function apply_to_each_field(f::MulByAccumUniformScaling, out, x, args...)
    @threaded for i in eachindex(x)
        out[i] = out[i] + x[i]
    end
end

"""
    eachdim(mesh)

Return an iterator over the indices that specify the location in relevant data structures
for the dimensions in `AbstractTree`.
In particular, not the dimensions themselves are returned.
"""
@inline eachdim(mesh) = Base.OneTo(ndims(mesh))

# iteration over all elements in a mesh
@inline function ndofs(mesh::DGMultiMesh, dg::DGMulti, other_args...)
    dg.basis.Np * mesh.md.num_elements
end
"""
    eachelement(mesh::DGMultiMesh, dg::DGMulti, other_args...)

Return an iterator over the indices that specify the location in relevant data structures
for the elements in `mesh`.
In particular, not the elements themselves are returned.
"""
@inline function eachelement(mesh::DGMultiMesh, dg::DGMulti, other_args...)
    Base.OneTo(mesh.md.num_elements)
end

# iteration over quantities in a single element
@inline nnodes(basis::RefElemData) = basis.Np

"""
    each_face_node(mesh::DGMultiMesh, dg::DGMulti, other_args...)

Return an iterator over the indices that specify the location in relevant data structures
for the face nodes in `dg`.
In particular, not the face_nodes themselves are returned.
"""
@inline function each_face_node(mesh::DGMultiMesh, dg::DGMulti, other_args...)
    Base.OneTo(dg.basis.Nfq)
end

"""
    each_quad_node(mesh::DGMultiMesh, dg::DGMulti, other_args...)

Return an iterator over the indices that specify the location in relevant data structures
for the quadrature nodes in `dg`.
In particular, not the quadrature nodes themselves are returned.
"""
@inline function each_quad_node(mesh::DGMultiMesh, dg::DGMulti, other_args...)
    Base.OneTo(dg.basis.Nq)
end

# iteration over quantities over the entire mesh (dofs, quad nodes, face nodes).
"""
    each_dof_global(mesh::DGMultiMesh, dg::DGMulti, other_args...)

Return an iterator over the indices that specify the location in relevant data structures
for the degrees of freedom (DOF) in `dg`.
In particular, not the DOFs themselves are returned.
"""
@inline function each_dof_global(mesh::DGMultiMesh, dg::DGMulti, other_args...)
    Base.OneTo(ndofs(mesh, dg, other_args...))
end

"""
    each_quad_node_global(mesh::DGMultiMesh, dg::DGMulti, other_args...)

Return an iterator over the indices that specify the location in relevant data structures
for the global quadrature nodes in `mesh`.
In particular, not the quadrature nodes themselves are returned.
"""
@inline function each_quad_node_global(mesh::DGMultiMesh, dg::DGMulti, other_args...)
    Base.OneTo(dg.basis.Nq * mesh.md.num_elements)
end

"""
    each_face_node_global(mesh::DGMultiMesh, dg::DGMulti, other_args...)

Return an iterator over the indices that specify the location in relevant data structures
for the face nodes in `mesh`.
In particular, not the face nodes themselves are returned.
"""
@inline function each_face_node_global(mesh::DGMultiMesh, dg::DGMulti, other_args...)
    Base.OneTo(dg.basis.Nfq * mesh.md.num_elements)
end

# interface with semidiscretization_hyperbolic
wrap_array(u_ode, mesh::DGMultiMesh, equations, dg::DGMulti, cache) = u_ode
wrap_array_native(u_ode, mesh::DGMultiMesh, equations, dg::DGMulti, cache) = u_ode
function digest_boundary_conditions(boundary_conditions::NamedTuple{Keys, ValueTypes},
                                    mesh::DGMultiMesh,
                                    dg::DGMulti,
                                    cache) where {Keys, ValueTypes <: NTuple{N, Any}
                                                  } where {N}
    return boundary_conditions
end

# Allocate nested array type for DGMulti solution storage.
function allocate_nested_array(uEltype, nvars, array_dimensions, dg)
    # store components as separate arrays, combine via StructArrays
    return StructArray{SVector{nvars, uEltype}}(ntuple(_ -> zeros(uEltype,
                                                                  array_dimensions...),
                                                       nvars))
end

function reset_du!(du, dg::DGMulti, other_args...)
    @threaded for i in eachindex(du)
        du[i] = zero(eltype(du))
    end

    return du
end

# Constructs cache variables for both affine and non-affine (curved) DGMultiMeshes
function create_cache(mesh::DGMultiMesh{NDIMS}, equations, dg::DGMultiWeakForm, RealT,
                      uEltype) where {NDIMS}
    rd = dg.basis
    md = mesh.md

    # volume quadrature weights, volume interpolation matrix, mass matrix, differentiation matrices
    @unpack wq, Vq, M, Drst = rd

    # ∫f(u) * dv/dx_i = ∑_j (Vq*Drst[i])'*diagm(wq)*(rstxyzJ[i,j].*f(Vq*u))
    weak_differentiation_matrices = map(D -> -M \ ((Vq * D)' * Diagonal(wq)), Drst)

    nvars = nvariables(equations)

    # storage for volume quadrature values, face quadrature values, flux values
    u_values = allocate_nested_array(uEltype, nvars, size(md.xq), dg)
    u_face_values = allocate_nested_array(uEltype, nvars, size(md.xf), dg)
    flux_face_values = allocate_nested_array(uEltype, nvars, size(md.xf), dg)
    if typeof(rd.approximation_type) <:
       Union{SBP, AbstractNonperiodicDerivativeOperator}
        lift_scalings = rd.wf ./ rd.wq[rd.Fmask] # lift scalings for diag-norm SBP operators
    else
        lift_scalings = nothing
    end

    # local storage for volume integral and source computations
    local_values_threaded = [allocate_nested_array(uEltype, nvars, (rd.Nq,), dg)
                             for _ in 1:Threads.nthreads()]

    # For curved meshes, we interpolate geometric terms from nodal points to quadrature points.
    # For affine meshes, we just access one element of this interpolated data.
    dxidxhatj = map(x -> rd.Vq * x, md.rstxyzJ)

    # interpolate J to quadrature points for weight-adjusted DG (WADG)
    invJ = inv.(rd.Vq * md.J)

    # for scaling by curved geometric terms (not used by affine DGMultiMesh)
    flux_threaded = [[allocate_nested_array(uEltype, nvars, (rd.Nq,), dg)
                      for _ in 1:NDIMS] for _ in 1:Threads.nthreads()]
    rotated_flux_threaded = [allocate_nested_array(uEltype, nvars, (rd.Nq,), dg)
                             for _ in 1:Threads.nthreads()]

    return (; md, weak_differentiation_matrices, lift_scalings, invJ, dxidxhatj,
            u_values, u_face_values, flux_face_values,
            local_values_threaded, flux_threaded, rotated_flux_threaded)
end

function allocate_coefficients(mesh::DGMultiMesh, equations, dg::DGMulti, cache)
    return allocate_nested_array(real(dg), nvariables(equations), size(mesh.md.x), dg)
end

function compute_coefficients!(u, initial_condition, t,
                               mesh::DGMultiMesh, equations, dg::DGMulti, cache)
    md = mesh.md
    rd = dg.basis
    @unpack u_values = cache

    # evaluate the initial condition at quadrature points
    @threaded for i in each_quad_node_global(mesh, dg, cache)
        u_values[i] = initial_condition(SVector(getindex.(md.xyzq, i)),
                                        t, equations)
    end

    # multiplying by Pq computes the L2 projection
    apply_to_each_field(mul_by!(rd.Pq), u, u_values)
end

# estimates the timestep based on polynomial degree and mesh. Does not account for physics (e.g.,
# computes an estimate of `dt` based on the advection equation with constant unit advection speed).
function estimate_dt(mesh::DGMultiMesh, dg::DGMulti)
    rd = dg.basis # RefElemData
    return StartUpDG.estimate_h(rd, mesh.md) / StartUpDG.inverse_trace_constant(rd)
end

dt_polydeg_scaling(dg::DGMulti) = inv(dg.basis.N + 1)
function dt_polydeg_scaling(dg::DGMulti{3, <:Wedge, <:TensorProductWedge})
    inv(maximum(dg.basis.N) + 1)
end

# for the stepsize callback
function max_dt(u, t, mesh::DGMultiMesh,
                constant_speed::False, equations, dg::DGMulti{NDIMS},
                cache) where {NDIMS}
    @unpack md = mesh
    rd = dg.basis

    dt_min = Inf
    for e in eachelement(mesh, dg, cache)
        h_e = StartUpDG.estimate_h(e, rd, md)
        max_speeds = ntuple(_ -> nextfloat(zero(t)), NDIMS)
        for i in Base.OneTo(rd.Np) # loop over nodes
            lambda_i = max_abs_speeds(u[i, e], equations)
            max_speeds = max.(max_speeds, lambda_i)
        end
        dt_min = min(dt_min, h_e / sum(max_speeds))
    end
    # This mimics `max_dt` for `TreeMesh`, except that `nnodes(dg)` is replaced by
    # `polydeg+1`. This is because `nnodes(dg)` returns the total number of
    # multi-dimensional nodes for DGMulti solver types, while `nnodes(dg)` returns
    # the number of 1D nodes for `DGSEM` solvers.
    return 2 * dt_min * dt_polydeg_scaling(dg)
end

function max_dt(u, t, mesh::DGMultiMesh,
                constant_speed::True, equations, dg::DGMulti{NDIMS},
                cache) where {NDIMS}
    @unpack md = mesh
    rd = dg.basis

    dt_min = Inf
    for e in eachelement(mesh, dg, cache)
        h_e = StartUpDG.estimate_h(e, rd, md)
        max_speeds = ntuple(_ -> nextfloat(zero(t)), NDIMS)
        for i in Base.OneTo(rd.Np) # loop over nodes
            max_speeds = max.(max_abs_speeds(equations), max_speeds)
        end
        dt_min = min(dt_min, h_e / sum(max_speeds))
    end
    # This mimics `max_dt` for `TreeMesh`, except that `nnodes(dg)` is replaced by
    # `polydeg+1`. This is because `nnodes(dg)` returns the total number of
    # multi-dimensional nodes for DGMulti solver types, while `nnodes(dg)` returns
    # the number of 1D nodes for `DGSEM` solvers.
    return 2 * dt_min * dt_polydeg_scaling(dg)
end

# interpolates from solution coefficients to face quadrature points
# We pass the `surface_integral` argument solely for dispatch
function prolong2interfaces!(cache, u, mesh::DGMultiMesh, equations,
                             surface_integral, dg::DGMulti)
    rd = dg.basis
    @unpack u_face_values = cache
    apply_to_each_field(mul_by!(rd.Vf), u_face_values, u)
end

# version for affine meshes
function calc_volume_integral!(du, u, mesh::DGMultiMesh,
                               have_nonconservative_terms::False, equations,
                               volume_integral::VolumeIntegralWeakForm, dg::DGMulti,
                               cache)
    rd = dg.basis
    md = mesh.md
    @unpack weak_differentiation_matrices, dxidxhatj, u_values, local_values_threaded = cache
    @unpack rstxyzJ = md # geometric terms

    # interpolate to quadrature points
    apply_to_each_field(mul_by!(rd.Vq), u_values, u)

    @threaded for e in eachelement(mesh, dg, cache)
        flux_values = local_values_threaded[Threads.threadid()]
        for i in eachdim(mesh)
            # Here, the broadcasting operation does allocate
            #flux_values .= flux.(view(u_values, :, e), i, equations)
            # Use loop instead
            for j in eachindex(flux_values)
                flux_values[j] = flux(u_values[j, e], i, equations)
            end
            for j in eachdim(mesh)
                apply_to_each_field(mul_by_accum!(weak_differentiation_matrices[j],
                                                  dxidxhatj[i, j][1, e]),
                                    view(du, :, e), flux_values)
            end
        end
    end
end

# version for curved meshes
function calc_volume_integral!(du, u, mesh::DGMultiMesh{NDIMS, <:NonAffine},
                               have_nonconservative_terms::False, equations,
                               volume_integral::VolumeIntegralWeakForm, dg::DGMulti,
                               cache) where {NDIMS}
    rd = dg.basis
    (; weak_differentiation_matrices, u_values) = cache
    (; dxidxhatj) = cache

    # interpolate to quadrature points
    apply_to_each_field(mul_by!(rd.Vq), u_values, u)

    @threaded for e in eachelement(mesh, dg, cache)
        flux_values = cache.flux_threaded[Threads.threadid()]
        for i in eachdim(mesh)
            # Here, the broadcasting operation does not allocate
            flux_values[i] .= flux.(view(u_values, :, e), i, equations)
        end

        # rotate flux with df_i/dx_i = sum_j d(x_i)/d(x̂_j) * d(f_i)/d(x̂_j).
        # Example: df_x/dx + df_y/dy = dr/dx * df_x/dr + ds/dx * df_x/ds
        #                  + dr/dy * df_y/dr + ds/dy * df_y/ds
        #                  = Dr * (dr/dx * fx + dr/dy * fy) + Ds * (...)
        #                  = Dr * (f_r) + Ds * (f_s)

        rotated_flux_values = cache.rotated_flux_threaded[Threads.threadid()]
        for j in eachdim(mesh)
            fill!(rotated_flux_values, zero(eltype(rotated_flux_values)))

            # compute rotated fluxes
            for i in eachdim(mesh)
                for ii in eachindex(rotated_flux_values)
                    flux_i_node = flux_values[i][ii]
                    dxidxhatj_node = dxidxhatj[i, j][ii, e]
                    rotated_flux_values[ii] = rotated_flux_values[ii] +
                                              dxidxhatj_node * flux_i_node
                end
            end

            # apply weak differentiation matrices to rotated fluxes
            apply_to_each_field(mul_by_accum!(weak_differentiation_matrices[j]),
                                view(du, :, e), rotated_flux_values)
        end
    end
end

function calc_interface_flux!(cache, surface_integral::SurfaceIntegralWeakForm,
                              mesh::DGMultiMesh,
                              have_nonconservative_terms::False, equations,
                              dg::DGMulti{NDIMS}) where {NDIMS}
    @unpack surface_flux = surface_integral
    md = mesh.md
    @unpack mapM, mapP, nxyzJ, Jf = md
    @unpack u_face_values, flux_face_values = cache

    @threaded for face_node_index in each_face_node_global(mesh, dg, cache)

        # inner (idM -> minus) and outer (idP -> plus) indices
        idM, idP = mapM[face_node_index], mapP[face_node_index]
        uM = u_face_values[idM]
        uP = u_face_values[idP]
        normal = SVector{NDIMS}(getindex.(nxyzJ, idM)) / Jf[idM]
        flux_face_values[idM] = surface_flux(uM, uP, normal, equations) * Jf[idM]
    end
end

function calc_interface_flux!(cache, surface_integral::SurfaceIntegralWeakForm,
                              mesh::DGMultiMesh,
                              have_nonconservative_terms::True, equations,
                              dg::DGMulti{NDIMS}) where {NDIMS}
    flux_conservative, flux_nonconservative = surface_integral.surface_flux
    md = mesh.md
    @unpack mapM, mapP, nxyzJ, Jf = md
    @unpack u_face_values, flux_face_values = cache

    @threaded for face_node_index in each_face_node_global(mesh, dg, cache)

        # inner (idM -> minus) and outer (idP -> plus) indices
        idM, idP = mapM[face_node_index], mapP[face_node_index]
        uM = u_face_values[idM]

        # compute flux if node is not a boundary node
        if idM != idP
            uP = u_face_values[idP]
            normal = SVector{NDIMS}(getindex.(nxyzJ, idM)) / Jf[idM]
            conservative_part = flux_conservative(uM, uP, normal, equations)

            # Two notes on the use of `flux_nonconservative`:
            # 1. In contrast to other mesh types, only one nonconservative part needs to be
            #    computed since we loop over the elements, not the unique interfaces.
            nonconservative_part = flux_nonconservative(uM, uP, normal, equations)
            # The factor 0.5 is necessary for the nonconservative fluxes based on the
            # interpretation of global SBP operators.
            flux_face_values[idM] = (conservative_part + 0.5 * nonconservative_part) *
                                    Jf[idM]
        end
    end
end

# assumes cache.flux_face_values is computed and filled with
# for polyomial discretizations, use dense LIFT matrix for surface contributions.
function calc_surface_integral!(du, u, mesh::DGMultiMesh, equations,
                                surface_integral::SurfaceIntegralWeakForm,
                                dg::DGMulti, cache)
    rd = dg.basis
    apply_to_each_field(mul_by_accum!(rd.LIFT), du, cache.flux_face_values)
end

# Specialize for nodal SBP discretizations. Uses that Vf*u = u[Fmask,:]
# We pass the `surface_integral` argument solely for dispatch
function prolong2interfaces!(cache, u, mesh::DGMultiMesh, equations, surface_integral,
                             dg::DGMultiSBP)
    rd = dg.basis
    @unpack Fmask = rd
    @unpack u_face_values = cache
    @threaded for e in eachelement(mesh, dg, cache)
        for (i, fid) in enumerate(Fmask)
            u_face_values[i, e] = u[fid, e]
        end
    end
end

# Specialize for nodal SBP discretizations. Uses that du = LIFT*u is equivalent to
# du[Fmask,:] .= u ./ rd.wq[rd.Fmask]
function calc_surface_integral!(du, u, mesh::DGMultiMesh, equations,
                                surface_integral::SurfaceIntegralWeakForm,
                                dg::DGMultiSBP, cache)
    rd = dg.basis
    @unpack flux_face_values, lift_scalings = cache

    @threaded for e in eachelement(mesh, dg, cache)
        for i in each_face_node(mesh, dg, cache)
            fid = rd.Fmask[i]
            du[fid, e] = du[fid, e] + flux_face_values[i, e] * lift_scalings[i]
        end
    end
end

# do nothing for periodic (default) boundary conditions
function calc_boundary_flux!(cache, t, boundary_conditions::BoundaryConditionPeriodic,
                             mesh, have_nonconservative_terms, equations, dg::DGMulti)
    nothing
end

function calc_boundary_flux!(cache, t, boundary_conditions, mesh,
                             have_nonconservative_terms, equations, dg::DGMulti)
    for (key, value) in zip(keys(boundary_conditions), boundary_conditions)
        calc_single_boundary_flux!(cache, t, value,
                                   key,
                                   mesh, have_nonconservative_terms, equations, dg)
    end
end

function calc_single_boundary_flux!(cache, t, boundary_condition, boundary_key, mesh,
                                    have_nonconservative_terms::False, equations,
                                    dg::DGMulti{NDIMS}) where {NDIMS}
    rd = dg.basis
    md = mesh.md
    @unpack u_face_values, flux_face_values = cache
    @unpack xyzf, nxyzJ, Jf = md
    @unpack surface_flux = dg.surface_integral

    # reshape face/normal arrays to have size = (num_points_on_face, num_faces_total).
    # mesh.boundary_faces indexes into the columns of these face-reshaped arrays.
    num_faces = StartUpDG.num_faces(rd.element_type)
    num_pts_per_face = rd.Nfq ÷ num_faces
    num_faces_total = num_faces * md.num_elements

    # This function was originally defined as
    # `reshape_by_face(u) = reshape(view(u, :), num_pts_per_face, num_faces_total)`.
    # This results in allocations due to https://github.com/JuliaLang/julia/issues/36313.
    # To avoid allocations, we use Tim Holy's suggestion:
    # https://github.com/JuliaLang/julia/issues/36313#issuecomment-782336300.
    reshape_by_face(u) = Base.ReshapedArray(u, (num_pts_per_face, num_faces_total), ())

    u_face_values = reshape_by_face(u_face_values)
    flux_face_values = reshape_by_face(flux_face_values)
    Jf = reshape_by_face(Jf)
    nxyzJ, xyzf = reshape_by_face.(nxyzJ), reshape_by_face.(xyzf) # broadcast over nxyzJ::NTuple{NDIMS,Matrix}

    # loop through boundary faces, which correspond to columns of reshaped u_face_values, ...
    for f in mesh.boundary_faces[boundary_key]
        for i in Base.OneTo(num_pts_per_face)
            face_normal = SVector{NDIMS}(getindex.(nxyzJ, i, f)) / Jf[i, f]
            face_coordinates = SVector{NDIMS}(getindex.(xyzf, i, f))
            flux_face_values[i, f] = boundary_condition(u_face_values[i, f],
                                                        face_normal, face_coordinates,
                                                        t,
                                                        surface_flux, equations) *
                                     Jf[i, f]
        end
    end

    # Note: modifying the values of the reshaped array modifies the values of cache.flux_face_values.
    # However, we don't have to re-reshape, since cache.flux_face_values still retains its original shape.
end

function calc_single_boundary_flux!(cache, t, boundary_condition, boundary_key, mesh,
                                    have_nonconservative_terms::True, equations,
                                    dg::DGMulti{NDIMS}) where {NDIMS}
    rd = dg.basis
    md = mesh.md
    surface_flux, nonconservative_flux = dg.surface_integral.surface_flux

    # reshape face/normal arrays to have size = (num_points_on_face, num_faces_total).
    # mesh.boundary_faces indexes into the columns of these face-reshaped arrays.
    num_pts_per_face = rd.Nfq ÷ StartUpDG.num_faces(rd.element_type)
    num_faces_total = StartUpDG.num_faces(rd.element_type) * md.num_elements

    # This function was originally defined as
    # `reshape_by_face(u) = reshape(view(u, :), num_pts_per_face, num_faces_total)`.
    # This results in allocations due to https://github.com/JuliaLang/julia/issues/36313.
    # To avoid allocations, we use Tim Holy's suggestion:
    # https://github.com/JuliaLang/julia/issues/36313#issuecomment-782336300.
    reshape_by_face(u) = Base.ReshapedArray(u, (num_pts_per_face, num_faces_total), ())

    u_face_values = reshape_by_face(cache.u_face_values)
    flux_face_values = reshape_by_face(cache.flux_face_values)
    Jf = reshape_by_face(md.Jf)
    nxyzJ, xyzf = reshape_by_face.(md.nxyzJ), reshape_by_face.(md.xyzf) # broadcast over nxyzJ::NTuple{NDIMS,Matrix}

    # loop through boundary faces, which correspond to columns of reshaped u_face_values, ...
    for f in mesh.boundary_faces[boundary_key]
        for i in Base.OneTo(num_pts_per_face)
            face_normal = SVector{NDIMS}(getindex.(nxyzJ, i, f)) / Jf[i, f]
            face_coordinates = SVector{NDIMS}(getindex.(xyzf, i, f))

            # Compute conservative and non-conservative fluxes separately.
            # This imposes boundary conditions on the conservative part of the flux.
            cons_flux_at_face_node = boundary_condition(u_face_values[i, f],
                                                        face_normal, face_coordinates,
                                                        t,
                                                        surface_flux, equations)

            # Compute pointwise nonconservative numerical flux at the boundary.
            noncons_flux_at_face_node = boundary_condition(u_face_values[i, f],
                                                           face_normal,
                                                           face_coordinates,
                                                           t,
                                                           nonconservative_flux,
                                                           equations)

            flux_face_values[i, f] = (cons_flux_at_face_node +
                                      0.5 * noncons_flux_at_face_node) * Jf[i, f]
        end
    end

    # Note: modifying the values of the reshaped array modifies the values of cache.flux_face_values.
    # However, we don't have to re-reshape, since cache.flux_face_values still retains its original shape.
end

# inverts Jacobian and scales by -1.0
function invert_jacobian!(du, mesh::DGMultiMesh, equations, dg::DGMulti, cache;
                          scaling = -1)
    @threaded for e in eachelement(mesh, dg, cache)
        invJ = cache.invJ[1, e]
        for i in axes(du, 1)
            du[i, e] *= scaling * invJ
        end
    end
end

# inverts Jacobian using weight-adjusted DG, and scales by -1.0.
# - Chan, Jesse, Russell J. Hewett, and Timothy Warburton.
#   "Weight-adjusted discontinuous Galerkin methods: curvilinear meshes."
#   https://doi.org/10.1137/16M1089198
function invert_jacobian!(du, mesh::DGMultiMesh{NDIMS, <:NonAffine}, equations,
                          dg::DGMulti, cache; scaling = -1) where {NDIMS}
    # Vq = interpolation matrix to quadrature points, Pq = quadrature-based L2 projection matrix
    (; Pq, Vq) = dg.basis
    (; local_values_threaded, invJ) = cache

    @threaded for e in eachelement(mesh, dg, cache)
        du_at_quad_points = local_values_threaded[Threads.threadid()]

        # interpolate solution to quadrature
        apply_to_each_field(mul_by!(Vq), du_at_quad_points, view(du, :, e))

        # scale by quadrature points
        for i in eachindex(du_at_quad_points)
            du_at_quad_points[i] *= scaling * invJ[i, e]
        end

        # project back to polynomials
        apply_to_each_field(mul_by!(Pq), view(du, :, e), du_at_quad_points)
    end
end

# Multiple calc_sources! to resolve method ambiguities
function calc_sources!(du, u, t, source_terms::Nothing,
                       mesh, equations, dg::DGMulti, cache)
    nothing
end
function calc_sources!(du, u, t, source_terms::Nothing,
                       mesh, equations, dg::DGMultiFluxDiffSBP, cache)
    nothing
end

# uses quadrature + projection to compute source terms.
function calc_sources!(du, u, t, source_terms,
                       mesh, equations, dg::DGMulti, cache)
    rd = dg.basis
    md = mesh.md
    @unpack Pq = rd
    @unpack u_values, local_values_threaded = cache
    @threaded for e in eachelement(mesh, dg, cache)
        source_values = local_values_threaded[Threads.threadid()]

        u_e = view(u_values, :, e) # u_values should already be computed from volume integral

        for i in each_quad_node(mesh, dg, cache)
            source_values[i] = source_terms(u_e[i], SVector(getindex.(md.xyzq, i, e)),
                                            t, equations)
        end
        apply_to_each_field(mul_by_accum!(Pq), view(du, :, e), source_values)
    end
end

function rhs!(du, u, t, mesh, equations,
              boundary_conditions::BC, source_terms::Source,
              dg::DGMulti, cache) where {BC, Source}
    @trixi_timeit timer() "reset ∂u/∂t" reset_du!(du, dg, cache)

    @trixi_timeit timer() "volume integral" begin
        calc_volume_integral!(du, u, mesh,
                              have_nonconservative_terms(equations), equations,
                              dg.volume_integral, dg, cache)
    end

    @trixi_timeit timer() "prolong2interfaces" begin
        prolong2interfaces!(cache, u, mesh, equations, dg.surface_integral, dg)
    end

    @trixi_timeit timer() "interface flux" begin
        calc_interface_flux!(cache, dg.surface_integral, mesh,
                             have_nonconservative_terms(equations), equations, dg)
    end

    @trixi_timeit timer() "boundary flux" begin
        calc_boundary_flux!(cache, t, boundary_conditions, mesh,
                            have_nonconservative_terms(equations), equations, dg)
    end

    @trixi_timeit timer() "surface integral" begin
        calc_surface_integral!(du, u, mesh, equations, dg.surface_integral, dg, cache)
    end

    @trixi_timeit timer() "Jacobian" invert_jacobian!(du, mesh, equations, dg, cache)

    @trixi_timeit timer() "source terms" begin
        calc_sources!(du, u, t, source_terms, mesh, equations, dg, cache)
    end

    return nothing
end
end # @muladd
