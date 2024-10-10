# version for standard (e.g., non-entropy stable or flux differencing) schemes
function create_cache_parabolic(mesh::DGMultiMesh,
                                equations_hyperbolic::AbstractEquations,
                                equations_parabolic::AbstractEquationsParabolic,
                                dg::DGMulti, parabolic_scheme, RealT, uEltype)

    # default to taking derivatives of all hyperbolic variables
    # TODO: parabolic; utilize the parabolic variables in `equations_parabolic` to reduce memory usage in the parabolic cache
    nvars = nvariables(equations_hyperbolic)

    (; M, Vq, Pq, Drst) = dg.basis

    # gradient operators: map from nodes to quadrature
    strong_differentiation_matrices = map(A -> Vq * A, Drst)
    gradient_lift_matrix = Vq * dg.basis.LIFT

    # divergence operators: map from quadrature to nodes
    weak_differentiation_matrices = map(A -> (M \ (-A' * M * Pq)), Drst)
    divergence_lift_matrix = dg.basis.LIFT
    projection_face_interpolation_matrix = dg.basis.Vf * dg.basis.Pq

    # evaluate geometric terms at quadrature points in case the mesh is curved
    (; md) = mesh
    J = dg.basis.Vq * md.J
    invJ = inv.(J)
    dxidxhatj = map(x -> dg.basis.Vq * x, md.rstxyzJ)

    # u_transformed stores "transformed" variables for computing the gradient
    u_transformed = allocate_nested_array(uEltype, nvars, size(md.x), dg)
    gradients = SVector{ndims(mesh)}(ntuple(_ -> similar(u_transformed,
                                                         (dg.basis.Nq,
                                                          mesh.md.num_elements)),
                                            ndims(mesh)))
    flux_viscous = similar.(gradients)

    u_face_values = allocate_nested_array(uEltype, nvars, size(md.xf), dg)
    scalar_flux_face_values = similar(u_face_values)
    gradients_face_values = ntuple(_ -> similar(u_face_values), ndims(mesh))

    local_u_values_threaded = [similar(u_transformed, dg.basis.Nq)
                               for _ in 1:Threads.nthreads()]
    local_flux_viscous_threaded = [SVector{ndims(mesh)}(ntuple(_ -> similar(u_transformed,
                                                                            dg.basis.Nq),
                                                               ndims(mesh)))
                                   for _ in 1:Threads.nthreads()]
    local_flux_face_values_threaded = [similar(scalar_flux_face_values[:, 1])
                                       for _ in 1:Threads.nthreads()]

    return (; u_transformed, gradients, flux_viscous,
            weak_differentiation_matrices, strong_differentiation_matrices,
            gradient_lift_matrix, projection_face_interpolation_matrix,
            divergence_lift_matrix,
            dxidxhatj, J, invJ, # geometric terms
            u_face_values, gradients_face_values, scalar_flux_face_values,
            local_u_values_threaded, local_flux_viscous_threaded,
            local_flux_face_values_threaded)
end

# Transform solution variables prior to taking the gradient
# (e.g., conservative to primitive variables). Defaults to doing nothing.
# TODO: can we avoid copying data?
function transform_variables!(u_transformed, u, mesh,
                              equations_parabolic::AbstractEquationsParabolic,
                              dg::DGMulti, parabolic_scheme, cache, cache_parabolic)
    transformation = gradient_variable_transformation(equations_parabolic)

    @threaded for i in eachindex(u)
        u_transformed[i] = transformation(u[i], equations_parabolic)
    end
end

# TODO: reuse entropy projection computations for DGMultiFluxDiff{<:Polynomial} (including `GaussSBP` solvers)
function calc_gradient_surface_integral!(gradients, u, scalar_flux_face_values,
                                         mesh, equations::AbstractEquationsParabolic,
                                         dg::DGMulti, cache, cache_parabolic)
    (; gradient_lift_matrix, local_flux_face_values_threaded) = cache_parabolic
    @threaded for e in eachelement(mesh, dg)
        local_flux_values = local_flux_face_values_threaded[Threads.threadid()]
        for dim in eachdim(mesh)
            for i in eachindex(local_flux_values)
                # compute flux * (nx, ny, nz)
                local_flux_values[i] = scalar_flux_face_values[i, e] *
                                       mesh.md.nxyzJ[dim][i, e]
            end
            apply_to_each_field(mul_by_accum!(gradient_lift_matrix),
                                view(gradients[dim], :, e), local_flux_values)
        end
    end
end

function calc_gradient_volume_integral!(gradients, u, mesh::DGMultiMesh,
                                        equations::AbstractEquationsParabolic,
                                        dg::DGMulti, cache, cache_parabolic)
    (; strong_differentiation_matrices) = cache_parabolic

    # compute volume contributions to gradients
    @threaded for e in eachelement(mesh, dg)
        for i in eachdim(mesh), j in eachdim(mesh)

            # We assume each element is affine (e.g., constant geometric terms) here.
            dxidxhatj = mesh.md.rstxyzJ[i, j][1, e]

            apply_to_each_field(mul_by_accum!(strong_differentiation_matrices[j],
                                              dxidxhatj),
                                view(gradients[i], :, e), view(u, :, e))
        end
    end
end

function calc_gradient_volume_integral!(gradients, u, mesh::DGMultiMesh{NDIMS, <:NonAffine},
                                        equations::AbstractEquationsParabolic,
                                        dg::DGMulti, cache, cache_parabolic) where {NDIMS}
    (; strong_differentiation_matrices, dxidxhatj, local_flux_viscous_threaded) = cache_parabolic

    # compute volume contributions to gradients
    @threaded for e in eachelement(mesh, dg)

        # compute gradients with respect to reference coordinates
        local_reference_gradients = local_flux_viscous_threaded[Threads.threadid()]
        for i in eachdim(mesh)
            apply_to_each_field(mul_by!(strong_differentiation_matrices[i]),
                                local_reference_gradients[i], view(u, :, e))
        end

        # rotate to physical frame on each element
        for i in eachdim(mesh), j in eachdim(mesh)
            for node in eachindex(local_reference_gradients[j])
                gradients[i][node, e] = gradients[i][node, e] +
                                        dxidxhatj[i, j][node, e] *
                                        local_reference_gradients[j][node]
            end
        end
    end
end

function calc_gradient!(gradients, u::StructArray, t, mesh::DGMultiMesh,
                        equations::AbstractEquationsParabolic,
                        boundary_conditions, dg::DGMulti, cache, cache_parabolic)
    for dim in eachindex(gradients)
        reset_du!(gradients[dim], dg)
    end

    calc_gradient_volume_integral!(gradients, u, mesh, equations, dg, cache,
                                   cache_parabolic)

    (; u_face_values) = cache_parabolic
    apply_to_each_field(mul_by!(dg.basis.Vf), u_face_values, u)

    # compute fluxes at interfaces
    (; scalar_flux_face_values) = cache_parabolic
    (; mapM, mapP) = mesh.md
    @threaded for face_node_index in each_face_node_global(mesh, dg)
        idM, idP = mapM[face_node_index], mapP[face_node_index]
        uM = u_face_values[idM]
        uP = u_face_values[idP]
        # Here, we use the "strong" formulation to compute the gradient. This guarantees that the parabolic
        # formulation is symmetric and stable on curved meshes with variable geometric terms.
        scalar_flux_face_values[idM] = 0.5 * (uP - uM)
    end

    calc_boundary_flux!(scalar_flux_face_values, u_face_values, t, Gradient(),
                        boundary_conditions,
                        mesh, equations, dg, cache, cache_parabolic)

    # compute surface contributions
    calc_gradient_surface_integral!(gradients, u, scalar_flux_face_values,
                                    mesh, equations, dg, cache, cache_parabolic)

    invert_jacobian_gradient!(gradients, mesh, equations, dg, cache, cache_parabolic)
end

# affine mesh - constant Jacobian version
function invert_jacobian_gradient!(gradients, mesh::DGMultiMesh, equations, dg::DGMulti,
                                   cache, cache_parabolic)
    @threaded for e in eachelement(mesh, dg)

        # Here, we exploit the fact that J is constant on affine elements,
        # so we only have to access invJ once per element.
        invJ = cache_parabolic.invJ[1, e]

        for dim in eachdim(mesh)
            for i in axes(gradients[dim], 1)
                gradients[dim][i, e] = gradients[dim][i, e] * invJ
            end
        end
    end
end

# non-affine mesh - variable Jacobian version
function invert_jacobian_gradient!(gradients, mesh::DGMultiMesh{NDIMS, <:NonAffine},
                                   equations,
                                   dg::DGMulti, cache, cache_parabolic) where {NDIMS}
    (; invJ) = cache_parabolic
    @threaded for e in eachelement(mesh, dg)
        for dim in eachdim(mesh)
            for i in axes(gradients[dim], 1)
                gradients[dim][i, e] = gradients[dim][i, e] * invJ[i, e]
            end
        end
    end
end

# do nothing for periodic domains
function calc_boundary_flux!(flux, u, t, operator_type, ::BoundaryConditionPeriodic,
                             mesh, equations::AbstractEquationsParabolic, dg::DGMulti,
                             cache, cache_parabolic)
    return nothing
end

# "lispy tuple programming" instead of for loop for type stability
function calc_boundary_flux!(flux, u, t, operator_type, boundary_conditions,
                             mesh, equations, dg::DGMulti, cache, cache_parabolic)

    # peel off first boundary condition
    calc_single_boundary_flux!(flux, u, t, operator_type, first(boundary_conditions),
                               first(keys(boundary_conditions)),
                               mesh, equations, dg, cache, cache_parabolic)

    # recurse on the remainder of the boundary conditions
    calc_boundary_flux!(flux, u, t, operator_type, Base.tail(boundary_conditions),
                        mesh, equations, dg, cache, cache_parabolic)
end

# terminate recursion
function calc_boundary_flux!(flux, u, t, operator_type,
                             boundary_conditions::NamedTuple{(), Tuple{}},
                             mesh, equations, dg::DGMulti, cache, cache_parabolic)
    nothing
end

function calc_single_boundary_flux!(flux_face_values, u_face_values, t,
                                    operator_type, boundary_condition, boundary_key,
                                    mesh, equations, dg::DGMulti{NDIMS}, cache,
                                    cache_parabolic) where {NDIMS}
    rd = dg.basis
    md = mesh.md

    num_faces = StartUpDG.num_faces(rd.element_type)
    num_pts_per_face = rd.Nfq ÷ num_faces
    (; xyzf, nxyz) = md
    for f in mesh.boundary_faces[boundary_key]
        for i in Base.OneTo(num_pts_per_face)

            # reverse engineer element + face node indices (avoids reshaping arrays)
            e = ((f - 1) ÷ num_faces) + 1
            fid = i + ((f - 1) % num_faces) * num_pts_per_face

            face_normal = SVector{NDIMS}(getindex.(nxyz, fid, e))
            face_coordinates = SVector{NDIMS}(getindex.(xyzf, fid, e))

            # for both the gradient and the divergence, the boundary flux is scalar valued.
            # for the gradient, it is the solution; for divergence, it is the normal flux.
            flux_face_values[fid, e] = boundary_condition(flux_face_values[fid, e],
                                                          u_face_values[fid, e],
                                                          face_normal, face_coordinates, t,
                                                          operator_type, equations)

            # Here, we use the "strong form" for the Gradient (and the "weak form" for Divergence).
            # `flux_face_values` should contain the boundary values for `u`, and we
            # subtract off `u_face_values[fid, e]` because we are using the strong formulation to
            # compute the gradient.
            if operator_type isa Gradient
                flux_face_values[fid, e] = flux_face_values[fid, e] - u_face_values[fid, e]
            end
        end
    end
    return nothing
end

function calc_viscous_fluxes!(flux_viscous, u, gradients, mesh::DGMultiMesh,
                              equations::AbstractEquationsParabolic,
                              dg::DGMulti, cache, cache_parabolic)
    for dim in eachdim(mesh)
        reset_du!(flux_viscous[dim], dg)
    end

    (; local_u_values_threaded) = cache_parabolic

    @threaded for e in eachelement(mesh, dg)

        # reset local storage for each element, interpolate u to quadrature points
        # TODO: DGMulti. Specialize for nodal collocation methods (SBP, GaussSBP)?
        local_u_values = local_u_values_threaded[Threads.threadid()]
        fill!(local_u_values, zero(eltype(local_u_values)))
        apply_to_each_field(mul_by!(dg.basis.Vq), local_u_values, view(u, :, e))

        # compute viscous flux at quad points
        for i in eachindex(local_u_values)
            u_i = local_u_values[i]
            gradients_i = getindex.(gradients, i, e)
            for dim in eachdim(mesh)
                flux_viscous_i = flux(u_i, gradients_i, dim, equations)
                setindex!(flux_viscous[dim], flux_viscous_i, i, e)
            end
        end
    end
end

# no penalization for a BR1 parabolic solver
function calc_viscous_penalty!(scalar_flux_face_values, u_face_values, t,
                               boundary_conditions,
                               mesh, equations::AbstractEquationsParabolic, dg::DGMulti,
                               parabolic_scheme::ViscousFormulationBassiRebay1, cache,
                               cache_parabolic)
    return nothing
end

function calc_viscous_penalty!(scalar_flux_face_values, u_face_values, t,
                               boundary_conditions,
                               mesh, equations::AbstractEquationsParabolic, dg::DGMulti,
                               parabolic_scheme, cache, cache_parabolic)
    # compute fluxes at interfaces
    (; scalar_flux_face_values) = cache_parabolic
    (; mapM, mapP) = mesh.md
    @threaded for face_node_index in each_face_node_global(mesh, dg)
        idM, idP = mapM[face_node_index], mapP[face_node_index]
        uM, uP = u_face_values[idM], u_face_values[idP]
        scalar_flux_face_values[idM] = scalar_flux_face_values[idM] +
                                       penalty(uP, uM, equations, parabolic_scheme)
    end
    return nothing
end

function calc_divergence_volume_integral!(du, u, flux_viscous, mesh::DGMultiMesh,
                                          equations::AbstractEquationsParabolic,
                                          dg::DGMulti, cache, cache_parabolic)
    (; weak_differentiation_matrices) = cache_parabolic

    # compute volume contributions to divergence
    @threaded for e in eachelement(mesh, dg)
        for i in eachdim(mesh), j in eachdim(mesh)
            dxidxhatj = mesh.md.rstxyzJ[i, j][1, e] # assumes mesh is affine
            apply_to_each_field(mul_by_accum!(weak_differentiation_matrices[j], dxidxhatj),
                                view(du, :, e), view(flux_viscous[i], :, e))
        end
    end
end

function calc_divergence_volume_integral!(du, u, flux_viscous,
                                          mesh::DGMultiMesh{NDIMS, <:NonAffine},
                                          equations::AbstractEquationsParabolic,
                                          dg::DGMulti, cache, cache_parabolic) where {NDIMS}
    (; weak_differentiation_matrices, dxidxhatj, local_flux_viscous_threaded) = cache_parabolic

    # compute volume contributions to divergence
    @threaded for e in eachelement(mesh, dg)
        local_viscous_flux = local_flux_viscous_threaded[Threads.threadid()][1]
        for i in eachdim(mesh)
            # rotate flux to reference coordinates
            fill!(local_viscous_flux, zero(eltype(local_viscous_flux)))
            for j in eachdim(mesh)
                for node in eachindex(local_viscous_flux)
                    local_viscous_flux[node] = local_viscous_flux[node] +
                                               dxidxhatj[j, i][node, e] *
                                               flux_viscous[j][node, e]
                end
            end

            # differentiate with respect to reference coordinates
            apply_to_each_field(mul_by_accum!(weak_differentiation_matrices[i]),
                                view(du, :, e), local_viscous_flux)
        end
    end
end

function calc_divergence!(du, u::StructArray, t, flux_viscous, mesh::DGMultiMesh,
                          equations::AbstractEquationsParabolic,
                          boundary_conditions, dg::DGMulti, parabolic_scheme, cache,
                          cache_parabolic)
    reset_du!(du, dg)

    calc_divergence_volume_integral!(du, u, flux_viscous, mesh, equations, dg, cache,
                                     cache_parabolic)

    # interpolates from solution coefficients to face quadrature points
    (; projection_face_interpolation_matrix) = cache_parabolic
    flux_viscous_face_values = cache_parabolic.gradients_face_values # reuse storage
    for dim in eachdim(mesh)
        apply_to_each_field(mul_by!(projection_face_interpolation_matrix),
                            flux_viscous_face_values[dim], flux_viscous[dim])
    end

    # compute fluxes at interfaces
    (; scalar_flux_face_values) = cache_parabolic
    (; mapM, mapP, nxyzJ) = mesh.md

    @threaded for face_node_index in each_face_node_global(mesh, dg, cache, cache_parabolic)
        idM, idP = mapM[face_node_index], mapP[face_node_index]

        # compute f(u, ∇u) ⋅ n
        flux_face_value = zero(eltype(scalar_flux_face_values))
        for dim in eachdim(mesh)
            fM = flux_viscous_face_values[dim][idM]
            fP = flux_viscous_face_values[dim][idP]
            # Here, we use the "weak" formulation to compute the divergence (to ensure stability on curved meshes).
            flux_face_value = flux_face_value +
                              0.5 * (fP + fM) * nxyzJ[dim][face_node_index]
        end
        scalar_flux_face_values[idM] = flux_face_value
    end

    calc_boundary_flux!(scalar_flux_face_values, cache_parabolic.u_face_values, t,
                        Divergence(),
                        boundary_conditions, mesh, equations, dg, cache, cache_parabolic)

    calc_viscous_penalty!(scalar_flux_face_values, cache_parabolic.u_face_values, t,
                          boundary_conditions, mesh, equations, dg, parabolic_scheme,
                          cache, cache_parabolic)

    # surface contributions
    apply_to_each_field(mul_by_accum!(cache_parabolic.divergence_lift_matrix), du,
                        scalar_flux_face_values)

    # Note: we do not flip the sign of the geometric Jacobian here.
    # This is because the parabolic fluxes are assumed to be of the form
    #   `du/dt + df/dx = dg/dx + source(x,t)`,
    # where f(u) is the inviscid flux and g(u) is the viscous flux.
    invert_jacobian!(du, mesh, equations, dg, cache; scaling = 1.0)
end

# assumptions: parabolic terms are of the form div(f(u, grad(u))) and
# will be discretized first order form as follows:
#               1. compute grad(u)
#               2. compute f(u, grad(u))
#               3. compute div(u)
# boundary conditions will be applied to both grad(u) and div(u).
function rhs_parabolic!(du, u, t, mesh::DGMultiMesh,
                        equations_parabolic::AbstractEquationsParabolic,
                        boundary_conditions, source_terms,
                        dg::DGMulti, parabolic_scheme, cache, cache_parabolic)
    reset_du!(du, dg)

    @trixi_timeit timer() "transform variables" begin
        (; u_transformed, gradients, flux_viscous) = cache_parabolic
        transform_variables!(u_transformed, u, mesh, equations_parabolic,
                             dg, parabolic_scheme, cache, cache_parabolic)
    end

    @trixi_timeit timer() "calc gradient" begin
        calc_gradient!(gradients, u_transformed, t, mesh, equations_parabolic,
                       boundary_conditions, dg, cache, cache_parabolic)
    end

    @trixi_timeit timer() "calc viscous fluxes" begin
        calc_viscous_fluxes!(flux_viscous, u_transformed, gradients,
                             mesh, equations_parabolic, dg, cache, cache_parabolic)
    end

    @trixi_timeit timer() "calc divergence" begin
        calc_divergence!(du, u_transformed, t, flux_viscous, mesh, equations_parabolic,
                         boundary_conditions, dg, parabolic_scheme, cache, cache_parabolic)
    end
    return nothing
end
