
"""
    DGMulti(approximation_type::AbstractDerivativeOperator;
            element_type::AbstractElemShape,
            surface_flux=flux_central,
            surface_integral=SurfaceIntegralWeakForm(surface_flux),
            volume_integral=VolumeIntegralWeakForm(),
            kwargs...)

Create a summation by parts (SBP) discretization on the given `element_type`
using a tensor product structure based on the 1D SBP derivative operator
passed as `approximation_type`.

For more info, see the documentations of
[StartUpDG.jl](https://jlchan.github.io/StartUpDG.jl/dev/)
and
[SummationByPartsOperators.jl](https://ranocha.de/SummationByPartsOperators.jl/stable/).
"""
function DGMulti(approximation_type::AbstractDerivativeOperator;
                 element_type::AbstractElemShape,
                 surface_flux = flux_central,
                 surface_integral = SurfaceIntegralWeakForm(surface_flux),
                 volume_integral = VolumeIntegralWeakForm(),
                 kwargs...)
    rd = RefElemData(element_type, approximation_type; kwargs...)
    # `nothing` is passed as `mortar`
    return DG(rd, nothing, surface_integral, volume_integral)
end

function DGMulti(element_type::AbstractElemShape,
                 approximation_type::AbstractDerivativeOperator,
                 volume_integral,
                 surface_integral;
                 kwargs...)
    return DGMulti(approximation_type, element_type = element_type,
                   surface_integral = surface_integral, volume_integral = volume_integral)
end

# type alias for specializing on a periodic SBP operator
const DGMultiPeriodicFDSBP{NDIMS, ApproxType, ElemType} = DGMulti{NDIMS, ElemType,
                                                                  ApproxType,
                                                                  SurfaceIntegral,
                                                                  VolumeIntegral} where {
                                                                                         NDIMS,
                                                                                         ElemType,
                                                                                         ApproxType <:
                                                                                         SummationByPartsOperators.AbstractPeriodicDerivativeOperator,
                                                                                         SurfaceIntegral,
                                                                                         VolumeIntegral
                                                                                         }

const DGMultiFluxDiffPeriodicFDSBP{NDIMS, ApproxType, ElemType} = DGMulti{NDIMS, ElemType,
                                                                          ApproxType,
                                                                          SurfaceIntegral,
                                                                          VolumeIntegral} where {
                                                                                                 NDIMS,
                                                                                                 ElemType,
                                                                                                 ApproxType <:
                                                                                                 SummationByPartsOperators.AbstractPeriodicDerivativeOperator,
                                                                                                 SurfaceIntegral <:
                                                                                                 SurfaceIntegralWeakForm,
                                                                                                 VolumeIntegral <:
                                                                                                 VolumeIntegralFluxDifferencing
                                                                                                 }

"""
    DGMultiMesh(dg::DGMulti)

Constructs a single-element [`DGMultiMesh`](@ref) for a single periodic element given
a DGMulti with `approximation_type` set to a periodic (finite difference) SBP operator from
SummationByPartsOperators.jl.
"""
function DGMultiMesh(dg::DGMultiPeriodicFDSBP{NDIMS};
                     coordinates_min = ntuple(_ -> -one(real(dg)), NDIMS),
                     coordinates_max = ntuple(_ -> one(real(dg)), NDIMS)) where {NDIMS}
    rd = dg.basis

    e = Ones{eltype(rd.r)}(size(rd.r))
    z = Zeros{eltype(rd.r)}(size(rd.r))

    VXYZ = ntuple(_ -> [], NDIMS)
    EToV = NaN # StartUpDG.jl uses size(EToV, 1) for the number of elements, this lets us reuse that.
    FToF = []

    # We need to scale the domain from `[-1, 1]^NDIMS` (default in StartUpDG.jl)
    # to the given `coordinates_min, coordinates_max`
    xyz = xyzq = map(copy, rd.rst)
    for dim in 1:NDIMS
        factor = (coordinates_max[dim] - coordinates_min[dim]) / 2
        @. xyz[dim] = factor * (xyz[dim] + 1) + coordinates_min[dim]
    end
    xyzf = ntuple(_ -> [], NDIMS)
    wJq = diag(rd.M)

    # arrays of connectivity indices between face nodes
    mapM = mapP = mapB = []

    # volume geofacs Gij = dx_i/dxhat_j
    coord_diffs = coordinates_max .- coordinates_min

    J_scalar = prod(coord_diffs) / 2^NDIMS
    J = e * J_scalar

    if NDIMS == 1
        rxJ = J_scalar * 2 / coord_diffs[1]
        rstxyzJ = @SMatrix [rxJ * e]
    elseif NDIMS == 2
        rxJ = J_scalar * 2 / coord_diffs[1]
        syJ = J_scalar * 2 / coord_diffs[2]
        rstxyzJ = @SMatrix [rxJ*e z; z syJ*e]
    elseif NDIMS == 3
        rxJ = J_scalar * 2 / coord_diffs[1]
        syJ = J_scalar * 2 / coord_diffs[2]
        tzJ = J_scalar * 2 / coord_diffs[3]
        rstxyzJ = @SMatrix [rxJ*e z z; z syJ*e z; z z tzJ*e]
    end

    # surface geofacs
    nxyzJ = ntuple(_ -> [], NDIMS)
    Jf = []

    periodicity = ntuple(_ -> true, NDIMS)

    if NDIMS == 1
        mesh_type = Line()
    elseif NDIMS == 2
        mesh_type = Quad()
    elseif NDIMS == 3
        mesh_type = Hex()
    end

    md = MeshData(StartUpDG.VertexMappedMesh(mesh_type, VXYZ, EToV), FToF, xyz, xyzf, xyzq,
                  wJq,
                  mapM, mapP, mapB, rstxyzJ, J, nxyzJ, Jf,
                  periodicity)

    boundary_faces = []
    return DGMultiMesh{NDIMS, rd.element_type, typeof(md),
                       typeof(boundary_faces)}(md, boundary_faces)
end

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# specialized for DGMultiPeriodicFDSBP since there are no face nodes
# and thus no inverse trace constant for periodic domains.
function estimate_dt(mesh::DGMultiMesh, dg::DGMultiPeriodicFDSBP)
    rd = dg.basis # RefElemData
    return StartUpDG.estimate_h(rd, mesh.md)
end

# do nothing for interface terms if using a periodic operator
function prolong2interfaces!(cache, u,
                             mesh::DGMultiMesh, equations, dg::DGMultiPeriodicFDSBP)
    @assert nelements(mesh, dg, cache) == 1
    return nothing
end

function calc_interface_flux!(cache, surface_integral::SurfaceIntegralWeakForm,
                              mesh::DGMultiMesh,
                              have_nonconservative_terms::False, equations,
                              dg::DGMultiPeriodicFDSBP)
    @assert nelements(mesh, dg, cache) == 1
    return nothing
end

function calc_surface_integral!(du, u, mesh::DGMultiMesh, equations,
                                surface_integral::SurfaceIntegralWeakForm,
                                dg::DGMultiPeriodicFDSBP, cache)
    @assert nelements(mesh, dg, cache) == 1
    return nothing
end

function create_cache(mesh::DGMultiMesh, equations,
                      dg::DGMultiFluxDiffPeriodicFDSBP, RealT, uEltype)
    md = mesh.md

    solution_container = initialize_dgmulti_solution_container(mesh, equations, dg,
                                                               uEltype)

    # since dg.basis.Drst is not skew-symmetric for CG operators, we explicitly construct
    # the skew-symmetric operators for flux differencing
    Qrst = map(A -> dg.basis.M * A, dg.basis.Drst)

    # check for skew-symmetry
    for dim in eachdim(mesh)
        Q = Qrst[dim]
        @assert isapprox(Q, -Q', atol = 1e-12, rtol = 0.0)
    end
    return (; solution_container, Qrst, invM = inv(dg.basis.M), invJ = inv.(md.J))
end

# Specialize calc_volume_integral for periodic SBP operators (assumes the operator is sparse).
function calc_volume_integral!(du, u, mesh::DGMultiMesh,
                               have_nonconservative_terms::False, equations,
                               volume_integral::VolumeIntegralFluxDifferencing,
                               dg::DGMultiFluxDiffPeriodicFDSBP, cache)
    @unpack volume_flux = volume_integral

    # We expect speedup over the serial version only when using two or more threads
    # since the threaded version below does not exploit the symmetry properties,
    # resulting in a performance penalty of 1/2
    if Threads.nthreads() > 1
        for dim in eachdim(mesh)
            normal_direction = get_contravariant_vector(1, dim, mesh, cache)

            # These are weak-form operators of the form `Q = M * D` where `M` is diagonal
            # and `Q` is skew-symmetric. 
            # TODO: DGMulti.
            # This would have to be changed if `have_nonconservative_terms = False()`
            # because then `volume_flux` is non-symmetric.
            A = cache.Qrst[dim]

            # sparse_operator_data retrieves column indices and row offsets, but because 
            # A is skew-symmetric, these are also the row indices and column offsets.
            A_base, row_ids, cols, vals = sparse_operator_data(A)

            @threaded for i in row_ids
                u_i = u[i]
                du_i = du[i]
                @inbounds for id in nzrange(A_base, i)
                    j = cols[id]
                    u_j = u[j]

                    # we use the negative of A_ij since A is skew-symmetric, 
                    # and we are accessing the transpose of A. 
                    A_ij = -vals[id]
                    AF_ij = 2 * A_ij *
                            volume_flux(u_i, u_j, normal_direction, equations)
                    du_i = du_i + AF_ij
                end
                du[i] = du_i
            end
        end

        # apply M^{-1} once after all spatial dimensions.
        @inbounds for i in eachindex(du)
            du[i] = du[i] * cache.invM.diag[i]
        end

    else # if using two threads or fewer

        # Exploit skew-symmetry to halve the number of flux evaluations (≈2x speedup).
        # A = Qrst[dim] is skew-symmetric for periodic SBP operators on uniform grids, so
        # A[i,j] = -A[j,i]. The stored CSC value vals[id] = A[j,i] = -A[i,j], hence
        # we use -vals[id] to recover A[i,j], matching the multithreaded branch above.
        for dim in eachdim(mesh)
            normal_direction = get_contravariant_vector(1, dim, mesh, cache)

            A = cache.Qrst[dim]
            # sparse_operator_data retrieves column indices and row offsets, but because 
            # A is skew-symmetric, these are also the row indices and column offsets.
            A_base, row_ids, cols, vals = sparse_operator_data(A)

            @inbounds for i in row_ids
                u_i = u[i]
                du_i = du[i]
                for id in nzrange(A_base, i)
                    j = cols[id]
                    # We use the symmetry of the volume flux and the anti-symmetry
                    # of the derivative operator to save half of the volume flux
                    # computations.
                    if j > i
                        A_ij = -vals[id]  # A[j,i] stored; skew-symmetry: -A[j,i] = A[i,j]
                        u_j = u[j]
                        AF_ij = 2 * A_ij *
                                volume_flux(u_i, u_j, normal_direction, equations)
                        du_i = du_i + AF_ij
                        du[j] = du[j] - AF_ij # Due to skew-symmetry
                    end
                end
                du[i] = du_i
            end
        end

        # apply M^{-1} only after all skew-symmetric contributions are 
        # accumulated over each dimension.
        @inbounds for i in eachindex(du)
            du[i] = du[i] * cache.invM.diag[i]
        end
    end

    return nothing
end
end # @muladd
