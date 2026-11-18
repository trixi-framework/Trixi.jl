# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Continuous Galerkin spectral element methods (CGSEM) build on the DGSEM kernels
# and containers of the `P4estMesh`. The `CGSEM` type is defined first since the
# gather-scatter functions dispatch on it.
include("cgsem/cgsem.jl")
include("cgsem/assembly.jl")

# Hyperbolic right-hand side of CGSEM on a conforming `P4estMesh{2}`, cf. the DGSEM
# version in `dgsem_tree/dg_2d.jl`. The differences are:
# - Interface and mortar terms are skipped; continuity is enforced by the assembly.
# - The interior entries of `surface_flux_values` are zero (see `create_cache`), so the
#   surface integral adds only the weakly imposed boundary fluxes.
# - The source terms are added before the assembly, see `calc_sources!` below.
# - `assemble_and_apply_jacobian!` replaces `apply_jacobian!`.
function rhs_hyperbolic!(backend::Nothing, du, u, t, mesh::P4estMesh{2}, equations,
                         boundary_conditions, source_terms::Source, solver::CGSEM,
                         cache) where {Source}
    (; dg) = solver

    # Reset du
    @trixi_timeit timer() "reset ∂u/∂t" begin
        set_zero!(du, dg, cache)
    end

    # Calculate volume integral. The DGSEM kernels include the element-boundary term
    # -M^{-1} B f(u) of the weak form, which cancels between neighboring elements for
    # continuous u once the residuals are summed in the assembly.
    @trixi_timeit timer() "volume integral" begin
        calc_volume_integral!(backend, du, u, mesh,
                              have_nonconservative_terms(equations), equations,
                              dg.volume_integral, dg, cache)
    end

    # Prolong solution to boundaries
    @trixi_timeit timer() "prolong2boundaries" begin
        prolong2boundaries!(backend, cache, u, mesh, equations, dg)
    end

    # Calculate boundary fluxes
    @trixi_timeit timer() "boundary flux" begin
        calc_boundary_flux!(backend, cache, t, boundary_conditions, mesh, equations,
                            dg.surface_integral, dg)
    end

    # Calculate surface integrals. Only the boundary faces contribute, since the
    # interior entries of `surface_flux_values` are zero.
    # TODO: Revisit this soon. The DGSEM kernel `calc_surface_integral!` loops over all
    # faces of all elements, so most of its work adds zeros at interior faces. We reuse
    # it for now since it keeps the option of adding surface terms at interior faces
    # later (e.g., for stabilization). Without such terms, a surface integral over the
    # boundary faces only would avoid the unnecessary work.
    @trixi_timeit timer() "surface integral" begin
        calc_surface_integral!(backend, du, u, mesh, equations,
                               dg.surface_integral, dg, cache)
    end

    # Calculate source terms before assembly/Jacobian application
    @trixi_timeit timer() "source terms" begin
        calc_sources!(backend, du, u, t, source_terms, equations, solver, cache)
    end

    # Sum the residuals of all copies of each node and apply the inverse of the
    # assembled lumped mass matrix
    @trixi_timeit timer() "assembly" begin
        assemble_and_apply_jacobian!(du, cache.assembly, mesh, equations, solver, cache)
    end

    return nothing
end

# Source terms of CGSEM. DGSEM adds them after `apply_jacobian!`, i.e., to the final
# time derivative. CGSEM adds them to the element-local residuals before the assembly,
# which scales the residuals by `-1 / sum_k J_k` (see `assemble_and_apply_jacobian!`),
# so the source terms are multiplied by `-J` here. The assembled time derivative then
# contains the Jacobian-weighted average of the source terms of all copies of a node,
# which is the same for all copies even though their node coordinates (and thus the
# source terms) differ by roundoff.
function calc_sources!(backend::Nothing, du, u, t, source_terms::Nothing,
                       equations::AbstractEquations{2}, solver::CGSEM, cache)
    return nothing
end

function calc_sources!(backend::Nothing, du, u, t, source_terms,
                       equations::AbstractEquations{2}, solver::CGSEM, cache)
    (; dg) = solver
    (; node_coordinates, inverse_jacobian) = cache.elements

    # Explicit bounds check, which allows us to assume inbounds access below
    @boundscheck begin
        check_axes(u, Val(2), equations, dg, cache)
        check_axes(du, Val(2), equations, dg, cache)
        check_axes(cache.elements, equations, dg, cache)
    end

    @threaded for element in eachelement(dg, cache)
        @inbounds begin
            for j in eachnode(dg), i in eachnode(dg)
                u_local = get_node_vars(u, equations, dg, i, j, element)
                x_local = get_node_coords(node_coordinates, equations, dg,
                                          i, j, element)
                source_local = source_terms(u_local, x_local, t, equations)
                factor = -inv(inverse_jacobian[i, j, element])
                multiply_add_to_node_vars!(du, factor, source_local, equations, dg,
                                           i, j, element)
            end
        end
    end

    return nothing
end
end # @muladd
