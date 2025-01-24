# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

struct FV{RealT <: Real, SurfaceFlux}
    order::Integer
    extended_reconstruction_stencil::Bool
    surface_flux::SurfaceFlux

    function FV(; order = 1, extended_reconstruction_stencil = false, RealT = Float64,
                surface_flux = flux_central)
        new{RealT, typeof(surface_flux)}(order, extended_reconstruction_stencil,
                                         surface_flux)
    end
end

function Base.show(io::IO, solver::FV)
    @nospecialize solver # reduce precompilation time

    print(io, "FV(")
    print(io, "order $(solver.order)")
    print(io, ", ", solver.surface_flux)
    print(io, ")")
end

function Base.show(io::IO, mime::MIME"text/plain", solver::FV)
    @nospecialize solver # reduce precompilation time

    if get(io, :compact, false)
        show(io, solver)
    else
        summary_header(io, "FV{" * string(real(solver)) * "}")
        summary_line(io, "order", solver.order)
        summary_line(io, "surface flux", solver.surface_flux)
        summary_footer(io)
    end
end

Base.summary(io::IO, solver::FV) = print(io, "FV(order=$(solver.order))")

@inline Base.real(solver::FV{RealT}) where {RealT} = RealT

@inline ndofs(mesh, solver::FV, cache) = ncells(mesh)

@inline nelements(mesh::T8codeMesh, solver::FV, cache) = ncells(mesh)
@inline function ndofsglobal(mesh, solver::FV, cache)
    nelementsglobal(mesh, solver, cache)
end

@inline function eachelement(mesh, solver::FV, cache)
    Base.OneTo(nelements(mesh, solver, cache))
end

@inline eachinterface(solver::FV, cache) = Base.OneTo(ninterfaces(solver, cache))
@inline eachboundary(solver::FV, cache) = Base.OneTo(nboundaries(solver, cache))

@inline function nelementsglobal(mesh, solver::FV, cache)
    if mpi_isparallel()
        Int(t8_forest_get_global_num_elements(mesh.forest))
    else
        nelements(mesh, solver, cache)
    end
end

@inline ninterfaces(solver::FV, cache) = ninterfaces(cache.interfaces)
@inline nboundaries(solver::FV, cache) = nboundaries(cache.boundaries)

@inline function get_node_coords(x, equations, solver::FV, indices...)
    SVector(ntuple(@inline(idx->x[idx, indices...]), Val(ndims(equations))))
end

@inline function get_node_vars(u, equations, solver::FV, element)
    SVector(ntuple(@inline(v->u[v, element]), Val(nvariables(equations))))
end

@inline function set_node_vars!(u, u_node, equations, solver::FV, element)
    for v in eachvariable(equations)
        u[v, element] = u_node[v]
    end
    return nothing
end

@inline function get_surface_node_vars(u, equations, solver::FV, indices...)
    # There is a cut-off at `n == 10` inside of the method
    # `ntuple(f::F, n::Integer) where F` in Base at ntuple.jl:17
    # in Julia `v1.5`, leading to type instabilities if
    # more than ten variables are used. That's why we use
    # `Val(...)` below.
    u_ll = SVector(ntuple(@inline(v->u[1, v, indices...]), Val(nvariables(equations))))
    u_rr = SVector(ntuple(@inline(v->u[2, v, indices...]), Val(nvariables(equations))))
    return u_ll, u_rr
end

function allocate_coefficients(mesh::T8codeMesh, equations, solver::FV, cache)
    # We must allocate a `Vector` in order to be able to `resize!` it (AMR).
    # cf. wrap_array
    zeros(eltype(cache.elements),
          nvariables(equations) * nelements(mesh, solver, cache))
end

# General fallback
@inline function wrap_array(u_ode::AbstractVector, mesh::AbstractMesh, equations,
                            solver::FV, cache)
    wrap_array_native(u_ode, mesh, equations, solver, cache)
end

# Like `wrap_array`, but guarantees to return a plain `Array`, which can be better
# for interfacing with external C libraries (MPI, HDF5, visualization),
# writing solution files etc.
@inline function wrap_array_native(u_ode::AbstractVector, mesh::AbstractMesh, equations,
                                   solver::FV, cache)
    @boundscheck begin
        @assert length(u_ode) ==
                nvariables(equations) * nelements(mesh, solver, cache)
    end
    unsafe_wrap(Array{eltype(u_ode), 2}, pointer(u_ode),
                (nvariables(equations), nelements(mesh, solver, cache)))
end

function compute_coefficients!(u, func, t, mesh::T8codeMesh,
                               equations, solver::FV, cache)
    for element in eachelement(mesh, solver, cache)
        x_node = get_node_coords(cache.elements.midpoint, equations, solver, element)
        u_node = func(x_node, t, equations)
        set_node_vars!(u, u_node, equations, solver, element)
    end
end

function create_cache(mesh::T8codeMesh, equations::AbstractEquations, solver::FV, ::Any,
                      ::Type{uEltype}) where {uEltype <: Real}
    count_required_surfaces!(mesh)

    # After I saved some data (e.g. normal) in the interfaces and boundaries,
    # the element data structure is not used anymore after this `create_cache` routine.
    # Possible to remove it and directly save the data in interface, boundars (and mortar) data structure?
    elements = init_elements(mesh, equations, solver, uEltype)
    interfaces = init_interfaces(mesh, equations, solver, uEltype)
    boundaries = init_boundaries(mesh, equations, solver, uEltype)
    # mortars = init_mortars(mesh, equations, solver, uEltype)

    fill_mesh_info_fv!(mesh, interfaces, boundaries,
                       mesh.boundary_names)

    # Data structure for exchange between MPI ranks.
    communication_data = init_communication_data!(mesh, equations)
    exchange_domain_data!(communication_data, elements, mesh, equations, solver)

    # Initialize reconstruction stencil
    if !solver.extended_reconstruction_stencil
        init_reconstruction_stencil!(elements, interfaces, boundaries,
                                     communication_data, mesh, equations, solver)
    end

    cache = (; elements, interfaces, boundaries, communication_data)

    return cache
end

function rhs!(du, u, t, mesh::T8codeMesh, equations,
              boundary_conditions, source_terms::Source,
              solver::FV, cache) where {Source}
    # Reset du
    @trixi_timeit timer() "reset ∂u/∂t" du.=zero(eltype(du))

    # Exchange solution between MPI ranks
    @trixi_timeit timer() "exchange_solution_data!" exchange_solution_data!(u, mesh,
                                                                            equations,
                                                                            solver,
                                                                            cache)

    @trixi_timeit timer() "gradient reconstruction" calc_gradient_reconstruction!(u,
                                                                                  mesh,
                                                                                  equations,
                                                                                  solver,
                                                                                  cache)

    # Prolong solution to interfaces
    @trixi_timeit timer() "prolong2interfaces" begin
        prolong2interfaces!(cache, mesh, equations, solver)
    end

    # Calculate interface fluxes
    @trixi_timeit timer() "interface flux" begin
        calc_interface_flux!(du, mesh, have_nonconservative_terms(equations), equations,
                             solver, cache)
    end

    # Prolong solution to boundaries
    @trixi_timeit timer() "prolong2boundaries!" begin
        prolong2boundaries!(cache, mesh, equations, solver)
    end

    # Calculate boundary fluxes
    @trixi_timeit timer() "calc boundary flux" begin
        calc_boundary_flux!(du, cache, t, boundary_conditions, mesh,
                            equations, solver)
    end

    @trixi_timeit timer() "volume" begin
        for element in eachelement(mesh, solver, cache)
            volume = cache.elements.volume[element]
            for v in eachvariable(equations)
                du[v, element] = (1 / volume) * du[v, element]
            end
        end
    end

    # Calculate source terms
    @trixi_timeit timer() "source terms" begin
        calc_sources!(du, u, t, source_terms, mesh, equations, solver, cache)
    end

    return nothing
end

function calc_gradient_reconstruction!(u, mesh::T8codeMesh{2}, equations, solver, cache)
    if solver.order == 1
        return nothing
    elseif solver.order > 2
        error("Order $(solver.order) is not supported yet!")
    end

    (; elements, communication_data) = cache
    (; reconstruction_stencil, reconstruction_distance, reconstruction_corner_elements, reconstruction_gradient, reconstruction_gradient_limited) = elements
    (; solution_data) = communication_data

    # A         N x 2 Matrix, where N is the number of stencil neighbors
    # A^T A     2 x 2 Matrix
    # b         N     Vector
    # A^T b     2     Vector

    # Matrix/vector notation
    # A^T A = [a1 a2; a2 a3]
    # (A^T A)^-1 = determinant_factor * [a3 -a2; -a2, a1]

    # A^T b = [d1; d2]
    # Note: A^T b depends on the variable v. Using structure [1/2, v]
    a = zeros(eltype(u), 3)             # [a1, a2, a3]
    d = zeros(eltype(u), size(u, 1), 2) # [d1(v), d2(v)]

    # Parameter for limiting weights
    lambda = [0.0, 1.0]
    # lambda = [1.0, 0.0] # No limiting
    r = 4
    epsilon = 1.0e-13

    for element in eachelement(mesh, solver, cache)
        # The actual number of used stencils is `n_stencil_neighbors + 1`, since the full stencil is additionally used once.
        if solver.extended_reconstruction_stencil
            # Number of faces = Number of corners
            n_stencil_neighbors = elements.num_faces[element]
        else
            # Number of direct (edge) neighbors
            n_stencil_neighbors = length(reconstruction_stencil[element])
        end

        for stencil in 1:(n_stencil_neighbors + 1)
            # Reset variables
            a .= zero(eltype(u))
            # A^T b = [d1(v), d2(v)]
            # Note: A^T b depends on the variable v. Using vectors with d[v, 1/2]
            d .= zero(eltype(u))

            if solver.extended_reconstruction_stencil
                calc_gradient_extended!(stencil, element, a, d,
                                        reconstruction_stencil, reconstruction_distance,
                                        reconstruction_corner_elements, solution_data,
                                        equations, solver, cache)
            else
                calc_gradient_simple!(stencil, n_stencil_neighbors, element, a, d,
                                      reconstruction_stencil, reconstruction_distance,
                                      solution_data, equations)
            end

            # Divide by determinant
            AT_A_determinant = a[1] * a[3] - a[2]^2
            if isapprox(AT_A_determinant, 0.0)
                for v in eachvariable(equations)
                    reconstruction_gradient[:, v, stencil, element] .= 0.0
                end
                continue
            end
            a .*= 1 / AT_A_determinant

            # Solving least square problem
            for v in eachvariable(equations)
                reconstruction_gradient[1, v, stencil, element] = a[3] * d[v, 1] -
                                                                  a[2] * d[v, 2]
                reconstruction_gradient[2, v, stencil, element] = -a[2] * d[v, 1] +
                                                                  a[1] * d[v, 2]
            end
        end

        # Get limited reconstruction gradient by weighting the just computed
        weight_sum = zero(eltype(reconstruction_gradient))
        for v in eachvariable(equations)
            reconstruction_gradient_limited[:, v, element] .= zero(eltype(reconstruction_gradient_limited))
            weight_sum = zero(eltype(reconstruction_gradient))
            for j in 1:(n_stencil_neighbors + 1)
                gradient = get_node_coords(reconstruction_gradient, equations, solver,
                                           v, j, element)
                indicator = sum(gradient .^ 2)
                lambda_j = (j == 1) ? lambda[1] : lambda[2]
                weight = (lambda_j / (indicator + epsilon))^r
                for dim in axes(reconstruction_gradient_limited, 1)
                    reconstruction_gradient_limited[dim, v, element] += weight *
                                                                        gradient[dim]
                end
                weight_sum += weight
            end
            for dim in axes(reconstruction_gradient_limited, 1)
                reconstruction_gradient_limited[dim, v, element] /= weight_sum
            end
        end
    end

    exchange_gradient_data!(reconstruction_gradient_limited, mesh, equations, solver,
                            cache)

    return nothing
end

@inline function calc_gradient_simple!(stencil, n_stencil_neighbors, element, a, d,
                                       reconstruction_stencil, reconstruction_distance,
                                       solution_data, equations::AbstractEquations{2})
    for i in 1:n_stencil_neighbors
        # stencil=1 contains information from all neighbors
        # stencil=2,...,n_stencil_neighbors+1 is computed without (stencil+1)-th neighbor's information
        if i + 1 != stencil
            neighbor = reconstruction_stencil[element][i]
            distance = reconstruction_distance[element][i]
            a[1] += distance[1]^2
            a[2] += distance[1] * distance[2]
            a[3] += distance[2]^2

            for v in eachvariable(equations)
                d[v, 1] += distance[1] *
                           (solution_data[neighbor].u[v] -
                            solution_data[element].u[v])
                d[v, 2] += distance[2] *
                           (solution_data[neighbor].u[v] -
                            solution_data[element].u[v])
            end
        end
    end

    return nothing
end

@inline function calc_gradient_extended!(stencil, element, a, d,
                                         reconstruction_stencil,
                                         reconstruction_distance,
                                         reconstruction_corner_elements, solution_data,
                                         equations::AbstractEquations{2}, solver, cache)
    if stencil == 1
        # Full stencil
        for i in eachindex(reconstruction_stencil[element])
            neighbor = reconstruction_stencil[element][i]
            distance = reconstruction_distance[element][i]
            a[1] += distance[1]^2
            a[2] += distance[1] * distance[2]
            a[3] += distance[2]^2

            for v in eachvariable(equations)
                d[v, 1] += distance[1] *
                           (solution_data[neighbor].u[v] -
                            solution_data[element].u[v])
                d[v, 2] += distance[2] *
                           (solution_data[neighbor].u[v] -
                            solution_data[element].u[v])
            end
        end
    else
        # Partial stencils
        midpoint_element = get_node_coords(cache.elements.midpoint, equations, solver,
                                           element)
        for neighbor in reconstruction_corner_elements[stencil - 1, element]
            midpoint_neighbor = get_node_coords(cache.elements.midpoint, equations,
                                                solver, neighbor)
            distance = midpoint_neighbor .- midpoint_element

            a[1] += distance[1]^2
            a[2] += distance[1] * distance[2]
            a[3] += distance[2]^2

            for v in eachvariable(equations)
                d[v, 1] += distance[1] *
                           (solution_data[neighbor].u[v] -
                            solution_data[element].u[v])
                d[v, 2] += distance[2] *
                           (solution_data[neighbor].u[v] -
                            solution_data[element].u[v])
            end
        end
    end

    return nothing
end

function calc_gradient_reconstruction!(u, mesh::T8codeMesh{3}, equations, solver, cache)
    if solver.order == 1
        return nothing
    elseif solver.order > 2
        error("Order $(solver.order) is not supported yet!")
    end

    (; elements, communication_data) = cache
    (; reconstruction_stencil, reconstruction_distance, reconstruction_corner_elements, reconstruction_gradient, reconstruction_gradient_limited) = elements
    (; solution_data) = communication_data

    # A         N x 3 Matrix, where N is the number of stencil neighbors
    # A^T A     3 x 3 Matrix
    # b         N     Vector
    # A^T b     3     Vector

    # Matrix/vector notation
    # A^T A = [a11 a12 a13; a12 a22 a23; a13 a23 a33]
    # det(A^T A) = a11 * (a22*a33 - a23^2) - a12 * (a12*a33 - a13*a23) + a13 * (a12*a23 - a22*a13)
    # (A^T A)^-1 = 1/det(A^T A) * [(a33*a22-a23^2) -(a33*a12-a23*a13) (a23*a12-a22*a13) ;
    #                               ***             (a11*a33-a13^2)   -(a23*a11-a12*a13);
    #                               ***            ***                (a11*a22-a12^2)   ]

    # A^T b = [d1; d2; d3]
    # Note: A^T b depends on the variable v. Using structure [1/2/3, v]
    a = zeros(eltype(u), 6)             # [a11, a12, a13, a22, a23, a33]
    d = zeros(eltype(u), size(u, 1), 3) # [d1(v), d2(v), d3(v)]

    # Parameter for limiting weights
    lambda = [0.0, 1.0]
    # lambda = [1.0, 0.0] # No limiting
    r = 4
    epsilon = 1.0e-13

    for element in eachelement(mesh, solver, cache)
        # The actual number of used stencils is `n_stencil_neighbors + 1`, since the full stencil is additionally used once.
        if solver.extended_reconstruction_stencil
            # Number of faces = Number of corners
            n_stencil_neighbors = elements.num_faces[element]
        else
            # Number of direct (surface) neighbors
            n_stencil_neighbors = length(reconstruction_stencil[element])
        end

        for stencil in 1:(n_stencil_neighbors + 1)
            # Reset variables
            a .= zero(eltype(u))
            # A^T b = [d1(v), d2(v), d3(v)]
            # Note: A^T b depends on the variable v. Using vectors with d[v, 1/2/3]
            d .= zero(eltype(u))

            if solver.extended_reconstruction_stencil
                calc_gradient_extended!(stencil, element, a, d,
                                        reconstruction_stencil, reconstruction_distance,
                                        reconstruction_corner_elements, solution_data,
                                        equations, solver, cache)
            else
                calc_gradient_simple!(stencil, n_stencil_neighbors, element, a, d,
                                      reconstruction_stencil, reconstruction_distance,
                                      solution_data, equations)
            end

            # Divide by determinant
            # Determinant    = a11  * (a22  * a33  -  a23^2 ) - a12  * (a12  * a33  - a13  * a23 ) + a13  * (a12  * a23  - a22  * a13 )
            AT_A_determinant = a[1] * (a[4] * a[6] - a[5]^2) -
                               a[2] * (a[2] * a[6] - a[3] * a[5]) +
                               a[3] * (a[2] * a[5] - a[4] * a[3])
            if isapprox(AT_A_determinant, 0.0)
                for v in eachvariable(equations)
                    reconstruction_gradient[:, v, stencil, element] .= 0.0
                end
                continue
            end

            # det(A^T A) = a11 * (a22*a33 - a23^2) - a12 * (a12*a33 - a13*a23) + a13 * (a12*a23 - a22*a13)
            # (A^T A)^-1 = 1/det(A^T A) * [(a33*a22-a23^2) -(a33*a12-a23*a13) (a23*a12-a22*a13) ;
            #                               ***             (a11*a33-a13^2)   -(a23*a11-a12*a13);
            #                               ***            ***                (a11*a22-a12^2)   ]
            # = [m11 m12 m13; m12 m22 m23; m13 m23 m33]
            m11 = a[6] * a[4] - a[5]^2
            m12 = -(a[6] * a[2] - a[5] * a[3])
            m13 = (a[5] * a[2] - a[4] * a[3])
            m22 = (a[1] * a[6] - a[3]^2)
            m23 = -(a[5] * a[1] - a[2] * a[3])
            m33 = (a[1] * a[4] - a[2]^2)

            # Solving least square problem
            for v in eachvariable(equations)
                reconstruction_gradient[1, v, stencil, element] = 1 / AT_A_determinant *
                                                                  (m11 * d[v, 1] +
                                                                   m12 * d[v, 2] +
                                                                   m13 * d[v, 3])
                reconstruction_gradient[2, v, stencil, element] = 1 / AT_A_determinant *
                                                                  (m12 * d[v, 1] +
                                                                   m22 * d[v, 2] +
                                                                   m23 * d[v, 3])
                reconstruction_gradient[3, v, stencil, element] = 1 / AT_A_determinant *
                                                                  (m13 * d[v, 1] +
                                                                   m23 * d[v, 2] +
                                                                   m33 * d[v, 3])
            end
        end

        # Get limited reconstruction gradient by weighting the just computed
        weight_sum = zero(eltype(reconstruction_gradient))
        for v in eachvariable(equations)
            reconstruction_gradient_limited[:, v, element] .= zero(eltype(reconstruction_gradient_limited))
            weight_sum = zero(eltype(reconstruction_gradient))
            for j in 1:(n_stencil_neighbors + 1)
                gradient = get_node_coords(reconstruction_gradient, equations, solver,
                                           v, j, element)
                indicator = sum(gradient .^ 2)
                lambda_j = (j == 1) ? lambda[1] : lambda[2]
                weight = (lambda_j / (indicator + epsilon))^r
                for dim in axes(reconstruction_gradient_limited, 1)
                    reconstruction_gradient_limited[dim, v, element] += weight *
                                                                        gradient[dim]
                end
                weight_sum += weight
            end
            for dim in axes(reconstruction_gradient_limited, 1)
                reconstruction_gradient_limited[dim, v, element] /= weight_sum
            end
        end
    end

    exchange_gradient_data!(reconstruction_gradient_limited, mesh, equations, solver,
                            cache)

    return nothing
end

@inline function calc_gradient_simple!(stencil, n_stencil_neighbors, element, a, d,
                                       reconstruction_stencil, reconstruction_distance,
                                       solution_data, equations::AbstractEquations{3})
    for i in 1:n_stencil_neighbors
        # stencil=1 contains information from all neighbors
        # stencil=2,...,n_stencil_neighbors+1 is computed without (stencil+1)-th neighbor's information
        if i + 1 != stencil
            neighbor = reconstruction_stencil[element][i]
            distance = reconstruction_distance[element][i]
            a[1] += distance[1]^2 # = a11
            a[2] += distance[1] * distance[2] # = a12
            a[3] += distance[1] * distance[3]
            a[4] += distance[2]^2
            a[5] += distance[2] * distance[3]
            a[6] += distance[3]^2

            for v in eachvariable(equations)
                d[v, 1] += distance[1] *
                           (solution_data[neighbor].u[v] -
                            solution_data[element].u[v])
                d[v, 2] += distance[2] *
                           (solution_data[neighbor].u[v] -
                            solution_data[element].u[v])
                d[v, 3] += distance[3] *
                           (solution_data[neighbor].u[v] -
                            solution_data[element].u[v])
            end
        end
    end

    return nothing
end

@inline function calc_gradient_extended!(stencil, element, a, d,
                                         reconstruction_stencil,
                                         reconstruction_distance,
                                         reconstruction_corner_elements, solution_data,
                                         equations::AbstractEquations{3}, solver, cache)
    if stencil == 1
        # Full stencil
        for i in eachindex(reconstruction_stencil[element])
            neighbor = reconstruction_stencil[element][i]
            distance = reconstruction_distance[element][i]
            a[1] += distance[1]^2 # = a11
            a[2] += distance[1] * distance[2] # = a12
            a[3] += distance[1] * distance[3]
            a[4] += distance[2]^2
            a[5] += distance[2] * distance[3]
            a[6] += distance[3]^2

            for v in eachvariable(equations)
                d[v, 1] += distance[1] *
                           (solution_data[neighbor].u[v] -
                            solution_data[element].u[v])
                d[v, 2] += distance[2] *
                           (solution_data[neighbor].u[v] -
                            solution_data[element].u[v])
                d[v, 3] += distance[3] *
                           (solution_data[neighbor].u[v] -
                            solution_data[element].u[v])
            end
        end
    else
        # Partial stencils
        midpoint_element = get_node_coords(cache.elements.midpoint, equations, solver,
                                           element)
        for neighbor in reconstruction_corner_elements[stencil - 1, element]
            midpoint_neighbor = get_node_coords(cache.elements.midpoint, equations,
                                                solver, neighbor)
            distance = midpoint_neighbor .- midpoint_element

            a[1] += distance[1]^2 # = a11
            a[2] += distance[1] * distance[2] # = a12
            a[3] += distance[1] * distance[3]
            a[4] += distance[2]^2
            a[5] += distance[2] * distance[3]
            a[6] += distance[3]^2

            for v in eachvariable(equations)
                d[v, 1] += distance[1] *
                           (solution_data[neighbor].u[v] -
                            solution_data[element].u[v])
                d[v, 2] += distance[2] *
                           (solution_data[neighbor].u[v] -
                            solution_data[element].u[v])
                d[v, 3] += distance[3] *
                           (solution_data[neighbor].u[v] -
                            solution_data[element].u[v])
            end
        end
    end

    return nothing
end

function prolong2interfaces!(cache, mesh::T8codeMesh, equations, solver::FV)
    (; interfaces, communication_data) = cache
    (; solution_data, domain_data, gradient_data) = communication_data

    for interface in eachinterface(solver, cache)
        element = interfaces.neighbor_ids[1, interface]
        neighbor = interfaces.neighbor_ids[2, interface]
        if solver.order == 1
            for v in eachvariable(equations)
                interfaces.u[1, v, interface] = solution_data[element].u[v]
                interfaces.u[2, v, interface] = solution_data[neighbor].u[v]
            end
        elseif solver.order == 2
            face_element = interfaces.faces[1, interface]
            face_neighbor = interfaces.faces[2, interface]

            face_midpoint_element = domain_data[element].face_midpoints[face_element]
            face_midpoint_neighbor = domain_data[neighbor].face_midpoints[face_neighbor]

            midpoint_element = domain_data[element].midpoint
            midpoint_neighbor = domain_data[neighbor].midpoint

            vector_element = face_midpoint_element .- midpoint_element
            vector_neighbor = face_midpoint_neighbor .- midpoint_neighbor
            for v in eachvariable(equations)
                gradient_v_element = gradient_data[element].reconstruction_gradient_limited[v]
                gradient_v_neighbor = gradient_data[neighbor].reconstruction_gradient_limited[v]
                interfaces.u[1, v, interface] = solution_data[element].u[v] +
                                                dot(gradient_v_element, vector_element)
                interfaces.u[2, v, interface] = solution_data[neighbor].u[v] +
                                                dot(gradient_v_neighbor,
                                                    vector_neighbor)
            end
        else
            error("Order $(solver.order) is not supported.")
        end
    end

    return nothing
end

function calc_interface_flux!(du, mesh::T8codeMesh,
                              nonconservative_terms::False, equations,
                              solver::FV, cache)
    (; surface_flux) = solver
    (; elements, interfaces) = cache

    for interface in eachinterface(solver, cache)
        element = interfaces.neighbor_ids[1, interface]
        neighbor = interfaces.neighbor_ids[2, interface]
        face = interfaces.faces[1, interface]

        # TODO: Save normal and face_areas in interface?
        normal = get_node_coords(elements.face_normals, equations, solver,
                                 face, element)
        u_ll, u_rr = get_surface_node_vars(interfaces.u, equations, solver,
                                           interface)
        flux = surface_flux(u_ll, u_rr, normal, equations)

        for v in eachvariable(equations)
            flux_ = elements.face_areas[face, element] * flux[v]
            du[v, element] -= flux_
            if !is_ghost_cell(neighbor, mesh)
                du[v, neighbor] += flux_
            end
        end
    end

    return nothing
end

function prolong2boundaries!(cache, mesh::T8codeMesh, equations, solver::FV)
    (; elements, boundaries, communication_data) = cache
    (; solution_data) = communication_data
    (; midpoint, face_midpoints, reconstruction_gradient_limited) = elements

    for boundary in eachboundary(solver, cache)
        element = boundaries.neighbor_ids[boundary]
        if solver.order == 1
            for v in eachvariable(equations)
                boundaries.u[v, boundary] = solution_data[element].u[v]
            end
        elseif solver.order == 2
            face_element = boundaries.faces[boundary]

            face_midpoint_element = get_node_coords(face_midpoints, equations, solver,
                                                    face_element, element)

            midpoint_element = get_node_coords(midpoint, equations, solver, element)

            vector_element = face_midpoint_element .- midpoint_element
            for v in eachvariable(equations)
                gradient_v_element = get_node_coords(reconstruction_gradient_limited,
                                                     equations, solver, v, element)
                boundaries.u[v, boundary] = solution_data[element].u[v] +
                                            dot(gradient_v_element, vector_element)
            end
        else
            error("Order $(solver.order) is not supported.")
        end
    end

    return nothing
end

function calc_boundary_flux!(du, cache, t,
                             boundary_condition::BoundaryConditionPeriodic,
                             mesh::T8codeMesh,
                             equations, solver::FV)
    @assert isempty(eachboundary(solver, cache))
end

# Function barrier for type stability
function calc_boundary_flux!(du, cache, t, boundary_conditions,
                             mesh::T8codeMesh,
                             equations, solver::FV)
    @unpack boundary_condition_types, boundary_indices = boundary_conditions

    calc_boundary_flux_by_type!(du, cache, t, boundary_condition_types,
                                boundary_indices, mesh, equations, solver)
    return nothing
end

# Iterate over tuples of boundary condition types and associated indices
# in a type-stable way using "lispy tuple programming".
function calc_boundary_flux_by_type!(du, cache, t, BCs::NTuple{N, Any},
                                     BC_indices::NTuple{N, Vector{Int}},
                                     mesh::T8codeMesh,
                                     equations, solver::FV) where {N}
    # Extract the boundary condition type and index vector
    boundary_condition = first(BCs)
    boundary_condition_indices = first(BC_indices)
    # Extract the remaining types and indices to be processed later
    remaining_boundary_conditions = Base.tail(BCs)
    remaining_boundary_condition_indices = Base.tail(BC_indices)

    # process the first boundary condition type
    calc_boundary_flux!(du, cache, t, boundary_condition, boundary_condition_indices,
                        mesh, equations, solver)

    # recursively call this method with the unprocessed boundary types
    calc_boundary_flux_by_type!(du, cache, t, remaining_boundary_conditions,
                                remaining_boundary_condition_indices,
                                mesh, equations, solver)

    return nothing
end

# terminate the type-stable iteration over tuples
function calc_boundary_flux_by_type!(du, cache, t, BCs::Tuple{}, BC_indices::Tuple{},
                                     mesh::T8codeMesh,
                                     equations, solver::FV)
    nothing
end

function calc_boundary_flux!(du, cache, t, boundary_condition::BC, boundary_indexing,
                             mesh::T8codeMesh,
                             equations, solver::FV) where {BC}
    (; elements, boundaries) = cache
    (; surface_flux) = solver

    for local_index in eachindex(boundary_indexing)
        # Use the local index to get the global boundary index from the pre-sorted list
        boundary = boundary_indexing[local_index]

        # Get information on the adjacent element, compute the surface fluxes,
        # and store them
        element = boundaries.neighbor_ids[boundary]
        face = boundaries.faces[boundary]

        # TODO: Save normal and face_areas in interface?
        normal = get_node_coords(cache.elements.face_normals, equations, solver,
                                 face, element)

        u_inner = get_node_vars(boundaries.u, equations, solver, boundary)

        # Coordinates at boundary node
        face_midpoint = get_node_coords(cache.elements.face_midpoints, equations,
                                        solver, face, element)

        flux = boundary_condition(u_inner, normal, face_midpoint, t, surface_flux,
                                  equations)
        for v in eachvariable(equations)
            flux_ = elements.face_areas[face, element] * flux[v]
            du[v, element] -= flux_
        end
    end

    return nothing
end

function calc_sources!(du, u, t, source_terms::Nothing, mesh::T8codeMesh,
                       equations::AbstractEquations, solver::FV, cache)
    return nothing
end

function calc_sources!(du, u, t, source_terms, mesh::T8codeMesh,
                       equations::AbstractEquations, solver::FV, cache)
    @threaded for element in eachelement(mesh, solver, cache)
        u_local = get_node_vars(u, equations, solver, element)
        x_local = get_node_coords(cache.elements.midpoint, equations, solver, element)
        du_local = source_terms(u_local, x_local, t, equations)
        for v in eachvariable(equations)
            du[v, element] += du_local[v]
        end
    end

    return nothing
end

function get_element_variables!(element_variables, u,
                                mesh, equations,
                                solver::FV, cache)
    return nothing
end

function get_node_variables!(node_variables, mesh,
                             equations, solver::FV, cache)
    return nothing
end

function SolutionAnalyzer(solver::FV; kwargs...)
end

function create_cache_analysis(analyzer, mesh,
                               equations, solver::FV, cache,
                               RealT, uEltype)
end

function T8codeMesh(cmesh::Ptr{t8_cmesh}, solver::DG; kwargs...)
    T8codeMesh(cmesh; kwargs...)
end

function T8codeMesh(cmesh::Ptr{t8_cmesh}, solver::FV; kwargs...)
    T8codeMesh(cmesh; polydeg = 0, kwargs...)
end

# Container data structures
include("containers.jl")
end # @muladd
