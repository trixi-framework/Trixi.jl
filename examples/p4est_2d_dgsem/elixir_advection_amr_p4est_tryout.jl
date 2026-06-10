using OrdinaryDiffEqLowStorageRK
using Trixi
using CUDA
using Adapt
using KernelAbstractions
using StaticArrays


module TrixiExtension
using Trixi
using KernelAbstractions
using StaticArrays

struct IndicatorSolutionIndependent{Cache <: NamedTuple} <: Trixi.AbstractIndicator
    cache::Cache
end

function IndicatorSolutionIndependent(semi)
    basis = semi.solver.basis
    alpha = Vector{real(basis)}()
    cache = (; semi.mesh, alpha)
    return IndicatorSolutionIndependent{typeof(cache)}(cache)
end

@inline function periodic_distance_2d(coordinates, center, domain_length)
    dx          = coordinates .- center
    dx_shifted  = abs.(dx .% domain_length)
    dx_periodic = min.(dx_shifted, domain_length .- dx_shifted)
    return sqrt(sum(dx_periodic .^ 2))
end

@inline function original_coordinates(coordinates, cell_length)
    offset      = coordinates .% cell_length
    offset_sign = sign.(offset)
    border      = coordinates - offset
    center = border + (offset_sign .* cell_length / 2)
    return center
end

@inline function compute_alpha_per_element(element, node_coordinates, t)
    advection_velocity = (0.2, -0.7)
    center             = t .* advection_velocity
    inner_distance     = 1.0
    outer_distance     = 1.85

    coordinates = SVector(
        0.5 * (node_coordinates[1, 1,   1, element] +
               node_coordinates[1, end, 1, element]),
        0.5 * (node_coordinates[2, 1,   1, element] +
               node_coordinates[2, 1, end, element])
    )

    cell_coordinates = original_coordinates(coordinates, 5.0 / 8.0)
    cell_distance    = periodic_distance_2d(cell_coordinates, center, 10.0)

    if cell_distance < (inner_distance + outer_distance) / 2
        cell_coordinates = original_coordinates(coordinates, 5.0 / 16.0)
        cell_distance    = periodic_distance_2d(cell_coordinates, center, 10.0)
    end

    target_level = (cell_distance < inner_distance) + (cell_distance < outer_distance)
    return target_level / 2.0
end

@kernel function indicator_kernel!(alpha, node_coordinates, t)
    element = @index(Global)
    @inbounds alpha[element] = compute_alpha_per_element(element, node_coordinates, t)
end

function (indicator::IndicatorSolutionIndependent)(::Nothing,
                                                    u::AbstractArray{<:Any,4},
                                                    mesh, equations, dg, cache;
                                                    t, kwargs...)
    alpha = indicator.cache.alpha
    resize!(alpha, nelements(dg, cache))
    @unpack node_coordinates = cache.elements
    Trixi.@threaded for element in eachindex(alpha)
        @inbounds alpha[element] = compute_alpha_per_element(element, node_coordinates, t)
    end
    return alpha
end

function (indicator::IndicatorSolutionIndependent)(backend::KernelAbstractions.Backend,
                                                    u::AbstractArray{<:Any,4},
                                                    mesh, equations, dg, cache;
                                                    t, kwargs...)
    alpha = indicator.cache.alpha
    resize!(alpha, nelements(dg, cache))
    @unpack node_coordinates = cache.elements
    nelems = nelements(dg, cache)
    nelems == 0 && return alpha

    alpha_gpu=CuArray{eltype(alpha)}(undef, nelems)

    kernel! = indicator_kernel!(backend)
    kernel!(alpha_gpu, node_coordinates, t; ndrange = nelems)
    KernelAbstractions.synchronize(backend)

    copyto!(alpha, alpha_gpu)

    return alpha
end

function (indicator::IndicatorSolutionIndependent)(u::AbstractArray{<:Any,4},
                                                    mesh, equations, dg, cache;
                                                    t, kwargs...)
    backend = Trixi.trixi_backend(u)
    return indicator(backend, u, mesh, equations, dg, cache; t = t, kwargs...)
end

end # module TrixiExtension
import .TrixiExtension


#semidiscretization

advection_velocity  = (0.2, -0.7)
equations           = LinearScalarAdvectionEquation2D(advection_velocity)
initial_condition   = initial_condition_gauss
solver              = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)
coordinates_min     = (-5.0, -5.0)
coordinates_max     = ( 5.0,  5.0)
trees_per_dimension = (1, 1)

mesh = P4estMesh(trees_per_dimension,
                 polydeg                  = 3,
                 coordinates_min          = coordinates_min,
                 coordinates_max          = coordinates_max,
                 initial_refinement_level = 4,
                 periodicity              = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_condition_periodic)
tspan = (0.0, 10.0)
ode   = semidiscretize(semi, tspan)

summary_callback  = SummaryCallback()
analysis_interval = 100

analysis_callback = AnalysisCallback(semi,
                                     interval = analysis_interval,
                                     extra_analysis_integrals = (entropy,))
alive_callback    = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval              = 100,
                                     save_initial_solution = true,
                                     save_final_solution   = true,
                                     solution_variables    = cons2prim)

amr_controller = ControllerThreeLevel(
    semi,
    TrixiExtension.IndicatorSolutionIndependent(semi),
    base_level    = 4,
    med_level     = 5, med_threshold = 0.1,
    max_level     = 6, max_threshold = 0.6
)

amr_callback = AMRCallback(semi, amr_controller,
                           interval                            = 5,
                           adapt_initial_condition             = true,
                           adapt_initial_condition_only_refine = true)

stepsize_callback = StepsizeCallback(cfl = 1.6)

#added a callback that dont have amr_callback(callbacks_no_amr is called when simulation is on gpu)
callbacks_no_amr = CallbackSet(summary_callback,
                               analysis_callback,
                               alive_callback,
                               save_solution,
                               stepsize_callback)


@inline function Trixi.multiply_dimensionwise!(data_out::AbstractArray{<:Any, 2}, matrix::AbstractMatrix,
                                               data_in::AbstractArray{<:Any, 2}) 
    @inbounds for i in axes(data_out, 2), v in axes(data_out, 1)
        res = zero(eltype(data_out))
        for ii in axes(data_in, 2) 
            res += matrix[i, ii] * data_in[v, ii]
        end
        data_out[v, i] = res
    end

    return nothing
end


@inline function prolong2mortars_per_mortar!(mortars_u, u, mortar,
                                             MeshT::Type{<:Union{Trixi.P4estMesh{2},
                                                                 Trixi.P4estMeshView{2},
                                                                 Trixi.T8codeMesh{2}}},
                                             equations,
                                             neighbor_ids, node_indices,
                                             index_range,
                                             forward_lower,
                                             forward_upper,
                                             ::Val{N}, ::Val{NVARS}, ::Val{T}, ::Val{L}) where {N, NVARS, T, L}

    @inbounds begin
        #N = length(index_range)
        #T = eltype(u)
        small_indices    = node_indices[1, mortar]
        i_small_start, i_small_step = Trixi.index_to_start_step_2d(small_indices[1], index_range)
        j_small_start, j_small_step = Trixi.index_to_start_step_2d(small_indices[2], index_range)

        for position in 1:2
            i_small = i_small_start
            j_small = j_small_start
            element  = neighbor_ids[position, mortar]
            for i in eachindex(index_range)
                for v in Base.OneTo(NVARS)
                    mortars_u[1, v, position, i, mortar] = u[v, i_small, j_small, element]
                end
                i_small += i_small_step
                j_small += j_small_step
            end
        end

        #can use for loop for it and it is mutable 
        u_buffer = MArray{Tuple{NVARS, N}, T, 2, NVARS * N}(undef)

        large_indices    = node_indices[2, mortar]
        i_large_start, i_large_step = Trixi.index_to_start_step_2d(large_indices[1], index_range)
        j_large_start, j_large_step = Trixi.index_to_start_step_2d(large_indices[2], index_range)

        i_large = i_large_start
        j_large = j_large_start
        element  = neighbor_ids[3, mortar]
        for i in eachindex(index_range)
            for v in Base.OneTo(NVARS)
                u_buffer[v, i] = u[v, i_large, j_large, element]
            end
            i_large += i_large_step
            j_large += j_large_step
        end


        # for i in eachnode(dg)
        #     for v in eachvariable(equations)
        #         u_buffer[v, i] = u[v, i_large, j_large, element]
        #     end
        #     i_large += i_large_step
        #     j_large += j_large_step
        # end


        #just trying
        val_lower = MMatrix{NVARS, N, T, NVARS * N}(undef)
        val_upper = MMatrix{NVARS, N, T, NVARS * N}(undef)

        Trixi.multiply_dimensionwise!(val_lower, forward_lower, u_buffer)
        Trixi.multiply_dimensionwise!(val_upper, forward_upper, u_buffer)

        
        for i in 1:N
            for v in Base.OneTo(NVARS)
                mortars_u[2, v, 1, i, mortar] = val_lower[v, i]
                mortars_u[2, v, 2, i, mortar] = val_upper[v, i]
            end
        end


        # Allocate the MMatrix registers using the static L
        # val_lower = MMatrix{NVARS, N, T, L}(undef)
        # val_upper = MMatrix{NVARS, N, T, L}(undef)

        # # Call the existing Trixi function!
        # Trixi.multiply_dimensionwise!(val_lower, forward_lower, u_buffer)
        # Trixi.multiply_dimensionwise!(val_upper, forward_upper, u_buffer)
        
        # # Store the calculated projections back into the global array
        # for i in 1:N
        #     for v in Base.OneTo(NVARS)
        #         mortars_u[2, v, 1, i, mortar] = val_lower[v, i]
        #         mortars_u[2, v, 2, i, mortar] = val_upper[v, i]
        #     end
        # end


        # in place of multiply dimensionwise(because used of MMatrix and SMatrix)
        # for i in 1:N
        #     for v in Base.OneTo(NVARS)
        #         val_lower = zero(T)
        #         val_upper = zero(T)
        #         for j in 1:N
        #             val_lower = val_lower + forward_lower[i, j] * u_buffer[v, j]
        #             val_upper = val_upper + forward_upper[i, j] * u_buffer[v, j]
        #         end
        #         mortars_u[2, v, 1, i, mortar] = val_lower
        #         mortars_u[2, v, 2, i, mortar] = val_upper
        #     end
        # end

    end # @inbounds

    return nothing
end


@kernel function prolong2mortars_KAkernel!(mortars_u, u,
                                            MeshT::Type{<:Union{Trixi.P4estMesh{2},
                                                                Trixi.P4estMeshView{2},
                                                                Trixi.T8codeMesh{2}}},
                                            equations,
                                            neighbor_ids, node_indices,
                                            index_range,
                                            forward_lower,
                                            forward_upper,
                                            ::Val{N}, ::Val{NVARS}, ::Val{T}, ::Val{L}) where {N, NVARS, T, L}
    mortar = @index(Global)
    prolong2mortars_per_mortar!(mortars_u, u, mortar, MeshT, equations,
                                 neighbor_ids, node_indices, index_range,
                                 forward_lower, forward_upper, Val(N), Val(NVARS), Val(T), Val(L))
end


function Trixi.prolong2mortars!(backend::KernelAbstractions.Backend, cache, u,
                                mesh::Union{Trixi.P4estMesh{2}, Trixi.P4estMeshView{2},
                                            Trixi.T8codeMesh{2}},
                                equations,
                                mortar_l2::Trixi.LobattoLegendreMortarL2,
                                dg::Trixi.DGSEM{<:Trixi.LobattoLegendreBasis})

    Trixi.nmortars(dg, cache) == 0 && return nothing

    @unpack mortars = cache
    @unpack neighbor_ids, node_indices = cache.mortars

    index_range = Trixi.eachnode(dg)

    N = Trixi.nnodes(dg)
    T = eltype(u)
    NVARS = Trixi.nvariables(equations)
    L = N * NVARS

    # Convert Matrix → SMatrix 
    # forward_lower = SMatrix{N, N, T, N * N}(Array(mortar_l2.forward_lower))
    # forward_upper = SMatrix{N, N, T, N * N}(Array(mortar_l2.forward_upper))

    kernel! = prolong2mortars_KAkernel!(backend)
    kernel!(mortars.u, u, typeof(mesh), equations,
            neighbor_ids, node_indices, index_range,
            mortar_l2.forward_lower, mortar_l2.forward_upper, 
            Val(N), Val(NVARS), Val(T), Val(L); 
            ndrange = Trixi.nmortars(dg, cache))

    return nothing
end


function Trixi.prolong2mortars!(cache, u::CuArray{<:Any, 4},
                                mesh::Union{Trixi.P4estMesh{2}, Trixi.P4estMeshView{2}, Trixi.T8codeMesh{2}},
                                equations,
                                mortar_l2::Trixi.LobattoLegendreMortarL2,
                                dg::Trixi.DGSEM{<:Trixi.LobattoLegendreBasis})
                                
    backend = KernelAbstractions.get_backend(u)
    Trixi.prolong2mortars!(backend, cache, u, mesh, equations, mortar_l2, dg)
    return nothing
end








@inline function Trixi.multiply_dimensionwise!(data_out::AbstractArray{<:Any, 2},
                                               matrix1::AbstractMatrix, data_in1::AbstractArray{<:Any, 2},
                                               matrix2::AbstractMatrix, data_in2::AbstractArray{<:Any, 2})
    @inbounds for i in axes(data_out, 2), v in axes(data_out, 1)
        res = zero(eltype(data_out))
        for ii in axes(data_in1, 2)
            res += matrix1[i, ii] * data_in1[v, ii] + matrix2[i, ii] * data_in2[v, ii]
        end
        data_out[v, i] = res
    end
    return nothing
end



@inline function mortar_fluxes_to_elements!(surface_flux_values,
                                            neighbor_ids, node_indices,
                                            reverse_lower, reverse_upper,
                                            mortar,
                                            fstar_primary_1, fstar_primary_2,
                                            fstar_s_lower, fstar_s_upper,
                                            u_buffer, N, NVARS)

    small_indices = node_indices[1, mortar]
    small_direction = Trixi.indices2direction(small_indices)



    for position in 1:2
        element = neighbor_ids[position, mortar]
        for i in 1:N
            for v in Base.OneTo(NVARS)
                surface_flux_values[v, i, small_direction, element] = position == 1 ? fstar_primary_1[v, i] : fstar_primary_2[v, i]
            end
        end
    end
    

    Trixi.multiply_dimensionwise!(u_buffer,
                                  reverse_upper, fstar_s_upper,
                                  reverse_lower, fstar_s_lower)

    u_buffer .*= -2

    large_element   = neighbor_ids[3, mortar]
    large_indices   = node_indices[2, mortar]
    large_direction = Trixi.indices2direction(large_indices)

    if :i_backward in large_indices
        for i in 1:N
            for v in Base.OneTo(NVARS)
            #for v in eachvariable(equations)
                surface_flux_values[v, N + 1 - i, large_direction, large_element] = u_buffer[v, i]
            end
        end
    else
        for i in 1:N
            for v in Base.OneTo(NVARS)
            #for v in eachvariable(equations)
                surface_flux_values[v, i, large_direction, large_element] = u_buffer[v, i]
            end
        end
    end

    return nothing
end



@inline function gpu_calc_mortar_flux!(fstar_p_1, fstar_p_2, fstar_s_1, fstar_s_2,
                                       MeshT,
                                       have_nonconservative_terms::Trixi.False, equations,
                                       pure_surface_flux, dg::Trixi.DGSEM, mortar_u,
                                       mortar_index, position_index, normal_direction,
                                       node_index)

    u_ll, u_rr = Trixi.get_surface_node_vars(mortar_u, equations, dg, position_index,
                                             node_index, mortar_index)

    flux = pure_surface_flux(u_ll, u_rr, normal_direction, equations)

    if position_index == 1
        Trixi.set_node_vars!(fstar_p_1, flux, equations, dg, node_index)
        Trixi.set_node_vars!(fstar_s_1, flux, equations, dg, node_index)
    else
        Trixi.set_node_vars!(fstar_p_2, flux, equations, dg, node_index)
        Trixi.set_node_vars!(fstar_s_2, flux, equations, dg, node_index)
    end
    return nothing
end


@inline function gpu_calc_mortar_flux!(fstar_p_1, fstar_p_2, fstar_s_1, fstar_s_2,
                                       MeshT,
                                       have_nonconservative_terms::Trixi.True, equations,
                                       pure_surface_flux, dg::Trixi.DGSEM, mortar_u,
                                       mortar_index, position_index, normal_direction,
                                       node_index)

    surface_flux, nonconservative_flux = pure_surface_flux

    u_ll, u_rr = Trixi.get_surface_node_vars(mortar_u, equations, dg, position_index,
                                             node_index, mortar_index)

    flux = surface_flux(u_ll, u_rr, normal_direction, equations)

    noncons_primary = nonconservative_flux(u_ll, u_rr, normal_direction, equations)
    noncons_secondary = nonconservative_flux(u_rr, u_ll, normal_direction, equations)

    flux_plus_noncons_primary = flux + 0.5f0 * noncons_primary
    flux_plus_noncons_secondary = flux + 0.5f0 * noncons_secondary

    if position_index == 1
        Trixi.set_node_vars!(fstar_p_1, flux_plus_noncons_primary, equations, dg, node_index)
        Trixi.set_node_vars!(fstar_s_1, flux_plus_noncons_secondary, equations, dg, node_index)
    else
        Trixi.set_node_vars!(fstar_p_2, flux_plus_noncons_primary, equations, dg, node_index)
        Trixi.set_node_vars!(fstar_s_2, flux_plus_noncons_secondary, equations, dg, node_index)
    end

    return nothing
end


@kernel function calc_mortar_flux_KAkernel!(surface_flux_values,
                                            MeshT::Type{<:Union{Trixi.P4estMesh{2},
                                                                Trixi.P4estMeshView{2},
                                                                Trixi.T8codeMesh{2}}},
                                            have_nonconservative_terms, equations,
                                            pure_surface_flux, dg::Trixi.DGSEM, 
                                            mortars_u, neighbor_ids, node_indices,
                                            contravariant_vectors,
                                            reverse_lower, reverse_upper, index_range,
                                            ::Val{N}, ::Val{NVARS}, ::Val{T}, ::Val{L}) where {N, NVARS, T, L}
    mortar = @index(Global)

    @inbounds begin
        fstar_p_1 = MMatrix{NVARS, N, T, L}(undef)
        fstar_p_2 = MMatrix{NVARS, N, T, L}(undef)
        fstar_s_1 = MMatrix{NVARS, N, T, L}(undef)
        fstar_s_2 = MMatrix{NVARS, N, T, L}(undef)
        u_buffer  = MMatrix{NVARS, N, T, L}(undef)
        

        small_indices = node_indices[1, mortar]
        small_direction = Trixi.indices2direction(small_indices)

        i_small_start, i_small_step = Trixi.index_to_start_step_2d(small_indices[1], index_range)
        j_small_start, j_small_step = Trixi.index_to_start_step_2d(small_indices[2], index_range)

        for position in 1:2
            i_small = i_small_start
            j_small = j_small_start
            element = neighbor_ids[position, mortar]
            
            for node in 1:N
                normal_direction = Trixi.get_normal_direction(small_direction,
                                                              contravariant_vectors,
                                                              i_small, j_small, element)

                gpu_calc_mortar_flux!(fstar_p_1, fstar_p_2, fstar_s_1, fstar_s_2,
                                        MeshT, have_nonconservative_terms, equations,
                                        pure_surface_flux, dg, mortars_u,
                                        mortar, position, normal_direction, node)

                i_small += i_small_step
                j_small += j_small_step
            end
        end

        mortar_fluxes_to_elements!(surface_flux_values,
                                       neighbor_ids, node_indices,
                                       reverse_lower, reverse_upper,
                                       mortar,
                                       fstar_p_1, fstar_p_2,
                                       fstar_s_1, fstar_s_2,
                                       u_buffer, N, NVARS)
    end
end



function Trixi.calc_mortar_flux!(backend::KernelAbstractions.Backend, surface_flux_values,
                                 mesh::Union{Trixi.P4estMesh{2}, Trixi.P4estMeshView{2}, Trixi.T8codeMesh{2}},
                                 have_nonconservative_terms, equations,
                                 mortar_l2::Trixi.LobattoLegendreMortarL2,
                                 surface_integral, dg::Trixi.DGSEM, cache)

    Trixi.nmortars(dg, cache) == 0 && return nothing

    @unpack neighbor_ids, node_indices = cache.mortars
    @unpack contravariant_vectors = cache.elements
    mortars_u = cache.mortars.u
    pure_surface_flux = surface_integral.surface_flux
    index_range = Trixi.eachnode(dg)

    N     = Trixi.nnodes(dg)
    NVARS = Trixi.nvariables(equations)
    T     = eltype(surface_flux_values)
    L     = N * NVARS

    kernel! = calc_mortar_flux_KAkernel!(backend)
    
    kernel!(surface_flux_values, typeof(mesh), have_nonconservative_terms,
            equations, pure_surface_flux, dg,
            mortars_u, neighbor_ids, node_indices, contravariant_vectors,
            mortar_l2.reverse_lower, mortar_l2.reverse_upper, index_range,
            Val(N), Val(NVARS), Val(T), Val(L);
            ndrange = Trixi.nmortars(dg, cache))

    return nothing
end

function Trixi.calc_mortar_flux!(surface_flux_values::CuArray,
                                 mesh::Union{Trixi.P4estMesh{2}, Trixi.P4estMeshView{2}, Trixi.T8codeMesh{2}},
                                 have_nonconservative_terms, equations,
                                 mortar_l2::Trixi.LobattoLegendreMortarL2,
                                 surface_integral, dg::Trixi.DGSEM, cache)
    
    backend = KernelAbstractions.get_backend(surface_flux_values)
    Trixi.calc_mortar_flux!(backend, surface_flux_values, mesh, have_nonconservative_terms,
                            equations, mortar_l2, surface_integral, dg, cache)
    return nothing
end





#added this to make prolong2mortar and mortar flux working(on CPU)
#make here false (for scalar indexing not allowed error)
CUDA.allowscalar(false)
const GPU_STEPS = 5

# Simulation state — u_cpu is a plain CPU Array between every burst
t_now         = tspan[1]
dt_now        = 1.0         # overwritten by StepsizeCallback on the first step
u_cpu         = Array(ode.u0)
global step_counter  = 0
burst_counter = 0


println(" GPU + AMR Hybrid Simulation")

# 
# Main GPU + AMR loop
# 
while t_now < tspan[2]

    global u_cpu
    
    global burst_counter += 1

    println("[Burst #$(burst_counter) | t = $(round(t_now; digits=6))]")

    #
    #allocated the problem on gpu 
    u0_gpu   = Trixi.trixi_adapt(CuArray, eltype(u_cpu), u_cpu)  
    
    #allocated the semi on gpu rather than cache directly 
    semi_gpu=Trixi.trixi_adapt(CuArray, eltype(u_cpu), semi)
    
    #created a new odeproblem with gpu array(CuArray) as u0
    prob_gpu = remake(ode; u0 = u0_gpu, p=semi_gpu, tspan = (t_now, tspan[2]))

    #solve on gpu (Note:callback_no_amr)
    global integrator_gpu = init(prob_gpu, CarpenterKennedy2N54(williamson_condition = false);
                          dt       = dt_now,
                          ode_default_options()...,
                          callback = callbacks_no_amr)

    # Check (if got CuArray then working fine)
    println("  [GPU] Cache type: $(typeof(integrator_gpu.cache.k))")

    for _ in 1:GPU_STEPS
        if integrator_gpu.t >= tspan[2]
            break
        end
        #finally run the simulation on gpu
        step!(integrator_gpu)      
        global step_counter += 1
    end

    
    
    #best way to use kernelabstarction here(for different gpu)
    # i think i should put it outside loop as no need to cumpute everytime
    backend = KernelAbstractions.get_backend(u_cpu) 
    KernelAbstractions.synchronize(backend)

    global t_now  = integrator_gpu.t
    global dt_now = integrator_gpu.dt
    println("  [GPU] Done. t = $(round(t_now; digits=6)), steps = $step_counter")

    if t_now >= tspan[2]
        break
    end



    #avoid this as allocating new memory everytime is expensive
    #global u_cpu = Array(integrator_gpu.u)


    println("Lengths -> CPU Cache: $(length(u_cpu)) | GPU VRAM: $(length(integrator_gpu.u))")
    #checked size of u_cpu and integrator_gpu.u before applying copyto!
    copyto!(u_cpu, integrator_gpu.u)

    
    prob_cpu = remake(ode; u0 = u_cpu, tspan = (t_now, tspan[2]))
    int_cpu  = init(prob_cpu, CarpenterKennedy2N54(williamson_condition = false);
                    dt = dt_now, ode_default_options()..., callback = callbacks_no_amr)

    n_before = Trixi.nelements(int_cpu.p.solver, int_cpu.p.cache)
    amr_callback.affect!(int_cpu)   
    n_after  = Trixi.nelements(int_cpu.p.solver, int_cpu.p.cache)

    #replacing the before amr by after amr
    u_cpu    = Array(int_cpu.u)
    dt_now   = int_cpu.dt

    delta = n_after - n_before
    #JUST CHECKING AFTER MESH SIZE
    println("  [AMR] : $n_before → $n_after ($delta)")
    

end

#added this just for analysis purpose

summary_callback()
println("  Simulation complete.")
println("  Final t        : $t_now")
println("  Total steps    : $step_counter")
println("  AMR bursts     : $burst_counter")


