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


@inline function prolong2mortars_per_mortar!(mortars_u, u, mortar,
                                             MeshT::Type{<:Union{Trixi.P4estMesh{2},
                                                                 Trixi.P4estMeshView{2},
                                                                 Trixi.T8codeMesh{2}}},
                                             equations,
                                             neighbor_ids, node_indices,
                                             index_range,
                                             forward_lower::SMatrix{N, N, T, NN},
                                             forward_upper::SMatrix{N, N, T, NN},
                                             ::Val{NVARS}) where {N, T, NN, NVARS}

    @inbounds begin
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
        # val_lower = MMatrix{NVARS, N, T, NVARS * N}(undef)
        # val_upper = MMatrix{NVARS, N, T, NVARS * N}(undef)

        # Trixi.multiply_dimensionwise!(val_lower, forward_lower, u_buffer)
        # Trixi.multiply_dimensionwise!(val_upper, forward_upper, u_buffer)

        
        # for i in 1:N
        #     for v in Base.OneTo(NVARS)
        #         mortars_u[2, v, 1, i, mortar] = val_lower[v, i]
        #         mortars_u[2, v, 2, i, mortar] = val_upper[v, i]
        #     end
        # end

        # in place of multiply dimensionwise(because used of MMatrix and SMatrix)
        for i in 1:N
            for v in Base.OneTo(NVARS)
                val_lower = zero(T)
                val_upper = zero(T)
                for j in 1:N
                    val_lower = val_lower + forward_lower[i, j] * u_buffer[v, j]
                    val_upper = val_upper + forward_upper[i, j] * u_buffer[v, j]
                end
                mortars_u[2, v, 1, i, mortar] = val_lower
                mortars_u[2, v, 2, i, mortar] = val_upper
            end
        end

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
                                            forward_lower::SMatrix,
                                            forward_upper::SMatrix,
                                            ::Val{NVARS}) where {NVARS}
    mortar = @index(Global)
    prolong2mortars_per_mortar!(mortars_u, u, mortar, MeshT, equations,
                                 neighbor_ids, node_indices, index_range,
                                 forward_lower, forward_upper, Val(NVARS))
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

    # Convert Matrix → SMatrix 
    forward_lower = SMatrix{N, N, T, N * N}(Array(mortar_l2.forward_lower))
    forward_upper = SMatrix{N, N, T, N * N}(Array(mortar_l2.forward_upper))

    kernel! = prolong2mortars_KAkernel!(backend)
    kernel!(mortars.u, u, typeof(mesh), equations,
            neighbor_ids, node_indices, index_range,
            forward_lower, forward_upper, 
            Val(NVARS); 
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



#added this to make prolong2mortar and mortar flux working(on CPU)
#make here false (for scalar indexing not allowed error)
CUDA.allowscalar(true)
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
