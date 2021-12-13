using OrdinaryDiffEq
using Trixi

# define new structs inside a module to allow re-evaluating the file
module TrixiExtensionFilter

using Trixi
using OrdinaryDiffEq
using SparseArrays
using IterativeSolvers
using IncompleteLU

# This is an example implementation for a simple stage callback (i.e., a callable
# that is executed after each Runge-Kutta *stage*), which uses the differential
# filter introduced by Germano to filter the solution in every stage
struct FilterStageCallback
    N::Int64
    N_Q::Int64
    LHS::SparseMatrixCSC{Float64,Int64}
    LHS_precondition::IncompleteLU.ILUFactorization{Float64,Int64}
    RHS::SparseMatrixCSC{Float64,Int64}

    # You can optionally define an inner constructor like the one below to set up
    # some required stuff. You can also create outer constructors (not demonstrated
    # here) for further customization options.
    function FilterStageCallback(
        N::Int64,
        N_Q::Int64,
        LHS::SparseMatrixCSC{Float64,Int64},
        LHS_precondition::IncompleteLU.ILUFactorization{Float64,Int64},
        RHS::SparseMatrixCSC{Float64,Int64})

        new(N, N_Q, LHS, LHS_precondition, RHS)
    end
end

# This method is called when the `FilterStageCallback` is used as `stage_limiter!`
# which gets called after every RK stage. There is no specific initialization
# method for such `stage_limiter!`s in OrdinaryDiffEq.jl.
function (filter_stage_callback::FilterStageCallback)(u_ode, f, semi, t)

    u = Trixi.wrap_array(u_ode, semi)

    stage_callback_filter(
        u,
        filter_stage_callback.N,
        filter_stage_callback.N_Q,
        filter_stage_callback.LHS,
        filter_stage_callback.LHS_precondition,
        filter_stage_callback.RHS,
        Trixi.mesh_equations_solver_cache(semi)...)

    return nothing
end

# takes the solution u and applies the differential filter
function stage_callback_filter(
    u,
    N::Int64,
    N_Q::Int64,
    LHS::SparseMatrixCSC{Float64,Int64},
    LHS_precondition::IncompleteLU.ILUFactorization{Float64,Int64},
    RHS::SparseMatrixCSC{Float64,Int64},
    mesh::TreeMesh{2},
    equations,
    dg::DGSEM,
    cache)


    # split u into 4 vectors (the matrix rows) to seperate the coordinates
    # U[k,:] describes the solution at all points for coordinate k
    U = reshape(u, 4, Int(length(u) / 4))
    U = convert(Array{Float64,2}, U') # transpose for matrix multiplication

    # apply the filter by solving the following system of equations.
    # the system gets solved for every stage, which takes a lot of time.
    # alternatively you can solve LHS\RHS = FilterMatrix beforehand and just
    # calculate FilterMatrix * U in every stage, but that method takes far longer.
    # LHS is already factorized for better runtime, RHS is sparse
    rhs = RHS * U
    U_filter = zero(U)
    for i = 1:4
        U_filter[:, i] = gmres(LHS, rhs[:, i], Pl = LHS_precondition)
    end

    # the 4 coordinates of the filtered solution, which were seperated, are now
    # being merged into one matrix again
    U_filter = convert(Array{Float64,2}, U_filter')
    u_filter = reshape(U_filter, 4, N + 1, N + 1, N_Q^2)

    # overwrite the value of u with the filtered value of u_filter for each
    # node in each element
    for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            u_node =
                Trixi.get_node_vars(u_filter, equations, dg, i, j, element)
            Trixi.set_node_vars!(u, u_node, equations, dg, i, j, element)
        end
    end

    return nothing
end



# This is an example implementation for a simple step callback (i.e., a callable
# that is potentially executed after each Runge-Kutta *step*), which records
# some values each time it is called. Its sole purpose here is to showcase
# how to implement a step callback for Trixi.
struct FilterStepCallback
    N::Int64
    N_Q::Int64
    LHS::SparseMatrixCSC{Float64,Int64}
    LHS_precondition::IncompleteLU.ILUFactorization{Float64,Int64}
    RHS::SparseMatrixCSC{Float64,Int64}
    filt::Bool
    filt_para::Array{Float64,1}

    # You can optionally define an inner constructor like the one below to set up
    # some required stuff. You can also create outer constructors (not demonstrated
    # here) for further customization options.
    function FilterStepCallback(
        N::Int64,
        N_Q::Int64,
        LHS::SparseMatrixCSC{Float64,Int64},
        LHS_precondition::IncompleteLU.ILUFactorization{Float64,Int64},
        RHS::SparseMatrixCSC{Float64,Int64},
        filt::Bool,
        filt_para::Array{Float64,1})

        new(N, N_Q, LHS, LHS_precondition, RHS, filt, filt_para)
    end
end

# This method is called when the `ExampleStepCallback` is used as callback
# which gets called after RK steps.
function (filter_step_callback::FilterStepCallback)(integrator)
    u_ode = integrator.u
    #t = integrator.t
    semi = integrator.p

    u = Trixi.wrap_array(u_ode, semi)

    step_callback_filter(
        u,
        filter_step_callback.N,
        filter_step_callback.N_Q,
        filter_step_callback.LHS,
        filter_step_callback.LHS_precondition,
        filter_step_callback.RHS,
        Trixi.mesh_equations_solver_cache(semi)...)

    # avoid re-evaluating possible FSAL stages
    u_modified!(integrator, false)

    return nothing
end

function step_callback_filter(
    u,
    N::Int64,
    N_Q::Int64,
    LHS::SparseMatrixCSC{Float64,Int64},
    LHS_precondition::IncompleteLU.ILUFactorization{Float64,Int64},
    RHS::SparseMatrixCSC{Float64,Int64},
    mesh::TreeMesh{2},
    equations,
    dg::DGSEM,
    cache)


    # split u into 4 vectors (the matrix rows) to seperate the coordinates
    # U[k,:] describes the solution at all points for coordinate k
    U = reshape(u, 4, Int(length(u) / 4))
    U = convert(Array{Float64,2}, U') # transpose for matrix multiplication

    # apply the filter by solving the following system of equations.
    # the system gets solved for every step, which takes a lot of time.
    # alternatively you can solve LHS\RHS = FilterMatrix beforehand and just
    # calculate FilterMatrix * U in every step, but that method takes far longer.
    # LHS is already factorized for better runtime, RHS is sparse
    rhs = RHS * U
    U_filter = zero(U)
    for i = 1:4
        U_filter[:, i] = gmres(LHS, rhs[:, i], Pl = LHS_precondition)
    end

    # the 4 coordinates of the filtered solution, which were seperated, are now
    # being merged into one matrix again
    U_filter = convert(Array{Float64,2}, U_filter')
    u_filter = reshape(U_filter, 4, N + 1, N + 1, N_Q^2)

    # overwrite the value of u with the filtered value of u_filter for each
    # node in each element
    for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            u_node = Trixi.get_node_vars(u_filter, equations, dg, i, j, element)
            Trixi.set_node_vars!(u, u_node, equations, dg, i, j, element)
        end
    end

    return nothing
end


# This method is used to wrap an `FilterStepCallback` inside a `DiscreteCallback`
# which gets called after every RK step. You can pass an additional initialization
# method and a separate condition specifying whether the callback shall be called.
function FilterStepCallback(;
    N::Int64,
    N_Q::Int64,
    LHS::SparseMatrixCSC{Float64,Int64},
    LHS_precondition::IncompleteLU.ILUFactorization{Float64,Int64},
    RHS::SparseMatrixCSC{Float64,Int64},
    filt::Bool,
    filt_para::Array{Float64,1})


    # Call the `FilterStepCallback` after every RK step.
    condition = (u_ode, t, integrator) -> true

    filter_step_callback =
        FilterStepCallback(N, N_Q, LHS, LHS_precondition, RHS, filt, filt_para)

    DiscreteCallback(condition,filter_step_callback,save_positions = (false, false))
end

# prin filter parameters
function Base.show(io::IO,::MIME"text/plain",cb::DiscreteCallback{<:Any,<:FilterStepCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        filter_step_callback = cb.affect!

        setup = [
            "Germano" => filter_step_callback.filt,
            "α²" => filter_step_callback.filt_para[1],
            "β²" => filter_step_callback.filt_para[2],
            "δ²" => filter_step_callback.filt_para[3],
        ]
        Trixi.summary_box(io, "FilterStepCallback", setup)
    end
end

end # module TrixiExtensionFilter

import .TrixiExtensionFilter


###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)

"""
    initial_condition_kelvin_helmholtz_instability(x, t, equations::CompressibleEulerEquations2D)
A version of the classical Kelvin-Helmholtz instability based on
- Andrés M. Rueda-Ramírez, Gregor J. Gassner (2021)
  A Subcell Finite Volume Positivity-Preserving Limiter for DGSEM Discretizations
  of the Euler Equations
  [arXiv: 2102.06017](https://arxiv.org/abs/2102.06017)
"""
function initial_condition_kelvin_helmholtz_instability(x, t, equations::CompressibleEulerEquations2D)
  # change discontinuity to tanh
  # typical resolution 128^2, 256^2
  # domain size is [-1,+1]^2
  slope = 15
  amplitude = 0.02
  B = tanh(slope * x[2] + 7.5) - tanh(slope * x[2] - 7.5)
  rho = 0.5 + 0.75 * B
  v1 = 0.5 * (B - 1)
  v2 = 0.1 * sin(2 * pi * x[1])
  p = 1.0
  return Trixi.prim2cons(SVector(rho, v1, v2, p), equations)
end

initial_condition = initial_condition_kelvin_helmholtz_instability

polydeg = 3
basis = LobattoLegendreBasis(polydeg)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.002,
                                         alpha_min=0.0001,
                                         alpha_smooth=true,
                                         variable=density_pressure)
surface_flux = flux_lax_friedrichs

# filtering should replace the shock-capturing
if filt_type == "none"
    volume_flux  = flux_chandrashekar
    volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                     volume_flux_dg=volume_flux,
                                                     volume_flux_fv=surface_flux)
else
    volume_flux  = flux_ranocha
    volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
end

solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-1.0, -1.0)
coordinates_max = ( 1.0,  1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=5,
                n_cells_max=100_000)
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, tEnd)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=20,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=1.3)

N = polydeg
# N_Q elements in each direction, all in all we have N_Q² elements
N_Q = Int(sqrt(size(semi.cache.elements.node_coordinates,4)))

if !post
    ElementMatrix,LHS,LHS_precondition,RHS,w_bary = prepareFilter(N,N_Q,
        coordinates_min[1],coordinates_max[1],filt,filt_para,solver)
end

if filt_type == "step"
    filter_step_callback = TrixiExtensionFilter.FilterStepCallback(N=N,N_Q=N_Q,
                                                                LHS=LHS,
                                                                LHS_precondition=LHS_precondition,
                                                                RHS=RHS,filt=filt,
                                                                filt_para=filt_para)
    callbacks = CallbackSet(summary_callback,
                            analysis_callback, alive_callback,
                            save_solution,
                            filter_step_callback,
                            stepsize_callback)
else
    callbacks = CallbackSet(summary_callback,
                            analysis_callback, alive_callback,
                            save_solution,
                            stepsize_callback)
end

if filt_type == "stage"
    # In OrdinaryDiffEq.jl, the `step_limiter!` is called after every Runge-Kutta step
    # but before possible RHS evaluations of the new value occur. Hence, it's possible
    # to modify the new solution value there without impacting the performance of FSAL
    # methods.
    # The `stage_limiter!` is called additionally after computing a Runge-Kutta stage
    # value but before evaluating the corresponding stage derivatives.
    # Hence, if a limiter should be called before each RHS evaluation, it needs to be
    # set both as `stage_limiter!` and as `step_limiter!`.
    filter_stage_callback! = TrixiExtensionFilter.FilterStageCallback(N,N_Q,LHS,LHS_precondition,RHS)
    stage_limiter! = filter_stage_callback!
    step_limiter! = filter_stage_callback!

    ############################################################################
    # run the simulation
    sol = solve(ode, CarpenterKennedy2N54(stage_limiter!, step_limiter!, williamson_condition=false),
                dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                save_everystep=false, callback=callbacks);

else

    ############################################################################
    # run the simulation
    sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
                dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                save_everystep=false, callback=callbacks);
end

summary_callback() # print the timer summary

if !post
    # converts the solution to primitive coordinates and plots them
    plot_sol(N,N_Q,coordinates_min[1],coordinates_max[1],
        convert(Array{Float64,1}, solver.basis.nodes),w_bary,
        ElementMatrix,sol[end])
end
