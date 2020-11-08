
#=
The code contained in this file is inspired by an analysis performed
using SnoopCompile, although most of it is written manually.

This kind of analysis was performed using the following code.
```julia
using SnoopCompile
inf_timing = @snoopi tmin=0.01 begin
  # below is mostly a copy of `examples/2d/elixir_advection_amr.jl`
  using Trixi
  using OrdinaryDiffEq

  ###############################################################################
  # semidiscretization of the linear advection equation

  advectionvelocity = (1.0, 1.0)
  equations = LinearScalarAdvectionEquation2D(advectionvelocity)

  initial_condition = initial_condition_gauss

  surface_flux = flux_lax_friedrichs
  basis = LobattoLegendreBasis(3)
  solver = DGSEM(basis, surface_flux)

  coordinates_min = (-5, -5)
  coordinates_max = ( 5,  5)
  mesh = TreeMesh(coordinates_min, coordinates_max,
                  initial_refinement_level=4,
                  n_cells_max=30_000)


  semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


  ###############################################################################
  # ODE solvers, callbacks etc.

  tspan = (0.0, 10.0)
  ode = semidiscretize(semi, tspan)

  summary_callback = SummaryCallback()

  amr_controller = ControllerThreeLevel(semi, IndicatorMax(semi, variable=first),
                                        base_level=4,
                                        med_level=5, med_threshold=0.1,
                                        max_level=6, max_threshold=0.6)
  amr_callback = AMRCallback(semi, amr_controller,
                            interval=5,
                            adapt_initial_condition=true,
                            adapt_initial_condition_only_refine=true)

  stepsize_callback = StepsizeCallback(cfl=1.6)

  save_solution = SaveSolutionCallback(interval=100,
                                      save_initial_solution=true,
                                      save_final_solution=true,
                                      solution_variables=:primitive)

  save_restart = SaveRestartCallback(interval=100,
                                    save_final_restart=true)

  analysis_interval = 100
  alive_callback = AliveCallback(analysis_interval=analysis_interval)
  analysis_callback = AnalysisCallback(equations, solver,
                                      interval=analysis_interval,
                                      extra_analysis_integrals=(entropy,))

  # TODO: Taal decide, first AMR or save solution etc.
  callbacks = CallbackSet(summary_callback, amr_callback, stepsize_callback,
                          save_restart, save_solution,
                          analysis_callback, alive_callback);


  ###############################################################################
  # run the simulation

  u_ode = copy(ode.u0)
  du_ode = similar(u_ode)
  Trixi.rhs!(du_ode, u_ode, semi, first(tspan))

  # You could also include a `solve` stage to generate possibly more precompile statements.
  # sol = Trixi.solve(ode, Trixi.SimpleAlgorithm2N45(),
  #                   dt=stepsize_callback(ode), # solve needs some value here but it will be overwritten by the stepsize_callback
  #                   save_everystep=false, callback=callbacks);
  summary_callback() # print the timer summary
end
pc = SnoopCompile.parcel(inf_timing)
SnoopCompile.write("dev/precompile", pc)

```
After running the code above, SnoopCompile has generated the file
`dev/precompile/precompile_Trixi.jl`, which contains precompile statements
in the function `_precompile_`.
More information on this process can be found in the docs of SnoopCompile,
in particular at https://timholy.github.io/SnoopCompile.jl/stable/snoopi/.

This kind of analysis helps finding type-unstable parts of the code, e.g.
`init_interfaces` etc. in https://github.com/trixi-framework/Trixi.jl/pull/307.
Moreover, it allows to generate precompile statements which reduce the latency
by caching type inference results.
=#


import StaticArrays


# manually generated precompile statements
function _precompile_manual_()
  ccall(:jl_generating_output, Cint, ()) == 1 || return nothing

  function equations_types_1d(RealT)
    ( LinearScalarAdvectionEquation1D{RealT},
      CompressibleEulerEquations1D{RealT},
    )
  end
  function equations_types_2d(RealT)
    ( LinearScalarAdvectionEquation2D{RealT},
      HyperbolicDiffusionEquations2D{RealT},
      CompressibleEulerEquations2D{RealT},
      IdealGlmMhdEquations2D{RealT},
    )
  end
  function equations_types_3d(RealT)
    ( LinearScalarAdvectionEquation3D{RealT},
      HyperbolicDiffusionEquations3D{RealT},
      CompressibleEulerEquations3D{RealT},
      IdealGlmMhdEquations3D{RealT},
    )
  end
  function equations_types(RealT)
    ( LinearScalarAdvectionEquation1D{RealT},
      LinearScalarAdvectionEquation2D{RealT},
      LinearScalarAdvectionEquation3D{RealT},
      HyperbolicDiffusionEquations2D{RealT},
      HyperbolicDiffusionEquations3D{RealT},
      CompressibleEulerEquations1D{RealT},
      CompressibleEulerEquations2D{RealT},
      CompressibleEulerEquations3D{RealT},
      IdealGlmMhdEquations2D{RealT},
      IdealGlmMhdEquations3D{RealT},
    )
  end

  # Constructors: mesh
  for RealT in (Int, Float64,)
    @assert Base.precompile(Tuple{Core.kwftype(typeof(Trixi.Type)),NamedTuple{(:initial_refinement_level, :n_cells_max),Tuple{Int,Int}},Type{TreeMesh},RealT,RealT})
    @assert Base.precompile(Tuple{Core.kwftype(typeof(Trixi.Type)),NamedTuple{(:initial_refinement_level, :n_cells_max),Tuple{Int,Int}},Type{TreeMesh},Tuple{RealT},Tuple{RealT}})
    @assert Base.precompile(Tuple{Core.kwftype(typeof(Trixi.Type)),NamedTuple{(:initial_refinement_level, :n_cells_max),Tuple{Int,Int}},Type{TreeMesh},Tuple{RealT,RealT},Tuple{RealT,RealT}})
    @assert Base.precompile(Tuple{Core.kwftype(typeof(Trixi.Type)),NamedTuple{(:initial_refinement_level, :n_cells_max),Tuple{Int,Int}},Type{TreeMesh},Tuple{RealT,RealT,RealT},Tuple{RealT,RealT,RealT}})
  end

  # Constructors: linear advection
  for RealT in (Float64,)
    @assert Base.precompile(Tuple{Type{LinearScalarAdvectionEquation1D},RealT})
    @assert Base.precompile(Tuple{Type{LinearScalarAdvectionEquation2D},RealT,RealT})
    @assert Base.precompile(Tuple{Type{LinearScalarAdvectionEquation2D},Tuple{RealT,RealT}})
    @assert Base.precompile(Tuple{Type{LinearScalarAdvectionEquation3D},RealT,RealT,RealT})
    @assert Base.precompile(Tuple{Type{LinearScalarAdvectionEquation3D},Tuple{RealT,RealT,RealT}})
  end

  # Constructors: hyperbolic diffusion
  for RealT in (Float64,)
    @assert Base.precompile(Tuple{Type{HyperbolicDiffusionEquations2D},RealT})
    @assert Base.precompile(Tuple{Type{HyperbolicDiffusionEquations3D},RealT})
  end

  # Constructors: Euler
  for RealT in (Float64,)
    @assert Base.precompile(Tuple{Type{CompressibleEulerEquations1D},RealT})
    @assert Base.precompile(Tuple{Type{CompressibleEulerEquations2D},RealT})
    @assert Base.precompile(Tuple{Type{CompressibleEulerEquations3D},RealT})
  end

  # Constructors: MHD
  for RealT in (Float64,)
    @assert Base.precompile(Tuple{Type{IdealGlmMhdEquations2D},RealT})
    @assert Base.precompile(Tuple{Type{IdealGlmMhdEquations3D},RealT})
  end

  # Constructors of the basis are inherently type-unstable since we pass integers
  # and use their values as parameters of static arrays.
  # Nevertheless, we can still precompile methods used to construct the bases.
  Base.precompile(Tuple{Type{LobattoLegendreBasis},Int})
  for RealT in (Float64,)
    Base.precompile(Tuple{Type{LobattoLegendreBasis},RealT,Int})
    @assert Base.precompile(Tuple{typeof(Trixi.calc_dhat),Vector{RealT},Vector{RealT}})
    @assert Base.precompile(Tuple{typeof(Trixi.calc_dsplit),Vector{RealT},Vector{RealT}})
    @assert Base.precompile(Tuple{typeof(Trixi.polynomial_derivative_matrix),Vector{RealT}})
    @assert Base.precompile(Tuple{typeof(Trixi.polynomial_interpolation_matrix),Vector{RealT},Vector{RealT}})
    @assert Base.precompile(Tuple{typeof(Trixi.barycentric_weights),Vector{RealT}})
    @assert Base.precompile(Tuple{typeof(Trixi.calc_lhat),RealT,Vector{RealT},Vector{RealT}})
    @assert Base.precompile(Tuple{typeof(Trixi.lagrange_interpolating_polynomials),RealT,Vector{RealT},Vector{RealT}})
    @assert Base.precompile(Tuple{typeof(Trixi.calc_q_and_l),Int,RealT})
    @assert Base.precompile(Tuple{typeof(Trixi.legendre_polynomial_and_derivative),Int,RealT})
    @assert Base.precompile(Tuple{typeof(Trixi.vandermonde_legendre),Vector{RealT}})
  end
  @assert Base.precompile(Tuple{typeof(Trixi.gauss_lobatto_nodes_weights),Int})
  @assert Base.precompile(Tuple{typeof(Trixi.gauss_nodes_weights),Int})
  @assert Base.precompile(Tuple{typeof(Trixi.calc_forward_upper),Int})
  @assert Base.precompile(Tuple{typeof(Trixi.calc_forward_lower),Int})
  @assert Base.precompile(Tuple{typeof(Trixi.calc_reverse_upper),Int,Val{:gauss}})
  @assert Base.precompile(Tuple{typeof(Trixi.calc_reverse_lower),Int,Val{:gauss}})
  @assert Base.precompile(Tuple{typeof(Trixi.calc_reverse_upper),Int,Val{:gauss_lobatto}})
  @assert Base.precompile(Tuple{typeof(Trixi.calc_reverse_lower),Int,Val{:gauss_lobatto}})

  # Constructors: mortars, analyzers, adaptors
  for RealT in (Float64,), polydeg in 1:7
    nnodes_ = polydeg + 1
    basis_type = LobattoLegendreBasis{RealT,nnodes_,Array{RealT,2},StaticArrays.SArray{Tuple{nnodes_,2},RealT,2,2*nnodes_},StaticArrays.SArray{Tuple{nnodes_,nnodes_},RealT,2,nnodes_^2}}
    @assert Base.precompile(Tuple{typeof(Trixi.MortarL2),basis_type})
    @assert Base.precompile(Tuple{Type{Trixi.SolutionAnalyzer},basis_type})
    @assert Base.precompile(Tuple{Type{Trixi.AdaptorL2},basis_type})
  end

  # Constructors: callbacks
  @assert Base.precompile(Tuple{Core.kwftype(typeof(Trixi.Type)),NamedTuple{(:analysis_interval,),Tuple{Int}},Type{AliveCallback}})
  for RealT in (Float64,)
    @assert Base.precompile(Tuple{Core.kwftype(typeof(Trixi.Type)),NamedTuple{(:cfl,),Tuple{RealT}},Type{StepsizeCallback}})
  end
  @assert Base.precompile(Tuple{Core.kwftype(typeof(Trixi.Type)),NamedTuple{(:interval, :save_final_restart),Tuple{Int,Bool}},Type{SaveRestartCallback}})
  @assert Base.precompile(Tuple{Core.kwftype(typeof(Trixi.Type)),NamedTuple{(:interval, :save_initial_solution, :save_final_solution, :solution_variables),Tuple{Int,Bool,Bool,Symbol}},Type{SaveSolutionCallback}})
  for RealT in (Float64,), polydeg in 1:7
    nnodes_ = polydeg + 1
    nnodes_analysis = 2*polydeg + 1
    @assert Base.precompile(Tuple{Type{AnalysisCallback},RealT,Int,Bool,String,String,Trixi.LobattoLegendreAnalyzer{RealT,nnodes_analysis,Array{RealT,2}},Array{Symbol,1},Tuple{typeof(Trixi.entropy_timederivative),typeof(entropy)},StaticArrays.SArray{Tuple{1},RealT,1,1}})
    # We would need to use all special cases instead of
    # Function,Trixi.AbstractVolumeIntegral
    # for equations_type in equations_types(RealT)
    #   @assert Base.precompile(Tuple{Core.kwftype(typeof(Trixi.Type)),NamedTuple{(:interval, :extra_analysis_integrals),Tuple{Int,Tuple{typeof(entropy)}}},Type{AnalysisCallback},equations_type,DG{RealT,LobattoLegendreBasis{RealT,nnodes_,Array{RealT,2},StaticArrays.SArray{Tuple{4,2},RealT,2,2*nnodes_},StaticArrays.SArray{Tuple{nnodes_,nnodes_},RealT,2,nnodes_^2}},Trixi.LobattoLegendreMortarL2{RealT,nnodes_,StaticArrays.SArray{Tuple{nnodes_,nnodes_},RealT,2,nnodes_^2}},Function,Trixi.AbstractVolumeIntegral}})
    # end
  end
  Base.precompile(Tuple{typeof(SummaryCallback)})
  # TODO: AMRCallback, ControllerThreeLevel, indicators

  # init_elements, interfaces, etc.
  for RealT in (Float64,), polydeg in 1:7
    nnodes_ = polydeg + 1

    # 2D, serial
    Base.precompile(Tuple{typeof(Trixi.init_boundaries),Array{Int,1},TreeMesh{1,Trixi.SerialTree{1}},Trixi.ElementContainer1D{RealT,1,polydeg}})
    Base.precompile(Tuple{typeof(Trixi.init_interfaces),Array{Int,1},TreeMesh{1,Trixi.SerialTree{1}},Trixi.ElementContainer1D{RealT,1,polydeg}})

    # 2D, serial
    Base.precompile(Tuple{typeof(Trixi.init_boundaries),Array{Int,1},TreeMesh{2,Trixi.SerialTree{2}},Trixi.ElementContainer2D{RealT,1,polydeg}})
    Base.precompile(Tuple{typeof(Trixi.init_interfaces),Array{Int,1},TreeMesh{2,Trixi.SerialTree{2}},Trixi.ElementContainer2D{RealT,1,polydeg}})
    Base.precompile(Tuple{typeof(Trixi.init_mortars),Array{Int,1},TreeMesh{2,Trixi.SerialTree{2}},Trixi.ElementContainer2D{RealT,1,polydeg},Trixi.LobattoLegendreMortarL2{RealT,nnodes_,StaticArrays.SArray{Tuple{nnodes_,nnodes_},RealT,2,nnodes_^2}}})

    # TODO: 2D, parallel

    # 3D, serial
    Base.precompile(Tuple{typeof(Trixi.init_boundaries),Array{Int,1},TreeMesh{3,Trixi.SerialTree{3}},Trixi.ElementContainer3D{RealT,1,polydeg}})
    Base.precompile(Tuple{typeof(Trixi.init_interfaces),Array{Int,1},TreeMesh{3,Trixi.SerialTree{3}},Trixi.ElementContainer3D{RealT,1,polydeg}})
    Base.precompile(Tuple{typeof(Trixi.init_mortars),Array{Int,1},TreeMesh{3,Trixi.SerialTree{3}},Trixi.ElementContainer3D{RealT,1,polydeg},Trixi.LobattoLegendreMortarL2{RealT,nnodes_,StaticArrays.SArray{Tuple{nnodes_,nnodes_},RealT,2,nnodes_^2}}})
  end

  return nothing
end

