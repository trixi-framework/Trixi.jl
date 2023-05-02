# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


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

  advection_velocity = (1.0, 1.0)
  equations = LinearScalarAdvectionEquation2D(advection_velocity)
  show(stdout, equations)
  show(stdout, MIME"text/plain"(), equations)

  initial_condition = initial_condition_gauss

  surface_flux = flux_lax_friedrichs
  basis = LobattoLegendreBasis(3)
  solver = DGSEM(basis, surface_flux)
  show(stdout, solver)
  show(stdout, MIME"text/plain"(), solver)

  coordinates_min = (-5, -5)
  coordinates_max = ( 5,  5)
  mesh = TreeMesh(coordinates_min, coordinates_max,
                  initial_refinement_level=4,
                  n_cells_max=30_000)
  show(stdout, mesh)
  show(stdout, MIME"text/plain"(), mesh)


  semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)
  show(stdout, semi)
  show(stdout, MIME"text/plain"(), semi)


  ###############################################################################
  # ODE solvers, callbacks etc.

  tspan = (0.0, 10.0)
  ode = semidiscretize(semi, tspan)

  summary_callback = SummaryCallback()
  show(stdout, summary_callback)
  show(stdout, MIME"text/plain"(), summary_callback)

  amr_controller = ControllerThreeLevel(semi, IndicatorMax(semi, variable=first),
                                        base_level=4,
                                        med_level=5, med_threshold=0.1,
                                        max_level=6, max_threshold=0.6)
  amr_callback = AMRCallback(semi, amr_controller,
                            interval=5,
                            adapt_initial_condition=true,
                            adapt_initial_condition_only_refine=true)
  show(stdout, amr_callback)
  show(stdout, MIME"text/plain"(), amr_callback)

  stepsize_callback = StepsizeCallback(cfl=1.6)
  show(stdout, stepsize_callback)
  show(stdout, MIME"text/plain"(), stepsize_callback)

  save_solution = SaveSolutionCallback(interval=100,
                                      save_initial_solution=true,
                                      save_final_solution=true,
                                      solution_variables=cons2prim)
  show(stdout, save_solution)
  show(stdout, MIME"text/plain"(), save_solution)

  save_restart = SaveRestartCallback(interval=100,
                                    save_final_restart=true)
  show(stdout, save_restart)
  show(stdout, MIME"text/plain"(), save_restart)

  analysis_interval = 100
  alive_callback = AliveCallback(analysis_interval=analysis_interval)
  show(stdout, alive_callback)
  show(stdout, MIME"text/plain"(), alive_callback)
  analysis_callback = AnalysisCallback(equations, solver,
                                      interval=analysis_interval,
                                      extra_analysis_integrals=(entropy,))
  show(stdout, analysis_callback)
  show(stdout, MIME"text/plain"(), analysis_callback)

  callbacks = CallbackSet(summary_callback,
                          save_restart, save_solution,
                          analysis_callback, alive_callback,
                          amr_callback, stepsize_callback);


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
The latency can be measured by running
```bash
julia --threads=1 -e '@time using Trixi; @time include(joinpath(examples_dir(), "2d", "elixir_advection_basic.jl"))'
```


We add `@assert` to the precompile statements below to make sure that we don't include
failing precompile statements, cf. https://timholy.github.io/SnoopCompile.jl/stable/snoopi/.
If any assertions below fail, it is generally safe to just disable the failing call
to precompile (or to not include precompile.jl at all).
To still get the same latency reductions, you should consider to adapt the failing precompile
statements in accordance with the changes in Trixi.jl's source code. Please, feel free to ping
the core developers of Trixi.jl to get help with that.
=#


import StaticArrays
import SciMLBase


# manually generated precompile statements
function _precompile_manual_()
  ccall(:jl_generating_output, Cint, ()) == 1 || return nothing

  function equations_types_1d(RealT)
    ( LinearScalarAdvectionEquation1D{RealT},
      HyperbolicDiffusionEquation1D{RealT},
      CompressibleEulerEquations1D{RealT},
      IdealGlmMhdEquations1D{RealT},
    )
  end
  function equations_types_2d(RealT)
    ( LinearScalarAdvectionEquation2D{RealT},
      HyperbolicDiffusionEquations2D{RealT},
      CompressibleEulerEquations2D{RealT},
      IdealGlmMhdEquations2D{RealT},
      LatticeBoltzmannEquations2D{RealT, typeof(Trixi.collision_bgk)},
    )
  end
  function equations_types_3d(RealT)
    ( LinearScalarAdvectionEquation3D{RealT},
      HyperbolicDiffusionEquations3D{RealT},
      CompressibleEulerEquations3D{RealT},
      IdealGlmMhdEquations3D{RealT},
      LatticeBoltzmannEquations3D{RealT, typeof(Trixi.collision_bgk)},
    )
  end
  function equations_types(RealT)
    ( LinearScalarAdvectionEquation1D{RealT},
      LinearScalarAdvectionEquation2D{RealT},
      LinearScalarAdvectionEquation3D{RealT},
      HyperbolicDiffusionEquations1D{RealT},
      HyperbolicDiffusionEquations2D{RealT},
      HyperbolicDiffusionEquations3D{RealT},
      CompressibleEulerEquations1D{RealT},
      CompressibleEulerEquations2D{RealT},
      CompressibleEulerEquations3D{RealT},
      IdealGlmMhdEquations1D{RealT},
      IdealGlmMhdEquations2D{RealT},
      IdealGlmMhdEquations3D{RealT},
      LatticeBoltzmannEquations2D{RealT, typeof(Trixi.collision_bgk)},
      LatticeBoltzmannEquations3D{RealT, typeof(Trixi.collision_bgk)},
    )
  end

  function basis_type_dgsem(RealT, nnodes_)
    LobattoLegendreBasis{RealT,nnodes_,
                         # VectorT
                         StaticArrays.SVector{nnodes_,RealT},
                         # InverseVandermondeLegendre
                         Matrix{RealT},
                         # BoundaryMatrix
                         #StaticArrays.SArray{Tuple{nnodes_,2},RealT,2,2*nnodes_},
                         Matrix{RealT},
                         # DerivativeMatrix
                         #StaticArrays.SArray{Tuple{nnodes_,nnodes_},RealT,2,nnodes_^2},
                         Matrix{RealT},
    }
  end

  function mortar_type_dgsem(RealT, nnodes_)
    LobattoLegendreMortarL2{RealT,nnodes_,
                            # ForwardMatrix
                            #StaticArrays.SArray{Tuple{nnodes_,nnodes_},RealT,2,nnodes_^2},
                            Matrix{RealT},
                            # ReverseMatrix
                            # StaticArrays.SArray{Tuple{nnodes_,nnodes_},RealT,2,nnodes_^2},
                            Matrix{RealT},
    }
  end

  function analyzer_type_dgsem(RealT, nnodes_)
    polydeg = nnodes_ - 1
    nnodes_analysis = 2 * polydeg + 1
    LobattoLegendreAnalyzer{RealT,nnodes_analysis,
                            # VectorT
                            StaticArrays.SVector{nnodes_analysis,RealT},
                            # Vandermonde
                            Array{RealT,2}
    }
  end

  function adaptor_type_dgsem(RealT, nnodes_)
    LobattoLegendreAdaptorL2{RealT,nnodes_,
                            # ForwardMatrix
                            StaticArrays.SArray{Tuple{nnodes_,nnodes_},RealT,2,nnodes_^2},
                            # Matrix{RealT},
                            # ReverseMatrix
                            StaticArrays.SArray{Tuple{nnodes_,nnodes_},RealT,2,nnodes_^2},
                            # Matrix{RealT},
    }
  end

  # Constructors: mesh
  for RealT in (Int, Float64,)
    @assert Base.precompile(Tuple{Core.kwftype(typeof(Trixi.Type)),NamedTuple{(:initial_refinement_level, :n_cells_max),Tuple{Int,Int}},Type{TreeMesh},RealT,RealT})
    @assert Base.precompile(Tuple{Core.kwftype(typeof(Trixi.Type)),NamedTuple{(:initial_refinement_level, :n_cells_max),Tuple{Int,Int}},Type{TreeMesh},Tuple{RealT},Tuple{RealT}})
    @assert Base.precompile(Tuple{Core.kwftype(typeof(Trixi.Type)),NamedTuple{(:initial_refinement_level, :n_cells_max),Tuple{Int,Int}},Type{TreeMesh},Tuple{RealT,RealT},Tuple{RealT,RealT}})
    @assert Base.precompile(Tuple{Core.kwftype(typeof(Trixi.Type)),NamedTuple{(:initial_refinement_level, :n_cells_max),Tuple{Int,Int}},Type{TreeMesh},Tuple{RealT,RealT,RealT},Tuple{RealT,RealT,RealT}})
  end
  for TreeType in (SerialTree, ParallelTree), NDIMS in 1:3
    @assert Base.precompile(Tuple{typeof(Trixi.initialize!),TreeMesh{NDIMS,TreeType{NDIMS}},Int,Tuple{},Tuple{}})
    @assert Base.precompile(Tuple{typeof(Trixi.save_mesh_file),TreeMesh{NDIMS,TreeType{NDIMS}},String,Int})
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
    @assert Base.precompile(Tuple{Type{HyperbolicDiffusionEquations1D},})
    @assert Base.precompile(Tuple{Type{HyperbolicDiffusionEquations2D},})
    @assert Base.precompile(Tuple{Type{HyperbolicDiffusionEquations3D},})
  end

  # Constructors: Euler
  for RealT in (Float64,)
    @assert Base.precompile(Tuple{Type{CompressibleEulerEquations1D},RealT})
    @assert Base.precompile(Tuple{Type{CompressibleEulerEquations2D},RealT})
    @assert Base.precompile(Tuple{Type{CompressibleEulerEquations3D},RealT})
  end

  # Constructors: MHD
  for RealT in (Float64,)
    @assert Base.precompile(Tuple{Type{IdealGlmMhdEquations1D},RealT})
    @assert Base.precompile(Tuple{Type{IdealGlmMhdEquations2D},RealT})
    @assert Base.precompile(Tuple{Type{IdealGlmMhdEquations3D},RealT})
  end

  # Constructors: LBM
  for RealT in (Float64,)
    @assert Base.precompile(Tuple{Core.kwftype(typeof(Trixi.Type)),NamedTuple{(:Ma, :Re), Tuple{RealT, RealT}},Type{LatticeBoltzmannEquations2D}})
    @assert Base.precompile(Tuple{Core.kwftype(typeof(Trixi.Type)),NamedTuple{(:Ma, :Re), Tuple{RealT, Int}},Type{LatticeBoltzmannEquations2D}})
    @assert Base.precompile(Tuple{Core.kwftype(typeof(Trixi.Type)),NamedTuple{(:Ma, :Re), Tuple{RealT, RealT}},Type{LatticeBoltzmannEquations3D}})
    @assert Base.precompile(Tuple{Core.kwftype(typeof(Trixi.Type)),NamedTuple{(:Ma, :Re), Tuple{RealT, Int}},Type{LatticeBoltzmannEquations3D}})
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
    basis_type = basis_type_dgsem(RealT, nnodes_)
    @assert Base.precompile(Tuple{typeof(Trixi.MortarL2),basis_type})
    @assert Base.precompile(Tuple{Type{Trixi.SolutionAnalyzer},basis_type})
    @assert Base.precompile(Tuple{Type{Trixi.AdaptorL2},basis_type})
  end

  # Constructors: callbacks
  @assert Base.precompile(Tuple{Core.kwftype(typeof(Trixi.Type)),NamedTuple{(:analysis_interval,),Tuple{Int}},Type{AliveCallback}})
  for RealT in (Float64,)
    @assert Base.precompile(Tuple{Core.kwftype(typeof(Trixi.Type)),NamedTuple{(:cfl,),Tuple{RealT}},Type{StepsizeCallback}})
    @assert Base.precompile(Tuple{Core.kwftype(typeof(Trixi.Type)),NamedTuple{(:glm_scale, :cfl),Tuple{RealT,RealT}},Type{GlmSpeedCallback}})
  end
  @assert Base.precompile(Tuple{Core.kwftype(typeof(Trixi.Type)),NamedTuple{(:interval, :save_final_restart),Tuple{Int,Bool}},Type{SaveRestartCallback}})
  @assert Base.precompile(Tuple{Core.kwftype(typeof(Trixi.Type)),NamedTuple{(:interval, :save_initial_solution, :save_final_solution, :solution_variables),Tuple{Int,Bool,Bool,typeof(cons2cons)}},Type{SaveSolutionCallback}})
  @assert Base.precompile(Tuple{Core.kwftype(typeof(Trixi.Type)),NamedTuple{(:interval, :save_initial_solution, :save_final_solution, :solution_variables),Tuple{Int,Bool,Bool,typeof(cons2prim)}},Type{SaveSolutionCallback}})
  # TODO: AnalysisCallback?
  # for RealT in (Float64,), polydeg in 1:7
  #   nnodes_ = polydeg + 1
  #   nnodes_analysis = 2*polydeg + 1
    # @assert Base.precompile(Tuple{Type{AnalysisCallback},RealT,Int,Bool,String,String,Trixi.LobattoLegendreAnalyzer{RealT,nnodes_analysis,Array{RealT,2}},Array{Symbol,1},Tuple{typeof(Trixi.entropy_timederivative),typeof(entropy)},StaticArrays.SArray{Tuple{1},RealT,1,1}})
    # We would need to use all special cases instead of
    # Function,Trixi.AbstractVolumeIntegral
    # for equations_type in equations_types(RealT)
    #   @assert Base.precompile(Tuple{Core.kwftype(typeof(Trixi.Type)),NamedTuple{(:interval, :extra_analysis_integrals),Tuple{Int,Tuple{typeof(entropy)}}},Type{AnalysisCallback},equations_type,DG{RealT,LobattoLegendreBasis{RealT,nnodes_,StaticArrays.SVector{nnodes_,RealT},Array{RealT,2},StaticArrays.SArray{Tuple{4,2},RealT,2,2*nnodes_},StaticArrays.SArray{Tuple{nnodes_,nnodes_},RealT,2,nnodes_^2}},Trixi.LobattoLegendreMortarL2{RealT,nnodes_,StaticArrays.SArray{Tuple{nnodes_,nnodes_},RealT,2,nnodes_^2}},Function,Trixi.AbstractVolumeIntegral}})
    # end
  # end
  @assert Base.precompile(Tuple{typeof(SummaryCallback)})
  @assert Base.precompile(Tuple{DiscreteCallback{typeof(Trixi.summary_callback), typeof(Trixi.summary_callback), typeof(Trixi.initialize_summary_callback), typeof(SciMLBase.FINALIZE_DEFAULT)}})
  @assert Base.precompile(Tuple{typeof(summary_box),typeof(stdout),String,Vector{Pair{String, Any}}})
  # TODO: AMRCallback, ControllerThreeLevel, indicators

  # init_elements, interfaces, etc.
  for RealT in (Float64,), polydeg in 1:7
    uEltype = RealT
    nnodes_ = polydeg + 1
    mortar_type = mortar_type_dgsem(RealT, nnodes_)

    # 1D, serial
    @assert Base.precompile(Tuple{typeof(Trixi.init_boundaries),Array{Int,1},TreeMesh{1,Trixi.SerialTree{1}},Trixi.ElementContainer1D{RealT,uEltype}})
    @assert Base.precompile(Tuple{typeof(Trixi.init_interfaces),Array{Int,1},TreeMesh{1,Trixi.SerialTree{1}},Trixi.ElementContainer1D{RealT,uEltype}})

    # 2D, serial
    @assert Base.precompile(Tuple{typeof(Trixi.init_boundaries),Array{Int,1},TreeMesh{2,Trixi.SerialTree{2}},Trixi.ElementContainer2D{RealT,uEltype}})
    @assert Base.precompile(Tuple{typeof(Trixi.init_interfaces),Array{Int,1},TreeMesh{2,Trixi.SerialTree{2}},Trixi.ElementContainer2D{RealT,uEltype}})
    @assert Base.precompile(Tuple{typeof(Trixi.init_mortars),Array{Int,1},TreeMesh{2,Trixi.SerialTree{2}},Trixi.ElementContainer2D{RealT,uEltype},mortar_type})

    # 2D, parallel
    @assert Base.precompile(Tuple{typeof(Trixi.init_boundaries),Array{Int,1},TreeMesh{2,Trixi.ParallelTree{2}},Trixi.ElementContainer2D{RealT,uEltype}})
    @assert Base.precompile(Tuple{typeof(Trixi.init_interfaces),Array{Int,1},TreeMesh{2,Trixi.ParallelTree{2}},Trixi.ElementContainer2D{RealT,uEltype}})
    @assert Base.precompile(Tuple{typeof(Trixi.init_mortars),Array{Int,1},TreeMesh{2,Trixi.ParallelTree{2}},Trixi.ElementContainer2D{RealT,uEltype},mortar_type})
    @assert Base.precompile(Tuple{typeof(Trixi.init_mpi_interfaces),Array{Int,1},TreeMesh{2,Trixi.ParallelTree{2}},Trixi.ElementContainer2D{RealT,uEltype}})

    # 3D, serial
    @assert Base.precompile(Tuple{typeof(Trixi.init_boundaries),Array{Int,1},TreeMesh{3,Trixi.SerialTree{3}},Trixi.ElementContainer3D{RealT,uEltype}})
    @assert Base.precompile(Tuple{typeof(Trixi.init_interfaces),Array{Int,1},TreeMesh{3,Trixi.SerialTree{3}},Trixi.ElementContainer3D{RealT,uEltype}})
    @assert Base.precompile(Tuple{typeof(Trixi.init_mortars),Array{Int,1},TreeMesh{3,Trixi.SerialTree{3}},Trixi.ElementContainer3D{RealT,uEltype},mortar_type})
  end

  # various `show` methods
  for RealT in (Float64,)
    # meshes
    for NDIMS in 1:3
      # serial
      @assert Base.precompile(Tuple{typeof(show),typeof(stdout),TreeMesh{NDIMS,Trixi.SerialTree{NDIMS}}})
      @assert Base.precompile(Tuple{typeof(show),IOContext{typeof(stdout)},MIME"text/plain",TreeMesh{NDIMS,Trixi.SerialTree{NDIMS}}})
      # parallel
      @assert Base.precompile(Tuple{typeof(show),typeof(stdout),TreeMesh{NDIMS,Trixi.ParallelTree{NDIMS}}})
      @assert Base.precompile(Tuple{typeof(show),IOContext{typeof(stdout)},MIME"text/plain",TreeMesh{NDIMS,Trixi.ParallelTree{NDIMS}}})
    end

    # equations
    for eq_type in equations_types(RealT)
      @assert Base.precompile(Tuple{typeof(show),typeof(stdout),eq_type})
      @assert Base.precompile(Tuple{typeof(show),IOContext{typeof(stdout)},MIME"text/plain",eq_type})
    end

    # mortars, analyzers, adaptors, DG
    for polydeg in 1:1
      nnodes_ = polydeg + 1
      basis_type    = basis_type_dgsem(RealT, nnodes_)
      mortar_type   = mortar_type_dgsem(RealT, nnodes_)
      analyzer_type = analyzer_type_dgsem(RealT, nnodes_)
      adaptor_type  = adaptor_type_dgsem(RealT, nnodes_)

      @assert Base.precompile(Tuple{typeof(show),typeof(stdout),basis_type})
      @assert Base.precompile(Tuple{typeof(show),IOContext{typeof(stdout)},MIME"text/plain",basis_type})

      @assert Base.precompile(Tuple{typeof(show),typeof(stdout),mortar_type})
      @assert Base.precompile(Tuple{typeof(show),IOContext{typeof(stdout)},MIME"text/plain",mortar_type})

      @assert Base.precompile(Tuple{typeof(show),typeof(stdout),analyzer_type})
      @assert Base.precompile(Tuple{typeof(show),IOContext{typeof(stdout)},MIME"text/plain",analyzer_type})

      @assert Base.precompile(Tuple{typeof(show),typeof(stdout),adaptor_type})
      @assert Base.precompile(Tuple{typeof(show),IOContext{typeof(stdout)},MIME"text/plain",adaptor_type})

      # we could also use more numerical fluxes and volume integral types here
      @assert Base.precompile(Tuple{typeof(show),typeof(stdout),DG{basis_type,mortar_type,typeof(flux_lax_friedrichs),VolumeIntegralWeakForm}})
      @assert Base.precompile(Tuple{typeof(show),IOContext{typeof(stdout)},MIME"text/plain",DG{basis_type,mortar_type,typeof(flux_lax_friedrichs),VolumeIntegralWeakForm}})
    end

    # semidiscretizations
    @assert Base.precompile(Tuple{typeof(show),IOContext{typeof(stdout)},MIME"text/plain",SemidiscretizationHyperbolic})

    # callbacks
    summary_callback_type = DiscreteCallback{typeof(Trixi.summary_callback),typeof(Trixi.summary_callback),typeof(Trixi.initialize_summary_callback),typeof(SciMLBase.FINALIZE_DEFAULT)}
    @assert Base.precompile(Tuple{typeof(show),typeof(stdout),summary_callback_type})
    @assert Base.precompile(Tuple{typeof(show),IOContext{typeof(stdout)},MIME"text/plain",summary_callback_type})
    @assert Base.precompile(Tuple{summary_callback_type,typeof(stdout)})

    # TODO: SteadyStateCallback, AnalysisCallback

    alive_callback_type = DiscreteCallback{AliveCallback,AliveCallback,typeof(Trixi.initialize!),typeof(SciMLBase.FINALIZE_DEFAULT)}
    @assert Base.precompile(Tuple{typeof(show),typeof(stdout),alive_callback_type})
    @assert Base.precompile(Tuple{typeof(show),IOContext{typeof(stdout)},MIME"text/plain",alive_callback_type})

    restart_callback_type = DiscreteCallback{SaveRestartCallback,SaveRestartCallback,typeof(Trixi.initialize!),typeof(SciMLBase.FINALIZE_DEFAULT)}
    @assert Base.precompile(Tuple{typeof(show),typeof(stdout),restart_callback_type})
    @assert Base.precompile(Tuple{typeof(show),IOContext{typeof(stdout)},MIME"text/plain",restart_callback_type})

    for solution_variables in (cons2cons, cons2prim)
      save_solution_callback_type = DiscreteCallback{SaveSolutionCallback{typeof(solution_variables)},SaveSolutionCallback{typeof(solution_variables)},typeof(Trixi.initialize!),typeof(SciMLBase.FINALIZE_DEFAULT)}
      @assert Base.precompile(Tuple{typeof(show),typeof(stdout),save_solution_callback_type})
      @assert Base.precompile(Tuple{typeof(show),IOContext{typeof(stdout)},MIME"text/plain",save_solution_callback_type})
    end

    # TODO: AMRCallback

    stepsize_callback_type = DiscreteCallback{StepsizeCallback{RealT},StepsizeCallback{RealT},typeof(Trixi.initialize!),typeof(SciMLBase.FINALIZE_DEFAULT)}
    @assert Base.precompile(Tuple{typeof(show),typeof(stdout),stepsize_callback_type})
    @assert Base.precompile(Tuple{typeof(show),IOContext{typeof(stdout)},MIME"text/plain",stepsize_callback_type})

    glm_speed_callback_type = DiscreteCallback{GlmSpeedCallback{RealT},GlmSpeedCallback{RealT},typeof(Trixi.initialize!),typeof(SciMLBase.FINALIZE_DEFAULT)}
    @assert Base.precompile(Tuple{typeof(show),typeof(stdout),glm_speed_callback_type})
    @assert Base.precompile(Tuple{typeof(show),IOContext{typeof(stdout)},MIME"text/plain",glm_speed_callback_type})

    lbm_collision_callback_type = DiscreteCallback{typeof(Trixi.lbm_collision_callback),typeof(Trixi.lbm_collision_callback),typeof(Trixi.initialize!),typeof(SciMLBase.FINALIZE_DEFAULT)}
    @assert Base.precompile(Tuple{typeof(show),typeof(stdout),lbm_collision_callback_type})
    @assert Base.precompile(Tuple{typeof(show),IOContext{typeof(stdout)},MIME"text/plain",lbm_collision_callback_type})

    # infrastructure, special elixirs
    @assert Base.precompile(Tuple{typeof(trixi_include),String})
  end

  # The following precompile statements do not seem to be taken
  # # `multiply_dimensionwise!` as used in the analysis callback
  # for RealT in (Float64,)
  #   # 1D version
  #   @assert Base.precompile(Tuple{typeof(multiply_dimensionwise!),Array{RealT, 2},Matrix{RealT},SubArray{RealT, 2, Array{RealT, 3}, Tuple{Base.Slice{Base.OneTo{Int}}, Base.Slice{Base.OneTo{Int}}, Int}, true}})
  #   # 2D version
  #   @assert Base.precompile(Tuple{typeof(multiply_dimensionwise!),Array{RealT, 3},Matrix{RealT},SubArray{RealT, 3, Array{RealT, 4}, Tuple{Base.Slice{Base.OneTo{Int}}, Base.Slice{Base.OneTo{Int}}, Base.Slice{Base.OneTo{Int}}, Int}, true},Array{RealT, 3}})
  #   # 3D version
  #   @assert Base.precompile(Tuple{typeof(multiply_dimensionwise!),Array{RealT, 4},Matrix{RealT},SubArray{RealT, 4, Array{RealT, 5}, Tuple{Base.Slice{Base.OneTo{Int}}, Base.Slice{Base.OneTo{Int}}, Base.Slice{Base.OneTo{Int}}, Base.Slice{Base.OneTo{Int}}, Int}, true},Array{RealT, 4},Array{RealT, 4}})
  # end

  return nothing
end


end # @muladd
