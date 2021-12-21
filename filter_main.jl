#= Add packages if necessary
Pkg.add("SparseArrays"); Pkg.add("LinearAlgebra"); Pkg.add("Plots")
Pkg.add("IncompleteLU"); Pkg.add("IterativeSolvers"); Pkg.add("Trixi")
Pkg.add("OrdinaryDiffEq"); Pkg.add("PrettyTables")
=#


using Trixi

# In this main file, you can configure all the parameters to run the simulation
# of the differential filters that was implemented with the Continuous Galerkin
# method. You can choose between the Germano filter introduced in the paper
"""
- Massimo Germano (1986)
  Differential filters for the large eddy numerical simulation of turbulent flows,
  The Physics of fluids Vol. 29 Nr. 6, pp. 1755-1757, American Institute of Physics
  (https://bugs.openfoam.org/file_download.php?file_id=681&type=bug)
"""
# and the newer one introduced in the paper
"""
- Alireza Najafi-Yazdi, Mostafa Najafi-Yazdi and Luc Mongeau (2015)
  A high resolution differential filter for large eddy simulation:
      Toward explicit filtering on unstructured grids,
  Journal of Computational Physics Vol. 292, pp. 272-286, Elsevier
"""
# You can apply the filters to some test functions via the a posteriori approach.
# Or you can apply them to the solution of the KHI problem calculated by Trixi.jl¹.
# The callback approach lets you integrate the filters into the solving process of
# Trixi.jl through the use of callbacks. The a posteriori approach applies the
# filters to the solution after it is calculated.

############################################

# First are the parameters for the a posteriori approach. You can start this
# simulation by running the function run_a_posteriori_approach().
# Second are the parameters for the callback approach. You can start this
# simulation by running the function run_callback_approach().

############################################
## A Posteriori approach
############################################

# Here we apply the filters a posteriori to some test functions or the solution
# of the KHI problem given by Trixi.jl

# Functions needed for the a posteriori approach
include("filter_a_posteriori.jl")

### Do NOT change the following parameters ###

# For post = 'false', we apply the callback approach, but here we have
# post = 'true', since we want to apply the a posteriori approach
post = true

# We don't want to apply a callback approach, but the a posteriori approach
# For that we have to set filt_type = "none"
# Only relevant if trixi = 'true'
filt_type = "none"


### The following parameters are changeable to calibrate the simulation
# you want to run ###

# Which differential filter do you want to apply?
# Germano filter: u_f - δ² * ∇² u_f = u,           u unfiltered, u_f filtered
# New filter: u_f + α² * ∇²  u_f = u + β² * ∇² u
# Set filt = 'true' for germano, and filt = 'false' for the new filter
filt = true

# Contains the filter parameters [α², β², δ²] for both filters
filt_para = [-0.008;-0.0001;0.001]

# The following parameters are only relevant if trixi = 'false', because for
# trixi = 'true' the parameters are set by Trixi.jl

# The polynomial degree of the approximation in each cell
N = 3
# N_Q² is the number of elements that we seperate Ω in
N_Q = [16;32]  # Has to be a vector even if it's only 1 value
# Solve problem on Ω = [a,b]²
a = 0.0; b = 1.0

# Set this integer as either 1, 2 oder 3 to pick the test function you want
# to aplly the filter to. The case input_func = 3 is special. For input_func = 3
# the method of manufactured solutions is used to test the accuracy of the program.
# This only works for the Germano filter, i.e for filt = 'true'
input_func = 1


# Set trixi = 'false' if you want to apply the filter to one of the test functions
# you can choose via the integer input_func.
# Set trixi = 'true' if you want to apply the filter to the solution of the
# KHI problem, that is calculated by the Trixi.jl elixir:
# elixir_euler_kelvin_helmholtz_instability_filter.jl
# Trixi.jl sets the parameters in this case to N=3, N_Q = 32, [a,b]² = [-1,1]²
trixi = false

# Run the Trixi.jl simulation until this point in time, only relevant if
# trixi = 'true'
tEnd = 3.0

# If you want to apply shock capturing to the unfiltered solution by having
# the volume integral calculated by the function VolumeIntegralShockCapturingHG,
# set this as true. Otherwise, the function VolumeIntegralFluxDifferencing is used.
# This is only relevant for trixi = 'true'. If you set this as 'false', the
# Trixi.jl simulation will break off at about t = 3.7
apply_shock_capturing = true


# If you want to calculate the runtime of the program, set this as 'true'
calc_runtime = false
# If you want to calculate just the runtime of the filtering process while
# ignoring the time needed to plot the results, set this as 'true'
# For this, calc_runtime has to be set to 'true' as well
runtime_without_plots = false

# You can either calculate the runtime with or without including the plotting process
# or you can just run the simulation.
# The Trixi.jl elixir has to be in the same folder as this file if you want to
# apply the filter to the KHI problem
function run_a_posteriori_approach()

    if calc_runtime
        @time main(N,N_Q,a,b,filt,filt_para,input_func,trixi,runtime_without_plots)
    else
        main(N,N_Q,a,b,filt,filt_para,input_func,trixi,false)
    end

end

run_a_posteriori_approach()




############################################
## Callback approach
############################################

# Here we integrate the differential filter into the framework Trixi.jl. We apply
# the filter to the KHI problem. Callbacks are algorithms that are run during
# the solving process. We wrote the filter as a callback that is applied in
# every time step (step-callback) or in every RK-step (stage-callback)

# Supporting functions needed for the callback approach
include("support_functions_khi.jl")


### Do NOT change the following parameters ###

# For post = 'true', we apply the a posteriori approach, but here we have
# post = 'false', since we want to apply the callback approach
post = false


### The following parameters are changeable to calibrate the simulation
#   you want to run ###

# Do you want to use a stage-callback-approach or a step-callback-approach?
# Alternatively set filt_type = "none" to not apply any filter
# Filter types: "stage", "step" and "none"
filt_type = "step"

# Which differential filter do you want to apply?
# Germano filter: u_f - δ² * ∇² u_f = u,           u unfiltered, u_f filtered
# New filter: u_f + α² * ∇²  u_f = u + β² * ∇² u
# Set filt = 'true' for germano, and filt = 'false' for the new filter
filt = true

# Contains the filter parameters [α², β², δ²] for both filters
filt_para = [-0.000002;-0.000001;0.0000008]

# Run the simulation until this point in time
tEnd = 3.0

# If you want to apply shock capturing to the unfiltered solution by having
# the volume integral calculated by the function VolumeIntegralShockCapturingHG,
# set this as true. Otherwise, the function VolumeIntegralFluxDifferencing is used.
# This is only relevant for filt_type = "none". If you set this as 'false' and
# don't apply any filter, the simulation will break off at about t = 3.7
apply_shock_capturing = false

# If you want to calculate the runtime of the entire program, set this as 'true'
calc_runtime = false
# If you want to calculate just the runtime of the filtering process while
# ignoring the time needed to plot the results, set this as 'true'
# For this, calc_runtime has to be set to 'true' as well
runtime_without_plots = false

# You can either calculate the runtime with or without including the plotting process
# or you can just run the simulation.
# The Trixi.jl elixir has to be in the same folder as this file
function run_callback_approach()

    if calc_runtime
        @time trixi_include(joinpath(@__DIR__,"elixir_euler_kelvin_helmholtz_instability_filter.jl"))
    else
        trixi_include(joinpath(@__DIR__,"elixir_euler_kelvin_helmholtz_instability_filter.jl"))
    end

end

run_callback_approach()




## ¹ This programm makes use of the framework Trixi.jl to run simulations of
#    the Kelvin Helmholtz instabilities. For more information concerning Trixi.jl
#    you can check out the following papers.
"""
- Hendrik Ranocha, Michael Schlottke-Lakemper, Andrew R. Winters,
       Erik Faulhaber, Jesse Chan and Gregor J. Gassner (2021)
  Adaptive numerical simulations with Trixi.jl: A case study of Julia for
  scientific computing
  [arXiv: 2108.06476](https://arxiv.org/abs/2108.06476)

 - Michael Schlottke-Lakemper, Andrew R. Winters, Hendrik Ranocha and
        Gregor J. Gassner (2021)
   A purely hyperbolic discontinuous Galerkin approach for self-gravitating
   gas dynamics
   Journal of Computational Physics Vol. 442, p. 110467, Elsevier
   [arXiv: 2008.10593](https://arxiv.org/abs/2008.10593)
"""

