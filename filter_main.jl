using Trixi

## Callback approach

# supporting functions needed for the callback approach
include("support_functions_khi.jl")

filt = false  # true for Germano, false for the new filter
# filter types: "stage", "step" and "none"
filt_type = "step"
post = false  # post = true for the a posteriori approach
post ? filt_type = "none" : nothing

# contains the filter parameters α², β² and δ² for both filters
# germano filter: u_f - δ² * u_f" = u
# new filter: u_f + α² * u_f" = u + β² * u"
filt_para = [-0.000002;-0.000001;0.0000008]

tEnd = 3.0  # the simulation runs until this point in time

# the elixir has to be in the same folder as this file filter_main.jl
trixi_include(joinpath(@__DIR__,"elixir_euler_kelvin_helmholtz_instability_filter.jl"))

# for comparing runtimes of different approaches
function testfilter()
    #@time main(N,N_Q,a,b,filt,filt_para,input_func,trixi)
    @time trixi_include(joinpath(@__DIR__,"elixir_euler_kelvin_helmholtz_instability_filter.jl"))
end


## A-Posteriori approach

# functions needed for the a posteriori approach
include("filter_a_posteriori.jl")

N = 3
# N_Q² is the number of elements of the CGSEM/DGSEM method
N_Q = [16]  # has to be a vector
# Solve problem on [a,b]²
a = 0.0; b = 1.0
# Boolean for the Germano filter (true) and the newer filter (false)
filt = true
# parameters [α²; β²; δ²] that determine the intensity of the filter
filt_para = [-0.008;-0.0001;0.001]

# if trixi=true, apply filter to the solution of the KHI problem via Trixi
trixi = true
# if trixi=false, you can choose the test function to apply the filter to via this Int
input_func = 1

# filter types: "stage", "step" and "none", only relevant if trixi=true
filt_type = "none"
post = true  # a posteriori approach for the solution calculated by Trixi
post ? filt_type = "none" : nothing

# if trixi=true, the simulation runs until this point in time
tEnd = 3.0

main(N,N_Q,a,b,filt,filt_para,input_func,trixi)
