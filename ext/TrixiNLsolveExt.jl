# Package extension for adding Convex-based features to Trixi.jl
module TrixiNLsolveExt

# Required for coefficient optimization in P-ERK scheme integrators
if isdefined(Base, :get_extension)
    using NLsolve: nlsolve
else
    # Until Julia v1.9 is the minimum required version for Trixi.jl, we still support Requires.jl
    using ..NLsolve: nlsolve
end

# Use other necessary libraries
using Random: seed!
using LinearAlgebra: norm

# Use functions and additional symbols that are not exported
using Trixi: Trixi, PairedExplicitRK3_butcher_tableau_objective_function, @muladd

# Use functions that are to be extended
using Trixi: Trixi, solve_a_unknown!

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Find the values of the a_{i, i-1} in the Butcher tableau matrix A by solving a system of
# non-linear equations that arise from the relation of the stability polynomial to the Butcher tableau.
# For details, see Proposition 3.2, Equation (3.3) from 
# Hairer, Wanner: Solving Ordinary Differential Equations 2
function Trixi.solve_a_unknown!(a_unknown, num_stages, monomial_coeffs, c_s2, c;
                                verbose)
    is_sol_valid = false

    # Define the objective_function
    function objective_function(x)
        return Trixi.PairedExplicitRK3_butcher_tableau_objective_function(x, num_stages,
                                                                          num_stages,
                                                                          monomial_coeffs,
                                                                          c_s2)
    end

    # Set the seed for reproducibility of the initial guess of a_unknown
    #seed!(5555)

    while !is_sol_valid
        # Initialize initial guess
        # The nonlinear system may have multiple valid solutions.
        # For the sake of reproducibility, we use a seeded random initial guess
        x0 = 0.1 .* rand(num_stages - 2)

        sol = nlsolve(objective_function, x0, method = :trust_region,
                      ftol = 4e-16, # Enforce objective up to machine precision
                      iterations = 10^4, xtol = 1e-13)

        a_unknown = sol.zero

        # Check if the values a[i, i-1] >= 0.0 (which stem from the nonlinear solver) 
        # and also c[i] - a[i, i-1] >= 0.0 since all coefficients are non-negative
        is_sol_valid = all(x -> !isnan(x) && x >= 0, a_unknown) &&
                       all(x -> !isnan(x) && x >= 0, c[3:end] .- a_unknown)

        if verbose && !is_sol_valid
            println("Solution invalid. Restart the process of solving non-linear system of equations again.")
        end
    end

    return a_unknown
end
end # @muladd

end # module TrixiNLsolveExt
