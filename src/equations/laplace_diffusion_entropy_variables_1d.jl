function LaplaceDiffusionEntropyVariables1D(diffusivity, equations_hyperbolic)
    LaplaceDiffusionEntropyVariables{1, typeof(equations_hyperbolic),
                                     nvariables(equations_hyperbolic),
                                     typeof(diffusivity)}(diffusivity, equations_hyperbolic)
end

# Note that here, `u` should be the transformed entropy variables, and 
# not the conservative variables.
function flux(u, gradients, orientation::Integer,
              equations::LaplaceDiffusionEntropyVariables{1})
    dudx = gradients
    diffusivity = jacobian_entropy2cons(u, equations)
    # if orientation == 1
    return SVector(diffusivity * dudx)
end
