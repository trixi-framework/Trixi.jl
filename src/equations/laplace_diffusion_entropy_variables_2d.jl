function LaplaceDiffusionEntropyVariables2D(diffusivity, equations_hyperbolic)
    LaplaceDiffusionEntropyVariables{2, typeof(equations_hyperbolic),
                                     nvariables(equations_hyperbolic),
                                     typeof(diffusivity)}(diffusivity, equations_hyperbolic)
end

function flux(u, gradients, orientation::Integer,
              equations::LaplaceDiffusionEntropyVariables{2})
    dudx, dudy = gradients
    diffusivity = jacobian_entropy2cons(u, equations)
    if orientation == 1
        return SVector(diffusivity * dudx)
    else # if orientation == 2
        return SVector(diffusivity * dudy)
    end
end
