function LaplaceDiffusionEntropyVariables2D(diffusivity, equations_hyperbolic)
    return LaplaceDiffusionEntropyVariables{2, typeof(equations_hyperbolic),
                                            nvariables(equations_hyperbolic),
                                            typeof(diffusivity)}(diffusivity,
                                                                 equations_hyperbolic)
end

function flux(u, gradients, orientation::Integer,
              equations::LaplaceDiffusionEntropyVariables{2})
    dudx, dudy = gradients
    if orientation == 1
        return equations.diffusivity *
               apply_jacobian_entropy2cons(dudx, u, equations)
    else # if orientation == 2
        return equations.diffusivity *
               apply_jacobian_entropy2cons(dudy, u, equations)
    end
end
