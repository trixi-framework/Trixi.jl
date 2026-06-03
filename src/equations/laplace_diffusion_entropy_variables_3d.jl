function LaplaceDiffusionEntropyVariables3D(diffusivity, equations_hyperbolic)
    return LaplaceDiffusionEntropyVariables{3, typeof(equations_hyperbolic),
                                            nvariables(equations_hyperbolic),
                                            typeof(diffusivity)}(diffusivity,
                                                                 equations_hyperbolic)
end

function flux(u, gradients, orientation::Integer,
              equations::LaplaceDiffusionEntropyVariables{3})
    dudx, dudy, dudz = gradients
    if orientation == 1
        return equations.diffusivity *
               apply_jacobian_entropy2cons(dudx, u, equations)
    elseif orientation == 2
        return equations.diffusivity *
               apply_jacobian_entropy2cons(dudy, u, equations)
    else # if orientation == 3
        return equations.diffusivity *
               apply_jacobian_entropy2cons(dudz, u, equations)
    end
end
