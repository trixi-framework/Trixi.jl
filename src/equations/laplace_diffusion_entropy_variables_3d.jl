function LaplaceDiffusionEntropyVariables3D(diffusivity, equations_hyperbolic)
    return LaplaceDiffusionEntropyVariables{3, typeof(equations_hyperbolic),
                                            nvariables(equations_hyperbolic),
                                            typeof(diffusivity)}(diffusivity,
                                                                 equations_hyperbolic)
end

function jacobian_entropy2cons(w,
                               equations::LaplaceDiffusionEntropyVariables{3,
                                                                           <:CompressibleEulerEquations3D})
    return equations.diffusivity *
           jacobian_entropy2cons(w, equations.equations_hyperbolic)
end

function flux(u, gradients, orientation::Integer,
              equations::LaplaceDiffusionEntropyVariables{3})
    dudx, dudy, dudz = gradients
    diffusivity = jacobian_entropy2cons(u, equations)
    if orientation == 1
        return SVector(diffusivity * dudx)
    elseif orientation == 2
        return SVector(diffusivity * dudy)
    else # if orientation == 3
        return SVector(diffusivity * dudz)
    end
end
