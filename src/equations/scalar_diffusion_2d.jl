
struct ScalarDiffusion2D{T} <: AbstractParabolicEquations{2, 1}
  diffusivity::T
end

# no orientation here since the flux is vector-valued
function flux(u, grad_u, equations::ScalarDiffusion2D)
  dudx, dudy = grad_u
  return equations.diffusivity * dudx, equations.diffusivity * dudy
end

