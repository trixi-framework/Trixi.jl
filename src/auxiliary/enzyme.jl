# Support for automatic differentiation with Enzyme.jl.

# The time step size of the CFL-based `StepsizeCallback` is not differentiated. Hence,
# reverse-mode derivatives through a time integration use the step sizes of the primal
# run as constants ("frozen" step size control), as in discrete adjoints of CFD codes.
EnzymeCore.EnzymeRules.inactive(::typeof(max_dt), args...) = nothing

# Reverse-mode rule for `octavian_matmul!`, i.e., Octavian.jl's `matmul!`, which uses
# LoopVectorization.jl internally and cannot be differentiated by Enzyme.jl.
#   out = A * x                 =>  x̄ += Aᵀ ō,    Ā += ō xᵀ,    ō = 0
#   out = α * A * x + β * out   =>  x̄ += α Aᵀ ō,  Ā += α ō xᵀ,  ᾱ = ō ⋅ (A x),  ō = β ō
# `A` and `α` are differentiated as well since they can depend on the mesh, e.g., for
# shape derivatives. `β` is a constant (`One()` in Trixi.jl).
function _shadows(config, x::EnzymeCore.Annotation)
    return EnzymeCore.EnzymeRules.width(config) == 1 ? (x.dval,) : x.dval
end

# ᾱ = ō ⋅ (A x)
function _matmul_dalpha(config, out, A, x, α::T) where {T}
    width = EnzymeCore.EnzymeRules.width(config)
    if out isa EnzymeCore.Const
        dαs = ntuple(_ -> zero(T), Val(width))
    else
        Ax = A * x
        out_shadows = _shadows(config, out)
        dαs = ntuple(i -> convert(T, dot(out_shadows[i], Ax))::T, Val(width))
    end
    return width == 1 ? dαs[1] : dαs
end

function EnzymeCore.EnzymeRules.augmented_primal(config,
                                                 func::EnzymeCore.Const{typeof(octavian_matmul!)},
                                                 ::Type{<:EnzymeCore.Const},
                                                 out::EnzymeCore.Annotation,
                                                 A::EnzymeCore.Annotation,
                                                 x::EnzymeCore.Annotation,
                                                 args::EnzymeCore.Annotation...)
    # Save `A` and `x` if they are needed in the reverse pass and may be overwritten
    overwritten = EnzymeCore.EnzymeRules.overwritten(config)
    active_α = length(args) == 2 && args[1] isa EnzymeCore.Active
    need_A = !(out isa EnzymeCore.Const) && (!(x isa EnzymeCore.Const) || active_α)
    need_x = !(out isa EnzymeCore.Const) && (!(A isa EnzymeCore.Const) || active_α)
    A_saved = need_A && overwritten[3] ? copy(A.val) : nothing
    x_saved = need_x && overwritten[4] ? copy(x.val) : nothing

    result = func.val(out.val, A.val, x.val, map(arg -> arg.val, args)...)
    primal = EnzymeCore.EnzymeRules.needs_primal(config) ? result : nothing

    return EnzymeCore.EnzymeRules.AugmentedReturn(primal, nothing, (A_saved, x_saved))
end

function EnzymeCore.EnzymeRules.reverse(config,
                                        ::EnzymeCore.Const{typeof(octavian_matmul!)},
                                        ::Type{<:EnzymeCore.Const}, tape,
                                        out::EnzymeCore.Annotation,
                                        A::EnzymeCore.Annotation,
                                        x::EnzymeCore.Annotation,
                                        args::EnzymeCore.Annotation...)
    if length(args) == 2
        α_annotation, β_annotation = args
        α = α_annotation.val
        β = β_annotation.val
    else
        α_annotation = nothing
        α = true
        β = false
    end
    active_α = α_annotation isa EnzymeCore.Active
    width = EnzymeCore.EnzymeRules.width(config)

    (A_saved, x_saved) = tape
    A_val = A_saved === nothing ? A.val : A_saved
    x_val = x_saved === nothing ? x.val : x_saved

    dα = active_α ? _matmul_dalpha(config, out, A_val, x_val, α) : nothing

    if !(out isa EnzymeCore.Const)
        # `α` may be a static number such as `One()`
        α_number = convert(eltype(x_val), α)
        for i in 1:width
            ō = _shadows(config, out)[i]
            if !(x isa EnzymeCore.Const)
                # x̄ += α Aᵀ ō
                mul!(_shadows(config, x)[i], transpose(A_val), ō, α_number, true)
            end
            if !(A isa EnzymeCore.Const)
                # Ā += α ō xᵀ
                mul!(_shadows(config, A)[i], ō, transpose(x_val), α_number, true)
            end
            # `out` was overwritten (β = 0) or scaled by β
            if iszero(β)
                fill!(ō, zero(eltype(ō)))
            elseif !isone(β)
                ō .*= β
            end
        end
    end

    if length(args) == 2
        return (nothing, nothing, nothing, dα, nothing)
    else
        return (nothing, nothing, nothing)
    end
end
