# Copies some of the LoopVectorization functionality, 
# but solely using Julia base functionality. It is equivalent to `@simd`
# at every loop level
macro turbo(exprs...)
    # Find the outermost for loop
    body = nothing
    for expr in exprs
        if Meta.isexpr(expr, :for)
            body = expr
        end
    end
    @assert body !== nothing

    # We want to visit each nested for loop and insert a `Loopinfo` expression at every level.
    function insert_loopinfo!(expr)
        recurse = Meta.isexpr(expr, :for) || Meta.isexpr(expr, :block) ||
                  Meta.isexpr(expr, :let)
        if recurse
            foreach(insert_loopinfo!, expr.args)
        end
        if Meta.isexpr(expr, :for)
            # We could insert additional LLVM loopinfo or `julia.ivdep`.
            # For now we just encourage vectorization.
            # `Expr(:loopinfo)` corresponds to https://llvm.org/docs/LangRef.html#llvm-loop with two additional nodes
            # `julia.simdloop` & `julia.ivdep`
            # x-ref: https://github.com/JuliaLang/julia/pull/31376
            push!(expr.args, Expr(:loopinfo, Symbol("julia.simdloop")))
        end
    end
    insert_loopinfo!(body)

    body = Expr(:block,
                Expr(:inbounds, true),
                body,
                Expr(:inbounds, :pop))
    return esc(body)
end
