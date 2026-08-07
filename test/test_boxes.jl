using Test
using Trixi

# Code for detecting `Core.Box`es adapted from
# <https://github.com/JuliaLang/julia/pull/60478> and
# <https://github.com/NumericalEarth/Breeze.jl/pull/400>.
# There's a chance this will be eventually integrated into
# Test or Aqua.jl; for the time being we vendor the code here.

function is_box_call(expr)
    if !(expr isa Expr)
        return false
    end
    if expr.head === :call
        callee = expr.args[1]
        return callee === Core.Box || (callee isa GlobalRef && callee.mod === Core && callee.name === :Box)
    elseif expr.head === :new
        callee = expr.args[1]
        return callee === Core.Box || (callee isa GlobalRef && callee.mod === Core && callee.name === :Box)
    end
    return false
end

function slot_name(ci, slot)
    if slot isa Core.SlotNumber
        idx = Int(slot.id)
        if 1 <= idx <= length(ci.slotnames)
            return string(ci.slotnames[idx])
        end
    end
    return string(slot)
end

function method_location(m::Method)
    file = m.file
    line = m.line
    file_str = file isa Symbol ? String(file) : string(file)
    if file_str == "none" || line == 0
        return ("", 0)
    end
    return (file_str, line)
end

function root_module(mod::Module)
    while true
        parent = parentmodule(mod)
        if parent === mod || parent === Main || parent === Core
            return mod
        end
        mod = parent
    end
end

function format_box_fields(var, m::Method)
    file, line = method_location(m)
    location = isempty(file) ? "" : string(file, ":", line)
    return (
        mod = string(root_module(m.module)),
        var = string(var),
        func = string(m.name),
        sig = string(m.sig),
        location = location,
    )
end

function escape_md(s)
    return replace(string(s), "|" => "\\|")
end

function md_code(s)
    return "`" * replace(string(s), "`" => "``") * "`"
end

function scan_method!(lines, m::Method, modules)
    root = string(root_module(m.module))
    if !isempty(modules) && !(root in modules)
        return
    end
    ci = try
        Base.uncompressed_ast(m)
    catch
        return
    end
    for stmt in ci.code
        if stmt isa Expr && stmt.head === :(=)
            lhs = stmt.args[1]
            rhs = stmt.args[2]
            if is_box_call(rhs)
                push!(lines, format_box_fields(slot_name(ci, lhs), m))
            end
        elseif is_box_call(stmt)
            push!(lines, format_box_fields("<unknown>", m))
        end
    end
end

function number_of_boxes(modules = Set(["Trixi"]))
    format = "markdown"
    lines = Vector{NamedTuple}()
    Base.visit(Core.methodtable) do m
        scan_method!(lines, m, modules)
    end
    sort!(lines, by = entry -> (entry.mod, entry.func, entry.var))
    if format == "plain"
        for entry in lines
            println("mod=", entry.mod,
                    "\tvar=", entry.var,
                    "\tfunc=", entry.func,
                    "\tsig=", entry.sig,
                    "\tlocation=", entry.location)
        end
    else
        # treat "markdown" and "markdown-table" as table output
        last_mod = ""
        for entry in lines
            if entry.mod != last_mod
                if !isempty(last_mod)
                    println()
                end
                println("## $(length(lines)) `Core.Box`es detected in module `", entry.mod, "`")
                println("| var | func | sig | location |")
                println("| --- | --- | --- | --- |")
                last_mod = entry.mod
            end
            println("| ", md_code(escape_md(entry.var)),
                    " | ", md_code(escape_md(entry.func)),
                    " | ", md_code(escape_md(entry.sig)),
                    " | ", md_code(escape_md(entry.location)),
                    " |")
        end
    end

    return length(lines)
end

################################################################################

@testset "No Core.Box" begin
    # Too complicated to adapt to v1.11-, skip it in that case.
    if VERSION < v"1.12"
        @info "Skipping Core.Box check on Julia versions < 1.12"
        return
    end
    @test number_of_boxes() == 33
end
