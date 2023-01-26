using Literate
Literate.markdown("./src/Getting_started_with_Trixi.jl", ".")
Literate.notebook("./src/Getting_started_with_Trixi.jl", "."; execute=false, credit=false)