# run tests on Travis CI in parallel
const TRIXI_TEST = get(ENV, "TRIXI_TEST", "all")

@time if TRIXI_TEST == "all" || TRIXI_TEST == "2D"
  include("test_examples_2d.jl")
end

@time if TRIXI_TEST == "all" || TRIXI_TEST == "3D"
  include("test_examples_3d.jl")
end

@time if TRIXI_TEST == "all" || TRIXI_TEST == "misc"
  include("test_manual.jl")
  include("test_elixirs.jl")
end

@time if TRIXI_TEST == "all" || TRIXI_TEST == "paper-self-gravitating-gas-dynamics"
  include("test_paper-self-gravitating-gas-dynamics.jl")
end
