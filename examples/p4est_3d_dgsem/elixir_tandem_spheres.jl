using Trixi

meshfile = "/home/daniel/Sciebo/Job/Doktorand/Content/Meshes/HighOrderCFDWorkshop/CS1/Pointwise/TandemSpheresHexMesh1P2.inp"
mesh_polydeg = 2
boundary_symbols = [:FrontSphere, :BackSphere, :FarField]

mesh = P4estMesh{3}(meshfile; boundary_symbols = boundary_symbols, polydeg = mesh_polydeg)