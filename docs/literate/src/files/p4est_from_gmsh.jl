#src # `P4estMesh` from [`gmsh`](https://gmsh.info/)

# Trixi.jl supports numerical approximations from structured and unstructured quadrilateral meshes
# with the [`P4estMesh`](@ref) mesh type.

# The purpose of this tutorial is to demonstrate how to use the `P4estMesh`
# functionality of Trixi.jl for existing meshes with straight-sided (bilinear) elements/cells.
# This begins by running and visualizing an available unstructured quadrilateral mesh example.
# Then, the tutorial will cover how to use existing meshes generated by [`gmsh`](https://gmsh.info/)
# or any other meshing software that can export to the Abaqus input `.inp` format.

# ## Running the simulation of a near-field flow around an airfoil

# Trixi.jl supports solving hyperbolic-parabolic problems on several mesh types.
# A somewhat complex example that employs the `P4estMesh` is the near-field simulation of a
# Mach 2 flow around the NACA6412 airfoil. 

using Trixi
redirect_stdio(stdout=devnull, stderr=devnull) do # code that prints annoying stuff we don't want to see here #hide #md
trixi_include(joinpath(examples_dir(), "p4est_2d_dgsem", "elixir_euler_NACA6412airfoil_mach2.jl"), tspan=(0.0, 0.5))
end #hide #md

# Conveniently, we use the Plots package to have a first look at the results:
# ```julia
# using Plots
# pd = PlotData2D(sol)
# plot(pd["rho"])
# plot!(getmesh(pd))
# ```

# ## Creating a mesh using `gmsh`

# The creation of an unstructured quadrilateral mesh using `gmsh` is driven by a **geometry file**. 
# There are plenty of possibilities for the user, see the [documentation](https://gmsh.info/doc/texinfo/gmsh.html) and [tutorials](https://gitlab.onelab.info/gmsh/gmsh/tree/master/tutorials).

# To begin, we provide a complete geometry file for the NACA6412 airfoil bounded by a rectangular box. After this we give a breakdown
# of the most important parts required for successful mesh generation that can later be used by the `p4est` library
# and Trixi.jl.
# We emphasize that this near-field mesh should only be used for instructive purposes and not for actual production runs.

# The associated `NACA6412.geo` file is given below:
# ```c++
#  // GMSH geometry script for a NACA 6412 airfoil with 11 degree angle of attack 
#  // in a box (near-field mesh).
#  // see https://github.com/cfsengineering/GMSH-Airfoil-2D
#  // for software to generate gmsh `.geo` geometry files for NACA airfoils.
#  
#  // outer bounding box
#  Point(1) = {-1.25, -0.5, 0, 1.0};
#  Point(2) = {1.25, -0.5, 0, 1.0};
#  Point(3) = {1.25, 0.5, 0, 1.0};
#  Point(4) = {-1.25, 0.5, 0, 1.0};
#
#  // lines of the bounding box
#  Line(1) = {1, 2};
#  Line(2) = {2, 3};
#  Line(3) = {3, 4};
#  Line(4) = {4, 1};
#  // outer box
#  Line Loop(8) = {1, 2, 3, 4};
#
#  // Settings
#  // This value gives the global element size factor (lower -> finer mesh)
#  Mesh.CharacteristicLengthFactor = 1.0 * 2^(-3);
#  // Insist on quads instead of default triangles
#  Mesh.RecombineAll = 1;
#  // Violet instead of green base color for better visibility
#  Mesh.ColorCarousel = 0;
#  
#  // points of the airfoil contour
#  // Format: {x, y, z, DesiredCellSize}. See the documentation: https://gmsh.info/doc/texinfo/gmsh.html#Points
#  Point(5) = {-0.4900332889206208, 0.09933466539753061, 0, 0.125};
#  Point(6) = {-0.4900274857651495, 0.1021542752054094, 0, 0.125};
#  Point(7) = {-0.4894921489729144, 0.1049830248247787, 0, 0.125};
#  Point(8) = {-0.4884253336670712, 0.1078191282319664, 0, 0.125};
#  Point(9) = {-0.4868257975566199, 0.1106599068424483, 0, 0.125};
#  Point(10) = {-0.4846930063965668, 0.1135018003016681, 0, 0.125};
#  Point(11) = {-0.4820271400142729, 0.1163403835785654, 0, 0.125};
#  Point(12) = {-0.4788290988083472, 0.1191703902233889, 0, 0.125};
#  Point(13) = {-0.4751005105908123, 0.1219857416089041, 0, 0.125};
#  Point(14) = {-0.4708437376101668, 0.1247795819332056, 0, 0.125};
#  Point(15) = {-0.4660618835629463, 0.1275443187232316, 0, 0.125};
#  Point(16) = {-0.4607588003749649, 0.1302716685409717, 0, 0.125};
#  Point(17) = {-0.4549390945110529, 0.132952707559475, 0, 0.125};
#  Point(18) = {-0.448608132554204, 0.1355779266432996, 0, 0.125};
#  Point(19) = {-0.4417720457819508, 0.138137290538182, 0, 0.125};
#  Point(20) = {-0.4344377334597768, 0.140620300747629, 0, 0.125};
#  Point(21) = {-0.4266128645686593, 0.1430160616500159, 0, 0.125};
#  Point(22) = {-0.4183058776865576, 0.1453133493887722, 0, 0.125};
#  Point(23) = {-0.4095259787518715, 0.147500683050503, 0, 0.125};
#  Point(24) = {-0.4002831364505879, 0.1495663976315875, 0, 0.125};
#  Point(25) = {-0.3905880749878933, 0.1514987182830453, 0, 0.125};
#  Point(26) = {-0.3804522640292948, 0.1532858353164163, 0, 0.125};
#  Point(27) = {-0.3698879056254708, 0.1549159794501833, 0, 0.125};
#  Point(28) = {-0.3589079179688306, 0.1563774967770029, 0, 0.125};
#  Point(29) = {-0.3475259158676376, 0.1576589229368209, 0, 0.125};
#  Point(30) = {-0.3357561878650377, 0.158749055989923, 0, 0.125};
#  Point(31) = {-0.3236136699747923, 0.1596370274972017, 0, 0.125};
#  Point(32) = {-0.3111139160522804, 0.1603123713324616, 0, 0.125};
#  Point(33) = {-0.298273064867608, 0.160765089773461, 0, 0.125};
#  Point(34) = {-0.2851078039966239, 0.1609857164445887, 0, 0.125};
#  Point(35) = {-0.2716353306943914, 0.160965375714529, 0, 0.125};
#  Point(36) = {-0.2578733099632437, 0.1606958381868515, 0, 0.125};
#  Point(37) = {-0.2438398300730194, 0.1601695719599709, 0, 0.125};
#  Point(38) = {-0.2295533558334121, 0.1593797893750759, 0, 0.125};
#  Point(39) = {-0.2150326799566391, 0.1583204890160489, 0, 0.125};
#  Point(40) = {-0.2002968728818922, 0.1569864927736143, 0, 0.125};
#  Point(41) = {-0.18536523146042, 0.1553734778363979, 0, 0.125};
#  Point(42) = {-0.1702572269208345, 0.1534780035235666, 0, 0.125};
#  Point(43) = {-0.1549924525477129, 0.1512975329264932, 0, 0.125};
#  Point(44) = {-0.1395905715122586, 0.1488304493795921, 0, 0.125};
#  Point(45) = {-0.1240712652914332, 0.1460760678321895, 0, 0.125};
#  Point(46) = {-0.1084541831014299, 0.1430346412430583, 0, 0.125};
#  Point(47) = {-0.09275889275279087, 0.1397073621660917, 0, 0.125};
#  Point(48) = {-0.07700483330818747, 0.1360963597385416, 0, 0.125};
#  Point(49) = {-0.06151286635366404, 0.1323050298149023, 0, 0.125};
#  Point(50) = {-0.04602933219022032, 0.1283521764905442, 0, 0.125};
#  Point(51) = {-0.03051345534800332, 0.1242331665904082, 0, 0.125};
#  Point(52) = {-0.01498163190522334, 0.1199540932779839, 0, 0.125};
#  Point(53) = {0.0005498526140696458, 0.1155214539466913, 0, 0.125};
#  Point(54) = {0.01606484191716884, 0.1109421303284033, 0, 0.125};
#  Point(55) = {0.03154732664394777, 0.106223368423828, 0, 0.125};
#  Point(56) = {0.0469814611314705, 0.1013727584299359, 0, 0.125};
#  Point(57) = {0.06235157928986135, 0.09639821481480275, 0, 0.125};
#  Point(58) = {0.07764220964363855, 0.09130795666388933, 0, 0.125};
#  Point(59) = {0.09283808959671735, 0.08611048839446452, 0, 0.125};
#  Point(60) = {0.1079241789809607, 0.08081458090718853, 0, 0.125};
#  Point(61) = {0.1228856729475325, 0.07542925321638272, 0, 0.125};
#  Point(62) = {0.1377080142575372, 0.06996375457378261, 0, 0.125};
#  Point(63) = {0.1523769050236616, 0.06442754707512513, 0, 0.125};
#  Point(64) = {0.1668783179480157, 0.05883028871526293, 0, 0.125};
#  Point(65) = {0.1811985070933818, 0.05318181683604975, 0, 0.125};
#  Point(66) = {0.1953240182159306, 0.04749213189240609, 0, 0.125};
#  Point(67) = {0.2092416986775084, 0.04177138144606024, 0, 0.125};
#  Point(68) = {0.2229387069452062, 0.03602984428372727, 0, 0.125};
#  Point(69) = {0.2364025216754475, 0.03027791454712048, 0, 0.125};
#  Point(70) = {0.2496209503696738, 0.02452608575629232, 0, 0.125};
#  Point(71) = {0.2625821375791982, 0.01878493460541621, 0, 0.125};
#  Point(72) = {0.2752745726282818, 0.01306510441121807, 0, 0.125};
#  Point(73) = {0.28768709681727, 0.007377288098728577, 0, 0.125};
#  Point(74) = {0.2998089100619555, 0.001732210616722449, 0, 0.125};
#  Point(75) = {0.3116295769214332, -0.003859389314124759, 0, 0.125};
#  Point(76) = {0.3231390319647309, -0.009386778203927332, 0, 0.125};
#  Point(77) = {0.3343275844265582, -0.01483924761490708, 0, 0.125};
#  Point(78) = {0.3451859221046181, -0.02020613485126957, 0, 0.125};
#  Point(79) = {0.3557051144551212, -0.02547684454806881, 0, 0.125};
#  Point(80) = {0.3658766148492779, -0.03064087116872238, 0, 0.125};
#  Point(81) = {0.3756922619615632, -0.0356878223992288, 0, 0.125};
#  Point(82) = {0.3851442802702071, -0.0406074434050937, 0, 0.125};
#  Point(83) = {0.394225279661484, -0.04538964189492445, 0, 0.125};
#  Point(84) = {0.4029282541416501, -0.05002451391298904, 0, 0.125};
#  Point(85) = {0.4112465796735204, -0.05450237026215737, 0, 0.125};
#  Point(86) = {0.4191740111683733, -0.05881376343890812, 0, 0.125};
#  Point(87) = {0.4267046786777481, -0.06294951494382847, 0, 0.125};
#  Point(88) = {0.4338330828434404, -0.06690074281456823, 0, 0.125};
#  Point(89) = {0.4405540896772232, -0.07065888921378868, 0, 0.125};
#  Point(90) = {0.4468629247542237, -0.07421574789251445, 0, 0.125};
#  Point(91) = {0.4527551669150955, -0.0775634913396257, 0, 0.125};
#  Point(92) = {0.4582267415819197, -0.08069469742118066, 0, 0.125};
#  Point(93) = {0.4632739138007936, -0.08360237530891265, 0, 0.125};
#  Point(94) = {0.4678932811302005, -0.08627999049569551, 0, 0.125};
#  Point(95) = {0.4720817664982195, -0.08872148869699745, 0, 0.125};
#  Point(96) = {0.4758366111533843, -0.09092131844134463, 0, 0.125};
#  Point(97) = {0.4791553678333992, -0.09287445215953141, 0, 0.125};
#  Point(98) = {0.4820358942729613, -0.09457640559161551, 0, 0.125};
#  Point(99) = {0.4844763471666588, -0.09602325534252773, 0, 0.125};
#  Point(100) = {0.4864751766953637, -0.09721165443119822, 0, 0.125};
#  Point(101) = {0.4880311217148797, -0.09813884569428721, 0, 0.125};
#  Point(102) = {0.4891432056939881, -0.09880267292366274, 0, 0.125};
#  Point(103) = {0.4898107334756874, -0.09920158963645126, 0, 0.125};
#  Point(104) = {0.4900332889206208, -0.09933466539753058, 0, 0.125};
#  Point(105) = {0.4897824225031319, -0.09926905587549506, 0, 0.125};
#  Point(106) = {0.4890301110661922, -0.09907236506934192, 0, 0.125};
#  Point(107) = {0.4877772173496635, -0.09874500608402761, 0, 0.125};
#  Point(108) = {0.48602517690576, -0.09828766683852558, 0, 0.125};
#  Point(109) = {0.4837759946062035, -0.09770130916007558, 0, 0.125};
#  Point(110) = {0.4810322398085871, -0.09698716747297723, 0, 0.125};
#  Point(111) = {0.4777970402368822, -0.09614674703990023, 0, 0.125};
#  Point(112) = {0.4740740746447117, -0.09518182170326678, 0, 0.125};
#  Point(113) = {0.4698675643422793, -0.09409443106501386, 0, 0.125};
#  Point(114) = {0.4651822636784212, -0.09288687703518478, 0, 0.125};
#  Point(115) = {0.460023449577924, -0.09156171967354482, 0, 0.125};
#  Point(116) = {0.4543969102408585, -0.09012177224394632, 0, 0.125};
#  Point(117) = {0.4483089331151018, -0.08857009539864649, 0, 0.125};
#  Point(118) = {0.4417662922553667, -0.08690999040934186, 0, 0.125};
#  Point(119) = {0.4347762351819332, -0.0851449913634191, 0, 0.125};
#  Point(120) = {0.4273464693498908, -0.08327885624791403, 0, 0.125};
#  Point(121) = {0.419485148335155, -0.08131555684993674, 0, 0.125};
#  Point(122) = {0.411200857836944, -0.07925926741086739, 0, 0.125};
#  Point(123) = {0.4025026015879757, -0.07711435198240155, 0, 0.125};
#  Point(124) = {0.3933997872536054, -0.07488535044544484, 0, 0.125};
#  Point(125) = {0.3839022123897198, -0.07257696316779733, 0, 0.125};
#  Point(126) = {0.3740200505167618, -0.07019403429336624, 0, 0.125};
#  Point(127) = {0.3637638373540689, -0.06774153367408606, 0, 0.125};
#  Point(128) = {0.3531444572451353, -0.06522453747557577, 0, 0.125};
#  Point(129) = {0.3421731297908021, -0.06264820750853495, 0, 0.125};
#  Point(130) = {0.3308613966940724, -0.06001776935966011, 0, 0.125};
#  Point(131) = {0.3192211088076166, -0.05733848941811218, 0, 0.125};
#  Point(132) = {0.3072644133633567, -0.05461565091590426, 0, 0.125};
#  Point(133) = {0.2950037413531683, -0.05185452912263369, 0, 0.125};
#  Point(134) = {0.2824517950208982, -0.04906036585632723, 0, 0.125};
#  Point(135) = {0.2696215354188702, -0.04623834349241404, 0, 0.125};
#  Point(136) = {0.2565261699769623, -0.04339355867155523, 0, 0.125};
#  Point(137) = {0.2431791400293651, -0.04053099592384862, 0, 0.125};
#  Point(138) = {0.2295941082432855, -0.03765550144139543, 0, 0.125};
#  Point(139) = {0.2157849458952252, -0.03477175724299444, 0, 0.125};
#  Point(140) = {0.2017657199439165, -0.03188425598348005, 0, 0.125};
#  Point(141) = {0.187550679854507, -0.02899727666564914, 0, 0.125};
#  Point(142) = {0.1731542441359161, -0.02611486151457043, 0, 0.125};
#  Point(143) = {0.1585909865622793, -0.02324079427214604, 0, 0.125};
#  Point(144) = {0.1438756220597465, -0.02037858016395433, 0, 0.125};
#  Point(145) = {0.129022992251319, -0.0175314277805827, 0, 0.125};
#  Point(146) = {0.1140480506645569, -0.01470223310184333, 0, 0.125};
#  Point(147) = {0.09896584761949168, -0.01189356587453844, 0, 0.125};
#  Point(148) = {0.08379151482656089, -0.009107658532933174, 0, 0.125};
#  Point(149) = {0.06854024973648176, -0.006346397826038436, 0, 0.125};
#  Point(150) = {0.05322729969528361, -0.003611319287478529, 0, 0.125};
#  Point(151) = {0.03786794596792287, -0.00090360465249055, 0, 0.125};
#  Point(152) = {0.0224774877026287, 0.00177591770710904, 0, 0.125};
#  Point(153) = {0.007071225915134205, 0.004426769294862437, 0, 0.125};
#  Point(154) = {-0.00833555242305456, 0.007048814950562587, 0, 0.125};
#  Point(155) = {-0.02372759010533726, 0.009642253300220296, 0, 0.125};
#  Point(156) = {-0.03908967513210498, 0.01220760427359278, 0, 0.125};
#  Point(157) = {-0.05440665578848514, 0.01474569380579989, 0, 0.125};
#  Point(158) = {-0.06966345527617318, 0.01725763587663899, 0, 0.125};
#  Point(159) = {-0.08484508582421563, 0.01974481207672138, 0, 0.125};
#  Point(160) = {-0.09987987792382108, 0.02219618763023203, 0, 0.125};
#  Point(161) = {-0.1145078729404739, 0.02450371976411331, 0, 0.125};
#  Point(162) = {-0.1290321771824579, 0.0267015185742735, 0, 0.125};
#  Point(163) = {-0.143440065923266, 0.02879471001709845, 0, 0.125};
#  Point(164) = {-0.1577189448447794, 0.03078883518202784, 0, 0.125};
#  Point(165) = {-0.1718563428491159, 0.03268980457290044, 0, 0.125};
#  Point(166) = {-0.1858399037768357, 0.03450385196323842, 0, 0.125};
#  Point(167) = {-0.1996573773370766, 0.03623748825421298, 0, 0.125};
#  Point(168) = {-0.2132966095779342, 0.03789745574015834, 0, 0.125};
#  Point(169) = {-0.2267455332406906, 0.0394906831577609, 0, 0.125};
#  Point(170) = {-0.2399921583489679, 0.04102424186233269, 0, 0.125};
#  Point(171) = {-0.2530245633834605, 0.04250530343879837, 0, 0.125};
#  Point(172) = {-0.2658308873846617, 0.04394109901707172, 0, 0.125};
#  Point(173) = {-0.2783993233102972, 0.04533888052223981, 0, 0.125};
#  Point(174) = {-0.2907181129514687, 0.04670588405019788, 0, 0.125};
#  Point(175) = {-0.3027755436824813, 0.0480492955198111, 0, 0.125};
#  Point(176) = {-0.3145599472847223, 0.04937621871394801, 0, 0.125};
#  Point(177) = {-0.3260597010456697, 0.05069364578437131, 0, 0.125};
#  Point(178) = {-0.337263231291058, 0.05200843025992359, 0, 0.125};
#  Point(179) = {-0.3481590194623916, 0.05332726256406103, 0, 0.125};
#  Point(180) = {-0.3587356108043638, 0.05465664801682354, 0, 0.125};
#  Point(181) = {-0.3689816256782782, 0.0560028872679817, 0, 0.125};
#  Point(182) = {-0.3788857734692287, 0.05737205908247899, 0, 0.125};
#  Point(183) = {-0.3884368690074614, 0.05877000537646382, 0, 0.125};
#  Point(184) = {-0.3976238513788748, 0.06020231838219783, 0, 0.125};
#  Point(185) = {-0.40643580495675, 0.06167432980291591, 0, 0.125};
#  Point(186) = {-0.4148619824472646, 0.06319110180426264, 0, 0.125};
#  Point(187) = {-0.4228918297057104, 0.06475741967717524, 0, 0.125};
#  Point(188) = {-0.43051501204915, 0.06637778599795482, 0, 0.125};
#  Point(189) = {-0.4377214417649294, 0.06805641610468524, 0, 0.125};
#  Point(190) = {-0.4445013064933708, 0.06979723470503821, 0, 0.125};
#  Point(191) = {-0.4508450981473512, 0.07160387342876083, 0, 0.125};
#  Point(192) = {-0.4567436420215075, 0.073479669138689, 0, 0.125};
#  Point(193) = {-0.4621881257395756, 0.07542766281688272, 0, 0.125};
#  Point(194) = {-0.4671701276898881, 0.07745059884734995, 0, 0.125};
#  Point(195) = {-0.471681644606229, 0.07955092452372269, 0, 0.125};
#  Point(196) = {-0.4757151179639407, 0.0817307896190848, 0, 0.125};
#  Point(197) = {-0.4792634588791559, 0.0839920458658267, 0, 0.125};
#  Point(198) = {-0.4823200712220043, 0.08633624620581726, 0, 0.125};
#  Point(199) = {-0.4848788726822436, 0.08876464368523246, 0, 0.125};
#  Point(200) = {-0.4869343135575803, 0.09127818988394577, 0, 0.125};
#  Point(201) = {-0.4884813930704814, 0.09387753278635144, 0, 0.125};
#  Point(202) = {-0.4895156730580155, 0.09656301401871749, 0, 0.125};
#  
#  // splines of the airfoil
#  Spline(5) = {5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104};
#  Spline(6) = {104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,5};
#  
#  // airfoil
#  Line Loop(9) = {5, 6};
#  // complete domain
#  Plane Surface(1) = {8, 9};
#  
#  // labeling of the boundary parts
#  Physical Line(1) = {4};      // inflow
#  Physical Line(2) = {2};      // outflow
#  Physical Line(3) = {1, 3};   // airfoil
#  Physical Line(4) = {5, 6};   // upper/lower wall
#  Physical Surface(1) = {10};
# ```
# From which we can construct a mesh like this:
# ![mesh_screenshot](https://github.com/trixi-framework/Trixi.jl/assets/75639095/67adfe3d-d403-4cd3-acaa-971a34df0709)
#
# The first four points define the bounding box = (near-field) domain:
# ```c++
#   // outer bounding box
# Point(1) = {-1.25, -0.5, 0, 1.0};
# Point(2) = {1.25, -0.5, 0, 1.0};
# Point(3) = {1.25, 0.5, 0, 1.0};
# Point(4) = {-1.25, 0.5, 0, 1.0};
# ```
# which is constructed from connecting the points in lines:
# ```c++
# // outer box
# Line(1) = {1, 2};
# Line(2) = {2, 3};
# Line(3) = {3, 4};
# Line(4) = {4, 1};
# // outer box
# Line Loop(8) = {1, 2, 3, 4};
# ```
#
# This is followed by a couple (in principle optional) settings where the most important one is
# ```c++
# // Insist on quads instead of default triangles
# Mesh.RecombineAll = 1;
# ```
# which forces `gmsh` to generate quadrilateral elements instead of the default triangles.
# This is strictly required to be able to use the mesh later with `p4est`, which supports only straight-sided quads,
# i.e., `C2D4, CPS4, S4` in 2D and `C3D` in 3D.
# See for more details the (short) [documentation](https://p4est.github.io/p4est-howto.pdf) on the interaction of `p4est` with `.inp` files.
# In principle, it should also be possible to use the `recombine` function of `gmsh` to convert the triangles to quads, 
# but this is observed to be less robust than enforcing quads from the beginning.
#
# Then the airfoil is defined by a set of points:
# ```c++
# // points of the airfoil contour
#  Point(5) = {-0.4900332889206208, 0.09933466539753061, 0, 0.125};
#  Point(6) = {-0.4900274857651495, 0.1021542752054094, 0, 0.125};
#  ...
# ```
# which are connected by splines for the upper and lower part of the airfoil:
# ```c++
# // splines of the airfoil
#  Spline(5) = {5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
#               ...
#               96,97,98,99,100,101,102,103,104};
#  Spline(6) = {104,105,106,107,108,109,110,111,112,113,114,115,
#               ...
#               200,201,202,5};
# ```
# which are then connected to form a single line loop for easy physical group assignment:
# ```c++
# // airfoil
#  Line Loop(9) = {5, 6};
# ```
#
# At the end of the file the physical groups are defined:
# ```c++
# // labeling of the boundary parts
#  Physical Line(1) = {4};      // Inflow. Label in Abaqus .inp file: PhysicalLine1
#  Physical Line(2) = {2};      // Outflow. Label in Abaqus .inp file: PhysicalLine2
#  Physical Line(3) = {1, 3};   // Airfoil. Label in Abaqus .inp file: PhysicalLine3
#  Physical Line(4) = {5, 6};   //Upper and lower wall/farfield/... Label in Abaqus .inp file: PhysicalLine4
# ```
# which are crucial for the correct assignment of boundary conditions in `Trixi.jl`.
# In particular, it is the responsibility of a user to keep track on the physical boundary names between the mesh generation and assignment of boundary condition functions in an elixir.
#
# After opening this file in `gmsh`, meshing the geometry and exporting to Abaqus `.inp` format, 
# we can have a look at the input file:
# ```
# *Heading
#  <something depending on gmsh>
# *NODE
# 1, -1.25, -0.5, 0
# 2, 1.25, -0.5, 0
# 3, 1.25, 0.5, 0
# 4, -1.25, 0.5, 0
# ...
# ******* E L E M E N T S *************
# *ELEMENT, type=T3D2, ELSET=Line1
# 1, 1, 7
# ...
# *ELEMENT, type=CPS4, ELSET=Surface1
# 191, 272, 46, 263, 807
# ...
# *NSET,NSET=PhysicalLine1
# 1, 4, 52, 53, 54, 55, 56, 57, 58, 
# *NSET,NSET=PhysicalLine2
# 2, 3, 26, 27, 28, 29, 30, 31, 32, 
# *NSET,NSET=PhysicalLine3
# 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 
# 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 
# 23, 24, 25, 33, 34, 35, 36, 37, 38, 39, 
# 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 
# 50, 51, 
# *NSET,NSET=PhysicalLine4
# 5, 6, 59, 60, 61, 62, 63, 64, 65, 66, 
# 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 
# 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 
# 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 
# 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 
# 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 
# 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 
# 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 
# 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 
# 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 
# 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 
# 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 
# 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 
# 187, 188, 189, 190,
# ```
#
# First, the coordinates of the nodes are listed, followed by the elements.
# Note that `gmsh` exports also line elements of type `T3D2` which are ignored by `p4est`.
# The relevant elements in 2D which form the gridcells are of type `CPS4` which are defined by their four corner nodes.
# This is followed by the nodesets encoded via `*NSET` which are used to assign boundary conditions in Trixi.jl.
# Trixi.jl parses the `.inp` file and assigns the edges (in 2D, surfaces in 3D) of elements to the corresponding boundary condition based on 
# the supplied `boundary_symbols` that have to be supplied to the `P4estMesh` constructor:
# ```julia
# # boundary symbols
# boundary_symbols = [:PhysicalLine1, :PhysicalLine2, :PhysicalLine3, :PhysicalLine4]
# mesh = P4estMesh{2}(mesh_file, polydeg = polydeg, boundary_symbols = boundary_symbols)
# ```
# The same boundary symbols have then also be supplied to the semidiscretization alongside the
# corresponding physical boundary conditions:
# ```julia
# # Supersonic inflow boundary condition.
# # Calculate the boundary flux entirely from the external solution state, i.e., set
# # external solution state values for everything entering the domain.
# @inline function boundary_condition_supersonic_inflow(u_inner,
#                                                       normal_direction::AbstractVector,
#                                                       x, t, surface_flux_function,
#                                                       equations::CompressibleEulerEquations2D)
#     u_boundary = initial_condition_mach2_flow(x, t, equations)
#     flux = Trixi.flux(u_boundary, normal_direction, equations)
#
#     return flux
# end
#
# # Supersonic outflow boundary condition.
# # Calculate the boundary flux entirely from the internal solution state. Analogous to supersonic inflow
# # except all the solution state values are set from the internal solution as everything leaves the domain
# @inline function boundary_condition_supersonic_outflow(u_inner,
#                                                        normal_direction::AbstractVector, x,
#                                                        t,
#                                                        surface_flux_function,
#                                                        equations::CompressibleEulerEquations2D)
# flux = Trixi.flux(u_inner, normal_direction, equations)
#
# boundary_conditions = Dict(:PhysicalLine1 => boundary_condition_supersonic_inflow, # Left boundary
#                            :PhysicalLine2 => boundary_condition_supersonic_outflow, # Right boundary
#                            :PhysicalLine3 => boundary_condition_slip_wall, # Airfoil
#                            :PhysicalLine4 => boundary_condition_supersonic_outflow) # Top and bottom boundary
# 
# semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
#                                     boundary_conditions = boundary_conditions)
# ```
# Note that you **have to** supply the `boundary_symbols` keyword to the `P4estMesh` constructor 
# to select the boundaries from the available nodesets in the `.inp` file.
# If the `boundary_symbols` keyword is not supplied, all boundaries will be assigned to the default set `:all`.

# ## Package versions

# These results were obtained using the following versions.

using InteractiveUtils
versioninfo()

using Pkg
Pkg.status(["Trixi", "OrdinaryDiffEq", "Plots", "Download"],
           mode=PKGMODE_MANIFEST)
