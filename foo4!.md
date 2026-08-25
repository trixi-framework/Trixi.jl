```julia
CodeInfo(
1 ──── %1     = Base.Threads.cglobal(:jl_n_threads_per_pool, Ptr{Int32})::Ptr{Ptr{Int32}}
│      %2     = Base.pointerref(%1, 1, 1)::Ptr{Int32}
│      %3     = Base.pointerref(%2, 2, 1)::Int32
│      %4     = Core.sext_int(Core.Int64, %3)::Int64
│      %5     = (%4 === 1)::Bool
└─────          goto #1196 if not %5
2 ──── %7     = Base.getfield(cache, :elements)::Trixi.TreeElementContainer3D{Float64, Float64}
│      %8     = Base.getfield(%7, :cell_ids)::Vector{Int64}
│      %9     = Base.arraylen(%8)::Int64
│      %10    = Base.slt_int(%9, 0)::Bool
│      %11    = Core.ifelse(%10, 0, %9)::Int64
│      %12    = Base.slt_int(%11, 1)::Bool
└─────          goto #4 if not %12
3 ────          goto #5
4 ────          goto #5
5 ┄─── %16    = φ (#3 => true, #4 => false)::Bool
│      %17    = φ (#4 => 1)::Int64
│      %18    = φ (#4 => 1)::Int64
│      %19    = Base.not_int(%16)::Bool
└─────          goto #1195 if not %19
6 ┄─── %21    = φ (#5 => %17, #1194 => %3456)::Int64
│      %22    = φ (#5 => %18, #1194 => %3457)::Int64
│      %23    = Base.getfield(dg, :basis)::LobattoLegendreBasis{Float64, 4, SVector{4, Float64}, Matrix{Float64}, Matrix{Float64}}
│      %24    = Base.getfield(%23, :derivative_split)::Matrix{Float64}
└─────          goto #1187 if not true
7 ┄─── %26    = φ (#6 => 1, #1186 => %3442)::Int64
│      %27    = φ (#6 => 1, #1186 => %3443)::Int64
└─────          goto #1182 if not true
8 ┄─── %29    = φ (#7 => 1, #1181 => %3432)::Int64
│      %30    = φ (#7 => 1, #1181 => %3431)::Int64
└─────          goto #1177 if not true
9 ┄─── %32    = φ (#8 => 1, #1176 => %3421)::Int64
│      %33    = φ (#8 => 1, #1176 => %3420)::Int64
│      %34    = StrideArraysCore.getfield(u, :data)::Vector{Float64}
│      %35    = $(Expr(:gc_preserve_begin, :(%34)))
│      %36    = StrideArraysCore.getfield(u, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #14 if not true
10 ─── %38    = Core.tuple(1, %32, %29, %26, %21)::NTuple{5, Int64}
│      %39    = StrideArraysCore.getfield(%36, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %40    = Core.getfield(%39, 5)::Int64
│      %41    = Base.bitcast(UInt64, %40)::UInt64
│      %42    = Base.bitcast(Int64, %41)::Int64
│      %43    = Base.sle_int(1, %32)::Bool
│      %44    = Base.sle_int(%32, 4)::Bool
│      %45    = Base.and_int(%43, %44)::Bool
│      %46    = Base.sle_int(1, %29)::Bool
│      %47    = Base.sle_int(%29, 4)::Bool
│      %48    = Base.and_int(%46, %47)::Bool
│      %49    = Base.sle_int(1, %26)::Bool
│      %50    = Base.sle_int(%26, 4)::Bool
│      %51    = Base.and_int(%49, %50)::Bool
│      %52    = Base.sub_int(%21, 1)::Int64
│      %53    = Base.bitcast(UInt64, %52)::UInt64
│      %54    = Base.bitcast(UInt64, %42)::UInt64
│      %55    = Base.ult_int(%53, %54)::Bool
│      %56    = Base.and_int(%55, true)::Bool
│      %57    = Base.and_int(%51, %56)::Bool
│      %58    = Base.and_int(%48, %57)::Bool
│      %59    = Base.and_int(%45, %58)::Bool
│      %60    = Base.and_int(true, %59)::Bool
└─────          goto #12 if not %60
11 ───          goto #13
12 ───          invoke Base.throw_boundserror(%36::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %38::NTuple{5, Int64})::Union{}
└─────          unreachable
13 ───          nothing::Nothing
14 ┄── %66    = StrideArraysCore.getfield(%36, :ptr)::Ptr{Float64}
│      %67    = Base.sub_int(%32, 1)::Int64
│      %68    = Base.sub_int(%29, 1)::Int64
│      %69    = Base.sub_int(%26, 1)::Int64
│      %70    = Base.sub_int(%21, 1)::Int64
└─────          goto #23 if not true
15 ┄── %72    = φ (#14 => 2, #22 => %84)::Int64
│      %73    = Base.sle_int(1, %72)::Bool
└─────          goto #17 if not %73
16 ─── %75    = Base.sle_int(%72, 5)::Bool
└─────          goto #18
17 ───          nothing::Nothing
18 ┄── %78    = φ (#16 => %75, #17 => false)::Bool
└─────          goto #20 if not %78
19 ───          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %72, true)::Static.True
│      %81    = Base.add_int(%72, 1)::Int64
└─────          goto #21
20 ───          goto #21
21 ┄── %84    = φ (#19 => %81)::Int64
│      %85    = φ (#19 => false, #20 => true)::Bool
│      %86    = Base.not_int(%85)::Bool
└─────          goto #23 if not %86
22 ───          goto #15
23 ┄──          goto #24
24 ───          goto #25
25 ─── %91    = Base.mul_int(%70, 4)::Int64
│      %92    = Base.add_int(%69, %91)::Int64
│      %93    = Base.mul_int(%92, 4)::Int64
│      %94    = Base.add_int(%68, %93)::Int64
│      %95    = Base.mul_int(%94, 4)::Int64
│      %96    = Base.add_int(%67, %95)::Int64
│      %97    = Base.mul_int(%96, 5)::Int64
│      %98    = Base.add_int(0, %97)::Int64
│      %99    = Base.mul_int(8, %98)::Int64
│      %100   = Core.bitcast(Core.UInt, %66)::UInt64
│      %101   = Base.bitcast(UInt64, %99)::UInt64
│      %102   = Base.add_ptr(%100, %101)::UInt64
│      %103   = Core.bitcast(Ptr{Float64}, %102)::Ptr{Float64}
└─────          goto #26
26 ─── %105   = Base.pointerref(%103, 1, 1)::Float64
└─────          goto #27
27 ───          goto #28
28 ───          $(Expr(:gc_preserve_end, :(%35)))
└─────          goto #29
29 ───          goto #30
30 ─── %111   = StrideArraysCore.getfield(u, :data)::Vector{Float64}
│      %112   = $(Expr(:gc_preserve_begin, :(%111)))
│      %113   = StrideArraysCore.getfield(u, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #35 if not true
31 ─── %115   = Core.tuple(2, %32, %29, %26, %21)::NTuple{5, Int64}
│      %116   = StrideArraysCore.getfield(%113, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %117   = Core.getfield(%116, 5)::Int64
│      %118   = Base.bitcast(UInt64, %117)::UInt64
│      %119   = Base.bitcast(Int64, %118)::Int64
│      %120   = Base.sle_int(1, %32)::Bool
│      %121   = Base.sle_int(%32, 4)::Bool
│      %122   = Base.and_int(%120, %121)::Bool
│      %123   = Base.sle_int(1, %29)::Bool
│      %124   = Base.sle_int(%29, 4)::Bool
│      %125   = Base.and_int(%123, %124)::Bool
│      %126   = Base.sle_int(1, %26)::Bool
│      %127   = Base.sle_int(%26, 4)::Bool
│      %128   = Base.and_int(%126, %127)::Bool
│      %129   = Base.sub_int(%21, 1)::Int64
│      %130   = Base.bitcast(UInt64, %129)::UInt64
│      %131   = Base.bitcast(UInt64, %119)::UInt64
│      %132   = Base.ult_int(%130, %131)::Bool
│      %133   = Base.and_int(%132, true)::Bool
│      %134   = Base.and_int(%128, %133)::Bool
│      %135   = Base.and_int(%125, %134)::Bool
│      %136   = Base.and_int(%122, %135)::Bool
│      %137   = Base.and_int(true, %136)::Bool
└─────          goto #33 if not %137
32 ───          goto #34
33 ───          invoke Base.throw_boundserror(%113::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %115::NTuple{5, Int64})::Union{}
└─────          unreachable
34 ───          nothing::Nothing
35 ┄── %143   = StrideArraysCore.getfield(%113, :ptr)::Ptr{Float64}
│      %144   = Base.sub_int(%32, 1)::Int64
│      %145   = Base.sub_int(%29, 1)::Int64
│      %146   = Base.sub_int(%26, 1)::Int64
│      %147   = Base.sub_int(%21, 1)::Int64
└─────          goto #44 if not true
36 ┄── %149   = φ (#35 => 2, #43 => %161)::Int64
│      %150   = Base.sle_int(1, %149)::Bool
└─────          goto #38 if not %150
37 ─── %152   = Base.sle_int(%149, 5)::Bool
└─────          goto #39
38 ───          nothing::Nothing
39 ┄── %155   = φ (#37 => %152, #38 => false)::Bool
└─────          goto #41 if not %155
40 ───          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %149, true)::Static.True
│      %158   = Base.add_int(%149, 1)::Int64
└─────          goto #42
41 ───          goto #42
42 ┄── %161   = φ (#40 => %158)::Int64
│      %162   = φ (#40 => false, #41 => true)::Bool
│      %163   = Base.not_int(%162)::Bool
└─────          goto #44 if not %163
43 ───          goto #36
44 ┄──          goto #45
45 ───          goto #46
46 ─── %168   = Base.mul_int(%147, 4)::Int64
│      %169   = Base.add_int(%146, %168)::Int64
│      %170   = Base.mul_int(%169, 4)::Int64
│      %171   = Base.add_int(%145, %170)::Int64
│      %172   = Base.mul_int(%171, 4)::Int64
│      %173   = Base.add_int(%144, %172)::Int64
│      %174   = Base.mul_int(%173, 5)::Int64
│      %175   = Base.add_int(1, %174)::Int64
│      %176   = Base.mul_int(8, %175)::Int64
│      %177   = Core.bitcast(Core.UInt, %143)::UInt64
│      %178   = Base.bitcast(UInt64, %176)::UInt64
│      %179   = Base.add_ptr(%177, %178)::UInt64
│      %180   = Core.bitcast(Ptr{Float64}, %179)::Ptr{Float64}
└─────          goto #47
47 ─── %182   = Base.pointerref(%180, 1, 1)::Float64
└─────          goto #48
48 ───          goto #49
49 ───          $(Expr(:gc_preserve_end, :(%112)))
└─────          goto #50
50 ───          goto #51
51 ─── %188   = StrideArraysCore.getfield(u, :data)::Vector{Float64}
│      %189   = $(Expr(:gc_preserve_begin, :(%188)))
│      %190   = StrideArraysCore.getfield(u, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #56 if not true
52 ─── %192   = Core.tuple(3, %32, %29, %26, %21)::NTuple{5, Int64}
│      %193   = StrideArraysCore.getfield(%190, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %194   = Core.getfield(%193, 5)::Int64
│      %195   = Base.bitcast(UInt64, %194)::UInt64
│      %196   = Base.bitcast(Int64, %195)::Int64
│      %197   = Base.sle_int(1, %32)::Bool
│      %198   = Base.sle_int(%32, 4)::Bool
│      %199   = Base.and_int(%197, %198)::Bool
│      %200   = Base.sle_int(1, %29)::Bool
│      %201   = Base.sle_int(%29, 4)::Bool
│      %202   = Base.and_int(%200, %201)::Bool
│      %203   = Base.sle_int(1, %26)::Bool
│      %204   = Base.sle_int(%26, 4)::Bool
│      %205   = Base.and_int(%203, %204)::Bool
│      %206   = Base.sub_int(%21, 1)::Int64
│      %207   = Base.bitcast(UInt64, %206)::UInt64
│      %208   = Base.bitcast(UInt64, %196)::UInt64
│      %209   = Base.ult_int(%207, %208)::Bool
│      %210   = Base.and_int(%209, true)::Bool
│      %211   = Base.and_int(%205, %210)::Bool
│      %212   = Base.and_int(%202, %211)::Bool
│      %213   = Base.and_int(%199, %212)::Bool
│      %214   = Base.and_int(true, %213)::Bool
└─────          goto #54 if not %214
53 ───          goto #55
54 ───          invoke Base.throw_boundserror(%190::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %192::NTuple{5, Int64})::Union{}
└─────          unreachable
55 ───          nothing::Nothing
56 ┄── %220   = StrideArraysCore.getfield(%190, :ptr)::Ptr{Float64}
│      %221   = Base.sub_int(%32, 1)::Int64
│      %222   = Base.sub_int(%29, 1)::Int64
│      %223   = Base.sub_int(%26, 1)::Int64
│      %224   = Base.sub_int(%21, 1)::Int64
└─────          goto #65 if not true
57 ┄── %226   = φ (#56 => 2, #64 => %238)::Int64
│      %227   = Base.sle_int(1, %226)::Bool
└─────          goto #59 if not %227
58 ─── %229   = Base.sle_int(%226, 5)::Bool
└─────          goto #60
59 ───          nothing::Nothing
60 ┄── %232   = φ (#58 => %229, #59 => false)::Bool
└─────          goto #62 if not %232
61 ───          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %226, true)::Static.True
│      %235   = Base.add_int(%226, 1)::Int64
└─────          goto #63
62 ───          goto #63
63 ┄── %238   = φ (#61 => %235)::Int64
│      %239   = φ (#61 => false, #62 => true)::Bool
│      %240   = Base.not_int(%239)::Bool
└─────          goto #65 if not %240
64 ───          goto #57
65 ┄──          goto #66
66 ───          goto #67
67 ─── %245   = Base.mul_int(%224, 4)::Int64
│      %246   = Base.add_int(%223, %245)::Int64
│      %247   = Base.mul_int(%246, 4)::Int64
│      %248   = Base.add_int(%222, %247)::Int64
│      %249   = Base.mul_int(%248, 4)::Int64
│      %250   = Base.add_int(%221, %249)::Int64
│      %251   = Base.mul_int(%250, 5)::Int64
│      %252   = Base.add_int(2, %251)::Int64
│      %253   = Base.mul_int(8, %252)::Int64
│      %254   = Core.bitcast(Core.UInt, %220)::UInt64
│      %255   = Base.bitcast(UInt64, %253)::UInt64
│      %256   = Base.add_ptr(%254, %255)::UInt64
│      %257   = Core.bitcast(Ptr{Float64}, %256)::Ptr{Float64}
└─────          goto #68
68 ─── %259   = Base.pointerref(%257, 1, 1)::Float64
└─────          goto #69
69 ───          goto #70
70 ───          $(Expr(:gc_preserve_end, :(%189)))
└─────          goto #71
71 ───          goto #72
72 ─── %265   = StrideArraysCore.getfield(u, :data)::Vector{Float64}
│      %266   = $(Expr(:gc_preserve_begin, :(%265)))
│      %267   = StrideArraysCore.getfield(u, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #77 if not true
73 ─── %269   = Core.tuple(4, %32, %29, %26, %21)::NTuple{5, Int64}
│      %270   = StrideArraysCore.getfield(%267, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %271   = Core.getfield(%270, 5)::Int64
│      %272   = Base.bitcast(UInt64, %271)::UInt64
│      %273   = Base.bitcast(Int64, %272)::Int64
│      %274   = Base.sle_int(1, %32)::Bool
│      %275   = Base.sle_int(%32, 4)::Bool
│      %276   = Base.and_int(%274, %275)::Bool
│      %277   = Base.sle_int(1, %29)::Bool
│      %278   = Base.sle_int(%29, 4)::Bool
│      %279   = Base.and_int(%277, %278)::Bool
│      %280   = Base.sle_int(1, %26)::Bool
│      %281   = Base.sle_int(%26, 4)::Bool
│      %282   = Base.and_int(%280, %281)::Bool
│      %283   = Base.sub_int(%21, 1)::Int64
│      %284   = Base.bitcast(UInt64, %283)::UInt64
│      %285   = Base.bitcast(UInt64, %273)::UInt64
│      %286   = Base.ult_int(%284, %285)::Bool
│      %287   = Base.and_int(%286, true)::Bool
│      %288   = Base.and_int(%282, %287)::Bool
│      %289   = Base.and_int(%279, %288)::Bool
│      %290   = Base.and_int(%276, %289)::Bool
│      %291   = Base.and_int(true, %290)::Bool
└─────          goto #75 if not %291
74 ───          goto #76
75 ───          invoke Base.throw_boundserror(%267::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %269::NTuple{5, Int64})::Union{}
└─────          unreachable
76 ───          nothing::Nothing
77 ┄── %297   = StrideArraysCore.getfield(%267, :ptr)::Ptr{Float64}
│      %298   = Base.sub_int(%32, 1)::Int64
│      %299   = Base.sub_int(%29, 1)::Int64
│      %300   = Base.sub_int(%26, 1)::Int64
│      %301   = Base.sub_int(%21, 1)::Int64
└─────          goto #86 if not true
78 ┄── %303   = φ (#77 => 2, #85 => %315)::Int64
│      %304   = Base.sle_int(1, %303)::Bool
└─────          goto #80 if not %304
79 ─── %306   = Base.sle_int(%303, 5)::Bool
└─────          goto #81
80 ───          nothing::Nothing
81 ┄── %309   = φ (#79 => %306, #80 => false)::Bool
└─────          goto #83 if not %309
82 ───          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %303, true)::Static.True
│      %312   = Base.add_int(%303, 1)::Int64
└─────          goto #84
83 ───          goto #84
84 ┄── %315   = φ (#82 => %312)::Int64
│      %316   = φ (#82 => false, #83 => true)::Bool
│      %317   = Base.not_int(%316)::Bool
└─────          goto #86 if not %317
85 ───          goto #78
86 ┄──          goto #87
87 ───          goto #88
88 ─── %322   = Base.mul_int(%301, 4)::Int64
│      %323   = Base.add_int(%300, %322)::Int64
│      %324   = Base.mul_int(%323, 4)::Int64
│      %325   = Base.add_int(%299, %324)::Int64
│      %326   = Base.mul_int(%325, 4)::Int64
│      %327   = Base.add_int(%298, %326)::Int64
│      %328   = Base.mul_int(%327, 5)::Int64
│      %329   = Base.add_int(3, %328)::Int64
│      %330   = Base.mul_int(8, %329)::Int64
│      %331   = Core.bitcast(Core.UInt, %297)::UInt64
│      %332   = Base.bitcast(UInt64, %330)::UInt64
│      %333   = Base.add_ptr(%331, %332)::UInt64
│      %334   = Core.bitcast(Ptr{Float64}, %333)::Ptr{Float64}
└─────          goto #89
89 ─── %336   = Base.pointerref(%334, 1, 1)::Float64
└─────          goto #90
90 ───          goto #91
91 ───          $(Expr(:gc_preserve_end, :(%266)))
└─────          goto #92
92 ───          goto #93
93 ─── %342   = StrideArraysCore.getfield(u, :data)::Vector{Float64}
│      %343   = $(Expr(:gc_preserve_begin, :(%342)))
│      %344   = StrideArraysCore.getfield(u, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #98 if not true
94 ─── %346   = Core.tuple(5, %32, %29, %26, %21)::NTuple{5, Int64}
│      %347   = StrideArraysCore.getfield(%344, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %348   = Core.getfield(%347, 5)::Int64
│      %349   = Base.bitcast(UInt64, %348)::UInt64
│      %350   = Base.bitcast(Int64, %349)::Int64
│      %351   = Base.sle_int(1, %32)::Bool
│      %352   = Base.sle_int(%32, 4)::Bool
│      %353   = Base.and_int(%351, %352)::Bool
│      %354   = Base.sle_int(1, %29)::Bool
│      %355   = Base.sle_int(%29, 4)::Bool
│      %356   = Base.and_int(%354, %355)::Bool
│      %357   = Base.sle_int(1, %26)::Bool
│      %358   = Base.sle_int(%26, 4)::Bool
│      %359   = Base.and_int(%357, %358)::Bool
│      %360   = Base.sub_int(%21, 1)::Int64
│      %361   = Base.bitcast(UInt64, %360)::UInt64
│      %362   = Base.bitcast(UInt64, %350)::UInt64
│      %363   = Base.ult_int(%361, %362)::Bool
│      %364   = Base.and_int(%363, true)::Bool
│      %365   = Base.and_int(%359, %364)::Bool
│      %366   = Base.and_int(%356, %365)::Bool
│      %367   = Base.and_int(%353, %366)::Bool
│      %368   = Base.and_int(true, %367)::Bool
└─────          goto #96 if not %368
95 ───          goto #97
96 ───          invoke Base.throw_boundserror(%344::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %346::NTuple{5, Int64})::Union{}
└─────          unreachable
97 ───          nothing::Nothing
98 ┄── %374   = StrideArraysCore.getfield(%344, :ptr)::Ptr{Float64}
│      %375   = Base.sub_int(%32, 1)::Int64
│      %376   = Base.sub_int(%29, 1)::Int64
│      %377   = Base.sub_int(%26, 1)::Int64
│      %378   = Base.sub_int(%21, 1)::Int64
└─────          goto #107 if not true
99 ┄── %380   = φ (#98 => 2, #106 => %392)::Int64
│      %381   = Base.sle_int(1, %380)::Bool
└─────          goto #101 if not %381
100 ── %383   = Base.sle_int(%380, 5)::Bool
└─────          goto #102
101 ──          nothing::Nothing
102 ┄─ %386   = φ (#100 => %383, #101 => false)::Bool
└─────          goto #104 if not %386
103 ──          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %380, true)::Static.True
│      %389   = Base.add_int(%380, 1)::Int64
└─────          goto #105
104 ──          goto #105
105 ┄─ %392   = φ (#103 => %389)::Int64
│      %393   = φ (#103 => false, #104 => true)::Bool
│      %394   = Base.not_int(%393)::Bool
└─────          goto #107 if not %394
106 ──          goto #99
107 ┄─          goto #108
108 ──          goto #109
109 ── %399   = Base.mul_int(%378, 4)::Int64
│      %400   = Base.add_int(%377, %399)::Int64
│      %401   = Base.mul_int(%400, 4)::Int64
│      %402   = Base.add_int(%376, %401)::Int64
│      %403   = Base.mul_int(%402, 4)::Int64
│      %404   = Base.add_int(%375, %403)::Int64
│      %405   = Base.mul_int(%404, 5)::Int64
│      %406   = Base.add_int(4, %405)::Int64
│      %407   = Base.mul_int(8, %406)::Int64
│      %408   = Core.bitcast(Core.UInt, %374)::UInt64
│      %409   = Base.bitcast(UInt64, %407)::UInt64
│      %410   = Base.add_ptr(%408, %409)::UInt64
│      %411   = Core.bitcast(Ptr{Float64}, %410)::Ptr{Float64}
└─────          goto #110
110 ── %413   = Base.pointerref(%411, 1, 1)::Float64
└─────          goto #111
111 ──          goto #112
112 ──          $(Expr(:gc_preserve_end, :(%343)))
└─────          goto #113
113 ──          goto #114
114 ──          goto #115
115 ──          goto #116
116 ── %421   = Base.add_int(%32, 1)::Int64
│      %422   = Base.sle_int(%421, 4)::Bool
└─────          goto #118 if not %422
117 ──          goto #119
118 ── %425   = Base.sub_int(%421, 1)::Int64
└─────          goto #119
119 ┄─ %427   = φ (#117 => 4, #118 => %425)::Int64
└─────          goto #120
120 ──          goto #121
121 ── %430   = Base.slt_int(%427, %421)::Bool
└─────          goto #123 if not %430
122 ──          goto #124
123 ──          goto #124
124 ┄─ %434   = φ (#122 => true, #123 => false)::Bool
│      %435   = φ (#123 => %421)::Int64
│      %436   = φ (#123 => %421)::Int64
│      %437   = Base.not_int(%434)::Bool
└─────          goto #468 if not %437
125 ┄─ %439   = φ (#124 => %435, #467 => %1413)::Int64
│      %440   = φ (#124 => %436, #467 => %1414)::Int64
│      %441   = StrideArraysCore.getfield(u, :data)::Vector{Float64}
│      %442   = $(Expr(:gc_preserve_begin, :(%441)))
│      %443   = StrideArraysCore.getfield(u, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #130 if not true
126 ── %445   = Core.tuple(1, %439, %29, %26, %21)::NTuple{5, Int64}
│      %446   = StrideArraysCore.getfield(%443, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %447   = Core.getfield(%446, 5)::Int64
│      %448   = Base.bitcast(UInt64, %447)::UInt64
│      %449   = Base.bitcast(Int64, %448)::Int64
│      %450   = Base.sle_int(1, %439)::Bool
│      %451   = Base.sle_int(%439, 4)::Bool
│      %452   = Base.and_int(%450, %451)::Bool
│      %453   = Base.sle_int(1, %29)::Bool
│      %454   = Base.sle_int(%29, 4)::Bool
│      %455   = Base.and_int(%453, %454)::Bool
│      %456   = Base.sle_int(1, %26)::Bool
│      %457   = Base.sle_int(%26, 4)::Bool
│      %458   = Base.and_int(%456, %457)::Bool
│      %459   = Base.sub_int(%21, 1)::Int64
│      %460   = Base.bitcast(UInt64, %459)::UInt64
│      %461   = Base.bitcast(UInt64, %449)::UInt64
│      %462   = Base.ult_int(%460, %461)::Bool
│      %463   = Base.and_int(%462, true)::Bool
│      %464   = Base.and_int(%458, %463)::Bool
│      %465   = Base.and_int(%455, %464)::Bool
│      %466   = Base.and_int(%452, %465)::Bool
│      %467   = Base.and_int(true, %466)::Bool
└─────          goto #128 if not %467
127 ──          goto #129
128 ──          invoke Base.throw_boundserror(%443::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %445::NTuple{5, Int64})::Union{}
└─────          unreachable
129 ──          nothing::Nothing
130 ┄─ %473   = StrideArraysCore.getfield(%443, :ptr)::Ptr{Float64}
│      %474   = Base.sub_int(%439, 1)::Int64
│      %475   = Base.sub_int(%29, 1)::Int64
│      %476   = Base.sub_int(%26, 1)::Int64
│      %477   = Base.sub_int(%21, 1)::Int64
└─────          goto #139 if not true
131 ┄─ %479   = φ (#130 => 2, #138 => %491)::Int64
│      %480   = Base.sle_int(1, %479)::Bool
└─────          goto #133 if not %480
132 ── %482   = Base.sle_int(%479, 5)::Bool
└─────          goto #134
133 ──          nothing::Nothing
134 ┄─ %485   = φ (#132 => %482, #133 => false)::Bool
└─────          goto #136 if not %485
135 ──          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %479, true)::Static.True
│      %488   = Base.add_int(%479, 1)::Int64
└─────          goto #137
136 ──          goto #137
137 ┄─ %491   = φ (#135 => %488)::Int64
│      %492   = φ (#135 => false, #136 => true)::Bool
│      %493   = Base.not_int(%492)::Bool
└─────          goto #139 if not %493
138 ──          goto #131
139 ┄─          goto #140
140 ──          goto #141
141 ── %498   = Base.mul_int(%477, 4)::Int64
│      %499   = Base.add_int(%476, %498)::Int64
│      %500   = Base.mul_int(%499, 4)::Int64
│      %501   = Base.add_int(%475, %500)::Int64
│      %502   = Base.mul_int(%501, 4)::Int64
│      %503   = Base.add_int(%474, %502)::Int64
│      %504   = Base.mul_int(%503, 5)::Int64
│      %505   = Base.add_int(0, %504)::Int64
│      %506   = Base.mul_int(8, %505)::Int64
│      %507   = Core.bitcast(Core.UInt, %473)::UInt64
│      %508   = Base.bitcast(UInt64, %506)::UInt64
│      %509   = Base.add_ptr(%507, %508)::UInt64
│      %510   = Core.bitcast(Ptr{Float64}, %509)::Ptr{Float64}
└─────          goto #142
142 ── %512   = Base.pointerref(%510, 1, 1)::Float64
└─────          goto #143
143 ──          goto #144
144 ──          $(Expr(:gc_preserve_end, :(%442)))
└─────          goto #145
145 ──          goto #146
146 ── %518   = StrideArraysCore.getfield(u, :data)::Vector{Float64}
│      %519   = $(Expr(:gc_preserve_begin, :(%518)))
│      %520   = StrideArraysCore.getfield(u, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #151 if not true
147 ── %522   = Core.tuple(2, %439, %29, %26, %21)::NTuple{5, Int64}
│      %523   = StrideArraysCore.getfield(%520, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %524   = Core.getfield(%523, 5)::Int64
│      %525   = Base.bitcast(UInt64, %524)::UInt64
│      %526   = Base.bitcast(Int64, %525)::Int64
│      %527   = Base.sle_int(1, %439)::Bool
│      %528   = Base.sle_int(%439, 4)::Bool
│      %529   = Base.and_int(%527, %528)::Bool
│      %530   = Base.sle_int(1, %29)::Bool
│      %531   = Base.sle_int(%29, 4)::Bool
│      %532   = Base.and_int(%530, %531)::Bool
│      %533   = Base.sle_int(1, %26)::Bool
│      %534   = Base.sle_int(%26, 4)::Bool
│      %535   = Base.and_int(%533, %534)::Bool
│      %536   = Base.sub_int(%21, 1)::Int64
│      %537   = Base.bitcast(UInt64, %536)::UInt64
│      %538   = Base.bitcast(UInt64, %526)::UInt64
│      %539   = Base.ult_int(%537, %538)::Bool
│      %540   = Base.and_int(%539, true)::Bool
│      %541   = Base.and_int(%535, %540)::Bool
│      %542   = Base.and_int(%532, %541)::Bool
│      %543   = Base.and_int(%529, %542)::Bool
│      %544   = Base.and_int(true, %543)::Bool
└─────          goto #149 if not %544
148 ──          goto #150
149 ──          invoke Base.throw_boundserror(%520::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %522::NTuple{5, Int64})::Union{}
└─────          unreachable
150 ──          nothing::Nothing
151 ┄─ %550   = StrideArraysCore.getfield(%520, :ptr)::Ptr{Float64}
│      %551   = Base.sub_int(%439, 1)::Int64
│      %552   = Base.sub_int(%29, 1)::Int64
│      %553   = Base.sub_int(%26, 1)::Int64
│      %554   = Base.sub_int(%21, 1)::Int64
└─────          goto #160 if not true
152 ┄─ %556   = φ (#151 => 2, #159 => %568)::Int64
│      %557   = Base.sle_int(1, %556)::Bool
└─────          goto #154 if not %557
153 ── %559   = Base.sle_int(%556, 5)::Bool
└─────          goto #155
154 ──          nothing::Nothing
155 ┄─ %562   = φ (#153 => %559, #154 => false)::Bool
└─────          goto #157 if not %562
156 ──          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %556, true)::Static.True
│      %565   = Base.add_int(%556, 1)::Int64
└─────          goto #158
157 ──          goto #158
158 ┄─ %568   = φ (#156 => %565)::Int64
│      %569   = φ (#156 => false, #157 => true)::Bool
│      %570   = Base.not_int(%569)::Bool
└─────          goto #160 if not %570
159 ──          goto #152
160 ┄─          goto #161
161 ──          goto #162
162 ── %575   = Base.mul_int(%554, 4)::Int64
│      %576   = Base.add_int(%553, %575)::Int64
│      %577   = Base.mul_int(%576, 4)::Int64
│      %578   = Base.add_int(%552, %577)::Int64
│      %579   = Base.mul_int(%578, 4)::Int64
│      %580   = Base.add_int(%551, %579)::Int64
│      %581   = Base.mul_int(%580, 5)::Int64
│      %582   = Base.add_int(1, %581)::Int64
│      %583   = Base.mul_int(8, %582)::Int64
│      %584   = Core.bitcast(Core.UInt, %550)::UInt64
│      %585   = Base.bitcast(UInt64, %583)::UInt64
│      %586   = Base.add_ptr(%584, %585)::UInt64
│      %587   = Core.bitcast(Ptr{Float64}, %586)::Ptr{Float64}
└─────          goto #163
163 ── %589   = Base.pointerref(%587, 1, 1)::Float64
└─────          goto #164
164 ──          goto #165
165 ──          $(Expr(:gc_preserve_end, :(%519)))
└─────          goto #166
166 ──          goto #167
167 ── %595   = StrideArraysCore.getfield(u, :data)::Vector{Float64}
│      %596   = $(Expr(:gc_preserve_begin, :(%595)))
│      %597   = StrideArraysCore.getfield(u, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #172 if not true
168 ── %599   = Core.tuple(3, %439, %29, %26, %21)::NTuple{5, Int64}
│      %600   = StrideArraysCore.getfield(%597, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %601   = Core.getfield(%600, 5)::Int64
│      %602   = Base.bitcast(UInt64, %601)::UInt64
│      %603   = Base.bitcast(Int64, %602)::Int64
│      %604   = Base.sle_int(1, %439)::Bool
│      %605   = Base.sle_int(%439, 4)::Bool
│      %606   = Base.and_int(%604, %605)::Bool
│      %607   = Base.sle_int(1, %29)::Bool
│      %608   = Base.sle_int(%29, 4)::Bool
│      %609   = Base.and_int(%607, %608)::Bool
│      %610   = Base.sle_int(1, %26)::Bool
│      %611   = Base.sle_int(%26, 4)::Bool
│      %612   = Base.and_int(%610, %611)::Bool
│      %613   = Base.sub_int(%21, 1)::Int64
│      %614   = Base.bitcast(UInt64, %613)::UInt64
│      %615   = Base.bitcast(UInt64, %603)::UInt64
│      %616   = Base.ult_int(%614, %615)::Bool
│      %617   = Base.and_int(%616, true)::Bool
│      %618   = Base.and_int(%612, %617)::Bool
│      %619   = Base.and_int(%609, %618)::Bool
│      %620   = Base.and_int(%606, %619)::Bool
│      %621   = Base.and_int(true, %620)::Bool
└─────          goto #170 if not %621
169 ──          goto #171
170 ──          invoke Base.throw_boundserror(%597::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %599::NTuple{5, Int64})::Union{}
└─────          unreachable
171 ──          nothing::Nothing
172 ┄─ %627   = StrideArraysCore.getfield(%597, :ptr)::Ptr{Float64}
│      %628   = Base.sub_int(%439, 1)::Int64
│      %629   = Base.sub_int(%29, 1)::Int64
│      %630   = Base.sub_int(%26, 1)::Int64
│      %631   = Base.sub_int(%21, 1)::Int64
└─────          goto #181 if not true
173 ┄─ %633   = φ (#172 => 2, #180 => %645)::Int64
│      %634   = Base.sle_int(1, %633)::Bool
└─────          goto #175 if not %634
174 ── %636   = Base.sle_int(%633, 5)::Bool
└─────          goto #176
175 ──          nothing::Nothing
176 ┄─ %639   = φ (#174 => %636, #175 => false)::Bool
└─────          goto #178 if not %639
177 ──          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %633, true)::Static.True
│      %642   = Base.add_int(%633, 1)::Int64
└─────          goto #179
178 ──          goto #179
179 ┄─ %645   = φ (#177 => %642)::Int64
│      %646   = φ (#177 => false, #178 => true)::Bool
│      %647   = Base.not_int(%646)::Bool
└─────          goto #181 if not %647
180 ──          goto #173
181 ┄─          goto #182
182 ──          goto #183
183 ── %652   = Base.mul_int(%631, 4)::Int64
│      %653   = Base.add_int(%630, %652)::Int64
│      %654   = Base.mul_int(%653, 4)::Int64
│      %655   = Base.add_int(%629, %654)::Int64
│      %656   = Base.mul_int(%655, 4)::Int64
│      %657   = Base.add_int(%628, %656)::Int64
│      %658   = Base.mul_int(%657, 5)::Int64
│      %659   = Base.add_int(2, %658)::Int64
│      %660   = Base.mul_int(8, %659)::Int64
│      %661   = Core.bitcast(Core.UInt, %627)::UInt64
│      %662   = Base.bitcast(UInt64, %660)::UInt64
│      %663   = Base.add_ptr(%661, %662)::UInt64
│      %664   = Core.bitcast(Ptr{Float64}, %663)::Ptr{Float64}
└─────          goto #184
184 ── %666   = Base.pointerref(%664, 1, 1)::Float64
└─────          goto #185
185 ──          goto #186
186 ──          $(Expr(:gc_preserve_end, :(%596)))
└─────          goto #187
187 ──          goto #188
188 ── %672   = StrideArraysCore.getfield(u, :data)::Vector{Float64}
│      %673   = $(Expr(:gc_preserve_begin, :(%672)))
│      %674   = StrideArraysCore.getfield(u, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #193 if not true
189 ── %676   = Core.tuple(4, %439, %29, %26, %21)::NTuple{5, Int64}
│      %677   = StrideArraysCore.getfield(%674, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %678   = Core.getfield(%677, 5)::Int64
│      %679   = Base.bitcast(UInt64, %678)::UInt64
│      %680   = Base.bitcast(Int64, %679)::Int64
│      %681   = Base.sle_int(1, %439)::Bool
│      %682   = Base.sle_int(%439, 4)::Bool
│      %683   = Base.and_int(%681, %682)::Bool
│      %684   = Base.sle_int(1, %29)::Bool
│      %685   = Base.sle_int(%29, 4)::Bool
│      %686   = Base.and_int(%684, %685)::Bool
│      %687   = Base.sle_int(1, %26)::Bool
│      %688   = Base.sle_int(%26, 4)::Bool
│      %689   = Base.and_int(%687, %688)::Bool
│      %690   = Base.sub_int(%21, 1)::Int64
│      %691   = Base.bitcast(UInt64, %690)::UInt64
│      %692   = Base.bitcast(UInt64, %680)::UInt64
│      %693   = Base.ult_int(%691, %692)::Bool
│      %694   = Base.and_int(%693, true)::Bool
│      %695   = Base.and_int(%689, %694)::Bool
│      %696   = Base.and_int(%686, %695)::Bool
│      %697   = Base.and_int(%683, %696)::Bool
│      %698   = Base.and_int(true, %697)::Bool
└─────          goto #191 if not %698
190 ──          goto #192
191 ──          invoke Base.throw_boundserror(%674::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %676::NTuple{5, Int64})::Union{}
└─────          unreachable
192 ──          nothing::Nothing
193 ┄─ %704   = StrideArraysCore.getfield(%674, :ptr)::Ptr{Float64}
│      %705   = Base.sub_int(%439, 1)::Int64
│      %706   = Base.sub_int(%29, 1)::Int64
│      %707   = Base.sub_int(%26, 1)::Int64
│      %708   = Base.sub_int(%21, 1)::Int64
└─────          goto #202 if not true
194 ┄─ %710   = φ (#193 => 2, #201 => %722)::Int64
│      %711   = Base.sle_int(1, %710)::Bool
└─────          goto #196 if not %711
195 ── %713   = Base.sle_int(%710, 5)::Bool
└─────          goto #197
196 ──          nothing::Nothing
197 ┄─ %716   = φ (#195 => %713, #196 => false)::Bool
└─────          goto #199 if not %716
198 ──          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %710, true)::Static.True
│      %719   = Base.add_int(%710, 1)::Int64
└─────          goto #200
199 ──          goto #200
200 ┄─ %722   = φ (#198 => %719)::Int64
│      %723   = φ (#198 => false, #199 => true)::Bool
│      %724   = Base.not_int(%723)::Bool
└─────          goto #202 if not %724
201 ──          goto #194
202 ┄─          goto #203
203 ──          goto #204
204 ── %729   = Base.mul_int(%708, 4)::Int64
│      %730   = Base.add_int(%707, %729)::Int64
│      %731   = Base.mul_int(%730, 4)::Int64
│      %732   = Base.add_int(%706, %731)::Int64
│      %733   = Base.mul_int(%732, 4)::Int64
│      %734   = Base.add_int(%705, %733)::Int64
│      %735   = Base.mul_int(%734, 5)::Int64
│      %736   = Base.add_int(3, %735)::Int64
│      %737   = Base.mul_int(8, %736)::Int64
│      %738   = Core.bitcast(Core.UInt, %704)::UInt64
│      %739   = Base.bitcast(UInt64, %737)::UInt64
│      %740   = Base.add_ptr(%738, %739)::UInt64
│      %741   = Core.bitcast(Ptr{Float64}, %740)::Ptr{Float64}
└─────          goto #205
205 ── %743   = Base.pointerref(%741, 1, 1)::Float64
└─────          goto #206
206 ──          goto #207
207 ──          $(Expr(:gc_preserve_end, :(%673)))
└─────          goto #208
208 ──          goto #209
209 ── %749   = StrideArraysCore.getfield(u, :data)::Vector{Float64}
│      %750   = $(Expr(:gc_preserve_begin, :(%749)))
│      %751   = StrideArraysCore.getfield(u, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #214 if not true
210 ── %753   = Core.tuple(5, %439, %29, %26, %21)::NTuple{5, Int64}
│      %754   = StrideArraysCore.getfield(%751, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %755   = Core.getfield(%754, 5)::Int64
│      %756   = Base.bitcast(UInt64, %755)::UInt64
│      %757   = Base.bitcast(Int64, %756)::Int64
│      %758   = Base.sle_int(1, %439)::Bool
│      %759   = Base.sle_int(%439, 4)::Bool
│      %760   = Base.and_int(%758, %759)::Bool
│      %761   = Base.sle_int(1, %29)::Bool
│      %762   = Base.sle_int(%29, 4)::Bool
│      %763   = Base.and_int(%761, %762)::Bool
│      %764   = Base.sle_int(1, %26)::Bool
│      %765   = Base.sle_int(%26, 4)::Bool
│      %766   = Base.and_int(%764, %765)::Bool
│      %767   = Base.sub_int(%21, 1)::Int64
│      %768   = Base.bitcast(UInt64, %767)::UInt64
│      %769   = Base.bitcast(UInt64, %757)::UInt64
│      %770   = Base.ult_int(%768, %769)::Bool
│      %771   = Base.and_int(%770, true)::Bool
│      %772   = Base.and_int(%766, %771)::Bool
│      %773   = Base.and_int(%763, %772)::Bool
│      %774   = Base.and_int(%760, %773)::Bool
│      %775   = Base.and_int(true, %774)::Bool
└─────          goto #212 if not %775
211 ──          goto #213
212 ──          invoke Base.throw_boundserror(%751::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %753::NTuple{5, Int64})::Union{}
└─────          unreachable
213 ──          nothing::Nothing
214 ┄─ %781   = StrideArraysCore.getfield(%751, :ptr)::Ptr{Float64}
│      %782   = Base.sub_int(%439, 1)::Int64
│      %783   = Base.sub_int(%29, 1)::Int64
│      %784   = Base.sub_int(%26, 1)::Int64
│      %785   = Base.sub_int(%21, 1)::Int64
└─────          goto #223 if not true
215 ┄─ %787   = φ (#214 => 2, #222 => %799)::Int64
│      %788   = Base.sle_int(1, %787)::Bool
└─────          goto #217 if not %788
216 ── %790   = Base.sle_int(%787, 5)::Bool
└─────          goto #218
217 ──          nothing::Nothing
218 ┄─ %793   = φ (#216 => %790, #217 => false)::Bool
└─────          goto #220 if not %793
219 ──          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %787, true)::Static.True
│      %796   = Base.add_int(%787, 1)::Int64
└─────          goto #221
220 ──          goto #221
221 ┄─ %799   = φ (#219 => %796)::Int64
│      %800   = φ (#219 => false, #220 => true)::Bool
│      %801   = Base.not_int(%800)::Bool
└─────          goto #223 if not %801
222 ──          goto #215
223 ┄─          goto #224
224 ──          goto #225
225 ── %806   = Base.mul_int(%785, 4)::Int64
│      %807   = Base.add_int(%784, %806)::Int64
│      %808   = Base.mul_int(%807, 4)::Int64
│      %809   = Base.add_int(%783, %808)::Int64
│      %810   = Base.mul_int(%809, 4)::Int64
│      %811   = Base.add_int(%782, %810)::Int64
│      %812   = Base.mul_int(%811, 5)::Int64
│      %813   = Base.add_int(4, %812)::Int64
│      %814   = Base.mul_int(8, %813)::Int64
│      %815   = Core.bitcast(Core.UInt, %781)::UInt64
│      %816   = Base.bitcast(UInt64, %814)::UInt64
│      %817   = Base.add_ptr(%815, %816)::UInt64
│      %818   = Core.bitcast(Ptr{Float64}, %817)::Ptr{Float64}
└─────          goto #226
226 ── %820   = Base.pointerref(%818, 1, 1)::Float64
└─────          goto #227
227 ──          goto #228
228 ──          $(Expr(:gc_preserve_end, :(%750)))
└─────          goto #229
229 ──          goto #230
230 ──          goto #231
231 ──          goto #232
232 ──          goto #234
233 ──          nothing::Nothing
234 ┄─          goto #236
235 ──          nothing::Nothing
236 ┄─          goto #237
237 ──          goto #239
238 ──          nothing::Nothing
239 ┄─          goto #240
240 ──          goto #242
241 ──          nothing::Nothing
242 ┄─          goto #244
243 ──          nothing::Nothing
244 ┄─          goto #245
245 ──          goto #247
246 ──          nothing::Nothing
247 ┄─          goto #248
248 ──          goto #250
249 ──          nothing::Nothing
250 ┄─          goto #252
251 ──          nothing::Nothing
252 ┄─          goto #253
253 ──          goto #255
254 ──          nothing::Nothing
255 ┄─          goto #256
256 ──          goto #258
257 ──          nothing::Nothing
258 ┄─          goto #260
259 ──          nothing::Nothing
260 ┄─          goto #261
261 ──          goto #263
262 ──          nothing::Nothing
263 ┄─          goto #264
264 ── %860   = Base.div_float(%182, %105)::Float64
│      %861   = Base.div_float(%259, %105)::Float64
│      %862   = Base.div_float(%336, %105)::Float64
│      %863   = Base.getfield(equations, :gamma)::Float64
│      %864   = Base.sub_float(%863, 1.0)::Float64
│      %865   = Base.mul_float(%182, %860)::Float64
│      %866   = Base.muladd_float(%259, %861, %865)::Float64
│      %867   = Base.muladd_float(%336, %862, %866)::Float64
│      %868   = Base.muladd_float(-0.5, %867, %413)::Float64
│      %869   = Base.mul_float(%864, %868)::Float64
└─────          goto #265
265 ──          goto #267
266 ──          nothing::Nothing
267 ┄─          goto #269
268 ──          nothing::Nothing
269 ┄─          goto #270
270 ──          goto #272
271 ──          nothing::Nothing
272 ┄─          goto #273
273 ──          goto #275
274 ──          nothing::Nothing
275 ┄─          goto #277
276 ──          nothing::Nothing
277 ┄─          goto #278
278 ──          goto #280
279 ──          nothing::Nothing
280 ┄─          goto #281
281 ──          goto #283
282 ──          nothing::Nothing
283 ┄─          goto #285
284 ──          nothing::Nothing
285 ┄─          goto #286
286 ──          goto #288
287 ──          nothing::Nothing
288 ┄─          goto #289
289 ──          goto #291
290 ──          nothing::Nothing
291 ┄─          goto #293
292 ──          nothing::Nothing
293 ┄─          goto #294
294 ──          goto #296
295 ──          nothing::Nothing
296 ┄─          goto #297
297 ──          goto #299
298 ──          nothing::Nothing
299 ┄─          goto #301
300 ──          nothing::Nothing
301 ┄─          goto #302
302 ──          goto #304
303 ──          nothing::Nothing
304 ┄─          goto #305
305 ──          goto #307
306 ──          nothing::Nothing
307 ┄─          goto #309
308 ──          nothing::Nothing
309 ┄─          goto #310
310 ──          goto #312
311 ──          nothing::Nothing
312 ┄─          goto #313
313 ──          goto #315
314 ──          nothing::Nothing
315 ┄─          goto #317
316 ──          nothing::Nothing
317 ┄─          goto #318
318 ──          goto #320
319 ──          nothing::Nothing
320 ┄─          goto #321
321 ──          goto #323
322 ──          nothing::Nothing
323 ┄─          goto #325
324 ──          nothing::Nothing
325 ┄─          goto #326
326 ──          goto #328
327 ──          nothing::Nothing
328 ┄─          goto #329
329 ── %935   = Base.div_float(%589, %512)::Float64
│      %936   = Base.div_float(%666, %512)::Float64
│      %937   = Base.div_float(%743, %512)::Float64
│      %938   = Base.getfield(equations, :gamma)::Float64
│      %939   = Base.sub_float(%938, 1.0)::Float64
│      %940   = Base.mul_float(%589, %935)::Float64
│      %941   = Base.muladd_float(%666, %936, %940)::Float64
│      %942   = Base.muladd_float(%743, %937, %941)::Float64
│      %943   = Base.muladd_float(-0.5, %942, %820)::Float64
│      %944   = Base.mul_float(%939, %943)::Float64
└─────          goto #330
330 ──          goto #332
331 ──          nothing::Nothing
332 ┄─          goto #334
333 ──          nothing::Nothing
334 ┄─          goto #335
335 ──          goto #337
336 ──          nothing::Nothing
337 ┄─          goto #338
338 ──          goto #340
339 ──          nothing::Nothing
340 ┄─          goto #342
341 ──          nothing::Nothing
342 ┄─          goto #343
343 ──          goto #345
344 ──          nothing::Nothing
345 ┄─          goto #346
346 ──          goto #348
347 ──          nothing::Nothing
348 ┄─          goto #350
349 ──          nothing::Nothing
350 ┄─          goto #351
351 ──          goto #353
352 ──          nothing::Nothing
353 ┄─          goto #354
354 ──          goto #356
355 ──          nothing::Nothing
356 ┄─          goto #358
357 ──          nothing::Nothing
358 ┄─          goto #359
359 ──          goto #361
360 ──          nothing::Nothing
361 ┄─          goto #362
362 ── %978   = Base.muladd_float(-2.0, %512, %105)::Float64
│      %979   = Base.mul_float(%105, %978)::Float64
│      %980   = Base.muladd_float(%512, %512, %979)::Float64
│      %981   = Base.muladd_float(2.0, %512, %105)::Float64
│      %982   = Base.mul_float(%105, %981)::Float64
│      %983   = Base.muladd_float(%512, %512, %982)::Float64
│      %984   = Base.div_float(%980, %983)::Float64
│      %985   = Base.lt_float(%984, 0.0001)::Bool
└─────          goto #364 if not %985
363 ── %987   = Base.add_float(%105, %512)::Float64
│      %988   = Base.muladd_float(%984, 0.2857142857142857, 0.4)::Float64
│      %989   = Base.muladd_float(%984, %988, 0.6666666666666666)::Float64
│      %990   = Base.muladd_float(%984, %989, 2.0)::Float64
│      %991   = Base.div_float(%987, %990)::Float64
└─────          goto #365
364 ── %993   = Base.sub_float(%512, %105)::Float64
│      %994   = Base.div_float(%512, %105)::Float64
│      %995   = $(Expr(:foreigncall, "llvm.log.f64", Float64, svec(Float64), 0, :(:llvmcall), :(%994), :(%994)))::Float64
│      %996   = Base.div_float(%993, %995)::Float64
└─────          goto #365
365 ┄─ %998   = φ (#363 => %991, #364 => %996)::Float64
│      %999   = Base.mul_float(%105, %944)::Float64
│      %1000  = Base.mul_float(%512, %869)::Float64
│      %1001  = Base.muladd_float(-2.0, %1000, %999)::Float64
│      %1002  = Base.mul_float(%999, %1001)::Float64
│      %1003  = Base.muladd_float(%1000, %1000, %1002)::Float64
│      %1004  = Base.muladd_float(2.0, %1000, %999)::Float64
│      %1005  = Base.mul_float(%999, %1004)::Float64
│      %1006  = Base.muladd_float(%1000, %1000, %1005)::Float64
│      %1007  = Base.div_float(%1003, %1006)::Float64
│      %1008  = Base.lt_float(%1007, 0.0001)::Bool
└─────          goto #367 if not %1008
366 ── %1010  = Base.muladd_float(%1007, 0.2857142857142857, 0.4)::Float64
│      %1011  = Base.muladd_float(%1007, %1010, 0.6666666666666666)::Float64
│      %1012  = Base.muladd_float(%1007, %1011, 2.0)::Float64
│      %1013  = Base.add_float(%999, %1000)::Float64
│      %1014  = Base.div_float(%1012, %1013)::Float64
└─────          goto #368
367 ── %1016  = Base.div_float(%1000, %999)::Float64
│      %1017  = $(Expr(:foreigncall, "llvm.log.f64", Float64, svec(Float64), 0, :(:llvmcall), :(%1016), :(%1016)))::Float64
│      %1018  = Base.sub_float(%1000, %999)::Float64
│      %1019  = Base.div_float(%1017, %1018)::Float64
└─────          goto #368
368 ┄─ %1021  = φ (#366 => %1014, #367 => %1019)::Float64
│      %1022  = Base.mul_float(%869, %944)::Float64
│      %1023  = Base.mul_float(%1022, %1021)::Float64
│      %1024  = Base.add_float(%860, %935)::Float64
│      %1025  = Base.mul_float(0.5, %1024)::Float64
│      %1026  = Base.add_float(%861, %936)::Float64
│      %1027  = Base.mul_float(0.5, %1026)::Float64
│      %1028  = Base.add_float(%862, %937)::Float64
│      %1029  = Base.mul_float(0.5, %1028)::Float64
│      %1030  = Base.add_float(%869, %944)::Float64
│      %1031  = Base.mul_float(0.5, %1030)::Float64
│      %1032  = Base.mul_float(%860, %935)::Float64
│      %1033  = Base.muladd_float(%861, %936, %1032)::Float64
│      %1034  = Base.muladd_float(%862, %937, %1033)::Float64
│      %1035  = Base.mul_float(0.5, %1034)::Float64
│      %1036  = Base.mul_float(%998, %1025)::Float64
│      %1037  = Base.muladd_float(%1036, %1025, %1031)::Float64
│      %1038  = Base.mul_float(%1036, %1027)::Float64
│      %1039  = Base.mul_float(%1036, %1029)::Float64
│      %1040  = Base.mul_float(%869, %935)::Float64
│      %1041  = Base.muladd_float(%944, %860, %1040)::Float64
│      %1042  = Base.getfield(equations, :inv_gamma_minus_one)::Float64
│      %1043  = Base.muladd_float(%1023, %1042, %1035)::Float64
│      %1044  = Base.mul_float(%1036, %1043)::Float64
│      %1045  = Base.muladd_float(0.5, %1041, %1044)::Float64
│      %1046  = Core.tuple(%1036, %1037, %1038, %1039, %1045)::NTuple{5, Float64}
└─────          goto #369
369 ── %1048  = Base.arrayref(false, %24, %32, %439)::Float64
│      %1049  = Base.copysign_float(0.0, %1048)::Float64
│      %1050  = Core.ifelse(true, %1048, %1049)::Float64
└─────          goto #415 if not true
370 ┄─ %1052  = φ (#369 => 1, #414 => %1221)::Int64
│      %1053  = φ (#369 => 1, #414 => %1222)::Int64
│      %1054  = Base.getfield(%1046, %1052, true)::Float64
│      %1055  = StrideArraysCore.getfield(du, :data)::Vector{Float64}
│      %1056  = $(Expr(:gc_preserve_begin, :(%1055)))
│      %1057  = StrideArraysCore.getfield(du, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #375 if not true
371 ── %1059  = Core.tuple(%1052, %32, %29, %26, %21)::NTuple{5, Int64}
│      %1060  = StrideArraysCore.getfield(%1057, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %1061  = Core.getfield(%1060, 5)::Int64
│      %1062  = Base.bitcast(UInt64, %1061)::UInt64
│      %1063  = Base.bitcast(Int64, %1062)::Int64
│      %1064  = Base.sle_int(1, %1052)::Bool
│      %1065  = Base.sle_int(%1052, 5)::Bool
│      %1066  = Base.and_int(%1064, %1065)::Bool
│      %1067  = Base.sle_int(1, %32)::Bool
│      %1068  = Base.sle_int(%32, 4)::Bool
│      %1069  = Base.and_int(%1067, %1068)::Bool
│      %1070  = Base.sle_int(1, %29)::Bool
│      %1071  = Base.sle_int(%29, 4)::Bool
│      %1072  = Base.and_int(%1070, %1071)::Bool
│      %1073  = Base.sle_int(1, %26)::Bool
│      %1074  = Base.sle_int(%26, 4)::Bool
│      %1075  = Base.and_int(%1073, %1074)::Bool
│      %1076  = Base.sub_int(%21, 1)::Int64
│      %1077  = Base.bitcast(UInt64, %1076)::UInt64
│      %1078  = Base.bitcast(UInt64, %1063)::UInt64
│      %1079  = Base.ult_int(%1077, %1078)::Bool
│      %1080  = Base.and_int(%1079, true)::Bool
│      %1081  = Base.and_int(%1075, %1080)::Bool
│      %1082  = Base.and_int(%1072, %1081)::Bool
│      %1083  = Base.and_int(%1069, %1082)::Bool
│      %1084  = Base.and_int(%1066, %1083)::Bool
└─────          goto #373 if not %1084
372 ──          goto #374
373 ──          invoke Base.throw_boundserror(%1057::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %1059::NTuple{5, Int64})::Union{}
└─────          unreachable
374 ──          nothing::Nothing
375 ┄─ %1090  = StrideArraysCore.getfield(%1057, :ptr)::Ptr{Float64}
│      %1091  = Base.sub_int(%1052, 1)::Int64
│      %1092  = Base.sub_int(%32, 1)::Int64
│      %1093  = Base.sub_int(%29, 1)::Int64
│      %1094  = Base.sub_int(%26, 1)::Int64
│      %1095  = Base.sub_int(%21, 1)::Int64
└─────          goto #384 if not true
376 ┄─ %1097  = φ (#375 => 2, #383 => %1109)::Int64
│      %1098  = Base.sle_int(1, %1097)::Bool
└─────          goto #378 if not %1098
377 ── %1100  = Base.sle_int(%1097, 5)::Bool
└─────          goto #379
378 ──          nothing::Nothing
379 ┄─ %1103  = φ (#377 => %1100, #378 => false)::Bool
└─────          goto #381 if not %1103
380 ──          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %1097, true)::Static.True
│      %1106  = Base.add_int(%1097, 1)::Int64
└─────          goto #382
381 ──          goto #382
382 ┄─ %1109  = φ (#380 => %1106)::Int64
│      %1110  = φ (#380 => false, #381 => true)::Bool
│      %1111  = Base.not_int(%1110)::Bool
└─────          goto #384 if not %1111
383 ──          goto #376
384 ┄─          goto #385
385 ──          goto #386
386 ── %1116  = Base.mul_int(%1095, 4)::Int64
│      %1117  = Base.add_int(%1094, %1116)::Int64
│      %1118  = Base.mul_int(%1117, 4)::Int64
│      %1119  = Base.add_int(%1093, %1118)::Int64
│      %1120  = Base.mul_int(%1119, 4)::Int64
│      %1121  = Base.add_int(%1092, %1120)::Int64
│      %1122  = Base.mul_int(%1121, 5)::Int64
│      %1123  = Base.add_int(%1091, %1122)::Int64
│      %1124  = Base.mul_int(8, %1123)::Int64
│      %1125  = Core.bitcast(Core.UInt, %1090)::UInt64
│      %1126  = Base.bitcast(UInt64, %1124)::UInt64
│      %1127  = Base.add_ptr(%1125, %1126)::UInt64
│      %1128  = Core.bitcast(Ptr{Float64}, %1127)::Ptr{Float64}
└─────          goto #387
387 ── %1130  = Base.pointerref(%1128, 1, 1)::Float64
└─────          goto #388
388 ──          goto #389
389 ──          $(Expr(:gc_preserve_end, :(%1056)))
└─────          goto #390
390 ── %1135  = Base.muladd_float(%1050, %1054, %1130)::Float64
│      %1136  = StrideArraysCore.getfield(du, :data)::Vector{Float64}
│      %1137  = $(Expr(:gc_preserve_begin, :(%1136)))
│      %1138  = StrideArraysCore.getfield(du, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #395 if not true
391 ── %1140  = Core.tuple(%1052, %32, %29, %26, %21)::NTuple{5, Int64}
│      %1141  = StrideArraysCore.getfield(%1138, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %1142  = Core.getfield(%1141, 5)::Int64
│      %1143  = Base.bitcast(UInt64, %1142)::UInt64
│      %1144  = Base.bitcast(Int64, %1143)::Int64
│      %1145  = Base.sle_int(1, %1052)::Bool
│      %1146  = Base.sle_int(%1052, 5)::Bool
│      %1147  = Base.and_int(%1145, %1146)::Bool
│      %1148  = Base.sle_int(1, %32)::Bool
│      %1149  = Base.sle_int(%32, 4)::Bool
│      %1150  = Base.and_int(%1148, %1149)::Bool
│      %1151  = Base.sle_int(1, %29)::Bool
│      %1152  = Base.sle_int(%29, 4)::Bool
│      %1153  = Base.and_int(%1151, %1152)::Bool
│      %1154  = Base.sle_int(1, %26)::Bool
│      %1155  = Base.sle_int(%26, 4)::Bool
│      %1156  = Base.and_int(%1154, %1155)::Bool
│      %1157  = Base.sub_int(%21, 1)::Int64
│      %1158  = Base.bitcast(UInt64, %1157)::UInt64
│      %1159  = Base.bitcast(UInt64, %1144)::UInt64
│      %1160  = Base.ult_int(%1158, %1159)::Bool
│      %1161  = Base.and_int(%1160, true)::Bool
│      %1162  = Base.and_int(%1156, %1161)::Bool
│      %1163  = Base.and_int(%1153, %1162)::Bool
│      %1164  = Base.and_int(%1150, %1163)::Bool
│      %1165  = Base.and_int(%1147, %1164)::Bool
└─────          goto #393 if not %1165
392 ──          goto #394
393 ──          invoke Base.throw_boundserror(%1138::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %1140::NTuple{5, Int64})::Union{}
└─────          unreachable
394 ──          nothing::Nothing
395 ┄─ %1171  = StrideArraysCore.getfield(%1138, :ptr)::Ptr{Float64}
│      %1172  = Base.sub_int(%1052, 1)::Int64
│      %1173  = Base.sub_int(%32, 1)::Int64
│      %1174  = Base.sub_int(%29, 1)::Int64
│      %1175  = Base.sub_int(%26, 1)::Int64
│      %1176  = Base.sub_int(%21, 1)::Int64
└─────          goto #404 if not true
396 ┄─ %1178  = φ (#395 => 2, #403 => %1190)::Int64
│      %1179  = Base.sle_int(1, %1178)::Bool
└─────          goto #398 if not %1179
397 ── %1181  = Base.sle_int(%1178, 5)::Bool
└─────          goto #399
398 ──          nothing::Nothing
399 ┄─ %1184  = φ (#397 => %1181, #398 => false)::Bool
└─────          goto #401 if not %1184
400 ──          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %1178, true)::Static.True
│      %1187  = Base.add_int(%1178, 1)::Int64
└─────          goto #402
401 ──          goto #402
402 ┄─ %1190  = φ (#400 => %1187)::Int64
│      %1191  = φ (#400 => false, #401 => true)::Bool
│      %1192  = Base.not_int(%1191)::Bool
└─────          goto #404 if not %1192
403 ──          goto #396
404 ┄─          goto #405
405 ──          goto #406
406 ── %1197  = Base.mul_int(%1176, 4)::Int64
│      %1198  = Base.add_int(%1175, %1197)::Int64
│      %1199  = Base.mul_int(%1198, 4)::Int64
│      %1200  = Base.add_int(%1174, %1199)::Int64
│      %1201  = Base.mul_int(%1200, 4)::Int64
│      %1202  = Base.add_int(%1173, %1201)::Int64
│      %1203  = Base.mul_int(%1202, 5)::Int64
│      %1204  = Base.add_int(%1172, %1203)::Int64
│      %1205  = Base.mul_int(8, %1204)::Int64
│      %1206  = Core.bitcast(Core.UInt, %1171)::UInt64
│      %1207  = Base.bitcast(UInt64, %1205)::UInt64
│      %1208  = Base.add_ptr(%1206, %1207)::UInt64
│      %1209  = Core.bitcast(Ptr{Float64}, %1208)::Ptr{Float64}
└─────          goto #407
407 ──          Base.pointerset(%1209, %1135, 1, 1)::Ptr{Float64}
└─────          goto #408
408 ──          goto #409
409 ──          $(Expr(:gc_preserve_end, :(%1137)))
└─────          goto #410
410 ── %1216  = (%1053 === 5)::Bool
└─────          goto #412 if not %1216
411 ──          goto #413
412 ── %1219  = Base.add_int(%1053, 1)::Int64
└─────          goto #413
413 ┄─ %1221  = φ (#412 => %1219)::Int64
│      %1222  = φ (#412 => %1219)::Int64
│      %1223  = φ (#411 => true, #412 => false)::Bool
│      %1224  = Base.not_int(%1223)::Bool
└─────          goto #415 if not %1224
414 ──          goto #370
415 ┄─          goto #416
416 ── %1228  = Base.arrayref(false, %24, %439, %32)::Float64
│      %1229  = Base.copysign_float(0.0, %1228)::Float64
│      %1230  = Core.ifelse(true, %1228, %1229)::Float64
└─────          goto #462 if not true
417 ┄─ %1232  = φ (#416 => 1, #461 => %1401)::Int64
│      %1233  = φ (#416 => 1, #461 => %1402)::Int64
│      %1234  = Base.getfield(%1046, %1232, true)::Float64
│      %1235  = StrideArraysCore.getfield(du, :data)::Vector{Float64}
│      %1236  = $(Expr(:gc_preserve_begin, :(%1235)))
│      %1237  = StrideArraysCore.getfield(du, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #422 if not true
418 ── %1239  = Core.tuple(%1232, %439, %29, %26, %21)::NTuple{5, Int64}
│      %1240  = StrideArraysCore.getfield(%1237, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %1241  = Core.getfield(%1240, 5)::Int64
│      %1242  = Base.bitcast(UInt64, %1241)::UInt64
│      %1243  = Base.bitcast(Int64, %1242)::Int64
│      %1244  = Base.sle_int(1, %1232)::Bool
│      %1245  = Base.sle_int(%1232, 5)::Bool
│      %1246  = Base.and_int(%1244, %1245)::Bool
│      %1247  = Base.sle_int(1, %439)::Bool
│      %1248  = Base.sle_int(%439, 4)::Bool
│      %1249  = Base.and_int(%1247, %1248)::Bool
│      %1250  = Base.sle_int(1, %29)::Bool
│      %1251  = Base.sle_int(%29, 4)::Bool
│      %1252  = Base.and_int(%1250, %1251)::Bool
│      %1253  = Base.sle_int(1, %26)::Bool
│      %1254  = Base.sle_int(%26, 4)::Bool
│      %1255  = Base.and_int(%1253, %1254)::Bool
│      %1256  = Base.sub_int(%21, 1)::Int64
│      %1257  = Base.bitcast(UInt64, %1256)::UInt64
│      %1258  = Base.bitcast(UInt64, %1243)::UInt64
│      %1259  = Base.ult_int(%1257, %1258)::Bool
│      %1260  = Base.and_int(%1259, true)::Bool
│      %1261  = Base.and_int(%1255, %1260)::Bool
│      %1262  = Base.and_int(%1252, %1261)::Bool
│      %1263  = Base.and_int(%1249, %1262)::Bool
│      %1264  = Base.and_int(%1246, %1263)::Bool
└─────          goto #420 if not %1264
419 ──          goto #421
420 ──          invoke Base.throw_boundserror(%1237::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %1239::NTuple{5, Int64})::Union{}
└─────          unreachable
421 ──          nothing::Nothing
422 ┄─ %1270  = StrideArraysCore.getfield(%1237, :ptr)::Ptr{Float64}
│      %1271  = Base.sub_int(%1232, 1)::Int64
│      %1272  = Base.sub_int(%439, 1)::Int64
│      %1273  = Base.sub_int(%29, 1)::Int64
│      %1274  = Base.sub_int(%26, 1)::Int64
│      %1275  = Base.sub_int(%21, 1)::Int64
└─────          goto #431 if not true
423 ┄─ %1277  = φ (#422 => 2, #430 => %1289)::Int64
│      %1278  = Base.sle_int(1, %1277)::Bool
└─────          goto #425 if not %1278
424 ── %1280  = Base.sle_int(%1277, 5)::Bool
└─────          goto #426
425 ──          nothing::Nothing
426 ┄─ %1283  = φ (#424 => %1280, #425 => false)::Bool
└─────          goto #428 if not %1283
427 ──          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %1277, true)::Static.True
│      %1286  = Base.add_int(%1277, 1)::Int64
└─────          goto #429
428 ──          goto #429
429 ┄─ %1289  = φ (#427 => %1286)::Int64
│      %1290  = φ (#427 => false, #428 => true)::Bool
│      %1291  = Base.not_int(%1290)::Bool
└─────          goto #431 if not %1291
430 ──          goto #423
431 ┄─          goto #432
432 ──          goto #433
433 ── %1296  = Base.mul_int(%1275, 4)::Int64
│      %1297  = Base.add_int(%1274, %1296)::Int64
│      %1298  = Base.mul_int(%1297, 4)::Int64
│      %1299  = Base.add_int(%1273, %1298)::Int64
│      %1300  = Base.mul_int(%1299, 4)::Int64
│      %1301  = Base.add_int(%1272, %1300)::Int64
│      %1302  = Base.mul_int(%1301, 5)::Int64
│      %1303  = Base.add_int(%1271, %1302)::Int64
│      %1304  = Base.mul_int(8, %1303)::Int64
│      %1305  = Core.bitcast(Core.UInt, %1270)::UInt64
│      %1306  = Base.bitcast(UInt64, %1304)::UInt64
│      %1307  = Base.add_ptr(%1305, %1306)::UInt64
│      %1308  = Core.bitcast(Ptr{Float64}, %1307)::Ptr{Float64}
└─────          goto #434
434 ── %1310  = Base.pointerref(%1308, 1, 1)::Float64
└─────          goto #435
435 ──          goto #436
436 ──          $(Expr(:gc_preserve_end, :(%1236)))
└─────          goto #437
437 ── %1315  = Base.muladd_float(%1230, %1234, %1310)::Float64
│      %1316  = StrideArraysCore.getfield(du, :data)::Vector{Float64}
│      %1317  = $(Expr(:gc_preserve_begin, :(%1316)))
│      %1318  = StrideArraysCore.getfield(du, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #442 if not true
438 ── %1320  = Core.tuple(%1232, %439, %29, %26, %21)::NTuple{5, Int64}
│      %1321  = StrideArraysCore.getfield(%1318, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %1322  = Core.getfield(%1321, 5)::Int64
│      %1323  = Base.bitcast(UInt64, %1322)::UInt64
│      %1324  = Base.bitcast(Int64, %1323)::Int64
│      %1325  = Base.sle_int(1, %1232)::Bool
│      %1326  = Base.sle_int(%1232, 5)::Bool
│      %1327  = Base.and_int(%1325, %1326)::Bool
│      %1328  = Base.sle_int(1, %439)::Bool
│      %1329  = Base.sle_int(%439, 4)::Bool
│      %1330  = Base.and_int(%1328, %1329)::Bool
│      %1331  = Base.sle_int(1, %29)::Bool
│      %1332  = Base.sle_int(%29, 4)::Bool
│      %1333  = Base.and_int(%1331, %1332)::Bool
│      %1334  = Base.sle_int(1, %26)::Bool
│      %1335  = Base.sle_int(%26, 4)::Bool
│      %1336  = Base.and_int(%1334, %1335)::Bool
│      %1337  = Base.sub_int(%21, 1)::Int64
│      %1338  = Base.bitcast(UInt64, %1337)::UInt64
│      %1339  = Base.bitcast(UInt64, %1324)::UInt64
│      %1340  = Base.ult_int(%1338, %1339)::Bool
│      %1341  = Base.and_int(%1340, true)::Bool
│      %1342  = Base.and_int(%1336, %1341)::Bool
│      %1343  = Base.and_int(%1333, %1342)::Bool
│      %1344  = Base.and_int(%1330, %1343)::Bool
│      %1345  = Base.and_int(%1327, %1344)::Bool
└─────          goto #440 if not %1345
439 ──          goto #441
440 ──          invoke Base.throw_boundserror(%1318::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %1320::NTuple{5, Int64})::Union{}
└─────          unreachable
441 ──          nothing::Nothing
442 ┄─ %1351  = StrideArraysCore.getfield(%1318, :ptr)::Ptr{Float64}
│      %1352  = Base.sub_int(%1232, 1)::Int64
│      %1353  = Base.sub_int(%439, 1)::Int64
│      %1354  = Base.sub_int(%29, 1)::Int64
│      %1355  = Base.sub_int(%26, 1)::Int64
│      %1356  = Base.sub_int(%21, 1)::Int64
└─────          goto #451 if not true
443 ┄─ %1358  = φ (#442 => 2, #450 => %1370)::Int64
│      %1359  = Base.sle_int(1, %1358)::Bool
└─────          goto #445 if not %1359
444 ── %1361  = Base.sle_int(%1358, 5)::Bool
└─────          goto #446
445 ──          nothing::Nothing
446 ┄─ %1364  = φ (#444 => %1361, #445 => false)::Bool
└─────          goto #448 if not %1364
447 ──          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %1358, true)::Static.True
│      %1367  = Base.add_int(%1358, 1)::Int64
└─────          goto #449
448 ──          goto #449
449 ┄─ %1370  = φ (#447 => %1367)::Int64
│      %1371  = φ (#447 => false, #448 => true)::Bool
│      %1372  = Base.not_int(%1371)::Bool
└─────          goto #451 if not %1372
450 ──          goto #443
451 ┄─          goto #452
452 ──          goto #453
453 ── %1377  = Base.mul_int(%1356, 4)::Int64
│      %1378  = Base.add_int(%1355, %1377)::Int64
│      %1379  = Base.mul_int(%1378, 4)::Int64
│      %1380  = Base.add_int(%1354, %1379)::Int64
│      %1381  = Base.mul_int(%1380, 4)::Int64
│      %1382  = Base.add_int(%1353, %1381)::Int64
│      %1383  = Base.mul_int(%1382, 5)::Int64
│      %1384  = Base.add_int(%1352, %1383)::Int64
│      %1385  = Base.mul_int(8, %1384)::Int64
│      %1386  = Core.bitcast(Core.UInt, %1351)::UInt64
│      %1387  = Base.bitcast(UInt64, %1385)::UInt64
│      %1388  = Base.add_ptr(%1386, %1387)::UInt64
│      %1389  = Core.bitcast(Ptr{Float64}, %1388)::Ptr{Float64}
└─────          goto #454
454 ──          Base.pointerset(%1389, %1315, 1, 1)::Ptr{Float64}
└─────          goto #455
455 ──          goto #456
456 ──          $(Expr(:gc_preserve_end, :(%1317)))
└─────          goto #457
457 ── %1396  = (%1233 === 5)::Bool
└─────          goto #459 if not %1396
458 ──          goto #460
459 ── %1399  = Base.add_int(%1233, 1)::Int64
└─────          goto #460
460 ┄─ %1401  = φ (#459 => %1399)::Int64
│      %1402  = φ (#459 => %1399)::Int64
│      %1403  = φ (#458 => true, #459 => false)::Bool
│      %1404  = Base.not_int(%1403)::Bool
└─────          goto #462 if not %1404
461 ──          goto #417
462 ┄─          goto #463
463 ── %1408  = (%440 === %427)::Bool
└─────          goto #465 if not %1408
464 ──          goto #466
465 ── %1411  = Base.add_int(%440, 1)::Int64
└─────          goto #466
466 ┄─ %1413  = φ (#465 => %1411)::Int64
│      %1414  = φ (#465 => %1411)::Int64
│      %1415  = φ (#464 => true, #465 => false)::Bool
│      %1416  = Base.not_int(%1415)::Bool
└─────          goto #468 if not %1416
467 ──          goto #125
468 ┄─ %1419  = Base.add_int(%29, 1)::Int64
│      %1420  = Base.sle_int(%1419, 4)::Bool
└─────          goto #470 if not %1420
469 ──          goto #471
470 ── %1423  = Base.sub_int(%1419, 1)::Int64
└─────          goto #471
471 ┄─ %1425  = φ (#469 => 4, #470 => %1423)::Int64
└─────          goto #472
472 ──          goto #473
473 ── %1428  = Base.slt_int(%1425, %1419)::Bool
└─────          goto #475 if not %1428
474 ──          goto #476
475 ──          goto #476
476 ┄─ %1432  = φ (#474 => true, #475 => false)::Bool
│      %1433  = φ (#475 => %1419)::Int64
│      %1434  = φ (#475 => %1419)::Int64
│      %1435  = Base.not_int(%1432)::Bool
└─────          goto #820 if not %1435
477 ┄─ %1437  = φ (#476 => %1433, #819 => %2411)::Int64
│      %1438  = φ (#476 => %1434, #819 => %2412)::Int64
│      %1439  = StrideArraysCore.getfield(u, :data)::Vector{Float64}
│      %1440  = $(Expr(:gc_preserve_begin, :(%1439)))
│      %1441  = StrideArraysCore.getfield(u, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #482 if not true
478 ── %1443  = Core.tuple(1, %32, %1437, %26, %21)::NTuple{5, Int64}
│      %1444  = StrideArraysCore.getfield(%1441, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %1445  = Core.getfield(%1444, 5)::Int64
│      %1446  = Base.bitcast(UInt64, %1445)::UInt64
│      %1447  = Base.bitcast(Int64, %1446)::Int64
│      %1448  = Base.sle_int(1, %32)::Bool
│      %1449  = Base.sle_int(%32, 4)::Bool
│      %1450  = Base.and_int(%1448, %1449)::Bool
│      %1451  = Base.sle_int(1, %1437)::Bool
│      %1452  = Base.sle_int(%1437, 4)::Bool
│      %1453  = Base.and_int(%1451, %1452)::Bool
│      %1454  = Base.sle_int(1, %26)::Bool
│      %1455  = Base.sle_int(%26, 4)::Bool
│      %1456  = Base.and_int(%1454, %1455)::Bool
│      %1457  = Base.sub_int(%21, 1)::Int64
│      %1458  = Base.bitcast(UInt64, %1457)::UInt64
│      %1459  = Base.bitcast(UInt64, %1447)::UInt64
│      %1460  = Base.ult_int(%1458, %1459)::Bool
│      %1461  = Base.and_int(%1460, true)::Bool
│      %1462  = Base.and_int(%1456, %1461)::Bool
│      %1463  = Base.and_int(%1453, %1462)::Bool
│      %1464  = Base.and_int(%1450, %1463)::Bool
│      %1465  = Base.and_int(true, %1464)::Bool
└─────          goto #480 if not %1465
479 ──          goto #481
480 ──          invoke Base.throw_boundserror(%1441::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %1443::NTuple{5, Int64})::Union{}
└─────          unreachable
481 ──          nothing::Nothing
482 ┄─ %1471  = StrideArraysCore.getfield(%1441, :ptr)::Ptr{Float64}
│      %1472  = Base.sub_int(%32, 1)::Int64
│      %1473  = Base.sub_int(%1437, 1)::Int64
│      %1474  = Base.sub_int(%26, 1)::Int64
│      %1475  = Base.sub_int(%21, 1)::Int64
└─────          goto #491 if not true
483 ┄─ %1477  = φ (#482 => 2, #490 => %1489)::Int64
│      %1478  = Base.sle_int(1, %1477)::Bool
└─────          goto #485 if not %1478
484 ── %1480  = Base.sle_int(%1477, 5)::Bool
└─────          goto #486
485 ──          nothing::Nothing
486 ┄─ %1483  = φ (#484 => %1480, #485 => false)::Bool
└─────          goto #488 if not %1483
487 ──          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %1477, true)::Static.True
│      %1486  = Base.add_int(%1477, 1)::Int64
└─────          goto #489
488 ──          goto #489
489 ┄─ %1489  = φ (#487 => %1486)::Int64
│      %1490  = φ (#487 => false, #488 => true)::Bool
│      %1491  = Base.not_int(%1490)::Bool
└─────          goto #491 if not %1491
490 ──          goto #483
491 ┄─          goto #492
492 ──          goto #493
493 ── %1496  = Base.mul_int(%1475, 4)::Int64
│      %1497  = Base.add_int(%1474, %1496)::Int64
│      %1498  = Base.mul_int(%1497, 4)::Int64
│      %1499  = Base.add_int(%1473, %1498)::Int64
│      %1500  = Base.mul_int(%1499, 4)::Int64
│      %1501  = Base.add_int(%1472, %1500)::Int64
│      %1502  = Base.mul_int(%1501, 5)::Int64
│      %1503  = Base.add_int(0, %1502)::Int64
│      %1504  = Base.mul_int(8, %1503)::Int64
│      %1505  = Core.bitcast(Core.UInt, %1471)::UInt64
│      %1506  = Base.bitcast(UInt64, %1504)::UInt64
│      %1507  = Base.add_ptr(%1505, %1506)::UInt64
│      %1508  = Core.bitcast(Ptr{Float64}, %1507)::Ptr{Float64}
└─────          goto #494
494 ── %1510  = Base.pointerref(%1508, 1, 1)::Float64
└─────          goto #495
495 ──          goto #496
496 ──          $(Expr(:gc_preserve_end, :(%1440)))
└─────          goto #497
497 ──          goto #498
498 ── %1516  = StrideArraysCore.getfield(u, :data)::Vector{Float64}
│      %1517  = $(Expr(:gc_preserve_begin, :(%1516)))
│      %1518  = StrideArraysCore.getfield(u, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #503 if not true
499 ── %1520  = Core.tuple(2, %32, %1437, %26, %21)::NTuple{5, Int64}
│      %1521  = StrideArraysCore.getfield(%1518, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %1522  = Core.getfield(%1521, 5)::Int64
│      %1523  = Base.bitcast(UInt64, %1522)::UInt64
│      %1524  = Base.bitcast(Int64, %1523)::Int64
│      %1525  = Base.sle_int(1, %32)::Bool
│      %1526  = Base.sle_int(%32, 4)::Bool
│      %1527  = Base.and_int(%1525, %1526)::Bool
│      %1528  = Base.sle_int(1, %1437)::Bool
│      %1529  = Base.sle_int(%1437, 4)::Bool
│      %1530  = Base.and_int(%1528, %1529)::Bool
│      %1531  = Base.sle_int(1, %26)::Bool
│      %1532  = Base.sle_int(%26, 4)::Bool
│      %1533  = Base.and_int(%1531, %1532)::Bool
│      %1534  = Base.sub_int(%21, 1)::Int64
│      %1535  = Base.bitcast(UInt64, %1534)::UInt64
│      %1536  = Base.bitcast(UInt64, %1524)::UInt64
│      %1537  = Base.ult_int(%1535, %1536)::Bool
│      %1538  = Base.and_int(%1537, true)::Bool
│      %1539  = Base.and_int(%1533, %1538)::Bool
│      %1540  = Base.and_int(%1530, %1539)::Bool
│      %1541  = Base.and_int(%1527, %1540)::Bool
│      %1542  = Base.and_int(true, %1541)::Bool
└─────          goto #501 if not %1542
500 ──          goto #502
501 ──          invoke Base.throw_boundserror(%1518::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %1520::NTuple{5, Int64})::Union{}
└─────          unreachable
502 ──          nothing::Nothing
503 ┄─ %1548  = StrideArraysCore.getfield(%1518, :ptr)::Ptr{Float64}
│      %1549  = Base.sub_int(%32, 1)::Int64
│      %1550  = Base.sub_int(%1437, 1)::Int64
│      %1551  = Base.sub_int(%26, 1)::Int64
│      %1552  = Base.sub_int(%21, 1)::Int64
└─────          goto #512 if not true
504 ┄─ %1554  = φ (#503 => 2, #511 => %1566)::Int64
│      %1555  = Base.sle_int(1, %1554)::Bool
└─────          goto #506 if not %1555
505 ── %1557  = Base.sle_int(%1554, 5)::Bool
└─────          goto #507
506 ──          nothing::Nothing
507 ┄─ %1560  = φ (#505 => %1557, #506 => false)::Bool
└─────          goto #509 if not %1560
508 ──          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %1554, true)::Static.True
│      %1563  = Base.add_int(%1554, 1)::Int64
└─────          goto #510
509 ──          goto #510
510 ┄─ %1566  = φ (#508 => %1563)::Int64
│      %1567  = φ (#508 => false, #509 => true)::Bool
│      %1568  = Base.not_int(%1567)::Bool
└─────          goto #512 if not %1568
511 ──          goto #504
512 ┄─          goto #513
513 ──          goto #514
514 ── %1573  = Base.mul_int(%1552, 4)::Int64
│      %1574  = Base.add_int(%1551, %1573)::Int64
│      %1575  = Base.mul_int(%1574, 4)::Int64
│      %1576  = Base.add_int(%1550, %1575)::Int64
│      %1577  = Base.mul_int(%1576, 4)::Int64
│      %1578  = Base.add_int(%1549, %1577)::Int64
│      %1579  = Base.mul_int(%1578, 5)::Int64
│      %1580  = Base.add_int(1, %1579)::Int64
│      %1581  = Base.mul_int(8, %1580)::Int64
│      %1582  = Core.bitcast(Core.UInt, %1548)::UInt64
│      %1583  = Base.bitcast(UInt64, %1581)::UInt64
│      %1584  = Base.add_ptr(%1582, %1583)::UInt64
│      %1585  = Core.bitcast(Ptr{Float64}, %1584)::Ptr{Float64}
└─────          goto #515
515 ── %1587  = Base.pointerref(%1585, 1, 1)::Float64
└─────          goto #516
516 ──          goto #517
517 ──          $(Expr(:gc_preserve_end, :(%1517)))
└─────          goto #518
518 ──          goto #519
519 ── %1593  = StrideArraysCore.getfield(u, :data)::Vector{Float64}
│      %1594  = $(Expr(:gc_preserve_begin, :(%1593)))
│      %1595  = StrideArraysCore.getfield(u, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #524 if not true
520 ── %1597  = Core.tuple(3, %32, %1437, %26, %21)::NTuple{5, Int64}
│      %1598  = StrideArraysCore.getfield(%1595, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %1599  = Core.getfield(%1598, 5)::Int64
│      %1600  = Base.bitcast(UInt64, %1599)::UInt64
│      %1601  = Base.bitcast(Int64, %1600)::Int64
│      %1602  = Base.sle_int(1, %32)::Bool
│      %1603  = Base.sle_int(%32, 4)::Bool
│      %1604  = Base.and_int(%1602, %1603)::Bool
│      %1605  = Base.sle_int(1, %1437)::Bool
│      %1606  = Base.sle_int(%1437, 4)::Bool
│      %1607  = Base.and_int(%1605, %1606)::Bool
│      %1608  = Base.sle_int(1, %26)::Bool
│      %1609  = Base.sle_int(%26, 4)::Bool
│      %1610  = Base.and_int(%1608, %1609)::Bool
│      %1611  = Base.sub_int(%21, 1)::Int64
│      %1612  = Base.bitcast(UInt64, %1611)::UInt64
│      %1613  = Base.bitcast(UInt64, %1601)::UInt64
│      %1614  = Base.ult_int(%1612, %1613)::Bool
│      %1615  = Base.and_int(%1614, true)::Bool
│      %1616  = Base.and_int(%1610, %1615)::Bool
│      %1617  = Base.and_int(%1607, %1616)::Bool
│      %1618  = Base.and_int(%1604, %1617)::Bool
│      %1619  = Base.and_int(true, %1618)::Bool
└─────          goto #522 if not %1619
521 ──          goto #523
522 ──          invoke Base.throw_boundserror(%1595::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %1597::NTuple{5, Int64})::Union{}
└─────          unreachable
523 ──          nothing::Nothing
524 ┄─ %1625  = StrideArraysCore.getfield(%1595, :ptr)::Ptr{Float64}
│      %1626  = Base.sub_int(%32, 1)::Int64
│      %1627  = Base.sub_int(%1437, 1)::Int64
│      %1628  = Base.sub_int(%26, 1)::Int64
│      %1629  = Base.sub_int(%21, 1)::Int64
└─────          goto #533 if not true
525 ┄─ %1631  = φ (#524 => 2, #532 => %1643)::Int64
│      %1632  = Base.sle_int(1, %1631)::Bool
└─────          goto #527 if not %1632
526 ── %1634  = Base.sle_int(%1631, 5)::Bool
└─────          goto #528
527 ──          nothing::Nothing
528 ┄─ %1637  = φ (#526 => %1634, #527 => false)::Bool
└─────          goto #530 if not %1637
529 ──          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %1631, true)::Static.True
│      %1640  = Base.add_int(%1631, 1)::Int64
└─────          goto #531
530 ──          goto #531
531 ┄─ %1643  = φ (#529 => %1640)::Int64
│      %1644  = φ (#529 => false, #530 => true)::Bool
│      %1645  = Base.not_int(%1644)::Bool
└─────          goto #533 if not %1645
532 ──          goto #525
533 ┄─          goto #534
534 ──          goto #535
535 ── %1650  = Base.mul_int(%1629, 4)::Int64
│      %1651  = Base.add_int(%1628, %1650)::Int64
│      %1652  = Base.mul_int(%1651, 4)::Int64
│      %1653  = Base.add_int(%1627, %1652)::Int64
│      %1654  = Base.mul_int(%1653, 4)::Int64
│      %1655  = Base.add_int(%1626, %1654)::Int64
│      %1656  = Base.mul_int(%1655, 5)::Int64
│      %1657  = Base.add_int(2, %1656)::Int64
│      %1658  = Base.mul_int(8, %1657)::Int64
│      %1659  = Core.bitcast(Core.UInt, %1625)::UInt64
│      %1660  = Base.bitcast(UInt64, %1658)::UInt64
│      %1661  = Base.add_ptr(%1659, %1660)::UInt64
│      %1662  = Core.bitcast(Ptr{Float64}, %1661)::Ptr{Float64}
└─────          goto #536
536 ── %1664  = Base.pointerref(%1662, 1, 1)::Float64
└─────          goto #537
537 ──          goto #538
538 ──          $(Expr(:gc_preserve_end, :(%1594)))
└─────          goto #539
539 ──          goto #540
540 ── %1670  = StrideArraysCore.getfield(u, :data)::Vector{Float64}
│      %1671  = $(Expr(:gc_preserve_begin, :(%1670)))
│      %1672  = StrideArraysCore.getfield(u, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #545 if not true
541 ── %1674  = Core.tuple(4, %32, %1437, %26, %21)::NTuple{5, Int64}
│      %1675  = StrideArraysCore.getfield(%1672, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %1676  = Core.getfield(%1675, 5)::Int64
│      %1677  = Base.bitcast(UInt64, %1676)::UInt64
│      %1678  = Base.bitcast(Int64, %1677)::Int64
│      %1679  = Base.sle_int(1, %32)::Bool
│      %1680  = Base.sle_int(%32, 4)::Bool
│      %1681  = Base.and_int(%1679, %1680)::Bool
│      %1682  = Base.sle_int(1, %1437)::Bool
│      %1683  = Base.sle_int(%1437, 4)::Bool
│      %1684  = Base.and_int(%1682, %1683)::Bool
│      %1685  = Base.sle_int(1, %26)::Bool
│      %1686  = Base.sle_int(%26, 4)::Bool
│      %1687  = Base.and_int(%1685, %1686)::Bool
│      %1688  = Base.sub_int(%21, 1)::Int64
│      %1689  = Base.bitcast(UInt64, %1688)::UInt64
│      %1690  = Base.bitcast(UInt64, %1678)::UInt64
│      %1691  = Base.ult_int(%1689, %1690)::Bool
│      %1692  = Base.and_int(%1691, true)::Bool
│      %1693  = Base.and_int(%1687, %1692)::Bool
│      %1694  = Base.and_int(%1684, %1693)::Bool
│      %1695  = Base.and_int(%1681, %1694)::Bool
│      %1696  = Base.and_int(true, %1695)::Bool
└─────          goto #543 if not %1696
542 ──          goto #544
543 ──          invoke Base.throw_boundserror(%1672::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %1674::NTuple{5, Int64})::Union{}
└─────          unreachable
544 ──          nothing::Nothing
545 ┄─ %1702  = StrideArraysCore.getfield(%1672, :ptr)::Ptr{Float64}
│      %1703  = Base.sub_int(%32, 1)::Int64
│      %1704  = Base.sub_int(%1437, 1)::Int64
│      %1705  = Base.sub_int(%26, 1)::Int64
│      %1706  = Base.sub_int(%21, 1)::Int64
└─────          goto #554 if not true
546 ┄─ %1708  = φ (#545 => 2, #553 => %1720)::Int64
│      %1709  = Base.sle_int(1, %1708)::Bool
└─────          goto #548 if not %1709
547 ── %1711  = Base.sle_int(%1708, 5)::Bool
└─────          goto #549
548 ──          nothing::Nothing
549 ┄─ %1714  = φ (#547 => %1711, #548 => false)::Bool
└─────          goto #551 if not %1714
550 ──          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %1708, true)::Static.True
│      %1717  = Base.add_int(%1708, 1)::Int64
└─────          goto #552
551 ──          goto #552
552 ┄─ %1720  = φ (#550 => %1717)::Int64
│      %1721  = φ (#550 => false, #551 => true)::Bool
│      %1722  = Base.not_int(%1721)::Bool
└─────          goto #554 if not %1722
553 ──          goto #546
554 ┄─          goto #555
555 ──          goto #556
556 ── %1727  = Base.mul_int(%1706, 4)::Int64
│      %1728  = Base.add_int(%1705, %1727)::Int64
│      %1729  = Base.mul_int(%1728, 4)::Int64
│      %1730  = Base.add_int(%1704, %1729)::Int64
│      %1731  = Base.mul_int(%1730, 4)::Int64
│      %1732  = Base.add_int(%1703, %1731)::Int64
│      %1733  = Base.mul_int(%1732, 5)::Int64
│      %1734  = Base.add_int(3, %1733)::Int64
│      %1735  = Base.mul_int(8, %1734)::Int64
│      %1736  = Core.bitcast(Core.UInt, %1702)::UInt64
│      %1737  = Base.bitcast(UInt64, %1735)::UInt64
│      %1738  = Base.add_ptr(%1736, %1737)::UInt64
│      %1739  = Core.bitcast(Ptr{Float64}, %1738)::Ptr{Float64}
└─────          goto #557
557 ── %1741  = Base.pointerref(%1739, 1, 1)::Float64
└─────          goto #558
558 ──          goto #559
559 ──          $(Expr(:gc_preserve_end, :(%1671)))
└─────          goto #560
560 ──          goto #561
561 ── %1747  = StrideArraysCore.getfield(u, :data)::Vector{Float64}
│      %1748  = $(Expr(:gc_preserve_begin, :(%1747)))
│      %1749  = StrideArraysCore.getfield(u, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #566 if not true
562 ── %1751  = Core.tuple(5, %32, %1437, %26, %21)::NTuple{5, Int64}
│      %1752  = StrideArraysCore.getfield(%1749, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %1753  = Core.getfield(%1752, 5)::Int64
│      %1754  = Base.bitcast(UInt64, %1753)::UInt64
│      %1755  = Base.bitcast(Int64, %1754)::Int64
│      %1756  = Base.sle_int(1, %32)::Bool
│      %1757  = Base.sle_int(%32, 4)::Bool
│      %1758  = Base.and_int(%1756, %1757)::Bool
│      %1759  = Base.sle_int(1, %1437)::Bool
│      %1760  = Base.sle_int(%1437, 4)::Bool
│      %1761  = Base.and_int(%1759, %1760)::Bool
│      %1762  = Base.sle_int(1, %26)::Bool
│      %1763  = Base.sle_int(%26, 4)::Bool
│      %1764  = Base.and_int(%1762, %1763)::Bool
│      %1765  = Base.sub_int(%21, 1)::Int64
│      %1766  = Base.bitcast(UInt64, %1765)::UInt64
│      %1767  = Base.bitcast(UInt64, %1755)::UInt64
│      %1768  = Base.ult_int(%1766, %1767)::Bool
│      %1769  = Base.and_int(%1768, true)::Bool
│      %1770  = Base.and_int(%1764, %1769)::Bool
│      %1771  = Base.and_int(%1761, %1770)::Bool
│      %1772  = Base.and_int(%1758, %1771)::Bool
│      %1773  = Base.and_int(true, %1772)::Bool
└─────          goto #564 if not %1773
563 ──          goto #565
564 ──          invoke Base.throw_boundserror(%1749::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %1751::NTuple{5, Int64})::Union{}
└─────          unreachable
565 ──          nothing::Nothing
566 ┄─ %1779  = StrideArraysCore.getfield(%1749, :ptr)::Ptr{Float64}
│      %1780  = Base.sub_int(%32, 1)::Int64
│      %1781  = Base.sub_int(%1437, 1)::Int64
│      %1782  = Base.sub_int(%26, 1)::Int64
│      %1783  = Base.sub_int(%21, 1)::Int64
└─────          goto #575 if not true
567 ┄─ %1785  = φ (#566 => 2, #574 => %1797)::Int64
│      %1786  = Base.sle_int(1, %1785)::Bool
└─────          goto #569 if not %1786
568 ── %1788  = Base.sle_int(%1785, 5)::Bool
└─────          goto #570
569 ──          nothing::Nothing
570 ┄─ %1791  = φ (#568 => %1788, #569 => false)::Bool
└─────          goto #572 if not %1791
571 ──          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %1785, true)::Static.True
│      %1794  = Base.add_int(%1785, 1)::Int64
└─────          goto #573
572 ──          goto #573
573 ┄─ %1797  = φ (#571 => %1794)::Int64
│      %1798  = φ (#571 => false, #572 => true)::Bool
│      %1799  = Base.not_int(%1798)::Bool
└─────          goto #575 if not %1799
574 ──          goto #567
575 ┄─          goto #576
576 ──          goto #577
577 ── %1804  = Base.mul_int(%1783, 4)::Int64
│      %1805  = Base.add_int(%1782, %1804)::Int64
│      %1806  = Base.mul_int(%1805, 4)::Int64
│      %1807  = Base.add_int(%1781, %1806)::Int64
│      %1808  = Base.mul_int(%1807, 4)::Int64
│      %1809  = Base.add_int(%1780, %1808)::Int64
│      %1810  = Base.mul_int(%1809, 5)::Int64
│      %1811  = Base.add_int(4, %1810)::Int64
│      %1812  = Base.mul_int(8, %1811)::Int64
│      %1813  = Core.bitcast(Core.UInt, %1779)::UInt64
│      %1814  = Base.bitcast(UInt64, %1812)::UInt64
│      %1815  = Base.add_ptr(%1813, %1814)::UInt64
│      %1816  = Core.bitcast(Ptr{Float64}, %1815)::Ptr{Float64}
└─────          goto #578
578 ── %1818  = Base.pointerref(%1816, 1, 1)::Float64
└─────          goto #579
579 ──          goto #580
580 ──          $(Expr(:gc_preserve_end, :(%1748)))
└─────          goto #581
581 ──          goto #582
582 ──          goto #583
583 ──          goto #584
584 ──          goto #586
585 ──          nothing::Nothing
586 ┄─          goto #588
587 ──          nothing::Nothing
588 ┄─          goto #589
589 ──          goto #591
590 ──          nothing::Nothing
591 ┄─          goto #592
592 ──          goto #594
593 ──          nothing::Nothing
594 ┄─          goto #596
595 ──          nothing::Nothing
596 ┄─          goto #597
597 ──          goto #599
598 ──          nothing::Nothing
599 ┄─          goto #600
600 ──          goto #602
601 ──          nothing::Nothing
602 ┄─          goto #604
603 ──          nothing::Nothing
604 ┄─          goto #605
605 ──          goto #607
606 ──          nothing::Nothing
607 ┄─          goto #608
608 ──          goto #610
609 ──          nothing::Nothing
610 ┄─          goto #612
611 ──          nothing::Nothing
612 ┄─          goto #613
613 ──          goto #615
614 ──          nothing::Nothing
615 ┄─          goto #616
616 ── %1858  = Base.div_float(%182, %105)::Float64
│      %1859  = Base.div_float(%259, %105)::Float64
│      %1860  = Base.div_float(%336, %105)::Float64
│      %1861  = Base.getfield(equations, :gamma)::Float64
│      %1862  = Base.sub_float(%1861, 1.0)::Float64
│      %1863  = Base.mul_float(%182, %1858)::Float64
│      %1864  = Base.muladd_float(%259, %1859, %1863)::Float64
│      %1865  = Base.muladd_float(%336, %1860, %1864)::Float64
│      %1866  = Base.muladd_float(-0.5, %1865, %413)::Float64
│      %1867  = Base.mul_float(%1862, %1866)::Float64
└─────          goto #617
617 ──          goto #619
618 ──          nothing::Nothing
619 ┄─          goto #621
620 ──          nothing::Nothing
621 ┄─          goto #622
622 ──          goto #624
623 ──          nothing::Nothing
624 ┄─          goto #625
625 ──          goto #627
626 ──          nothing::Nothing
627 ┄─          goto #629
628 ──          nothing::Nothing
629 ┄─          goto #630
630 ──          goto #632
631 ──          nothing::Nothing
632 ┄─          goto #633
633 ──          goto #635
634 ──          nothing::Nothing
635 ┄─          goto #637
636 ──          nothing::Nothing
637 ┄─          goto #638
638 ──          goto #640
639 ──          nothing::Nothing
640 ┄─          goto #641
641 ──          goto #643
642 ──          nothing::Nothing
643 ┄─          goto #645
644 ──          nothing::Nothing
645 ┄─          goto #646
646 ──          goto #648
647 ──          nothing::Nothing
648 ┄─          goto #649
649 ──          goto #651
650 ──          nothing::Nothing
651 ┄─          goto #653
652 ──          nothing::Nothing
653 ┄─          goto #654
654 ──          goto #656
655 ──          nothing::Nothing
656 ┄─          goto #657
657 ──          goto #659
658 ──          nothing::Nothing
659 ┄─          goto #661
660 ──          nothing::Nothing
661 ┄─          goto #662
662 ──          goto #664
663 ──          nothing::Nothing
664 ┄─          goto #665
665 ──          goto #667
666 ──          nothing::Nothing
667 ┄─          goto #669
668 ──          nothing::Nothing
669 ┄─          goto #670
670 ──          goto #672
671 ──          nothing::Nothing
672 ┄─          goto #673
673 ──          goto #675
674 ──          nothing::Nothing
675 ┄─          goto #677
676 ──          nothing::Nothing
677 ┄─          goto #678
678 ──          goto #680
679 ──          nothing::Nothing
680 ┄─          goto #681
681 ── %1933  = Base.div_float(%1587, %1510)::Float64
│      %1934  = Base.div_float(%1664, %1510)::Float64
│      %1935  = Base.div_float(%1741, %1510)::Float64
│      %1936  = Base.getfield(equations, :gamma)::Float64
│      %1937  = Base.sub_float(%1936, 1.0)::Float64
│      %1938  = Base.mul_float(%1587, %1933)::Float64
│      %1939  = Base.muladd_float(%1664, %1934, %1938)::Float64
│      %1940  = Base.muladd_float(%1741, %1935, %1939)::Float64
│      %1941  = Base.muladd_float(-0.5, %1940, %1818)::Float64
│      %1942  = Base.mul_float(%1937, %1941)::Float64
└─────          goto #682
682 ──          goto #684
683 ──          nothing::Nothing
684 ┄─          goto #686
685 ──          nothing::Nothing
686 ┄─          goto #687
687 ──          goto #689
688 ──          nothing::Nothing
689 ┄─          goto #690
690 ──          goto #692
691 ──          nothing::Nothing
692 ┄─          goto #694
693 ──          nothing::Nothing
694 ┄─          goto #695
695 ──          goto #697
696 ──          nothing::Nothing
697 ┄─          goto #698
698 ──          goto #700
699 ──          nothing::Nothing
700 ┄─          goto #702
701 ──          nothing::Nothing
702 ┄─          goto #703
703 ──          goto #705
704 ──          nothing::Nothing
705 ┄─          goto #706
706 ──          goto #708
707 ──          nothing::Nothing
708 ┄─          goto #710
709 ──          nothing::Nothing
710 ┄─          goto #711
711 ──          goto #713
712 ──          nothing::Nothing
713 ┄─          goto #714
714 ── %1976  = Base.muladd_float(-2.0, %1510, %105)::Float64
│      %1977  = Base.mul_float(%105, %1976)::Float64
│      %1978  = Base.muladd_float(%1510, %1510, %1977)::Float64
│      %1979  = Base.muladd_float(2.0, %1510, %105)::Float64
│      %1980  = Base.mul_float(%105, %1979)::Float64
│      %1981  = Base.muladd_float(%1510, %1510, %1980)::Float64
│      %1982  = Base.div_float(%1978, %1981)::Float64
│      %1983  = Base.lt_float(%1982, 0.0001)::Bool
└─────          goto #716 if not %1983
715 ── %1985  = Base.add_float(%105, %1510)::Float64
│      %1986  = Base.muladd_float(%1982, 0.2857142857142857, 0.4)::Float64
│      %1987  = Base.muladd_float(%1982, %1986, 0.6666666666666666)::Float64
│      %1988  = Base.muladd_float(%1982, %1987, 2.0)::Float64
│      %1989  = Base.div_float(%1985, %1988)::Float64
└─────          goto #717
716 ── %1991  = Base.sub_float(%1510, %105)::Float64
│      %1992  = Base.div_float(%1510, %105)::Float64
│      %1993  = $(Expr(:foreigncall, "llvm.log.f64", Float64, svec(Float64), 0, :(:llvmcall), :(%1992), :(%1992)))::Float64
│      %1994  = Base.div_float(%1991, %1993)::Float64
└─────          goto #717
717 ┄─ %1996  = φ (#715 => %1989, #716 => %1994)::Float64
│      %1997  = Base.mul_float(%105, %1942)::Float64
│      %1998  = Base.mul_float(%1510, %1867)::Float64
│      %1999  = Base.muladd_float(-2.0, %1998, %1997)::Float64
│      %2000  = Base.mul_float(%1997, %1999)::Float64
│      %2001  = Base.muladd_float(%1998, %1998, %2000)::Float64
│      %2002  = Base.muladd_float(2.0, %1998, %1997)::Float64
│      %2003  = Base.mul_float(%1997, %2002)::Float64
│      %2004  = Base.muladd_float(%1998, %1998, %2003)::Float64
│      %2005  = Base.div_float(%2001, %2004)::Float64
│      %2006  = Base.lt_float(%2005, 0.0001)::Bool
└─────          goto #719 if not %2006
718 ── %2008  = Base.muladd_float(%2005, 0.2857142857142857, 0.4)::Float64
│      %2009  = Base.muladd_float(%2005, %2008, 0.6666666666666666)::Float64
│      %2010  = Base.muladd_float(%2005, %2009, 2.0)::Float64
│      %2011  = Base.add_float(%1997, %1998)::Float64
│      %2012  = Base.div_float(%2010, %2011)::Float64
└─────          goto #720
719 ── %2014  = Base.div_float(%1998, %1997)::Float64
│      %2015  = $(Expr(:foreigncall, "llvm.log.f64", Float64, svec(Float64), 0, :(:llvmcall), :(%2014), :(%2014)))::Float64
│      %2016  = Base.sub_float(%1998, %1997)::Float64
│      %2017  = Base.div_float(%2015, %2016)::Float64
└─────          goto #720
720 ┄─ %2019  = φ (#718 => %2012, #719 => %2017)::Float64
│      %2020  = Base.mul_float(%1867, %1942)::Float64
│      %2021  = Base.mul_float(%2020, %2019)::Float64
│      %2022  = Base.add_float(%1858, %1933)::Float64
│      %2023  = Base.mul_float(0.5, %2022)::Float64
│      %2024  = Base.add_float(%1859, %1934)::Float64
│      %2025  = Base.mul_float(0.5, %2024)::Float64
│      %2026  = Base.add_float(%1860, %1935)::Float64
│      %2027  = Base.mul_float(0.5, %2026)::Float64
│      %2028  = Base.add_float(%1867, %1942)::Float64
│      %2029  = Base.mul_float(0.5, %2028)::Float64
│      %2030  = Base.mul_float(%1858, %1933)::Float64
│      %2031  = Base.muladd_float(%1859, %1934, %2030)::Float64
│      %2032  = Base.muladd_float(%1860, %1935, %2031)::Float64
│      %2033  = Base.mul_float(0.5, %2032)::Float64
│      %2034  = Base.mul_float(%1996, %2025)::Float64
│      %2035  = Base.mul_float(%2034, %2023)::Float64
│      %2036  = Base.muladd_float(%2034, %2025, %2029)::Float64
│      %2037  = Base.mul_float(%2034, %2027)::Float64
│      %2038  = Base.mul_float(%1867, %1934)::Float64
│      %2039  = Base.muladd_float(%1942, %1859, %2038)::Float64
│      %2040  = Base.getfield(equations, :inv_gamma_minus_one)::Float64
│      %2041  = Base.muladd_float(%2021, %2040, %2033)::Float64
│      %2042  = Base.mul_float(%2034, %2041)::Float64
│      %2043  = Base.muladd_float(0.5, %2039, %2042)::Float64
│      %2044  = Core.tuple(%2034, %2035, %2036, %2037, %2043)::NTuple{5, Float64}
└─────          goto #721
721 ── %2046  = Base.arrayref(false, %24, %29, %1437)::Float64
│      %2047  = Base.copysign_float(0.0, %2046)::Float64
│      %2048  = Core.ifelse(true, %2046, %2047)::Float64
└─────          goto #767 if not true
722 ┄─ %2050  = φ (#721 => 1, #766 => %2219)::Int64
│      %2051  = φ (#721 => 1, #766 => %2220)::Int64
│      %2052  = Base.getfield(%2044, %2050, true)::Float64
│      %2053  = StrideArraysCore.getfield(du, :data)::Vector{Float64}
│      %2054  = $(Expr(:gc_preserve_begin, :(%2053)))
│      %2055  = StrideArraysCore.getfield(du, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #727 if not true
723 ── %2057  = Core.tuple(%2050, %32, %29, %26, %21)::NTuple{5, Int64}
│      %2058  = StrideArraysCore.getfield(%2055, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %2059  = Core.getfield(%2058, 5)::Int64
│      %2060  = Base.bitcast(UInt64, %2059)::UInt64
│      %2061  = Base.bitcast(Int64, %2060)::Int64
│      %2062  = Base.sle_int(1, %2050)::Bool
│      %2063  = Base.sle_int(%2050, 5)::Bool
│      %2064  = Base.and_int(%2062, %2063)::Bool
│      %2065  = Base.sle_int(1, %32)::Bool
│      %2066  = Base.sle_int(%32, 4)::Bool
│      %2067  = Base.and_int(%2065, %2066)::Bool
│      %2068  = Base.sle_int(1, %29)::Bool
│      %2069  = Base.sle_int(%29, 4)::Bool
│      %2070  = Base.and_int(%2068, %2069)::Bool
│      %2071  = Base.sle_int(1, %26)::Bool
│      %2072  = Base.sle_int(%26, 4)::Bool
│      %2073  = Base.and_int(%2071, %2072)::Bool
│      %2074  = Base.sub_int(%21, 1)::Int64
│      %2075  = Base.bitcast(UInt64, %2074)::UInt64
│      %2076  = Base.bitcast(UInt64, %2061)::UInt64
│      %2077  = Base.ult_int(%2075, %2076)::Bool
│      %2078  = Base.and_int(%2077, true)::Bool
│      %2079  = Base.and_int(%2073, %2078)::Bool
│      %2080  = Base.and_int(%2070, %2079)::Bool
│      %2081  = Base.and_int(%2067, %2080)::Bool
│      %2082  = Base.and_int(%2064, %2081)::Bool
└─────          goto #725 if not %2082
724 ──          goto #726
725 ──          invoke Base.throw_boundserror(%2055::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %2057::NTuple{5, Int64})::Union{}
└─────          unreachable
726 ──          nothing::Nothing
727 ┄─ %2088  = StrideArraysCore.getfield(%2055, :ptr)::Ptr{Float64}
│      %2089  = Base.sub_int(%2050, 1)::Int64
│      %2090  = Base.sub_int(%32, 1)::Int64
│      %2091  = Base.sub_int(%29, 1)::Int64
│      %2092  = Base.sub_int(%26, 1)::Int64
│      %2093  = Base.sub_int(%21, 1)::Int64
└─────          goto #736 if not true
728 ┄─ %2095  = φ (#727 => 2, #735 => %2107)::Int64
│      %2096  = Base.sle_int(1, %2095)::Bool
└─────          goto #730 if not %2096
729 ── %2098  = Base.sle_int(%2095, 5)::Bool
└─────          goto #731
730 ──          nothing::Nothing
731 ┄─ %2101  = φ (#729 => %2098, #730 => false)::Bool
└─────          goto #733 if not %2101
732 ──          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %2095, true)::Static.True
│      %2104  = Base.add_int(%2095, 1)::Int64
└─────          goto #734
733 ──          goto #734
734 ┄─ %2107  = φ (#732 => %2104)::Int64
│      %2108  = φ (#732 => false, #733 => true)::Bool
│      %2109  = Base.not_int(%2108)::Bool
└─────          goto #736 if not %2109
735 ──          goto #728
736 ┄─          goto #737
737 ──          goto #738
738 ── %2114  = Base.mul_int(%2093, 4)::Int64
│      %2115  = Base.add_int(%2092, %2114)::Int64
│      %2116  = Base.mul_int(%2115, 4)::Int64
│      %2117  = Base.add_int(%2091, %2116)::Int64
│      %2118  = Base.mul_int(%2117, 4)::Int64
│      %2119  = Base.add_int(%2090, %2118)::Int64
│      %2120  = Base.mul_int(%2119, 5)::Int64
│      %2121  = Base.add_int(%2089, %2120)::Int64
│      %2122  = Base.mul_int(8, %2121)::Int64
│      %2123  = Core.bitcast(Core.UInt, %2088)::UInt64
│      %2124  = Base.bitcast(UInt64, %2122)::UInt64
│      %2125  = Base.add_ptr(%2123, %2124)::UInt64
│      %2126  = Core.bitcast(Ptr{Float64}, %2125)::Ptr{Float64}
└─────          goto #739
739 ── %2128  = Base.pointerref(%2126, 1, 1)::Float64
└─────          goto #740
740 ──          goto #741
741 ──          $(Expr(:gc_preserve_end, :(%2054)))
└─────          goto #742
742 ── %2133  = Base.muladd_float(%2048, %2052, %2128)::Float64
│      %2134  = StrideArraysCore.getfield(du, :data)::Vector{Float64}
│      %2135  = $(Expr(:gc_preserve_begin, :(%2134)))
│      %2136  = StrideArraysCore.getfield(du, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #747 if not true
743 ── %2138  = Core.tuple(%2050, %32, %29, %26, %21)::NTuple{5, Int64}
│      %2139  = StrideArraysCore.getfield(%2136, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %2140  = Core.getfield(%2139, 5)::Int64
│      %2141  = Base.bitcast(UInt64, %2140)::UInt64
│      %2142  = Base.bitcast(Int64, %2141)::Int64
│      %2143  = Base.sle_int(1, %2050)::Bool
│      %2144  = Base.sle_int(%2050, 5)::Bool
│      %2145  = Base.and_int(%2143, %2144)::Bool
│      %2146  = Base.sle_int(1, %32)::Bool
│      %2147  = Base.sle_int(%32, 4)::Bool
│      %2148  = Base.and_int(%2146, %2147)::Bool
│      %2149  = Base.sle_int(1, %29)::Bool
│      %2150  = Base.sle_int(%29, 4)::Bool
│      %2151  = Base.and_int(%2149, %2150)::Bool
│      %2152  = Base.sle_int(1, %26)::Bool
│      %2153  = Base.sle_int(%26, 4)::Bool
│      %2154  = Base.and_int(%2152, %2153)::Bool
│      %2155  = Base.sub_int(%21, 1)::Int64
│      %2156  = Base.bitcast(UInt64, %2155)::UInt64
│      %2157  = Base.bitcast(UInt64, %2142)::UInt64
│      %2158  = Base.ult_int(%2156, %2157)::Bool
│      %2159  = Base.and_int(%2158, true)::Bool
│      %2160  = Base.and_int(%2154, %2159)::Bool
│      %2161  = Base.and_int(%2151, %2160)::Bool
│      %2162  = Base.and_int(%2148, %2161)::Bool
│      %2163  = Base.and_int(%2145, %2162)::Bool
└─────          goto #745 if not %2163
744 ──          goto #746
745 ──          invoke Base.throw_boundserror(%2136::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %2138::NTuple{5, Int64})::Union{}
└─────          unreachable
746 ──          nothing::Nothing
747 ┄─ %2169  = StrideArraysCore.getfield(%2136, :ptr)::Ptr{Float64}
│      %2170  = Base.sub_int(%2050, 1)::Int64
│      %2171  = Base.sub_int(%32, 1)::Int64
│      %2172  = Base.sub_int(%29, 1)::Int64
│      %2173  = Base.sub_int(%26, 1)::Int64
│      %2174  = Base.sub_int(%21, 1)::Int64
└─────          goto #756 if not true
748 ┄─ %2176  = φ (#747 => 2, #755 => %2188)::Int64
│      %2177  = Base.sle_int(1, %2176)::Bool
└─────          goto #750 if not %2177
749 ── %2179  = Base.sle_int(%2176, 5)::Bool
└─────          goto #751
750 ──          nothing::Nothing
751 ┄─ %2182  = φ (#749 => %2179, #750 => false)::Bool
└─────          goto #753 if not %2182
752 ──          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %2176, true)::Static.True
│      %2185  = Base.add_int(%2176, 1)::Int64
└─────          goto #754
753 ──          goto #754
754 ┄─ %2188  = φ (#752 => %2185)::Int64
│      %2189  = φ (#752 => false, #753 => true)::Bool
│      %2190  = Base.not_int(%2189)::Bool
└─────          goto #756 if not %2190
755 ──          goto #748
756 ┄─          goto #757
757 ──          goto #758
758 ── %2195  = Base.mul_int(%2174, 4)::Int64
│      %2196  = Base.add_int(%2173, %2195)::Int64
│      %2197  = Base.mul_int(%2196, 4)::Int64
│      %2198  = Base.add_int(%2172, %2197)::Int64
│      %2199  = Base.mul_int(%2198, 4)::Int64
│      %2200  = Base.add_int(%2171, %2199)::Int64
│      %2201  = Base.mul_int(%2200, 5)::Int64
│      %2202  = Base.add_int(%2170, %2201)::Int64
│      %2203  = Base.mul_int(8, %2202)::Int64
│      %2204  = Core.bitcast(Core.UInt, %2169)::UInt64
│      %2205  = Base.bitcast(UInt64, %2203)::UInt64
│      %2206  = Base.add_ptr(%2204, %2205)::UInt64
│      %2207  = Core.bitcast(Ptr{Float64}, %2206)::Ptr{Float64}
└─────          goto #759
759 ──          Base.pointerset(%2207, %2133, 1, 1)::Ptr{Float64}
└─────          goto #760
760 ──          goto #761
761 ──          $(Expr(:gc_preserve_end, :(%2135)))
└─────          goto #762
762 ── %2214  = (%2051 === 5)::Bool
└─────          goto #764 if not %2214
763 ──          goto #765
764 ── %2217  = Base.add_int(%2051, 1)::Int64
└─────          goto #765
765 ┄─ %2219  = φ (#764 => %2217)::Int64
│      %2220  = φ (#764 => %2217)::Int64
│      %2221  = φ (#763 => true, #764 => false)::Bool
│      %2222  = Base.not_int(%2221)::Bool
└─────          goto #767 if not %2222
766 ──          goto #722
767 ┄─          goto #768
768 ── %2226  = Base.arrayref(false, %24, %1437, %29)::Float64
│      %2227  = Base.copysign_float(0.0, %2226)::Float64
│      %2228  = Core.ifelse(true, %2226, %2227)::Float64
└─────          goto #814 if not true
769 ┄─ %2230  = φ (#768 => 1, #813 => %2399)::Int64
│      %2231  = φ (#768 => 1, #813 => %2400)::Int64
│      %2232  = Base.getfield(%2044, %2230, true)::Float64
│      %2233  = StrideArraysCore.getfield(du, :data)::Vector{Float64}
│      %2234  = $(Expr(:gc_preserve_begin, :(%2233)))
│      %2235  = StrideArraysCore.getfield(du, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #774 if not true
770 ── %2237  = Core.tuple(%2230, %32, %1437, %26, %21)::NTuple{5, Int64}
│      %2238  = StrideArraysCore.getfield(%2235, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %2239  = Core.getfield(%2238, 5)::Int64
│      %2240  = Base.bitcast(UInt64, %2239)::UInt64
│      %2241  = Base.bitcast(Int64, %2240)::Int64
│      %2242  = Base.sle_int(1, %2230)::Bool
│      %2243  = Base.sle_int(%2230, 5)::Bool
│      %2244  = Base.and_int(%2242, %2243)::Bool
│      %2245  = Base.sle_int(1, %32)::Bool
│      %2246  = Base.sle_int(%32, 4)::Bool
│      %2247  = Base.and_int(%2245, %2246)::Bool
│      %2248  = Base.sle_int(1, %1437)::Bool
│      %2249  = Base.sle_int(%1437, 4)::Bool
│      %2250  = Base.and_int(%2248, %2249)::Bool
│      %2251  = Base.sle_int(1, %26)::Bool
│      %2252  = Base.sle_int(%26, 4)::Bool
│      %2253  = Base.and_int(%2251, %2252)::Bool
│      %2254  = Base.sub_int(%21, 1)::Int64
│      %2255  = Base.bitcast(UInt64, %2254)::UInt64
│      %2256  = Base.bitcast(UInt64, %2241)::UInt64
│      %2257  = Base.ult_int(%2255, %2256)::Bool
│      %2258  = Base.and_int(%2257, true)::Bool
│      %2259  = Base.and_int(%2253, %2258)::Bool
│      %2260  = Base.and_int(%2250, %2259)::Bool
│      %2261  = Base.and_int(%2247, %2260)::Bool
│      %2262  = Base.and_int(%2244, %2261)::Bool
└─────          goto #772 if not %2262
771 ──          goto #773
772 ──          invoke Base.throw_boundserror(%2235::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %2237::NTuple{5, Int64})::Union{}
└─────          unreachable
773 ──          nothing::Nothing
774 ┄─ %2268  = StrideArraysCore.getfield(%2235, :ptr)::Ptr{Float64}
│      %2269  = Base.sub_int(%2230, 1)::Int64
│      %2270  = Base.sub_int(%32, 1)::Int64
│      %2271  = Base.sub_int(%1437, 1)::Int64
│      %2272  = Base.sub_int(%26, 1)::Int64
│      %2273  = Base.sub_int(%21, 1)::Int64
└─────          goto #783 if not true
775 ┄─ %2275  = φ (#774 => 2, #782 => %2287)::Int64
│      %2276  = Base.sle_int(1, %2275)::Bool
└─────          goto #777 if not %2276
776 ── %2278  = Base.sle_int(%2275, 5)::Bool
└─────          goto #778
777 ──          nothing::Nothing
778 ┄─ %2281  = φ (#776 => %2278, #777 => false)::Bool
└─────          goto #780 if not %2281
779 ──          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %2275, true)::Static.True
│      %2284  = Base.add_int(%2275, 1)::Int64
└─────          goto #781
780 ──          goto #781
781 ┄─ %2287  = φ (#779 => %2284)::Int64
│      %2288  = φ (#779 => false, #780 => true)::Bool
│      %2289  = Base.not_int(%2288)::Bool
└─────          goto #783 if not %2289
782 ──          goto #775
783 ┄─          goto #784
784 ──          goto #785
785 ── %2294  = Base.mul_int(%2273, 4)::Int64
│      %2295  = Base.add_int(%2272, %2294)::Int64
│      %2296  = Base.mul_int(%2295, 4)::Int64
│      %2297  = Base.add_int(%2271, %2296)::Int64
│      %2298  = Base.mul_int(%2297, 4)::Int64
│      %2299  = Base.add_int(%2270, %2298)::Int64
│      %2300  = Base.mul_int(%2299, 5)::Int64
│      %2301  = Base.add_int(%2269, %2300)::Int64
│      %2302  = Base.mul_int(8, %2301)::Int64
│      %2303  = Core.bitcast(Core.UInt, %2268)::UInt64
│      %2304  = Base.bitcast(UInt64, %2302)::UInt64
│      %2305  = Base.add_ptr(%2303, %2304)::UInt64
│      %2306  = Core.bitcast(Ptr{Float64}, %2305)::Ptr{Float64}
└─────          goto #786
786 ── %2308  = Base.pointerref(%2306, 1, 1)::Float64
└─────          goto #787
787 ──          goto #788
788 ──          $(Expr(:gc_preserve_end, :(%2234)))
└─────          goto #789
789 ── %2313  = Base.muladd_float(%2228, %2232, %2308)::Float64
│      %2314  = StrideArraysCore.getfield(du, :data)::Vector{Float64}
│      %2315  = $(Expr(:gc_preserve_begin, :(%2314)))
│      %2316  = StrideArraysCore.getfield(du, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #794 if not true
790 ── %2318  = Core.tuple(%2230, %32, %1437, %26, %21)::NTuple{5, Int64}
│      %2319  = StrideArraysCore.getfield(%2316, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %2320  = Core.getfield(%2319, 5)::Int64
│      %2321  = Base.bitcast(UInt64, %2320)::UInt64
│      %2322  = Base.bitcast(Int64, %2321)::Int64
│      %2323  = Base.sle_int(1, %2230)::Bool
│      %2324  = Base.sle_int(%2230, 5)::Bool
│      %2325  = Base.and_int(%2323, %2324)::Bool
│      %2326  = Base.sle_int(1, %32)::Bool
│      %2327  = Base.sle_int(%32, 4)::Bool
│      %2328  = Base.and_int(%2326, %2327)::Bool
│      %2329  = Base.sle_int(1, %1437)::Bool
│      %2330  = Base.sle_int(%1437, 4)::Bool
│      %2331  = Base.and_int(%2329, %2330)::Bool
│      %2332  = Base.sle_int(1, %26)::Bool
│      %2333  = Base.sle_int(%26, 4)::Bool
│      %2334  = Base.and_int(%2332, %2333)::Bool
│      %2335  = Base.sub_int(%21, 1)::Int64
│      %2336  = Base.bitcast(UInt64, %2335)::UInt64
│      %2337  = Base.bitcast(UInt64, %2322)::UInt64
│      %2338  = Base.ult_int(%2336, %2337)::Bool
│      %2339  = Base.and_int(%2338, true)::Bool
│      %2340  = Base.and_int(%2334, %2339)::Bool
│      %2341  = Base.and_int(%2331, %2340)::Bool
│      %2342  = Base.and_int(%2328, %2341)::Bool
│      %2343  = Base.and_int(%2325, %2342)::Bool
└─────          goto #792 if not %2343
791 ──          goto #793
792 ──          invoke Base.throw_boundserror(%2316::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %2318::NTuple{5, Int64})::Union{}
└─────          unreachable
793 ──          nothing::Nothing
794 ┄─ %2349  = StrideArraysCore.getfield(%2316, :ptr)::Ptr{Float64}
│      %2350  = Base.sub_int(%2230, 1)::Int64
│      %2351  = Base.sub_int(%32, 1)::Int64
│      %2352  = Base.sub_int(%1437, 1)::Int64
│      %2353  = Base.sub_int(%26, 1)::Int64
│      %2354  = Base.sub_int(%21, 1)::Int64
└─────          goto #803 if not true
795 ┄─ %2356  = φ (#794 => 2, #802 => %2368)::Int64
│      %2357  = Base.sle_int(1, %2356)::Bool
└─────          goto #797 if not %2357
796 ── %2359  = Base.sle_int(%2356, 5)::Bool
└─────          goto #798
797 ──          nothing::Nothing
798 ┄─ %2362  = φ (#796 => %2359, #797 => false)::Bool
└─────          goto #800 if not %2362
799 ──          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %2356, true)::Static.True
│      %2365  = Base.add_int(%2356, 1)::Int64
└─────          goto #801
800 ──          goto #801
801 ┄─ %2368  = φ (#799 => %2365)::Int64
│      %2369  = φ (#799 => false, #800 => true)::Bool
│      %2370  = Base.not_int(%2369)::Bool
└─────          goto #803 if not %2370
802 ──          goto #795
803 ┄─          goto #804
804 ──          goto #805
805 ── %2375  = Base.mul_int(%2354, 4)::Int64
│      %2376  = Base.add_int(%2353, %2375)::Int64
│      %2377  = Base.mul_int(%2376, 4)::Int64
│      %2378  = Base.add_int(%2352, %2377)::Int64
│      %2379  = Base.mul_int(%2378, 4)::Int64
│      %2380  = Base.add_int(%2351, %2379)::Int64
│      %2381  = Base.mul_int(%2380, 5)::Int64
│      %2382  = Base.add_int(%2350, %2381)::Int64
│      %2383  = Base.mul_int(8, %2382)::Int64
│      %2384  = Core.bitcast(Core.UInt, %2349)::UInt64
│      %2385  = Base.bitcast(UInt64, %2383)::UInt64
│      %2386  = Base.add_ptr(%2384, %2385)::UInt64
│      %2387  = Core.bitcast(Ptr{Float64}, %2386)::Ptr{Float64}
└─────          goto #806
806 ──          Base.pointerset(%2387, %2313, 1, 1)::Ptr{Float64}
└─────          goto #807
807 ──          goto #808
808 ──          $(Expr(:gc_preserve_end, :(%2315)))
└─────          goto #809
809 ── %2394  = (%2231 === 5)::Bool
└─────          goto #811 if not %2394
810 ──          goto #812
811 ── %2397  = Base.add_int(%2231, 1)::Int64
└─────          goto #812
812 ┄─ %2399  = φ (#811 => %2397)::Int64
│      %2400  = φ (#811 => %2397)::Int64
│      %2401  = φ (#810 => true, #811 => false)::Bool
│      %2402  = Base.not_int(%2401)::Bool
└─────          goto #814 if not %2402
813 ──          goto #769
814 ┄─          goto #815
815 ── %2406  = (%1438 === %1425)::Bool
└─────          goto #817 if not %2406
816 ──          goto #818
817 ── %2409  = Base.add_int(%1438, 1)::Int64
└─────          goto #818
818 ┄─ %2411  = φ (#817 => %2409)::Int64
│      %2412  = φ (#817 => %2409)::Int64
│      %2413  = φ (#816 => true, #817 => false)::Bool
│      %2414  = Base.not_int(%2413)::Bool
└─────          goto #820 if not %2414
819 ──          goto #477
820 ┄─ %2417  = Base.add_int(%26, 1)::Int64
│      %2418  = Base.sle_int(%2417, 4)::Bool
└─────          goto #822 if not %2418
821 ──          goto #823
822 ── %2421  = Base.sub_int(%2417, 1)::Int64
└─────          goto #823
823 ┄─ %2423  = φ (#821 => 4, #822 => %2421)::Int64
└─────          goto #824
824 ──          goto #825
825 ── %2426  = Base.slt_int(%2423, %2417)::Bool
└─────          goto #827 if not %2426
826 ──          goto #828
827 ──          goto #828
828 ┄─ %2430  = φ (#826 => true, #827 => false)::Bool
│      %2431  = φ (#827 => %2417)::Int64
│      %2432  = φ (#827 => %2417)::Int64
│      %2433  = Base.not_int(%2430)::Bool
└─────          goto #1172 if not %2433
829 ┄─ %2435  = φ (#828 => %2431, #1171 => %3409)::Int64
│      %2436  = φ (#828 => %2432, #1171 => %3410)::Int64
│      %2437  = StrideArraysCore.getfield(u, :data)::Vector{Float64}
│      %2438  = $(Expr(:gc_preserve_begin, :(%2437)))
│      %2439  = StrideArraysCore.getfield(u, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #834 if not true
830 ── %2441  = Core.tuple(1, %32, %29, %2435, %21)::NTuple{5, Int64}
│      %2442  = StrideArraysCore.getfield(%2439, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %2443  = Core.getfield(%2442, 5)::Int64
│      %2444  = Base.bitcast(UInt64, %2443)::UInt64
│      %2445  = Base.bitcast(Int64, %2444)::Int64
│      %2446  = Base.sle_int(1, %32)::Bool
│      %2447  = Base.sle_int(%32, 4)::Bool
│      %2448  = Base.and_int(%2446, %2447)::Bool
│      %2449  = Base.sle_int(1, %29)::Bool
│      %2450  = Base.sle_int(%29, 4)::Bool
│      %2451  = Base.and_int(%2449, %2450)::Bool
│      %2452  = Base.sle_int(1, %2435)::Bool
│      %2453  = Base.sle_int(%2435, 4)::Bool
│      %2454  = Base.and_int(%2452, %2453)::Bool
│      %2455  = Base.sub_int(%21, 1)::Int64
│      %2456  = Base.bitcast(UInt64, %2455)::UInt64
│      %2457  = Base.bitcast(UInt64, %2445)::UInt64
│      %2458  = Base.ult_int(%2456, %2457)::Bool
│      %2459  = Base.and_int(%2458, true)::Bool
│      %2460  = Base.and_int(%2454, %2459)::Bool
│      %2461  = Base.and_int(%2451, %2460)::Bool
│      %2462  = Base.and_int(%2448, %2461)::Bool
│      %2463  = Base.and_int(true, %2462)::Bool
└─────          goto #832 if not %2463
831 ──          goto #833
832 ──          invoke Base.throw_boundserror(%2439::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %2441::NTuple{5, Int64})::Union{}
└─────          unreachable
833 ──          nothing::Nothing
834 ┄─ %2469  = StrideArraysCore.getfield(%2439, :ptr)::Ptr{Float64}
│      %2470  = Base.sub_int(%32, 1)::Int64
│      %2471  = Base.sub_int(%29, 1)::Int64
│      %2472  = Base.sub_int(%2435, 1)::Int64
│      %2473  = Base.sub_int(%21, 1)::Int64
└─────          goto #843 if not true
835 ┄─ %2475  = φ (#834 => 2, #842 => %2487)::Int64
│      %2476  = Base.sle_int(1, %2475)::Bool
└─────          goto #837 if not %2476
836 ── %2478  = Base.sle_int(%2475, 5)::Bool
└─────          goto #838
837 ──          nothing::Nothing
838 ┄─ %2481  = φ (#836 => %2478, #837 => false)::Bool
└─────          goto #840 if not %2481
839 ──          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %2475, true)::Static.True
│      %2484  = Base.add_int(%2475, 1)::Int64
└─────          goto #841
840 ──          goto #841
841 ┄─ %2487  = φ (#839 => %2484)::Int64
│      %2488  = φ (#839 => false, #840 => true)::Bool
│      %2489  = Base.not_int(%2488)::Bool
└─────          goto #843 if not %2489
842 ──          goto #835
843 ┄─          goto #844
844 ──          goto #845
845 ── %2494  = Base.mul_int(%2473, 4)::Int64
│      %2495  = Base.add_int(%2472, %2494)::Int64
│      %2496  = Base.mul_int(%2495, 4)::Int64
│      %2497  = Base.add_int(%2471, %2496)::Int64
│      %2498  = Base.mul_int(%2497, 4)::Int64
│      %2499  = Base.add_int(%2470, %2498)::Int64
│      %2500  = Base.mul_int(%2499, 5)::Int64
│      %2501  = Base.add_int(0, %2500)::Int64
│      %2502  = Base.mul_int(8, %2501)::Int64
│      %2503  = Core.bitcast(Core.UInt, %2469)::UInt64
│      %2504  = Base.bitcast(UInt64, %2502)::UInt64
│      %2505  = Base.add_ptr(%2503, %2504)::UInt64
│      %2506  = Core.bitcast(Ptr{Float64}, %2505)::Ptr{Float64}
└─────          goto #846
846 ── %2508  = Base.pointerref(%2506, 1, 1)::Float64
└─────          goto #847
847 ──          goto #848
848 ──          $(Expr(:gc_preserve_end, :(%2438)))
└─────          goto #849
849 ──          goto #850
850 ── %2514  = StrideArraysCore.getfield(u, :data)::Vector{Float64}
│      %2515  = $(Expr(:gc_preserve_begin, :(%2514)))
│      %2516  = StrideArraysCore.getfield(u, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #855 if not true
851 ── %2518  = Core.tuple(2, %32, %29, %2435, %21)::NTuple{5, Int64}
│      %2519  = StrideArraysCore.getfield(%2516, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %2520  = Core.getfield(%2519, 5)::Int64
│      %2521  = Base.bitcast(UInt64, %2520)::UInt64
│      %2522  = Base.bitcast(Int64, %2521)::Int64
│      %2523  = Base.sle_int(1, %32)::Bool
│      %2524  = Base.sle_int(%32, 4)::Bool
│      %2525  = Base.and_int(%2523, %2524)::Bool
│      %2526  = Base.sle_int(1, %29)::Bool
│      %2527  = Base.sle_int(%29, 4)::Bool
│      %2528  = Base.and_int(%2526, %2527)::Bool
│      %2529  = Base.sle_int(1, %2435)::Bool
│      %2530  = Base.sle_int(%2435, 4)::Bool
│      %2531  = Base.and_int(%2529, %2530)::Bool
│      %2532  = Base.sub_int(%21, 1)::Int64
│      %2533  = Base.bitcast(UInt64, %2532)::UInt64
│      %2534  = Base.bitcast(UInt64, %2522)::UInt64
│      %2535  = Base.ult_int(%2533, %2534)::Bool
│      %2536  = Base.and_int(%2535, true)::Bool
│      %2537  = Base.and_int(%2531, %2536)::Bool
│      %2538  = Base.and_int(%2528, %2537)::Bool
│      %2539  = Base.and_int(%2525, %2538)::Bool
│      %2540  = Base.and_int(true, %2539)::Bool
└─────          goto #853 if not %2540
852 ──          goto #854
853 ──          invoke Base.throw_boundserror(%2516::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %2518::NTuple{5, Int64})::Union{}
└─────          unreachable
854 ──          nothing::Nothing
855 ┄─ %2546  = StrideArraysCore.getfield(%2516, :ptr)::Ptr{Float64}
│      %2547  = Base.sub_int(%32, 1)::Int64
│      %2548  = Base.sub_int(%29, 1)::Int64
│      %2549  = Base.sub_int(%2435, 1)::Int64
│      %2550  = Base.sub_int(%21, 1)::Int64
└─────          goto #864 if not true
856 ┄─ %2552  = φ (#855 => 2, #863 => %2564)::Int64
│      %2553  = Base.sle_int(1, %2552)::Bool
└─────          goto #858 if not %2553
857 ── %2555  = Base.sle_int(%2552, 5)::Bool
└─────          goto #859
858 ──          nothing::Nothing
859 ┄─ %2558  = φ (#857 => %2555, #858 => false)::Bool
└─────          goto #861 if not %2558
860 ──          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %2552, true)::Static.True
│      %2561  = Base.add_int(%2552, 1)::Int64
└─────          goto #862
861 ──          goto #862
862 ┄─ %2564  = φ (#860 => %2561)::Int64
│      %2565  = φ (#860 => false, #861 => true)::Bool
│      %2566  = Base.not_int(%2565)::Bool
└─────          goto #864 if not %2566
863 ──          goto #856
864 ┄─          goto #865
865 ──          goto #866
866 ── %2571  = Base.mul_int(%2550, 4)::Int64
│      %2572  = Base.add_int(%2549, %2571)::Int64
│      %2573  = Base.mul_int(%2572, 4)::Int64
│      %2574  = Base.add_int(%2548, %2573)::Int64
│      %2575  = Base.mul_int(%2574, 4)::Int64
│      %2576  = Base.add_int(%2547, %2575)::Int64
│      %2577  = Base.mul_int(%2576, 5)::Int64
│      %2578  = Base.add_int(1, %2577)::Int64
│      %2579  = Base.mul_int(8, %2578)::Int64
│      %2580  = Core.bitcast(Core.UInt, %2546)::UInt64
│      %2581  = Base.bitcast(UInt64, %2579)::UInt64
│      %2582  = Base.add_ptr(%2580, %2581)::UInt64
│      %2583  = Core.bitcast(Ptr{Float64}, %2582)::Ptr{Float64}
└─────          goto #867
867 ── %2585  = Base.pointerref(%2583, 1, 1)::Float64
└─────          goto #868
868 ──          goto #869
869 ──          $(Expr(:gc_preserve_end, :(%2515)))
└─────          goto #870
870 ──          goto #871
871 ── %2591  = StrideArraysCore.getfield(u, :data)::Vector{Float64}
│      %2592  = $(Expr(:gc_preserve_begin, :(%2591)))
│      %2593  = StrideArraysCore.getfield(u, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #876 if not true
872 ── %2595  = Core.tuple(3, %32, %29, %2435, %21)::NTuple{5, Int64}
│      %2596  = StrideArraysCore.getfield(%2593, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %2597  = Core.getfield(%2596, 5)::Int64
│      %2598  = Base.bitcast(UInt64, %2597)::UInt64
│      %2599  = Base.bitcast(Int64, %2598)::Int64
│      %2600  = Base.sle_int(1, %32)::Bool
│      %2601  = Base.sle_int(%32, 4)::Bool
│      %2602  = Base.and_int(%2600, %2601)::Bool
│      %2603  = Base.sle_int(1, %29)::Bool
│      %2604  = Base.sle_int(%29, 4)::Bool
│      %2605  = Base.and_int(%2603, %2604)::Bool
│      %2606  = Base.sle_int(1, %2435)::Bool
│      %2607  = Base.sle_int(%2435, 4)::Bool
│      %2608  = Base.and_int(%2606, %2607)::Bool
│      %2609  = Base.sub_int(%21, 1)::Int64
│      %2610  = Base.bitcast(UInt64, %2609)::UInt64
│      %2611  = Base.bitcast(UInt64, %2599)::UInt64
│      %2612  = Base.ult_int(%2610, %2611)::Bool
│      %2613  = Base.and_int(%2612, true)::Bool
│      %2614  = Base.and_int(%2608, %2613)::Bool
│      %2615  = Base.and_int(%2605, %2614)::Bool
│      %2616  = Base.and_int(%2602, %2615)::Bool
│      %2617  = Base.and_int(true, %2616)::Bool
└─────          goto #874 if not %2617
873 ──          goto #875
874 ──          invoke Base.throw_boundserror(%2593::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %2595::NTuple{5, Int64})::Union{}
└─────          unreachable
875 ──          nothing::Nothing
876 ┄─ %2623  = StrideArraysCore.getfield(%2593, :ptr)::Ptr{Float64}
│      %2624  = Base.sub_int(%32, 1)::Int64
│      %2625  = Base.sub_int(%29, 1)::Int64
│      %2626  = Base.sub_int(%2435, 1)::Int64
│      %2627  = Base.sub_int(%21, 1)::Int64
└─────          goto #885 if not true
877 ┄─ %2629  = φ (#876 => 2, #884 => %2641)::Int64
│      %2630  = Base.sle_int(1, %2629)::Bool
└─────          goto #879 if not %2630
878 ── %2632  = Base.sle_int(%2629, 5)::Bool
└─────          goto #880
879 ──          nothing::Nothing
880 ┄─ %2635  = φ (#878 => %2632, #879 => false)::Bool
└─────          goto #882 if not %2635
881 ──          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %2629, true)::Static.True
│      %2638  = Base.add_int(%2629, 1)::Int64
└─────          goto #883
882 ──          goto #883
883 ┄─ %2641  = φ (#881 => %2638)::Int64
│      %2642  = φ (#881 => false, #882 => true)::Bool
│      %2643  = Base.not_int(%2642)::Bool
└─────          goto #885 if not %2643
884 ──          goto #877
885 ┄─          goto #886
886 ──          goto #887
887 ── %2648  = Base.mul_int(%2627, 4)::Int64
│      %2649  = Base.add_int(%2626, %2648)::Int64
│      %2650  = Base.mul_int(%2649, 4)::Int64
│      %2651  = Base.add_int(%2625, %2650)::Int64
│      %2652  = Base.mul_int(%2651, 4)::Int64
│      %2653  = Base.add_int(%2624, %2652)::Int64
│      %2654  = Base.mul_int(%2653, 5)::Int64
│      %2655  = Base.add_int(2, %2654)::Int64
│      %2656  = Base.mul_int(8, %2655)::Int64
│      %2657  = Core.bitcast(Core.UInt, %2623)::UInt64
│      %2658  = Base.bitcast(UInt64, %2656)::UInt64
│      %2659  = Base.add_ptr(%2657, %2658)::UInt64
│      %2660  = Core.bitcast(Ptr{Float64}, %2659)::Ptr{Float64}
└─────          goto #888
888 ── %2662  = Base.pointerref(%2660, 1, 1)::Float64
└─────          goto #889
889 ──          goto #890
890 ──          $(Expr(:gc_preserve_end, :(%2592)))
└─────          goto #891
891 ──          goto #892
892 ── %2668  = StrideArraysCore.getfield(u, :data)::Vector{Float64}
│      %2669  = $(Expr(:gc_preserve_begin, :(%2668)))
│      %2670  = StrideArraysCore.getfield(u, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #897 if not true
893 ── %2672  = Core.tuple(4, %32, %29, %2435, %21)::NTuple{5, Int64}
│      %2673  = StrideArraysCore.getfield(%2670, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %2674  = Core.getfield(%2673, 5)::Int64
│      %2675  = Base.bitcast(UInt64, %2674)::UInt64
│      %2676  = Base.bitcast(Int64, %2675)::Int64
│      %2677  = Base.sle_int(1, %32)::Bool
│      %2678  = Base.sle_int(%32, 4)::Bool
│      %2679  = Base.and_int(%2677, %2678)::Bool
│      %2680  = Base.sle_int(1, %29)::Bool
│      %2681  = Base.sle_int(%29, 4)::Bool
│      %2682  = Base.and_int(%2680, %2681)::Bool
│      %2683  = Base.sle_int(1, %2435)::Bool
│      %2684  = Base.sle_int(%2435, 4)::Bool
│      %2685  = Base.and_int(%2683, %2684)::Bool
│      %2686  = Base.sub_int(%21, 1)::Int64
│      %2687  = Base.bitcast(UInt64, %2686)::UInt64
│      %2688  = Base.bitcast(UInt64, %2676)::UInt64
│      %2689  = Base.ult_int(%2687, %2688)::Bool
│      %2690  = Base.and_int(%2689, true)::Bool
│      %2691  = Base.and_int(%2685, %2690)::Bool
│      %2692  = Base.and_int(%2682, %2691)::Bool
│      %2693  = Base.and_int(%2679, %2692)::Bool
│      %2694  = Base.and_int(true, %2693)::Bool
└─────          goto #895 if not %2694
894 ──          goto #896
895 ──          invoke Base.throw_boundserror(%2670::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %2672::NTuple{5, Int64})::Union{}
└─────          unreachable
896 ──          nothing::Nothing
897 ┄─ %2700  = StrideArraysCore.getfield(%2670, :ptr)::Ptr{Float64}
│      %2701  = Base.sub_int(%32, 1)::Int64
│      %2702  = Base.sub_int(%29, 1)::Int64
│      %2703  = Base.sub_int(%2435, 1)::Int64
│      %2704  = Base.sub_int(%21, 1)::Int64
└─────          goto #906 if not true
898 ┄─ %2706  = φ (#897 => 2, #905 => %2718)::Int64
│      %2707  = Base.sle_int(1, %2706)::Bool
└─────          goto #900 if not %2707
899 ── %2709  = Base.sle_int(%2706, 5)::Bool
└─────          goto #901
900 ──          nothing::Nothing
901 ┄─ %2712  = φ (#899 => %2709, #900 => false)::Bool
└─────          goto #903 if not %2712
902 ──          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %2706, true)::Static.True
│      %2715  = Base.add_int(%2706, 1)::Int64
└─────          goto #904
903 ──          goto #904
904 ┄─ %2718  = φ (#902 => %2715)::Int64
│      %2719  = φ (#902 => false, #903 => true)::Bool
│      %2720  = Base.not_int(%2719)::Bool
└─────          goto #906 if not %2720
905 ──          goto #898
906 ┄─          goto #907
907 ──          goto #908
908 ── %2725  = Base.mul_int(%2704, 4)::Int64
│      %2726  = Base.add_int(%2703, %2725)::Int64
│      %2727  = Base.mul_int(%2726, 4)::Int64
│      %2728  = Base.add_int(%2702, %2727)::Int64
│      %2729  = Base.mul_int(%2728, 4)::Int64
│      %2730  = Base.add_int(%2701, %2729)::Int64
│      %2731  = Base.mul_int(%2730, 5)::Int64
│      %2732  = Base.add_int(3, %2731)::Int64
│      %2733  = Base.mul_int(8, %2732)::Int64
│      %2734  = Core.bitcast(Core.UInt, %2700)::UInt64
│      %2735  = Base.bitcast(UInt64, %2733)::UInt64
│      %2736  = Base.add_ptr(%2734, %2735)::UInt64
│      %2737  = Core.bitcast(Ptr{Float64}, %2736)::Ptr{Float64}
└─────          goto #909
909 ── %2739  = Base.pointerref(%2737, 1, 1)::Float64
└─────          goto #910
910 ──          goto #911
911 ──          $(Expr(:gc_preserve_end, :(%2669)))
└─────          goto #912
912 ──          goto #913
913 ── %2745  = StrideArraysCore.getfield(u, :data)::Vector{Float64}
│      %2746  = $(Expr(:gc_preserve_begin, :(%2745)))
│      %2747  = StrideArraysCore.getfield(u, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #918 if not true
914 ── %2749  = Core.tuple(5, %32, %29, %2435, %21)::NTuple{5, Int64}
│      %2750  = StrideArraysCore.getfield(%2747, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %2751  = Core.getfield(%2750, 5)::Int64
│      %2752  = Base.bitcast(UInt64, %2751)::UInt64
│      %2753  = Base.bitcast(Int64, %2752)::Int64
│      %2754  = Base.sle_int(1, %32)::Bool
│      %2755  = Base.sle_int(%32, 4)::Bool
│      %2756  = Base.and_int(%2754, %2755)::Bool
│      %2757  = Base.sle_int(1, %29)::Bool
│      %2758  = Base.sle_int(%29, 4)::Bool
│      %2759  = Base.and_int(%2757, %2758)::Bool
│      %2760  = Base.sle_int(1, %2435)::Bool
│      %2761  = Base.sle_int(%2435, 4)::Bool
│      %2762  = Base.and_int(%2760, %2761)::Bool
│      %2763  = Base.sub_int(%21, 1)::Int64
│      %2764  = Base.bitcast(UInt64, %2763)::UInt64
│      %2765  = Base.bitcast(UInt64, %2753)::UInt64
│      %2766  = Base.ult_int(%2764, %2765)::Bool
│      %2767  = Base.and_int(%2766, true)::Bool
│      %2768  = Base.and_int(%2762, %2767)::Bool
│      %2769  = Base.and_int(%2759, %2768)::Bool
│      %2770  = Base.and_int(%2756, %2769)::Bool
│      %2771  = Base.and_int(true, %2770)::Bool
└─────          goto #916 if not %2771
915 ──          goto #917
916 ──          invoke Base.throw_boundserror(%2747::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %2749::NTuple{5, Int64})::Union{}
└─────          unreachable
917 ──          nothing::Nothing
918 ┄─ %2777  = StrideArraysCore.getfield(%2747, :ptr)::Ptr{Float64}
│      %2778  = Base.sub_int(%32, 1)::Int64
│      %2779  = Base.sub_int(%29, 1)::Int64
│      %2780  = Base.sub_int(%2435, 1)::Int64
│      %2781  = Base.sub_int(%21, 1)::Int64
└─────          goto #927 if not true
919 ┄─ %2783  = φ (#918 => 2, #926 => %2795)::Int64
│      %2784  = Base.sle_int(1, %2783)::Bool
└─────          goto #921 if not %2784
920 ── %2786  = Base.sle_int(%2783, 5)::Bool
└─────          goto #922
921 ──          nothing::Nothing
922 ┄─ %2789  = φ (#920 => %2786, #921 => false)::Bool
└─────          goto #924 if not %2789
923 ──          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %2783, true)::Static.True
│      %2792  = Base.add_int(%2783, 1)::Int64
└─────          goto #925
924 ──          goto #925
925 ┄─ %2795  = φ (#923 => %2792)::Int64
│      %2796  = φ (#923 => false, #924 => true)::Bool
│      %2797  = Base.not_int(%2796)::Bool
└─────          goto #927 if not %2797
926 ──          goto #919
927 ┄─          goto #928
928 ──          goto #929
929 ── %2802  = Base.mul_int(%2781, 4)::Int64
│      %2803  = Base.add_int(%2780, %2802)::Int64
│      %2804  = Base.mul_int(%2803, 4)::Int64
│      %2805  = Base.add_int(%2779, %2804)::Int64
│      %2806  = Base.mul_int(%2805, 4)::Int64
│      %2807  = Base.add_int(%2778, %2806)::Int64
│      %2808  = Base.mul_int(%2807, 5)::Int64
│      %2809  = Base.add_int(4, %2808)::Int64
│      %2810  = Base.mul_int(8, %2809)::Int64
│      %2811  = Core.bitcast(Core.UInt, %2777)::UInt64
│      %2812  = Base.bitcast(UInt64, %2810)::UInt64
│      %2813  = Base.add_ptr(%2811, %2812)::UInt64
│      %2814  = Core.bitcast(Ptr{Float64}, %2813)::Ptr{Float64}
└─────          goto #930
930 ── %2816  = Base.pointerref(%2814, 1, 1)::Float64
└─────          goto #931
931 ──          goto #932
932 ──          $(Expr(:gc_preserve_end, :(%2746)))
└─────          goto #933
933 ──          goto #934
934 ──          goto #935
935 ──          goto #936
936 ──          goto #938
937 ──          nothing::Nothing
938 ┄─          goto #940
939 ──          nothing::Nothing
940 ┄─          goto #941
941 ──          goto #943
942 ──          nothing::Nothing
943 ┄─          goto #944
944 ──          goto #946
945 ──          nothing::Nothing
946 ┄─          goto #948
947 ──          nothing::Nothing
948 ┄─          goto #949
949 ──          goto #951
950 ──          nothing::Nothing
951 ┄─          goto #952
952 ──          goto #954
953 ──          nothing::Nothing
954 ┄─          goto #956
955 ──          nothing::Nothing
956 ┄─          goto #957
957 ──          goto #959
958 ──          nothing::Nothing
959 ┄─          goto #960
960 ──          goto #962
961 ──          nothing::Nothing
962 ┄─          goto #964
963 ──          nothing::Nothing
964 ┄─          goto #965
965 ──          goto #967
966 ──          nothing::Nothing
967 ┄─          goto #968
968 ── %2856  = Base.div_float(%182, %105)::Float64
│      %2857  = Base.div_float(%259, %105)::Float64
│      %2858  = Base.div_float(%336, %105)::Float64
│      %2859  = Base.getfield(equations, :gamma)::Float64
│      %2860  = Base.sub_float(%2859, 1.0)::Float64
│      %2861  = Base.mul_float(%182, %2856)::Float64
│      %2862  = Base.muladd_float(%259, %2857, %2861)::Float64
│      %2863  = Base.muladd_float(%336, %2858, %2862)::Float64
│      %2864  = Base.muladd_float(-0.5, %2863, %413)::Float64
│      %2865  = Base.mul_float(%2860, %2864)::Float64
└─────          goto #969
969 ──          goto #971
970 ──          nothing::Nothing
971 ┄─          goto #973
972 ──          nothing::Nothing
973 ┄─          goto #974
974 ──          goto #976
975 ──          nothing::Nothing
976 ┄─          goto #977
977 ──          goto #979
978 ──          nothing::Nothing
979 ┄─          goto #981
980 ──          nothing::Nothing
981 ┄─          goto #982
982 ──          goto #984
983 ──          nothing::Nothing
984 ┄─          goto #985
985 ──          goto #987
986 ──          nothing::Nothing
987 ┄─          goto #989
988 ──          nothing::Nothing
989 ┄─          goto #990
990 ──          goto #992
991 ──          nothing::Nothing
992 ┄─          goto #993
993 ──          goto #995
994 ──          nothing::Nothing
995 ┄─          goto #997
996 ──          nothing::Nothing
997 ┄─          goto #998
998 ──          goto #1000
999 ──          nothing::Nothing
1000 ┄          goto #1001
1001 ─          goto #1003
1002 ─          nothing::Nothing
1003 ┄          goto #1005
1004 ─          nothing::Nothing
1005 ┄          goto #1006
1006 ─          goto #1008
1007 ─          nothing::Nothing
1008 ┄          goto #1009
1009 ─          goto #1011
1010 ─          nothing::Nothing
1011 ┄          goto #1013
1012 ─          nothing::Nothing
1013 ┄          goto #1014
1014 ─          goto #1016
1015 ─          nothing::Nothing
1016 ┄          goto #1017
1017 ─          goto #1019
1018 ─          nothing::Nothing
1019 ┄          goto #1021
1020 ─          nothing::Nothing
1021 ┄          goto #1022
1022 ─          goto #1024
1023 ─          nothing::Nothing
1024 ┄          goto #1025
1025 ─          goto #1027
1026 ─          nothing::Nothing
1027 ┄          goto #1029
1028 ─          nothing::Nothing
1029 ┄          goto #1030
1030 ─          goto #1032
1031 ─          nothing::Nothing
1032 ┄          goto #1033
1033 ─ %2931  = Base.div_float(%2585, %2508)::Float64
│      %2932  = Base.div_float(%2662, %2508)::Float64
│      %2933  = Base.div_float(%2739, %2508)::Float64
│      %2934  = Base.getfield(equations, :gamma)::Float64
│      %2935  = Base.sub_float(%2934, 1.0)::Float64
│      %2936  = Base.mul_float(%2585, %2931)::Float64
│      %2937  = Base.muladd_float(%2662, %2932, %2936)::Float64
│      %2938  = Base.muladd_float(%2739, %2933, %2937)::Float64
│      %2939  = Base.muladd_float(-0.5, %2938, %2816)::Float64
│      %2940  = Base.mul_float(%2935, %2939)::Float64
└─────          goto #1034
1034 ─          goto #1036
1035 ─          nothing::Nothing
1036 ┄          goto #1038
1037 ─          nothing::Nothing
1038 ┄          goto #1039
1039 ─          goto #1041
1040 ─          nothing::Nothing
1041 ┄          goto #1042
1042 ─          goto #1044
1043 ─          nothing::Nothing
1044 ┄          goto #1046
1045 ─          nothing::Nothing
1046 ┄          goto #1047
1047 ─          goto #1049
1048 ─          nothing::Nothing
1049 ┄          goto #1050
1050 ─          goto #1052
1051 ─          nothing::Nothing
1052 ┄          goto #1054
1053 ─          nothing::Nothing
1054 ┄          goto #1055
1055 ─          goto #1057
1056 ─          nothing::Nothing
1057 ┄          goto #1058
1058 ─          goto #1060
1059 ─          nothing::Nothing
1060 ┄          goto #1062
1061 ─          nothing::Nothing
1062 ┄          goto #1063
1063 ─          goto #1065
1064 ─          nothing::Nothing
1065 ┄          goto #1066
1066 ─ %2974  = Base.muladd_float(-2.0, %2508, %105)::Float64
│      %2975  = Base.mul_float(%105, %2974)::Float64
│      %2976  = Base.muladd_float(%2508, %2508, %2975)::Float64
│      %2977  = Base.muladd_float(2.0, %2508, %105)::Float64
│      %2978  = Base.mul_float(%105, %2977)::Float64
│      %2979  = Base.muladd_float(%2508, %2508, %2978)::Float64
│      %2980  = Base.div_float(%2976, %2979)::Float64
│      %2981  = Base.lt_float(%2980, 0.0001)::Bool
└─────          goto #1068 if not %2981
1067 ─ %2983  = Base.add_float(%105, %2508)::Float64
│      %2984  = Base.muladd_float(%2980, 0.2857142857142857, 0.4)::Float64
│      %2985  = Base.muladd_float(%2980, %2984, 0.6666666666666666)::Float64
│      %2986  = Base.muladd_float(%2980, %2985, 2.0)::Float64
│      %2987  = Base.div_float(%2983, %2986)::Float64
└─────          goto #1069
1068 ─ %2989  = Base.sub_float(%2508, %105)::Float64
│      %2990  = Base.div_float(%2508, %105)::Float64
│      %2991  = $(Expr(:foreigncall, "llvm.log.f64", Float64, svec(Float64), 0, :(:llvmcall), :(%2990), :(%2990)))::Float64
│      %2992  = Base.div_float(%2989, %2991)::Float64
└─────          goto #1069
1069 ┄ %2994  = φ (#1067 => %2987, #1068 => %2992)::Float64
│      %2995  = Base.mul_float(%105, %2940)::Float64
│      %2996  = Base.mul_float(%2508, %2865)::Float64
│      %2997  = Base.muladd_float(-2.0, %2996, %2995)::Float64
│      %2998  = Base.mul_float(%2995, %2997)::Float64
│      %2999  = Base.muladd_float(%2996, %2996, %2998)::Float64
│      %3000  = Base.muladd_float(2.0, %2996, %2995)::Float64
│      %3001  = Base.mul_float(%2995, %3000)::Float64
│      %3002  = Base.muladd_float(%2996, %2996, %3001)::Float64
│      %3003  = Base.div_float(%2999, %3002)::Float64
│      %3004  = Base.lt_float(%3003, 0.0001)::Bool
└─────          goto #1071 if not %3004
1070 ─ %3006  = Base.muladd_float(%3003, 0.2857142857142857, 0.4)::Float64
│      %3007  = Base.muladd_float(%3003, %3006, 0.6666666666666666)::Float64
│      %3008  = Base.muladd_float(%3003, %3007, 2.0)::Float64
│      %3009  = Base.add_float(%2995, %2996)::Float64
│      %3010  = Base.div_float(%3008, %3009)::Float64
└─────          goto #1072
1071 ─ %3012  = Base.div_float(%2996, %2995)::Float64
│      %3013  = $(Expr(:foreigncall, "llvm.log.f64", Float64, svec(Float64), 0, :(:llvmcall), :(%3012), :(%3012)))::Float64
│      %3014  = Base.sub_float(%2996, %2995)::Float64
│      %3015  = Base.div_float(%3013, %3014)::Float64
└─────          goto #1072
1072 ┄ %3017  = φ (#1070 => %3010, #1071 => %3015)::Float64
│      %3018  = Base.mul_float(%2865, %2940)::Float64
│      %3019  = Base.mul_float(%3018, %3017)::Float64
│      %3020  = Base.add_float(%2856, %2931)::Float64
│      %3021  = Base.mul_float(0.5, %3020)::Float64
│      %3022  = Base.add_float(%2857, %2932)::Float64
│      %3023  = Base.mul_float(0.5, %3022)::Float64
│      %3024  = Base.add_float(%2858, %2933)::Float64
│      %3025  = Base.mul_float(0.5, %3024)::Float64
│      %3026  = Base.add_float(%2865, %2940)::Float64
│      %3027  = Base.mul_float(0.5, %3026)::Float64
│      %3028  = Base.mul_float(%2856, %2931)::Float64
│      %3029  = Base.muladd_float(%2857, %2932, %3028)::Float64
│      %3030  = Base.muladd_float(%2858, %2933, %3029)::Float64
│      %3031  = Base.mul_float(0.5, %3030)::Float64
│      %3032  = Base.mul_float(%2994, %3025)::Float64
│      %3033  = Base.mul_float(%3032, %3021)::Float64
│      %3034  = Base.mul_float(%3032, %3023)::Float64
│      %3035  = Base.muladd_float(%3032, %3025, %3027)::Float64
│      %3036  = Base.mul_float(%2865, %2933)::Float64
│      %3037  = Base.muladd_float(%2940, %2858, %3036)::Float64
│      %3038  = Base.getfield(equations, :inv_gamma_minus_one)::Float64
│      %3039  = Base.muladd_float(%3019, %3038, %3031)::Float64
│      %3040  = Base.mul_float(%3032, %3039)::Float64
│      %3041  = Base.muladd_float(0.5, %3037, %3040)::Float64
│      %3042  = Core.tuple(%3032, %3033, %3034, %3035, %3041)::NTuple{5, Float64}
└─────          goto #1073
1073 ─ %3044  = Base.arrayref(false, %24, %26, %2435)::Float64
│      %3045  = Base.copysign_float(0.0, %3044)::Float64
│      %3046  = Core.ifelse(true, %3044, %3045)::Float64
└─────          goto #1119 if not true
1074 ┄ %3048  = φ (#1073 => 1, #1118 => %3217)::Int64
│      %3049  = φ (#1073 => 1, #1118 => %3218)::Int64
│      %3050  = Base.getfield(%3042, %3048, true)::Float64
│      %3051  = StrideArraysCore.getfield(du, :data)::Vector{Float64}
│      %3052  = $(Expr(:gc_preserve_begin, :(%3051)))
│      %3053  = StrideArraysCore.getfield(du, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #1079 if not true
1075 ─ %3055  = Core.tuple(%3048, %32, %29, %26, %21)::NTuple{5, Int64}
│      %3056  = StrideArraysCore.getfield(%3053, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %3057  = Core.getfield(%3056, 5)::Int64
│      %3058  = Base.bitcast(UInt64, %3057)::UInt64
│      %3059  = Base.bitcast(Int64, %3058)::Int64
│      %3060  = Base.sle_int(1, %3048)::Bool
│      %3061  = Base.sle_int(%3048, 5)::Bool
│      %3062  = Base.and_int(%3060, %3061)::Bool
│      %3063  = Base.sle_int(1, %32)::Bool
│      %3064  = Base.sle_int(%32, 4)::Bool
│      %3065  = Base.and_int(%3063, %3064)::Bool
│      %3066  = Base.sle_int(1, %29)::Bool
│      %3067  = Base.sle_int(%29, 4)::Bool
│      %3068  = Base.and_int(%3066, %3067)::Bool
│      %3069  = Base.sle_int(1, %26)::Bool
│      %3070  = Base.sle_int(%26, 4)::Bool
│      %3071  = Base.and_int(%3069, %3070)::Bool
│      %3072  = Base.sub_int(%21, 1)::Int64
│      %3073  = Base.bitcast(UInt64, %3072)::UInt64
│      %3074  = Base.bitcast(UInt64, %3059)::UInt64
│      %3075  = Base.ult_int(%3073, %3074)::Bool
│      %3076  = Base.and_int(%3075, true)::Bool
│      %3077  = Base.and_int(%3071, %3076)::Bool
│      %3078  = Base.and_int(%3068, %3077)::Bool
│      %3079  = Base.and_int(%3065, %3078)::Bool
│      %3080  = Base.and_int(%3062, %3079)::Bool
└─────          goto #1077 if not %3080
1076 ─          goto #1078
1077 ─          invoke Base.throw_boundserror(%3053::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %3055::NTuple{5, Int64})::Union{}
└─────          unreachable
1078 ─          nothing::Nothing
1079 ┄ %3086  = StrideArraysCore.getfield(%3053, :ptr)::Ptr{Float64}
│      %3087  = Base.sub_int(%3048, 1)::Int64
│      %3088  = Base.sub_int(%32, 1)::Int64
│      %3089  = Base.sub_int(%29, 1)::Int64
│      %3090  = Base.sub_int(%26, 1)::Int64
│      %3091  = Base.sub_int(%21, 1)::Int64
└─────          goto #1088 if not true
1080 ┄ %3093  = φ (#1079 => 2, #1087 => %3105)::Int64
│      %3094  = Base.sle_int(1, %3093)::Bool
└─────          goto #1082 if not %3094
1081 ─ %3096  = Base.sle_int(%3093, 5)::Bool
└─────          goto #1083
1082 ─          nothing::Nothing
1083 ┄ %3099  = φ (#1081 => %3096, #1082 => false)::Bool
└─────          goto #1085 if not %3099
1084 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %3093, true)::Static.True
│      %3102  = Base.add_int(%3093, 1)::Int64
└─────          goto #1086
1085 ─          goto #1086
1086 ┄ %3105  = φ (#1084 => %3102)::Int64
│      %3106  = φ (#1084 => false, #1085 => true)::Bool
│      %3107  = Base.not_int(%3106)::Bool
└─────          goto #1088 if not %3107
1087 ─          goto #1080
1088 ┄          goto #1089
1089 ─          goto #1090
1090 ─ %3112  = Base.mul_int(%3091, 4)::Int64
│      %3113  = Base.add_int(%3090, %3112)::Int64
│      %3114  = Base.mul_int(%3113, 4)::Int64
│      %3115  = Base.add_int(%3089, %3114)::Int64
│      %3116  = Base.mul_int(%3115, 4)::Int64
│      %3117  = Base.add_int(%3088, %3116)::Int64
│      %3118  = Base.mul_int(%3117, 5)::Int64
│      %3119  = Base.add_int(%3087, %3118)::Int64
│      %3120  = Base.mul_int(8, %3119)::Int64
│      %3121  = Core.bitcast(Core.UInt, %3086)::UInt64
│      %3122  = Base.bitcast(UInt64, %3120)::UInt64
│      %3123  = Base.add_ptr(%3121, %3122)::UInt64
│      %3124  = Core.bitcast(Ptr{Float64}, %3123)::Ptr{Float64}
└─────          goto #1091
1091 ─ %3126  = Base.pointerref(%3124, 1, 1)::Float64
└─────          goto #1092
1092 ─          goto #1093
1093 ─          $(Expr(:gc_preserve_end, :(%3052)))
└─────          goto #1094
1094 ─ %3131  = Base.muladd_float(%3046, %3050, %3126)::Float64
│      %3132  = StrideArraysCore.getfield(du, :data)::Vector{Float64}
│      %3133  = $(Expr(:gc_preserve_begin, :(%3132)))
│      %3134  = StrideArraysCore.getfield(du, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #1099 if not true
1095 ─ %3136  = Core.tuple(%3048, %32, %29, %26, %21)::NTuple{5, Int64}
│      %3137  = StrideArraysCore.getfield(%3134, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %3138  = Core.getfield(%3137, 5)::Int64
│      %3139  = Base.bitcast(UInt64, %3138)::UInt64
│      %3140  = Base.bitcast(Int64, %3139)::Int64
│      %3141  = Base.sle_int(1, %3048)::Bool
│      %3142  = Base.sle_int(%3048, 5)::Bool
│      %3143  = Base.and_int(%3141, %3142)::Bool
│      %3144  = Base.sle_int(1, %32)::Bool
│      %3145  = Base.sle_int(%32, 4)::Bool
│      %3146  = Base.and_int(%3144, %3145)::Bool
│      %3147  = Base.sle_int(1, %29)::Bool
│      %3148  = Base.sle_int(%29, 4)::Bool
│      %3149  = Base.and_int(%3147, %3148)::Bool
│      %3150  = Base.sle_int(1, %26)::Bool
│      %3151  = Base.sle_int(%26, 4)::Bool
│      %3152  = Base.and_int(%3150, %3151)::Bool
│      %3153  = Base.sub_int(%21, 1)::Int64
│      %3154  = Base.bitcast(UInt64, %3153)::UInt64
│      %3155  = Base.bitcast(UInt64, %3140)::UInt64
│      %3156  = Base.ult_int(%3154, %3155)::Bool
│      %3157  = Base.and_int(%3156, true)::Bool
│      %3158  = Base.and_int(%3152, %3157)::Bool
│      %3159  = Base.and_int(%3149, %3158)::Bool
│      %3160  = Base.and_int(%3146, %3159)::Bool
│      %3161  = Base.and_int(%3143, %3160)::Bool
└─────          goto #1097 if not %3161
1096 ─          goto #1098
1097 ─          invoke Base.throw_boundserror(%3134::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %3136::NTuple{5, Int64})::Union{}
└─────          unreachable
1098 ─          nothing::Nothing
1099 ┄ %3167  = StrideArraysCore.getfield(%3134, :ptr)::Ptr{Float64}
│      %3168  = Base.sub_int(%3048, 1)::Int64
│      %3169  = Base.sub_int(%32, 1)::Int64
│      %3170  = Base.sub_int(%29, 1)::Int64
│      %3171  = Base.sub_int(%26, 1)::Int64
│      %3172  = Base.sub_int(%21, 1)::Int64
└─────          goto #1108 if not true
1100 ┄ %3174  = φ (#1099 => 2, #1107 => %3186)::Int64
│      %3175  = Base.sle_int(1, %3174)::Bool
└─────          goto #1102 if not %3175
1101 ─ %3177  = Base.sle_int(%3174, 5)::Bool
└─────          goto #1103
1102 ─          nothing::Nothing
1103 ┄ %3180  = φ (#1101 => %3177, #1102 => false)::Bool
└─────          goto #1105 if not %3180
1104 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %3174, true)::Static.True
│      %3183  = Base.add_int(%3174, 1)::Int64
└─────          goto #1106
1105 ─          goto #1106
1106 ┄ %3186  = φ (#1104 => %3183)::Int64
│      %3187  = φ (#1104 => false, #1105 => true)::Bool
│      %3188  = Base.not_int(%3187)::Bool
└─────          goto #1108 if not %3188
1107 ─          goto #1100
1108 ┄          goto #1109
1109 ─          goto #1110
1110 ─ %3193  = Base.mul_int(%3172, 4)::Int64
│      %3194  = Base.add_int(%3171, %3193)::Int64
│      %3195  = Base.mul_int(%3194, 4)::Int64
│      %3196  = Base.add_int(%3170, %3195)::Int64
│      %3197  = Base.mul_int(%3196, 4)::Int64
│      %3198  = Base.add_int(%3169, %3197)::Int64
│      %3199  = Base.mul_int(%3198, 5)::Int64
│      %3200  = Base.add_int(%3168, %3199)::Int64
│      %3201  = Base.mul_int(8, %3200)::Int64
│      %3202  = Core.bitcast(Core.UInt, %3167)::UInt64
│      %3203  = Base.bitcast(UInt64, %3201)::UInt64
│      %3204  = Base.add_ptr(%3202, %3203)::UInt64
│      %3205  = Core.bitcast(Ptr{Float64}, %3204)::Ptr{Float64}
└─────          goto #1111
1111 ─          Base.pointerset(%3205, %3131, 1, 1)::Ptr{Float64}
└─────          goto #1112
1112 ─          goto #1113
1113 ─          $(Expr(:gc_preserve_end, :(%3133)))
└─────          goto #1114
1114 ─ %3212  = (%3049 === 5)::Bool
└─────          goto #1116 if not %3212
1115 ─          goto #1117
1116 ─ %3215  = Base.add_int(%3049, 1)::Int64
└─────          goto #1117
1117 ┄ %3217  = φ (#1116 => %3215)::Int64
│      %3218  = φ (#1116 => %3215)::Int64
│      %3219  = φ (#1115 => true, #1116 => false)::Bool
│      %3220  = Base.not_int(%3219)::Bool
└─────          goto #1119 if not %3220
1118 ─          goto #1074
1119 ┄          goto #1120
1120 ─ %3224  = Base.arrayref(false, %24, %2435, %26)::Float64
│      %3225  = Base.copysign_float(0.0, %3224)::Float64
│      %3226  = Core.ifelse(true, %3224, %3225)::Float64
└─────          goto #1166 if not true
1121 ┄ %3228  = φ (#1120 => 1, #1165 => %3397)::Int64
│      %3229  = φ (#1120 => 1, #1165 => %3398)::Int64
│      %3230  = Base.getfield(%3042, %3228, true)::Float64
│      %3231  = StrideArraysCore.getfield(du, :data)::Vector{Float64}
│      %3232  = $(Expr(:gc_preserve_begin, :(%3231)))
│      %3233  = StrideArraysCore.getfield(du, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #1126 if not true
1122 ─ %3235  = Core.tuple(%3228, %32, %29, %2435, %21)::NTuple{5, Int64}
│      %3236  = StrideArraysCore.getfield(%3233, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %3237  = Core.getfield(%3236, 5)::Int64
│      %3238  = Base.bitcast(UInt64, %3237)::UInt64
│      %3239  = Base.bitcast(Int64, %3238)::Int64
│      %3240  = Base.sle_int(1, %3228)::Bool
│      %3241  = Base.sle_int(%3228, 5)::Bool
│      %3242  = Base.and_int(%3240, %3241)::Bool
│      %3243  = Base.sle_int(1, %32)::Bool
│      %3244  = Base.sle_int(%32, 4)::Bool
│      %3245  = Base.and_int(%3243, %3244)::Bool
│      %3246  = Base.sle_int(1, %29)::Bool
│      %3247  = Base.sle_int(%29, 4)::Bool
│      %3248  = Base.and_int(%3246, %3247)::Bool
│      %3249  = Base.sle_int(1, %2435)::Bool
│      %3250  = Base.sle_int(%2435, 4)::Bool
│      %3251  = Base.and_int(%3249, %3250)::Bool
│      %3252  = Base.sub_int(%21, 1)::Int64
│      %3253  = Base.bitcast(UInt64, %3252)::UInt64
│      %3254  = Base.bitcast(UInt64, %3239)::UInt64
│      %3255  = Base.ult_int(%3253, %3254)::Bool
│      %3256  = Base.and_int(%3255, true)::Bool
│      %3257  = Base.and_int(%3251, %3256)::Bool
│      %3258  = Base.and_int(%3248, %3257)::Bool
│      %3259  = Base.and_int(%3245, %3258)::Bool
│      %3260  = Base.and_int(%3242, %3259)::Bool
└─────          goto #1124 if not %3260
1123 ─          goto #1125
1124 ─          invoke Base.throw_boundserror(%3233::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %3235::NTuple{5, Int64})::Union{}
└─────          unreachable
1125 ─          nothing::Nothing
1126 ┄ %3266  = StrideArraysCore.getfield(%3233, :ptr)::Ptr{Float64}
│      %3267  = Base.sub_int(%3228, 1)::Int64
│      %3268  = Base.sub_int(%32, 1)::Int64
│      %3269  = Base.sub_int(%29, 1)::Int64
│      %3270  = Base.sub_int(%2435, 1)::Int64
│      %3271  = Base.sub_int(%21, 1)::Int64
└─────          goto #1135 if not true
1127 ┄ %3273  = φ (#1126 => 2, #1134 => %3285)::Int64
│      %3274  = Base.sle_int(1, %3273)::Bool
└─────          goto #1129 if not %3274
1128 ─ %3276  = Base.sle_int(%3273, 5)::Bool
└─────          goto #1130
1129 ─          nothing::Nothing
1130 ┄ %3279  = φ (#1128 => %3276, #1129 => false)::Bool
└─────          goto #1132 if not %3279
1131 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %3273, true)::Static.True
│      %3282  = Base.add_int(%3273, 1)::Int64
└─────          goto #1133
1132 ─          goto #1133
1133 ┄ %3285  = φ (#1131 => %3282)::Int64
│      %3286  = φ (#1131 => false, #1132 => true)::Bool
│      %3287  = Base.not_int(%3286)::Bool
└─────          goto #1135 if not %3287
1134 ─          goto #1127
1135 ┄          goto #1136
1136 ─          goto #1137
1137 ─ %3292  = Base.mul_int(%3271, 4)::Int64
│      %3293  = Base.add_int(%3270, %3292)::Int64
│      %3294  = Base.mul_int(%3293, 4)::Int64
│      %3295  = Base.add_int(%3269, %3294)::Int64
│      %3296  = Base.mul_int(%3295, 4)::Int64
│      %3297  = Base.add_int(%3268, %3296)::Int64
│      %3298  = Base.mul_int(%3297, 5)::Int64
│      %3299  = Base.add_int(%3267, %3298)::Int64
│      %3300  = Base.mul_int(8, %3299)::Int64
│      %3301  = Core.bitcast(Core.UInt, %3266)::UInt64
│      %3302  = Base.bitcast(UInt64, %3300)::UInt64
│      %3303  = Base.add_ptr(%3301, %3302)::UInt64
│      %3304  = Core.bitcast(Ptr{Float64}, %3303)::Ptr{Float64}
└─────          goto #1138
1138 ─ %3306  = Base.pointerref(%3304, 1, 1)::Float64
└─────          goto #1139
1139 ─          goto #1140
1140 ─          $(Expr(:gc_preserve_end, :(%3232)))
└─────          goto #1141
1141 ─ %3311  = Base.muladd_float(%3226, %3230, %3306)::Float64
│      %3312  = StrideArraysCore.getfield(du, :data)::Vector{Float64}
│      %3313  = $(Expr(:gc_preserve_begin, :(%3312)))
│      %3314  = StrideArraysCore.getfield(du, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #1146 if not true
1142 ─ %3316  = Core.tuple(%3228, %32, %29, %2435, %21)::NTuple{5, Int64}
│      %3317  = StrideArraysCore.getfield(%3314, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %3318  = Core.getfield(%3317, 5)::Int64
│      %3319  = Base.bitcast(UInt64, %3318)::UInt64
│      %3320  = Base.bitcast(Int64, %3319)::Int64
│      %3321  = Base.sle_int(1, %3228)::Bool
│      %3322  = Base.sle_int(%3228, 5)::Bool
│      %3323  = Base.and_int(%3321, %3322)::Bool
│      %3324  = Base.sle_int(1, %32)::Bool
│      %3325  = Base.sle_int(%32, 4)::Bool
│      %3326  = Base.and_int(%3324, %3325)::Bool
│      %3327  = Base.sle_int(1, %29)::Bool
│      %3328  = Base.sle_int(%29, 4)::Bool
│      %3329  = Base.and_int(%3327, %3328)::Bool
│      %3330  = Base.sle_int(1, %2435)::Bool
│      %3331  = Base.sle_int(%2435, 4)::Bool
│      %3332  = Base.and_int(%3330, %3331)::Bool
│      %3333  = Base.sub_int(%21, 1)::Int64
│      %3334  = Base.bitcast(UInt64, %3333)::UInt64
│      %3335  = Base.bitcast(UInt64, %3320)::UInt64
│      %3336  = Base.ult_int(%3334, %3335)::Bool
│      %3337  = Base.and_int(%3336, true)::Bool
│      %3338  = Base.and_int(%3332, %3337)::Bool
│      %3339  = Base.and_int(%3329, %3338)::Bool
│      %3340  = Base.and_int(%3326, %3339)::Bool
│      %3341  = Base.and_int(%3323, %3340)::Bool
└─────          goto #1144 if not %3341
1143 ─          goto #1145
1144 ─          invoke Base.throw_boundserror(%3314::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %3316::NTuple{5, Int64})::Union{}
└─────          unreachable
1145 ─          nothing::Nothing
1146 ┄ %3347  = StrideArraysCore.getfield(%3314, :ptr)::Ptr{Float64}
│      %3348  = Base.sub_int(%3228, 1)::Int64
│      %3349  = Base.sub_int(%32, 1)::Int64
│      %3350  = Base.sub_int(%29, 1)::Int64
│      %3351  = Base.sub_int(%2435, 1)::Int64
│      %3352  = Base.sub_int(%21, 1)::Int64
└─────          goto #1155 if not true
1147 ┄ %3354  = φ (#1146 => 2, #1154 => %3366)::Int64
│      %3355  = Base.sle_int(1, %3354)::Bool
└─────          goto #1149 if not %3355
1148 ─ %3357  = Base.sle_int(%3354, 5)::Bool
└─────          goto #1150
1149 ─          nothing::Nothing
1150 ┄ %3360  = φ (#1148 => %3357, #1149 => false)::Bool
└─────          goto #1152 if not %3360
1151 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %3354, true)::Static.True
│      %3363  = Base.add_int(%3354, 1)::Int64
└─────          goto #1153
1152 ─          goto #1153
1153 ┄ %3366  = φ (#1151 => %3363)::Int64
│      %3367  = φ (#1151 => false, #1152 => true)::Bool
│      %3368  = Base.not_int(%3367)::Bool
└─────          goto #1155 if not %3368
1154 ─          goto #1147
1155 ┄          goto #1156
1156 ─          goto #1157
1157 ─ %3373  = Base.mul_int(%3352, 4)::Int64
│      %3374  = Base.add_int(%3351, %3373)::Int64
│      %3375  = Base.mul_int(%3374, 4)::Int64
│      %3376  = Base.add_int(%3350, %3375)::Int64
│      %3377  = Base.mul_int(%3376, 4)::Int64
│      %3378  = Base.add_int(%3349, %3377)::Int64
│      %3379  = Base.mul_int(%3378, 5)::Int64
│      %3380  = Base.add_int(%3348, %3379)::Int64
│      %3381  = Base.mul_int(8, %3380)::Int64
│      %3382  = Core.bitcast(Core.UInt, %3347)::UInt64
│      %3383  = Base.bitcast(UInt64, %3381)::UInt64
│      %3384  = Base.add_ptr(%3382, %3383)::UInt64
│      %3385  = Core.bitcast(Ptr{Float64}, %3384)::Ptr{Float64}
└─────          goto #1158
1158 ─          Base.pointerset(%3385, %3311, 1, 1)::Ptr{Float64}
└─────          goto #1159
1159 ─          goto #1160
1160 ─          $(Expr(:gc_preserve_end, :(%3313)))
└─────          goto #1161
1161 ─ %3392  = (%3229 === 5)::Bool
└─────          goto #1163 if not %3392
1162 ─          goto #1164
1163 ─ %3395  = Base.add_int(%3229, 1)::Int64
└─────          goto #1164
1164 ┄ %3397  = φ (#1163 => %3395)::Int64
│      %3398  = φ (#1163 => %3395)::Int64
│      %3399  = φ (#1162 => true, #1163 => false)::Bool
│      %3400  = Base.not_int(%3399)::Bool
└─────          goto #1166 if not %3400
1165 ─          goto #1121
1166 ┄          goto #1167
1167 ─ %3404  = (%2436 === %2423)::Bool
└─────          goto #1169 if not %3404
1168 ─          goto #1170
1169 ─ %3407  = Base.add_int(%2436, 1)::Int64
└─────          goto #1170
1170 ┄ %3409  = φ (#1169 => %3407)::Int64
│      %3410  = φ (#1169 => %3407)::Int64
│      %3411  = φ (#1168 => true, #1169 => false)::Bool
│      %3412  = Base.not_int(%3411)::Bool
└─────          goto #1172 if not %3412
1171 ─          goto #829
1172 ┄ %3415  = (%33 === 4)::Bool
└─────          goto #1174 if not %3415
1173 ─          goto #1175
1174 ─ %3418  = Base.add_int(%33, 1)::Int64
└─────          goto #1175
1175 ┄ %3420  = φ (#1174 => %3418)::Int64
│      %3421  = φ (#1174 => %3418)::Int64
│      %3422  = φ (#1173 => true, #1174 => false)::Bool
│      %3423  = Base.not_int(%3422)::Bool
└─────          goto #1177 if not %3423
1176 ─          goto #9
1177 ┄ %3426  = (%30 === 4)::Bool
└─────          goto #1179 if not %3426
1178 ─          goto #1180
1179 ─ %3429  = Base.add_int(%30, 1)::Int64
└─────          goto #1180
1180 ┄ %3431  = φ (#1179 => %3429)::Int64
│      %3432  = φ (#1179 => %3429)::Int64
│      %3433  = φ (#1178 => true, #1179 => false)::Bool
│      %3434  = Base.not_int(%3433)::Bool
└─────          goto #1182 if not %3434
1181 ─          goto #8
1182 ┄ %3437  = (%27 === 4)::Bool
└─────          goto #1184 if not %3437
1183 ─          goto #1185
1184 ─ %3440  = Base.add_int(%27, 1)::Int64
└─────          goto #1185
1185 ┄ %3442  = φ (#1184 => %3440)::Int64
│      %3443  = φ (#1184 => %3440)::Int64
│      %3444  = φ (#1183 => true, #1184 => false)::Bool
│      %3445  = Base.not_int(%3444)::Bool
└─────          goto #1187 if not %3445
1186 ─          goto #7
1187 ┄          goto #1188
1188 ─          goto #1189
1189 ─          goto #1190
1190 ─ %3451  = (%22 === %11)::Bool
└─────          goto #1192 if not %3451
1191 ─          goto #1193
1192 ─ %3454  = Base.add_int(%22, 1)::Int64
└─────          goto #1193
1193 ┄ %3456  = φ (#1192 => %3454)::Int64
│      %3457  = φ (#1192 => %3454)::Int64
│      %3458  = φ (#1191 => true, #1192 => false)::Bool
│      %3459  = Base.not_int(%3458)::Bool
└─────          goto #1195 if not %3459
1194 ─          goto #6
1195 ┄          goto #3619
1196 ─ %3463  = Base.getfield(cache, :elements)::Trixi.TreeElementContainer3D{Float64, Float64}
│      %3464  = Base.getfield(%3463, :cell_ids)::Vector{Int64}
│      %3465  = Base.arraylen(%3464)::Int64
│      %3466  = Base.slt_int(%3465, 0)::Bool
│      %3467  = Core.ifelse(%3466, 0, %3465)::Int64
│      %3468  = Base.slt_int(%4, %3467)::Bool
│      %3469  = Core.ifelse(%3468, %4, %3467)::Int64
│      %3470  = Base.slt_int(0, %3467)::Bool
└─────          goto #3617 if not %3470
1197 ─ %3472  = Base.slt_int(%3467, %3469)::Bool
└─────          goto #1199 if not %3472
1198 ─          nothing::Nothing
1199 ┄ %3475  = φ (#1198 => %3467, #1197 => %3469)::Int64
│      %3476  = Base.bitcast(UInt64, %3467)::UInt64
│      %3477  = (%3475 === 0)::Bool
└─────          goto #1201 if not %3477
1200 ─          goto #2426
1201 ─ %3480  = Base.sub_int(%3475, 1)::Int64
│      %3481  = Base.trunc_int(UInt32, %3480)::UInt32
│      %3482  = PolyesterWeave.WORKERS::Base.RefValue{NTuple{8, UInt64}}
│      %3483  = $(Expr(:foreigncall, :(:jl_value_ptr), Ptr{Nothing}, svec(Any), 0, :(:ccall), :(%3482)))::Ptr{Nothing}
│      %3484  = Base.bitcast(Ptr{UInt64}, %3483)::Ptr{UInt64}
└─────          goto #1204 if not true
1202 ─ %3486  = Base.bitcast(Int32, %3481)::Int32
│      %3487  = Base.sle_int(%3486, 0)::Bool
└─────          goto #1204 if not %3487
1203 ─          goto #1212
1204 ┄ %3490  = Base.llvmcall("%p = inttoptr i64 %0 to i64*\n%v = atomicrmw xchg i64* %p, i64 %1 acq_rel\nret i64 %v\n", UInt64, Tuple{Ptr{UInt64}, UInt64}, %3484, 0x0000000000000000)::UInt64
│      %3491  = Base.ctpop_int(%3490)::UInt64
│      %3492  = Base.bitcast(Int64, %3491)::Int64
│      %3493  = Base.trunc_int(UInt32, %3492)::UInt32
│      %3494  = Base.sub_int(%3481, %3493)::UInt32
│      %3495  = Base.bitcast(Int32, %3494)::Int32
│      %3496  = Core.sext_int(Core.Int64, %3495)::Int64
│      %3497  = Base.sle_int(0, %3496)::Bool
└─────          goto #1206 if not %3497
1205 ─ %3499  = Base.ctpop_int(%3490)::UInt64
│      %3500  = Base.bitcast(Int64, %3499)::Int64
│      %3501  = Base.trunc_int(UInt32, %3500)::UInt32
└─────          goto #1212
1206 ─ %3503  = Base.ctlz_int(%3490)::UInt64
│      %3504  = Base.bitcast(Int64, %3503)::Int64
└───── %3505  = Base.trunc_int(UInt32, %3504)::UInt32
1207 ┄ %3506  = φ (#1206 => %3505, #1210 => %3511)::UInt32
│      %3507  = φ (#1206 => %3494, #1210 => %3520)::UInt32
│      %3508  = φ (#1206 => %3490, #1210 => %3522)::UInt64
└─────          goto #1211 if not true
1208 ─ %3510  = Base.neg_int(%3507)::UInt32
│      %3511  = Base.add_int(%3506, %3510)::UInt32
│      %3512  = Base.sub_int(0x00000040, %3511)::UInt32
│      %3513  = Base.shl_int(0x0000000000000001, %3512)::UInt64
│      %3514  = Base.sub_int(%3513, 0x0000000000000001)::UInt64
│      %3515  = Base.and_int(%3508, %3514)::UInt64
│      %3516  = Base.xor_int(%3515, %3508)::UInt64
│      %3517  = Base.ctpop_int(%3516)::UInt64
│      %3518  = Base.bitcast(Int64, %3517)::Int64
│      %3519  = Base.trunc_int(UInt32, %3518)::UInt32
│      %3520  = Base.add_int(%3507, %3519)::UInt32
│      %3521  = Base.not_int(%3516)::UInt64
│      %3522  = Base.and_int(%3508, %3521)::UInt64
│      %3523  = (%3520 === 0x00000000)::Bool
└─────          goto #1210 if not %3523
1209 ─          goto #1211
1210 ─          goto #1207
1211 ┄ %3527  = φ (#1209 => %3522, #1207 => %3508)::UInt64
│      %3528  = Base.not_int(%3527)::UInt64
│      %3529  = Base.and_int(%3490, %3528)::UInt64
│               Base.llvmcall("%p = inttoptr i64 %0 to i64*\nstore atomic i64 %1, i64* %p release, align 16\nret void\n", ThreadingUtilities.Cvoid, Tuple{Ptr{UInt64}, UInt64}, %3484, %3529)::Nothing
└─────          goto #1212
1212 ┄ %3532  = φ (#1203 => 0x00000000, #1205 => %3501, #1211 => %3481)::UInt32
│      %3533  = φ (#1203 => 0x0000000000000000, #1205 => %3490, #1211 => %3527)::UInt64
│      %3534  = φ (#1203 => 0x0000000000000000, #1205 => %3490, #1211 => %3527)::UInt64
└─────          goto #1213
1213 ─          goto #1214
1214 ─          goto #1215
1215 ─ %3538  = Core.zext_int(Core.UInt64, %3532)::UInt64
│      %3539  = Base.trunc_int(Int32, %3538)::Int32
│      %3540  = Base.sle_int(%3539, 0)::Bool
└─────          goto #1217 if not %3540
1216 ─          goto #2426
1217 ─ %3543  = Base.add_int(%3538, 0x0000000000000001)::UInt64
│      %3544  = Base.udiv_int(%3476, %3543)::UInt64
│      %3545  = Base.mul_int(%3544, %3543)::UInt64
│      %3546  = Base.sub_int(%3476, %3545)::UInt64
│      %3547  = Base.bitcast(Int64, %3546)::Int64
│      %3548  = Base.add_int(%3544, 0x0000000000000001)::UInt64
│      %3549  = StrideArraysCore.getfield(du, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
│      %3550  = StrideArraysCore.getfield(du, :data)::Vector{Float64}
│      %3551  = StrideArraysCore.getfield(u, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
│      %3552  = StrideArraysCore.getfield(u, :data)::Vector{Float64}
│      %3553  = Core.tuple(static(1), static(1), $(QuoteNode(Polyester.NoLoop())), $(QuoteNode(Polyester.CombineIndices())), %3549, %3551, $(QuoteNode(Polyester.WrapType{TreeMesh{3, Trixi.SerialTree{3, Float64}, Float64}}())), static(false), equations, $(QuoteNode(VolumeIntegralFluxDifferencing{typeof(flux_ranocha)}(Trixi.flux_ranocha))), dg, cache)::Tuple{Static.StaticInt{1}, Static.StaticInt{1}, Polyester.NoLoop, Polyester.CombineIndices, StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, Polyester.WrapType{TreeMesh{3, Trixi.SerialTree{3, Float64}, Float64}}, False, CompressibleEulerEquations3D{Float64}, VolumeIntegralFluxDifferencing{typeof(flux_ranocha)}, DGSEM{LobattoLegendreBasis{Float64, 4, SVector{4, Float64}, Matrix{Float64}, Matrix{Float64}}, Trixi.LobattoLegendreMortarL2{Float64, 4, Matrix{Float64}, Matrix{Float64}}, SurfaceIntegralWeakForm{FluxLaxFriedrichs{typeof(max_abs_speed_naive)}}, VolumeIntegralFluxDifferencing{typeof(flux_ranocha)}}, @NamedTuple{elements::Trixi.TreeElementContainer3D{Float64, Float64}, interfaces::Trixi.TreeInterfaceContainer3D{Float64}, boundaries::Trixi.TreeBoundaryContainer3D{Float64, Float64}, mortars::Trixi.TreeL2MortarContainer3D{Float64}, fstar_primary_upper_left_threaded::Vector{Array{Float64, 3}}, fstar_primary_upper_right_threaded::Vector{Array{Float64, 3}}, fstar_primary_lower_left_threaded::Vector{Array{Float64, 3}}, fstar_primary_lower_right_threaded::Vector{Array{Float64, 3}}, fstar_secondary_upper_left_threaded::Vector{Array{Float64, 3}}, fstar_secondary_upper_right_threaded::Vector{Array{Float64, 3}}, fstar_secondary_lower_left_threaded::Vector{Array{Float64, 3}}, fstar_secondary_lower_right_threaded::Vector{Array{Float64, 3}}, fstar_tmp1_threaded::Vector{Array{Float64, 3}}}}
│      %3554  = %new(ManualMemory.Reference{Tuple{Static.StaticInt{1}, Static.StaticInt{1}, Polyester.NoLoop, Polyester.CombineIndices, StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, Polyester.WrapType{TreeMesh{3, Trixi.SerialTree{3, Float64}, Float64}}, False, CompressibleEulerEquations3D{Float64}, VolumeIntegralFluxDifferencing{typeof(flux_ranocha)}, DGSEM{LobattoLegendreBasis{Float64, 4, SVector{4, Float64}, Matrix{Float64}, Matrix{Float64}}, Trixi.LobattoLegendreMortarL2{Float64, 4, Matrix{Float64}, Matrix{Float64}}, SurfaceIntegralWeakForm{FluxLaxFriedrichs{typeof(max_abs_speed_naive)}}, VolumeIntegralFluxDifferencing{typeof(flux_ranocha)}}, @NamedTuple{elements::Trixi.TreeElementContainer3D{Float64, Float64}, interfaces::Trixi.TreeInterfaceContainer3D{Float64}, boundaries::Trixi.TreeBoundaryContainer3D{Float64, Float64}, mortars::Trixi.TreeL2MortarContainer3D{Float64}, fstar_primary_upper_left_threaded::Vector{Array{Float64, 3}}, fstar_primary_upper_right_threaded::Vector{Array{Float64, 3}}, fstar_primary_lower_left_threaded::Vector{Array{Float64, 3}}, fstar_primary_lower_right_threaded::Vector{Array{Float64, 3}}, fstar_secondary_upper_left_threaded::Vector{Array{Float64, 3}}, fstar_secondary_upper_right_threaded::Vector{Array{Float64, 3}}, fstar_secondary_lower_left_threaded::Vector{Array{Float64, 3}}, fstar_secondary_lower_right_threaded::Vector{Array{Float64, 3}}, fstar_tmp1_threaded::Vector{Array{Float64, 3}}}}}, %3553)::ManualMemory.Reference{Tuple{Static.StaticInt{1}, Static.StaticInt{1}, Polyester.NoLoop, Polyester.CombineIndices, StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, Polyester.WrapType{TreeMesh{3, Trixi.SerialTree{3, Float64}, Float64}}, False, CompressibleEulerEquations3D{Float64}, VolumeIntegralFluxDifferencing{typeof(flux_ranocha)}, DGSEM{LobattoLegendreBasis{Float64, 4, SVector{4, Float64}, Matrix{Float64}, Matrix{Float64}}, Trixi.LobattoLegendreMortarL2{Float64, 4, Matrix{Float64}, Matrix{Float64}}, SurfaceIntegralWeakForm{FluxLaxFriedrichs{typeof(max_abs_speed_naive)}}, VolumeIntegralFluxDifferencing{typeof(flux_ranocha)}}, @NamedTuple{elements::Trixi.TreeElementContainer3D{Float64, Float64}, interfaces::Trixi.TreeInterfaceContainer3D{Float64}, boundaries::Trixi.TreeBoundaryContainer3D{Float64, Float64}, mortars::Trixi.TreeL2MortarContainer3D{Float64}, fstar_primary_upper_left_threaded::Vector{Array{Float64, 3}}, fstar_primary_upper_right_threaded::Vector{Array{Float64, 3}}, fstar_primary_lower_left_threaded::Vector{Array{Float64, 3}}, fstar_primary_lower_right_threaded::Vector{Array{Float64, 3}}, fstar_secondary_upper_left_threaded::Vector{Array{Float64, 3}}, fstar_secondary_upper_right_threaded::Vector{Array{Float64, 3}}, fstar_secondary_lower_left_threaded::Vector{Array{Float64, 3}}, fstar_secondary_lower_right_threaded::Vector{Array{Float64, 3}}, fstar_tmp1_threaded::Vector{Array{Float64, 3}}}}}
│      %3555  = $(Expr(:cfunction, Ptr{Nothing}, :($(QuoteNode(Polyester.BatchClosure{var"#68#69", ManualMemory.Reference{Tuple{Static.StaticInt{1}, Static.StaticInt{1}, Polyester.NoLoop, Polyester.CombineIndices, StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, Polyester.WrapType{TreeMesh{3, Trixi.SerialTree{3, Float64}, Float64}}, False, CompressibleEulerEquations3D{Float64}, VolumeIntegralFluxDifferencing{typeof(flux_ranocha)}, DGSEM{LobattoLegendreBasis{Float64, 4, SVector{4, Float64}, Matrix{Float64}, Matrix{Float64}}, Trixi.LobattoLegendreMortarL2{Float64, 4, Matrix{Float64}, Matrix{Float64}}, SurfaceIntegralWeakForm{FluxLaxFriedrichs{typeof(max_abs_speed_naive)}}, VolumeIntegralFluxDifferencing{typeof(flux_ranocha)}}, @NamedTuple{elements::Trixi.TreeElementContainer3D{Float64, Float64}, interfaces::Trixi.TreeInterfaceContainer3D{Float64}, boundaries::Trixi.TreeBoundaryContainer3D{Float64, Float64}, mortars::Trixi.TreeL2MortarContainer3D{Float64}, fstar_primary_upper_left_threaded::Vector{Array{Float64, 3}}, fstar_primary_upper_right_threaded::Vector{Array{Float64, 3}}, fstar_primary_lower_left_threaded::Vector{Array{Float64, 3}}, fstar_primary_lower_right_threaded::Vector{Array{Float64, 3}}, fstar_secondary_upper_left_threaded::Vector{Array{Float64, 3}}, fstar_secondary_upper_right_threaded::Vector{Array{Float64, 3}}, fstar_secondary_lower_left_threaded::Vector{Array{Float64, 3}}, fstar_secondary_lower_right_threaded::Vector{Array{Float64, 3}}, fstar_tmp1_threaded::Vector{Array{Float64, 3}}}}}, false, Tuple{}}(var"#68#69"())))), Nothing, svec(Ptr{UInt64}), :(:ccall)))::Ptr{Nothing}
│      %3556  = $(Expr(:gc_preserve_begin, :(%3555), :(%3554), nothing, nothing, nothing, nothing, :(%3550), :(%3552), nothing, nothing, nothing, nothing, Core.Argument(9), Core.Argument(10)))
└─────          goto #1219
1218 ─          nothing::Nothing
1219 ┄          goto #1221
1220 ─          nothing::Nothing
1221 ┄          goto #1222
1222 ─          goto #1224
1223 ─          nothing::Nothing
1224 ┄          goto #1226
1225 ─          nothing::Nothing
1226 ┄          goto #1227
1227 ─          goto #1228
1228 ─          goto #1229
1229 ─          goto #1230
1230 ─          goto #1231
1231 ─          goto #1240 if not true
1232 ─          nothing::Nothing
1233 ┄ %3573  = φ (#1232 => 0x00000000, #1238 => %3596)::UInt32
│      %3574  = φ (#1232 => 0x0000000000000000, #1238 => %3593)::UInt64
│      %3575  = φ (#1232 => 0x00000000, #1238 => %3594)::UInt32
│      %3576  = φ (#1232 => %3533, #1238 => %3597)::UInt64
│      %3577  = (%3575 === %3532)::Bool
│      %3578  = Base.not_int(%3577)::Bool
└─────          goto #1239 if not %3578
1234 ─ %3580  = (%3576 === 0x0000000000000000)::Bool
│      %3581  = Base.not_int(%3580)::Bool
│      %3582  = Core.tuple("    declare void @llvm.assume(i1)\n\n    define void @entry(i8 %byte) alwaysinline {\n    top:\n      %bit = trunc i8 %byte to i1\n      call void @llvm.assume(i1 %bit)\n      ret void\n    }\n", "entry")::Tuple{String, String}
│               Base.llvmcall(%3582, PolyesterWeave.Cvoid, Tuple{Bool}, %3581)::Nothing
│      %3584  = Base.cttz_int(%3576)::UInt64
│      %3585  = Base.bitcast(Int64, %3584)::Int64
│      %3586  = Base.trunc_int(UInt32, %3585)::UInt32
│      %3587  = Base.sle_int(0, %3547)::Bool
│      %3588  = Base.bitcast(UInt64, %3547)::UInt64
│      %3589  = Core.zext_int(Core.UInt64, %3575)::UInt64
│      %3590  = Base.ult_int(%3589, %3588)::Bool
│      %3591  = Base.and_int(%3587, %3590)::Bool
│      %3592  = Core.ifelse(%3591, %3548, %3544)::UInt64
│      %3593  = Base.add_int(%3574, %3592)::UInt64
│      %3594  = Base.add_int(%3575, 0x00000001)::UInt32
│      %3595  = Base.add_int(%3586, 0x00000001)::UInt32
│      %3596  = Base.add_int(%3573, %3595)::UInt32
│      %3597  = Base.lshr_int(%3576, %3595)::UInt64
│      %3598  = ThreadingUtilities.THREADPOOLPTR::Base.RefValue{Ptr{UInt64}}
│      %3599  = Base.getfield(%3598, :x)::Ptr{UInt64}
│      %3600  = Base.mul_int(%3596, 0x00000200)::UInt32
│      %3601  = Core.bitcast(Core.UInt, %3599)::UInt64
│      %3602  = Core.zext_int(Core.UInt64, %3600)::UInt64
│      %3603  = Base.add_ptr(%3601, %3602)::UInt64
│      %3604  = Core.bitcast(Ptr{UInt64}, %3603)::Ptr{UInt64}
│      %3605  = Core.bitcast(Core.UInt, %3604)::UInt64
│      %3606  = Base.add_ptr(%3605, 0x0000000000000008)::UInt64
│      %3607  = Core.bitcast(Ptr{UInt64}, %3606)::Ptr{UInt64}
│      %3608  = Base.bitcast(Ptr{Ptr{Nothing}}, %3607)::Ptr{Ptr{Nothing}}
│               Base.pointerset(%3608, %3555, 1, 1)::Ptr{Ptr{Nothing}}
│      %3610  = Core.bitcast(Core.UInt, %3604)::UInt64
│      %3611  = Base.add_ptr(%3610, 0x0000000000000010)::UInt64
│      %3612  = Core.bitcast(Ptr{UInt64}, %3611)::Ptr{UInt64}
│      %3613  = Base.bitcast(Ptr{ManualMemory.Reference{Tuple{Static.StaticInt{1}, Static.StaticInt{1}, Polyester.NoLoop, Polyester.CombineIndices, StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, Polyester.WrapType{TreeMesh{3, Trixi.SerialTree{3, Float64}, Float64}}, False, CompressibleEulerEquations3D{Float64}, VolumeIntegralFluxDifferencing{typeof(flux_ranocha)}, DGSEM{LobattoLegendreBasis{Float64, 4, SVector{4, Float64}, Matrix{Float64}, Matrix{Float64}}, Trixi.LobattoLegendreMortarL2{Float64, 4, Matrix{Float64}, Matrix{Float64}}, SurfaceIntegralWeakForm{FluxLaxFriedrichs{typeof(max_abs_speed_naive)}}, VolumeIntegralFluxDifferencing{typeof(flux_ranocha)}}, @NamedTuple{elements::Trixi.TreeElementContainer3D{Float64, Float64}, interfaces::Trixi.TreeInterfaceContainer3D{Float64}, boundaries::Trixi.TreeBoundaryContainer3D{Float64, Float64}, mortars::Trixi.TreeL2MortarContainer3D{Float64}, fstar_primary_upper_left_threaded::Vector{Array{Float64, 3}}, fstar_primary_upper_right_threaded::Vector{Array{Float64, 3}}, fstar_primary_lower_left_threaded::Vector{Array{Float64, 3}}, fstar_primary_lower_right_threaded::Vector{Array{Float64, 3}}, fstar_secondary_upper_left_threaded::Vector{Array{Float64, 3}}, fstar_secondary_upper_right_threaded::Vector{Array{Float64, 3}}, fstar_secondary_lower_left_threaded::Vector{Array{Float64, 3}}, fstar_secondary_lower_right_threaded::Vector{Array{Float64, 3}}, fstar_tmp1_threaded::Vector{Array{Float64, 3}}}}}}, %3612)::Ptr{ManualMemory.Reference{Tuple{Static.StaticInt{1}, Static.StaticInt{1}, Polyester.NoLoop, Polyester.CombineIndices, StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, Polyester.WrapType{TreeMesh{3, Trixi.SerialTree{3, Float64}, Float64}}, False, CompressibleEulerEquations3D{Float64}, VolumeIntegralFluxDifferencing{typeof(flux_ranocha)}, DGSEM{LobattoLegendreBasis{Float64, 4, SVector{4, Float64}, Matrix{Float64}, Matrix{Float64}}, Trixi.LobattoLegendreMortarL2{Float64, 4, Matrix{Float64}, Matrix{Float64}}, SurfaceIntegralWeakForm{FluxLaxFriedrichs{typeof(max_abs_speed_naive)}}, VolumeIntegralFluxDifferencing{typeof(flux_ranocha)}}, @NamedTuple{elements::Trixi.TreeElementContainer3D{Float64, Float64}, interfaces::Trixi.TreeInterfaceContainer3D{Float64}, boundaries::Trixi.TreeBoundaryContainer3D{Float64, Float64}, mortars::Trixi.TreeL2MortarContainer3D{Float64}, fstar_primary_upper_left_threaded::Vector{Array{Float64, 3}}, fstar_primary_upper_right_threaded::Vector{Array{Float64, 3}}, fstar_primary_lower_left_threaded::Vector{Array{Float64, 3}}, fstar_primary_lower_right_threaded::Vector{Array{Float64, 3}}, fstar_secondary_upper_left_threaded::Vector{Array{Float64, 3}}, fstar_secondary_upper_right_threaded::Vector{Array{Float64, 3}}, fstar_secondary_lower_left_threaded::Vector{Array{Float64, 3}}, fstar_secondary_lower_right_threaded::Vector{Array{Float64, 3}}, fstar_tmp1_threaded::Vector{Array{Float64, 3}}}}}}
│      %3614  = Base.bitcast(Ptr{Ptr{Nothing}}, %3613)::Ptr{Ptr{Nothing}}
│      %3615  = $(Expr(:foreigncall, :(:jl_value_ptr), Ptr{Nothing}, svec(Any), 0, :(:ccall), :(%3554)))::Ptr{Nothing}
│               Base.pointerset(%3614, %3615, 1, 1)::Ptr{Ptr{Nothing}}
│      %3617  = Core.bitcast(Core.UInt, %3604)::UInt64
│      %3618  = Base.add_ptr(%3617, 0x0000000000000018)::UInt64
│      %3619  = Core.bitcast(Ptr{UInt64}, %3618)::Ptr{UInt64}
│               Base.pointerset(%3619, %3574, 1, 1)::Ptr{UInt64}
│      %3621  = Core.bitcast(Core.UInt, %3604)::UInt64
│      %3622  = Base.add_ptr(%3621, 0x0000000000000020)::UInt64
│      %3623  = Core.bitcast(Ptr{UInt64}, %3622)::Ptr{UInt64}
│               Base.pointerset(%3623, %3593, 1, 1)::Ptr{UInt64}
│      %3625  = Base.bitcast(Ptr{UInt32}, %3604)::Ptr{UInt32}
│      %3626  = Base.llvmcall("%p = inttoptr i64 %0 to i32*\n%v = atomicrmw xchg i32* %p, i32 %1 acq_rel\nret i32 %v\n", UInt32, Tuple{Ptr{UInt32}, UInt32}, %3625, 0x00000000)::UInt32
│      %3627  = Base.bitcast(ThreadingUtilities.ThreadState, %3626)::ThreadingUtilities.ThreadState
│      %3628  = ThreadingUtilities.WAIT::ThreadingUtilities.ThreadState
│      %3629  = (%3627 === %3628)::Bool
└─────          goto #1236 if not %3629
1235 ─          invoke ThreadingUtilities.wake_thread!(%3596::UInt32)::Any
1236 ┄          goto #1237
1237 ─          goto #1238
1238 ─          goto #1233
1239 ─          nothing::Nothing
1240 ┄ %3636  = φ (#1239 => %3574, #1231 => 0x0000000000000000)::UInt64
│      %3637  = Base.add_int(%3636, 0x0000000000000001)::UInt64
│      %3638  = Base.bitcast(Int64, %3637)::Int64
│      %3639  = Base.bitcast(Int64, %3476)::Int64
│      %3640  = Base.mul_int(%3638, 1)::Int64
│      %3641  = Base.add_int(%3640, 1)::Int64
│      %3642  = Base.sub_int(%3641, 1)::Int64
│      %3643  = Base.mul_int(%3639, 1)::Int64
│      %3644  = Base.add_int(%3643, 1)::Int64
└───── %3645  = Base.sub_int(%3644, 1)::Int64
1241 ┄ %3646  = φ (#1240 => %3642, #2396 => %6919)::Int64
│      %3647  = Base.sle_int(%3646, %3645)::Bool
└─────          goto #2397 if not %3647
1242 ─          goto #2396 if not true
1243 ─ %3650  = Base.getfield(dg, :basis)::LobattoLegendreBasis{Float64, 4, SVector{4, Float64}, Matrix{Float64}, Matrix{Float64}}
│      %3651  = Base.getfield(%3650, :derivative_split)::Matrix{Float64}
└─────          goto #2392 if not true
1244 ┄ %3653  = φ (#1243 => 1, #2391 => %6909)::Int64
│      %3654  = φ (#1243 => 1, #2391 => %6910)::Int64
└─────          goto #2387 if not true
1245 ┄ %3656  = φ (#1244 => 1, #2386 => %6899)::Int64
│      %3657  = φ (#1244 => 1, #2386 => %6898)::Int64
└─────          goto #2382 if not true
1246 ┄ %3659  = φ (#1245 => 1, #2381 => %6888)::Int64
│      %3660  = φ (#1245 => 1, #2381 => %6887)::Int64
└─────          goto #1251 if not true
1247 ─ %3662  = Core.tuple(1, %3659, %3656, %3653, %3646)::NTuple{5, Int64}
│      %3663  = StrideArraysCore.getfield(%3551, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %3664  = Core.getfield(%3663, 5)::Int64
│      %3665  = Base.bitcast(UInt64, %3664)::UInt64
│      %3666  = Base.bitcast(Int64, %3665)::Int64
│      %3667  = Base.sle_int(1, %3659)::Bool
│      %3668  = Base.sle_int(%3659, 4)::Bool
│      %3669  = Base.and_int(%3667, %3668)::Bool
│      %3670  = Base.sle_int(1, %3656)::Bool
│      %3671  = Base.sle_int(%3656, 4)::Bool
│      %3672  = Base.and_int(%3670, %3671)::Bool
│      %3673  = Base.sle_int(1, %3653)::Bool
│      %3674  = Base.sle_int(%3653, 4)::Bool
│      %3675  = Base.and_int(%3673, %3674)::Bool
│      %3676  = Base.sub_int(%3646, 1)::Int64
│      %3677  = Base.bitcast(UInt64, %3676)::UInt64
│      %3678  = Base.bitcast(UInt64, %3666)::UInt64
│      %3679  = Base.ult_int(%3677, %3678)::Bool
│      %3680  = Base.and_int(%3679, true)::Bool
│      %3681  = Base.and_int(%3675, %3680)::Bool
│      %3682  = Base.and_int(%3672, %3681)::Bool
│      %3683  = Base.and_int(%3669, %3682)::Bool
│      %3684  = Base.and_int(true, %3683)::Bool
└─────          goto #1249 if not %3684
1248 ─          goto #1250
1249 ─          invoke Base.throw_boundserror(%3551::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %3662::NTuple{5, Int64})::Union{}
└─────          unreachable
1250 ─          nothing::Nothing
1251 ┄ %3690  = StrideArraysCore.getfield(%3551, :ptr)::Ptr{Float64}
│      %3691  = Base.sub_int(%3659, 1)::Int64
│      %3692  = Base.sub_int(%3656, 1)::Int64
│      %3693  = Base.sub_int(%3653, 1)::Int64
│      %3694  = Base.sub_int(%3646, 1)::Int64
└─────          goto #1260 if not true
1252 ┄ %3696  = φ (#1251 => 2, #1259 => %3708)::Int64
│      %3697  = Base.sle_int(1, %3696)::Bool
└─────          goto #1254 if not %3697
1253 ─ %3699  = Base.sle_int(%3696, 5)::Bool
└─────          goto #1255
1254 ─          nothing::Nothing
1255 ┄ %3702  = φ (#1253 => %3699, #1254 => false)::Bool
└─────          goto #1257 if not %3702
1256 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %3696, true)::Static.True
│      %3705  = Base.add_int(%3696, 1)::Int64
└─────          goto #1258
1257 ─          goto #1258
1258 ┄ %3708  = φ (#1256 => %3705)::Int64
│      %3709  = φ (#1256 => false, #1257 => true)::Bool
│      %3710  = Base.not_int(%3709)::Bool
└─────          goto #1260 if not %3710
1259 ─          goto #1252
1260 ┄          goto #1261
1261 ─          goto #1262
1262 ─ %3715  = Base.mul_int(%3694, 4)::Int64
│      %3716  = Base.add_int(%3693, %3715)::Int64
│      %3717  = Base.mul_int(%3716, 4)::Int64
│      %3718  = Base.add_int(%3692, %3717)::Int64
│      %3719  = Base.mul_int(%3718, 4)::Int64
│      %3720  = Base.add_int(%3691, %3719)::Int64
│      %3721  = Base.mul_int(%3720, 5)::Int64
│      %3722  = Base.add_int(0, %3721)::Int64
│      %3723  = Base.mul_int(8, %3722)::Int64
│      %3724  = Core.bitcast(Core.UInt, %3690)::UInt64
│      %3725  = Base.bitcast(UInt64, %3723)::UInt64
│      %3726  = Base.add_ptr(%3724, %3725)::UInt64
│      %3727  = Core.bitcast(Ptr{Float64}, %3726)::Ptr{Float64}
└─────          goto #1263
1263 ─ %3729  = Base.pointerref(%3727, 1, 1)::Float64
└─────          goto #1264
1264 ─          goto #1265
1265 ─          goto #1266
1266 ─          goto #1271 if not true
1267 ─ %3734  = Core.tuple(2, %3659, %3656, %3653, %3646)::NTuple{5, Int64}
│      %3735  = StrideArraysCore.getfield(%3551, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %3736  = Core.getfield(%3735, 5)::Int64
│      %3737  = Base.bitcast(UInt64, %3736)::UInt64
│      %3738  = Base.bitcast(Int64, %3737)::Int64
│      %3739  = Base.sle_int(1, %3659)::Bool
│      %3740  = Base.sle_int(%3659, 4)::Bool
│      %3741  = Base.and_int(%3739, %3740)::Bool
│      %3742  = Base.sle_int(1, %3656)::Bool
│      %3743  = Base.sle_int(%3656, 4)::Bool
│      %3744  = Base.and_int(%3742, %3743)::Bool
│      %3745  = Base.sle_int(1, %3653)::Bool
│      %3746  = Base.sle_int(%3653, 4)::Bool
│      %3747  = Base.and_int(%3745, %3746)::Bool
│      %3748  = Base.sub_int(%3646, 1)::Int64
│      %3749  = Base.bitcast(UInt64, %3748)::UInt64
│      %3750  = Base.bitcast(UInt64, %3738)::UInt64
│      %3751  = Base.ult_int(%3749, %3750)::Bool
│      %3752  = Base.and_int(%3751, true)::Bool
│      %3753  = Base.and_int(%3747, %3752)::Bool
│      %3754  = Base.and_int(%3744, %3753)::Bool
│      %3755  = Base.and_int(%3741, %3754)::Bool
│      %3756  = Base.and_int(true, %3755)::Bool
└─────          goto #1269 if not %3756
1268 ─          goto #1270
1269 ─          invoke Base.throw_boundserror(%3551::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %3734::NTuple{5, Int64})::Union{}
└─────          unreachable
1270 ─          nothing::Nothing
1271 ┄ %3762  = StrideArraysCore.getfield(%3551, :ptr)::Ptr{Float64}
│      %3763  = Base.sub_int(%3659, 1)::Int64
│      %3764  = Base.sub_int(%3656, 1)::Int64
│      %3765  = Base.sub_int(%3653, 1)::Int64
│      %3766  = Base.sub_int(%3646, 1)::Int64
└─────          goto #1280 if not true
1272 ┄ %3768  = φ (#1271 => 2, #1279 => %3780)::Int64
│      %3769  = Base.sle_int(1, %3768)::Bool
└─────          goto #1274 if not %3769
1273 ─ %3771  = Base.sle_int(%3768, 5)::Bool
└─────          goto #1275
1274 ─          nothing::Nothing
1275 ┄ %3774  = φ (#1273 => %3771, #1274 => false)::Bool
└─────          goto #1277 if not %3774
1276 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %3768, true)::Static.True
│      %3777  = Base.add_int(%3768, 1)::Int64
└─────          goto #1278
1277 ─          goto #1278
1278 ┄ %3780  = φ (#1276 => %3777)::Int64
│      %3781  = φ (#1276 => false, #1277 => true)::Bool
│      %3782  = Base.not_int(%3781)::Bool
└─────          goto #1280 if not %3782
1279 ─          goto #1272
1280 ┄          goto #1281
1281 ─          goto #1282
1282 ─ %3787  = Base.mul_int(%3766, 4)::Int64
│      %3788  = Base.add_int(%3765, %3787)::Int64
│      %3789  = Base.mul_int(%3788, 4)::Int64
│      %3790  = Base.add_int(%3764, %3789)::Int64
│      %3791  = Base.mul_int(%3790, 4)::Int64
│      %3792  = Base.add_int(%3763, %3791)::Int64
│      %3793  = Base.mul_int(%3792, 5)::Int64
│      %3794  = Base.add_int(1, %3793)::Int64
│      %3795  = Base.mul_int(8, %3794)::Int64
│      %3796  = Core.bitcast(Core.UInt, %3762)::UInt64
│      %3797  = Base.bitcast(UInt64, %3795)::UInt64
│      %3798  = Base.add_ptr(%3796, %3797)::UInt64
│      %3799  = Core.bitcast(Ptr{Float64}, %3798)::Ptr{Float64}
└─────          goto #1283
1283 ─ %3801  = Base.pointerref(%3799, 1, 1)::Float64
└─────          goto #1284
1284 ─          goto #1285
1285 ─          goto #1286
1286 ─          goto #1291 if not true
1287 ─ %3806  = Core.tuple(3, %3659, %3656, %3653, %3646)::NTuple{5, Int64}
│      %3807  = StrideArraysCore.getfield(%3551, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %3808  = Core.getfield(%3807, 5)::Int64
│      %3809  = Base.bitcast(UInt64, %3808)::UInt64
│      %3810  = Base.bitcast(Int64, %3809)::Int64
│      %3811  = Base.sle_int(1, %3659)::Bool
│      %3812  = Base.sle_int(%3659, 4)::Bool
│      %3813  = Base.and_int(%3811, %3812)::Bool
│      %3814  = Base.sle_int(1, %3656)::Bool
│      %3815  = Base.sle_int(%3656, 4)::Bool
│      %3816  = Base.and_int(%3814, %3815)::Bool
│      %3817  = Base.sle_int(1, %3653)::Bool
│      %3818  = Base.sle_int(%3653, 4)::Bool
│      %3819  = Base.and_int(%3817, %3818)::Bool
│      %3820  = Base.sub_int(%3646, 1)::Int64
│      %3821  = Base.bitcast(UInt64, %3820)::UInt64
│      %3822  = Base.bitcast(UInt64, %3810)::UInt64
│      %3823  = Base.ult_int(%3821, %3822)::Bool
│      %3824  = Base.and_int(%3823, true)::Bool
│      %3825  = Base.and_int(%3819, %3824)::Bool
│      %3826  = Base.and_int(%3816, %3825)::Bool
│      %3827  = Base.and_int(%3813, %3826)::Bool
│      %3828  = Base.and_int(true, %3827)::Bool
└─────          goto #1289 if not %3828
1288 ─          goto #1290
1289 ─          invoke Base.throw_boundserror(%3551::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %3806::NTuple{5, Int64})::Union{}
└─────          unreachable
1290 ─          nothing::Nothing
1291 ┄ %3834  = StrideArraysCore.getfield(%3551, :ptr)::Ptr{Float64}
│      %3835  = Base.sub_int(%3659, 1)::Int64
│      %3836  = Base.sub_int(%3656, 1)::Int64
│      %3837  = Base.sub_int(%3653, 1)::Int64
│      %3838  = Base.sub_int(%3646, 1)::Int64
└─────          goto #1300 if not true
1292 ┄ %3840  = φ (#1291 => 2, #1299 => %3852)::Int64
│      %3841  = Base.sle_int(1, %3840)::Bool
└─────          goto #1294 if not %3841
1293 ─ %3843  = Base.sle_int(%3840, 5)::Bool
└─────          goto #1295
1294 ─          nothing::Nothing
1295 ┄ %3846  = φ (#1293 => %3843, #1294 => false)::Bool
└─────          goto #1297 if not %3846
1296 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %3840, true)::Static.True
│      %3849  = Base.add_int(%3840, 1)::Int64
└─────          goto #1298
1297 ─          goto #1298
1298 ┄ %3852  = φ (#1296 => %3849)::Int64
│      %3853  = φ (#1296 => false, #1297 => true)::Bool
│      %3854  = Base.not_int(%3853)::Bool
└─────          goto #1300 if not %3854
1299 ─          goto #1292
1300 ┄          goto #1301
1301 ─          goto #1302
1302 ─ %3859  = Base.mul_int(%3838, 4)::Int64
│      %3860  = Base.add_int(%3837, %3859)::Int64
│      %3861  = Base.mul_int(%3860, 4)::Int64
│      %3862  = Base.add_int(%3836, %3861)::Int64
│      %3863  = Base.mul_int(%3862, 4)::Int64
│      %3864  = Base.add_int(%3835, %3863)::Int64
│      %3865  = Base.mul_int(%3864, 5)::Int64
│      %3866  = Base.add_int(2, %3865)::Int64
│      %3867  = Base.mul_int(8, %3866)::Int64
│      %3868  = Core.bitcast(Core.UInt, %3834)::UInt64
│      %3869  = Base.bitcast(UInt64, %3867)::UInt64
│      %3870  = Base.add_ptr(%3868, %3869)::UInt64
│      %3871  = Core.bitcast(Ptr{Float64}, %3870)::Ptr{Float64}
└─────          goto #1303
1303 ─ %3873  = Base.pointerref(%3871, 1, 1)::Float64
└─────          goto #1304
1304 ─          goto #1305
1305 ─          goto #1306
1306 ─          goto #1311 if not true
1307 ─ %3878  = Core.tuple(4, %3659, %3656, %3653, %3646)::NTuple{5, Int64}
│      %3879  = StrideArraysCore.getfield(%3551, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %3880  = Core.getfield(%3879, 5)::Int64
│      %3881  = Base.bitcast(UInt64, %3880)::UInt64
│      %3882  = Base.bitcast(Int64, %3881)::Int64
│      %3883  = Base.sle_int(1, %3659)::Bool
│      %3884  = Base.sle_int(%3659, 4)::Bool
│      %3885  = Base.and_int(%3883, %3884)::Bool
│      %3886  = Base.sle_int(1, %3656)::Bool
│      %3887  = Base.sle_int(%3656, 4)::Bool
│      %3888  = Base.and_int(%3886, %3887)::Bool
│      %3889  = Base.sle_int(1, %3653)::Bool
│      %3890  = Base.sle_int(%3653, 4)::Bool
│      %3891  = Base.and_int(%3889, %3890)::Bool
│      %3892  = Base.sub_int(%3646, 1)::Int64
│      %3893  = Base.bitcast(UInt64, %3892)::UInt64
│      %3894  = Base.bitcast(UInt64, %3882)::UInt64
│      %3895  = Base.ult_int(%3893, %3894)::Bool
│      %3896  = Base.and_int(%3895, true)::Bool
│      %3897  = Base.and_int(%3891, %3896)::Bool
│      %3898  = Base.and_int(%3888, %3897)::Bool
│      %3899  = Base.and_int(%3885, %3898)::Bool
│      %3900  = Base.and_int(true, %3899)::Bool
└─────          goto #1309 if not %3900
1308 ─          goto #1310
1309 ─          invoke Base.throw_boundserror(%3551::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %3878::NTuple{5, Int64})::Union{}
└─────          unreachable
1310 ─          nothing::Nothing
1311 ┄ %3906  = StrideArraysCore.getfield(%3551, :ptr)::Ptr{Float64}
│      %3907  = Base.sub_int(%3659, 1)::Int64
│      %3908  = Base.sub_int(%3656, 1)::Int64
│      %3909  = Base.sub_int(%3653, 1)::Int64
│      %3910  = Base.sub_int(%3646, 1)::Int64
└─────          goto #1320 if not true
1312 ┄ %3912  = φ (#1311 => 2, #1319 => %3924)::Int64
│      %3913  = Base.sle_int(1, %3912)::Bool
└─────          goto #1314 if not %3913
1313 ─ %3915  = Base.sle_int(%3912, 5)::Bool
└─────          goto #1315
1314 ─          nothing::Nothing
1315 ┄ %3918  = φ (#1313 => %3915, #1314 => false)::Bool
└─────          goto #1317 if not %3918
1316 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %3912, true)::Static.True
│      %3921  = Base.add_int(%3912, 1)::Int64
└─────          goto #1318
1317 ─          goto #1318
1318 ┄ %3924  = φ (#1316 => %3921)::Int64
│      %3925  = φ (#1316 => false, #1317 => true)::Bool
│      %3926  = Base.not_int(%3925)::Bool
└─────          goto #1320 if not %3926
1319 ─          goto #1312
1320 ┄          goto #1321
1321 ─          goto #1322
1322 ─ %3931  = Base.mul_int(%3910, 4)::Int64
│      %3932  = Base.add_int(%3909, %3931)::Int64
│      %3933  = Base.mul_int(%3932, 4)::Int64
│      %3934  = Base.add_int(%3908, %3933)::Int64
│      %3935  = Base.mul_int(%3934, 4)::Int64
│      %3936  = Base.add_int(%3907, %3935)::Int64
│      %3937  = Base.mul_int(%3936, 5)::Int64
│      %3938  = Base.add_int(3, %3937)::Int64
│      %3939  = Base.mul_int(8, %3938)::Int64
│      %3940  = Core.bitcast(Core.UInt, %3906)::UInt64
│      %3941  = Base.bitcast(UInt64, %3939)::UInt64
│      %3942  = Base.add_ptr(%3940, %3941)::UInt64
│      %3943  = Core.bitcast(Ptr{Float64}, %3942)::Ptr{Float64}
└─────          goto #1323
1323 ─ %3945  = Base.pointerref(%3943, 1, 1)::Float64
└─────          goto #1324
1324 ─          goto #1325
1325 ─          goto #1326
1326 ─          goto #1331 if not true
1327 ─ %3950  = Core.tuple(5, %3659, %3656, %3653, %3646)::NTuple{5, Int64}
│      %3951  = StrideArraysCore.getfield(%3551, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %3952  = Core.getfield(%3951, 5)::Int64
│      %3953  = Base.bitcast(UInt64, %3952)::UInt64
│      %3954  = Base.bitcast(Int64, %3953)::Int64
│      %3955  = Base.sle_int(1, %3659)::Bool
│      %3956  = Base.sle_int(%3659, 4)::Bool
│      %3957  = Base.and_int(%3955, %3956)::Bool
│      %3958  = Base.sle_int(1, %3656)::Bool
│      %3959  = Base.sle_int(%3656, 4)::Bool
│      %3960  = Base.and_int(%3958, %3959)::Bool
│      %3961  = Base.sle_int(1, %3653)::Bool
│      %3962  = Base.sle_int(%3653, 4)::Bool
│      %3963  = Base.and_int(%3961, %3962)::Bool
│      %3964  = Base.sub_int(%3646, 1)::Int64
│      %3965  = Base.bitcast(UInt64, %3964)::UInt64
│      %3966  = Base.bitcast(UInt64, %3954)::UInt64
│      %3967  = Base.ult_int(%3965, %3966)::Bool
│      %3968  = Base.and_int(%3967, true)::Bool
│      %3969  = Base.and_int(%3963, %3968)::Bool
│      %3970  = Base.and_int(%3960, %3969)::Bool
│      %3971  = Base.and_int(%3957, %3970)::Bool
│      %3972  = Base.and_int(true, %3971)::Bool
└─────          goto #1329 if not %3972
1328 ─          goto #1330
1329 ─          invoke Base.throw_boundserror(%3551::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %3950::NTuple{5, Int64})::Union{}
└─────          unreachable
1330 ─          nothing::Nothing
1331 ┄ %3978  = StrideArraysCore.getfield(%3551, :ptr)::Ptr{Float64}
│      %3979  = Base.sub_int(%3659, 1)::Int64
│      %3980  = Base.sub_int(%3656, 1)::Int64
│      %3981  = Base.sub_int(%3653, 1)::Int64
│      %3982  = Base.sub_int(%3646, 1)::Int64
└─────          goto #1340 if not true
1332 ┄ %3984  = φ (#1331 => 2, #1339 => %3996)::Int64
│      %3985  = Base.sle_int(1, %3984)::Bool
└─────          goto #1334 if not %3985
1333 ─ %3987  = Base.sle_int(%3984, 5)::Bool
└─────          goto #1335
1334 ─          nothing::Nothing
1335 ┄ %3990  = φ (#1333 => %3987, #1334 => false)::Bool
└─────          goto #1337 if not %3990
1336 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %3984, true)::Static.True
│      %3993  = Base.add_int(%3984, 1)::Int64
└─────          goto #1338
1337 ─          goto #1338
1338 ┄ %3996  = φ (#1336 => %3993)::Int64
│      %3997  = φ (#1336 => false, #1337 => true)::Bool
│      %3998  = Base.not_int(%3997)::Bool
└─────          goto #1340 if not %3998
1339 ─          goto #1332
1340 ┄          goto #1341
1341 ─          goto #1342
1342 ─ %4003  = Base.mul_int(%3982, 4)::Int64
│      %4004  = Base.add_int(%3981, %4003)::Int64
│      %4005  = Base.mul_int(%4004, 4)::Int64
│      %4006  = Base.add_int(%3980, %4005)::Int64
│      %4007  = Base.mul_int(%4006, 4)::Int64
│      %4008  = Base.add_int(%3979, %4007)::Int64
│      %4009  = Base.mul_int(%4008, 5)::Int64
│      %4010  = Base.add_int(4, %4009)::Int64
│      %4011  = Base.mul_int(8, %4010)::Int64
│      %4012  = Core.bitcast(Core.UInt, %3978)::UInt64
│      %4013  = Base.bitcast(UInt64, %4011)::UInt64
│      %4014  = Base.add_ptr(%4012, %4013)::UInt64
│      %4015  = Core.bitcast(Ptr{Float64}, %4014)::Ptr{Float64}
└─────          goto #1343
1343 ─ %4017  = Base.pointerref(%4015, 1, 1)::Float64
└─────          goto #1344
1344 ─          goto #1345
1345 ─          goto #1346
1346 ─          goto #1347
1347 ─          goto #1348
1348 ─ %4023  = Base.add_int(%3659, 1)::Int64
│      %4024  = Base.sle_int(%4023, 4)::Bool
└─────          goto #1350 if not %4024
1349 ─          goto #1351
1350 ─ %4027  = Base.sub_int(%4023, 1)::Int64
└─────          goto #1351
1351 ┄ %4029  = φ (#1349 => 4, #1350 => %4027)::Int64
└─────          goto #1352
1352 ─          goto #1353
1353 ─ %4032  = Base.slt_int(%4029, %4023)::Bool
└─────          goto #1355 if not %4032
1354 ─          goto #1356
1355 ─          goto #1356
1356 ┄ %4036  = φ (#1354 => true, #1355 => false)::Bool
│      %4037  = φ (#1355 => %4023)::Int64
│      %4038  = φ (#1355 => %4023)::Int64
│      %4039  = Base.not_int(%4036)::Bool
└─────          goto #1691 if not %4039
1357 ┄ %4041  = φ (#1356 => %4037, #1690 => %4970)::Int64
│      %4042  = φ (#1356 => %4038, #1690 => %4971)::Int64
└─────          goto #1362 if not true
1358 ─ %4044  = Core.tuple(1, %4041, %3656, %3653, %3646)::NTuple{5, Int64}
│      %4045  = StrideArraysCore.getfield(%3551, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %4046  = Core.getfield(%4045, 5)::Int64
│      %4047  = Base.bitcast(UInt64, %4046)::UInt64
│      %4048  = Base.bitcast(Int64, %4047)::Int64
│      %4049  = Base.sle_int(1, %4041)::Bool
│      %4050  = Base.sle_int(%4041, 4)::Bool
│      %4051  = Base.and_int(%4049, %4050)::Bool
│      %4052  = Base.sle_int(1, %3656)::Bool
│      %4053  = Base.sle_int(%3656, 4)::Bool
│      %4054  = Base.and_int(%4052, %4053)::Bool
│      %4055  = Base.sle_int(1, %3653)::Bool
│      %4056  = Base.sle_int(%3653, 4)::Bool
│      %4057  = Base.and_int(%4055, %4056)::Bool
│      %4058  = Base.sub_int(%3646, 1)::Int64
│      %4059  = Base.bitcast(UInt64, %4058)::UInt64
│      %4060  = Base.bitcast(UInt64, %4048)::UInt64
│      %4061  = Base.ult_int(%4059, %4060)::Bool
│      %4062  = Base.and_int(%4061, true)::Bool
│      %4063  = Base.and_int(%4057, %4062)::Bool
│      %4064  = Base.and_int(%4054, %4063)::Bool
│      %4065  = Base.and_int(%4051, %4064)::Bool
│      %4066  = Base.and_int(true, %4065)::Bool
└─────          goto #1360 if not %4066
1359 ─          goto #1361
1360 ─          invoke Base.throw_boundserror(%3551::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %4044::NTuple{5, Int64})::Union{}
└─────          unreachable
1361 ─          nothing::Nothing
1362 ┄ %4072  = StrideArraysCore.getfield(%3551, :ptr)::Ptr{Float64}
│      %4073  = Base.sub_int(%4041, 1)::Int64
│      %4074  = Base.sub_int(%3656, 1)::Int64
│      %4075  = Base.sub_int(%3653, 1)::Int64
│      %4076  = Base.sub_int(%3646, 1)::Int64
└─────          goto #1371 if not true
1363 ┄ %4078  = φ (#1362 => 2, #1370 => %4090)::Int64
│      %4079  = Base.sle_int(1, %4078)::Bool
└─────          goto #1365 if not %4079
1364 ─ %4081  = Base.sle_int(%4078, 5)::Bool
└─────          goto #1366
1365 ─          nothing::Nothing
1366 ┄ %4084  = φ (#1364 => %4081, #1365 => false)::Bool
└─────          goto #1368 if not %4084
1367 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %4078, true)::Static.True
│      %4087  = Base.add_int(%4078, 1)::Int64
└─────          goto #1369
1368 ─          goto #1369
1369 ┄ %4090  = φ (#1367 => %4087)::Int64
│      %4091  = φ (#1367 => false, #1368 => true)::Bool
│      %4092  = Base.not_int(%4091)::Bool
└─────          goto #1371 if not %4092
1370 ─          goto #1363
1371 ┄          goto #1372
1372 ─          goto #1373
1373 ─ %4097  = Base.mul_int(%4076, 4)::Int64
│      %4098  = Base.add_int(%4075, %4097)::Int64
│      %4099  = Base.mul_int(%4098, 4)::Int64
│      %4100  = Base.add_int(%4074, %4099)::Int64
│      %4101  = Base.mul_int(%4100, 4)::Int64
│      %4102  = Base.add_int(%4073, %4101)::Int64
│      %4103  = Base.mul_int(%4102, 5)::Int64
│      %4104  = Base.add_int(0, %4103)::Int64
│      %4105  = Base.mul_int(8, %4104)::Int64
│      %4106  = Core.bitcast(Core.UInt, %4072)::UInt64
│      %4107  = Base.bitcast(UInt64, %4105)::UInt64
│      %4108  = Base.add_ptr(%4106, %4107)::UInt64
│      %4109  = Core.bitcast(Ptr{Float64}, %4108)::Ptr{Float64}
└─────          goto #1374
1374 ─ %4111  = Base.pointerref(%4109, 1, 1)::Float64
└─────          goto #1375
1375 ─          goto #1376
1376 ─          goto #1377
1377 ─          goto #1382 if not true
1378 ─ %4116  = Core.tuple(2, %4041, %3656, %3653, %3646)::NTuple{5, Int64}
│      %4117  = StrideArraysCore.getfield(%3551, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %4118  = Core.getfield(%4117, 5)::Int64
│      %4119  = Base.bitcast(UInt64, %4118)::UInt64
│      %4120  = Base.bitcast(Int64, %4119)::Int64
│      %4121  = Base.sle_int(1, %4041)::Bool
│      %4122  = Base.sle_int(%4041, 4)::Bool
│      %4123  = Base.and_int(%4121, %4122)::Bool
│      %4124  = Base.sle_int(1, %3656)::Bool
│      %4125  = Base.sle_int(%3656, 4)::Bool
│      %4126  = Base.and_int(%4124, %4125)::Bool
│      %4127  = Base.sle_int(1, %3653)::Bool
│      %4128  = Base.sle_int(%3653, 4)::Bool
│      %4129  = Base.and_int(%4127, %4128)::Bool
│      %4130  = Base.sub_int(%3646, 1)::Int64
│      %4131  = Base.bitcast(UInt64, %4130)::UInt64
│      %4132  = Base.bitcast(UInt64, %4120)::UInt64
│      %4133  = Base.ult_int(%4131, %4132)::Bool
│      %4134  = Base.and_int(%4133, true)::Bool
│      %4135  = Base.and_int(%4129, %4134)::Bool
│      %4136  = Base.and_int(%4126, %4135)::Bool
│      %4137  = Base.and_int(%4123, %4136)::Bool
│      %4138  = Base.and_int(true, %4137)::Bool
└─────          goto #1380 if not %4138
1379 ─          goto #1381
1380 ─          invoke Base.throw_boundserror(%3551::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %4116::NTuple{5, Int64})::Union{}
└─────          unreachable
1381 ─          nothing::Nothing
1382 ┄ %4144  = StrideArraysCore.getfield(%3551, :ptr)::Ptr{Float64}
│      %4145  = Base.sub_int(%4041, 1)::Int64
│      %4146  = Base.sub_int(%3656, 1)::Int64
│      %4147  = Base.sub_int(%3653, 1)::Int64
│      %4148  = Base.sub_int(%3646, 1)::Int64
└─────          goto #1391 if not true
1383 ┄ %4150  = φ (#1382 => 2, #1390 => %4162)::Int64
│      %4151  = Base.sle_int(1, %4150)::Bool
└─────          goto #1385 if not %4151
1384 ─ %4153  = Base.sle_int(%4150, 5)::Bool
└─────          goto #1386
1385 ─          nothing::Nothing
1386 ┄ %4156  = φ (#1384 => %4153, #1385 => false)::Bool
└─────          goto #1388 if not %4156
1387 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %4150, true)::Static.True
│      %4159  = Base.add_int(%4150, 1)::Int64
└─────          goto #1389
1388 ─          goto #1389
1389 ┄ %4162  = φ (#1387 => %4159)::Int64
│      %4163  = φ (#1387 => false, #1388 => true)::Bool
│      %4164  = Base.not_int(%4163)::Bool
└─────          goto #1391 if not %4164
1390 ─          goto #1383
1391 ┄          goto #1392
1392 ─          goto #1393
1393 ─ %4169  = Base.mul_int(%4148, 4)::Int64
│      %4170  = Base.add_int(%4147, %4169)::Int64
│      %4171  = Base.mul_int(%4170, 4)::Int64
│      %4172  = Base.add_int(%4146, %4171)::Int64
│      %4173  = Base.mul_int(%4172, 4)::Int64
│      %4174  = Base.add_int(%4145, %4173)::Int64
│      %4175  = Base.mul_int(%4174, 5)::Int64
│      %4176  = Base.add_int(1, %4175)::Int64
│      %4177  = Base.mul_int(8, %4176)::Int64
│      %4178  = Core.bitcast(Core.UInt, %4144)::UInt64
│      %4179  = Base.bitcast(UInt64, %4177)::UInt64
│      %4180  = Base.add_ptr(%4178, %4179)::UInt64
│      %4181  = Core.bitcast(Ptr{Float64}, %4180)::Ptr{Float64}
└─────          goto #1394
1394 ─ %4183  = Base.pointerref(%4181, 1, 1)::Float64
└─────          goto #1395
1395 ─          goto #1396
1396 ─          goto #1397
1397 ─          goto #1402 if not true
1398 ─ %4188  = Core.tuple(3, %4041, %3656, %3653, %3646)::NTuple{5, Int64}
│      %4189  = StrideArraysCore.getfield(%3551, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %4190  = Core.getfield(%4189, 5)::Int64
│      %4191  = Base.bitcast(UInt64, %4190)::UInt64
│      %4192  = Base.bitcast(Int64, %4191)::Int64
│      %4193  = Base.sle_int(1, %4041)::Bool
│      %4194  = Base.sle_int(%4041, 4)::Bool
│      %4195  = Base.and_int(%4193, %4194)::Bool
│      %4196  = Base.sle_int(1, %3656)::Bool
│      %4197  = Base.sle_int(%3656, 4)::Bool
│      %4198  = Base.and_int(%4196, %4197)::Bool
│      %4199  = Base.sle_int(1, %3653)::Bool
│      %4200  = Base.sle_int(%3653, 4)::Bool
│      %4201  = Base.and_int(%4199, %4200)::Bool
│      %4202  = Base.sub_int(%3646, 1)::Int64
│      %4203  = Base.bitcast(UInt64, %4202)::UInt64
│      %4204  = Base.bitcast(UInt64, %4192)::UInt64
│      %4205  = Base.ult_int(%4203, %4204)::Bool
│      %4206  = Base.and_int(%4205, true)::Bool
│      %4207  = Base.and_int(%4201, %4206)::Bool
│      %4208  = Base.and_int(%4198, %4207)::Bool
│      %4209  = Base.and_int(%4195, %4208)::Bool
│      %4210  = Base.and_int(true, %4209)::Bool
└─────          goto #1400 if not %4210
1399 ─          goto #1401
1400 ─          invoke Base.throw_boundserror(%3551::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %4188::NTuple{5, Int64})::Union{}
└─────          unreachable
1401 ─          nothing::Nothing
1402 ┄ %4216  = StrideArraysCore.getfield(%3551, :ptr)::Ptr{Float64}
│      %4217  = Base.sub_int(%4041, 1)::Int64
│      %4218  = Base.sub_int(%3656, 1)::Int64
│      %4219  = Base.sub_int(%3653, 1)::Int64
│      %4220  = Base.sub_int(%3646, 1)::Int64
└─────          goto #1411 if not true
1403 ┄ %4222  = φ (#1402 => 2, #1410 => %4234)::Int64
│      %4223  = Base.sle_int(1, %4222)::Bool
└─────          goto #1405 if not %4223
1404 ─ %4225  = Base.sle_int(%4222, 5)::Bool
└─────          goto #1406
1405 ─          nothing::Nothing
1406 ┄ %4228  = φ (#1404 => %4225, #1405 => false)::Bool
└─────          goto #1408 if not %4228
1407 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %4222, true)::Static.True
│      %4231  = Base.add_int(%4222, 1)::Int64
└─────          goto #1409
1408 ─          goto #1409
1409 ┄ %4234  = φ (#1407 => %4231)::Int64
│      %4235  = φ (#1407 => false, #1408 => true)::Bool
│      %4236  = Base.not_int(%4235)::Bool
└─────          goto #1411 if not %4236
1410 ─          goto #1403
1411 ┄          goto #1412
1412 ─          goto #1413
1413 ─ %4241  = Base.mul_int(%4220, 4)::Int64
│      %4242  = Base.add_int(%4219, %4241)::Int64
│      %4243  = Base.mul_int(%4242, 4)::Int64
│      %4244  = Base.add_int(%4218, %4243)::Int64
│      %4245  = Base.mul_int(%4244, 4)::Int64
│      %4246  = Base.add_int(%4217, %4245)::Int64
│      %4247  = Base.mul_int(%4246, 5)::Int64
│      %4248  = Base.add_int(2, %4247)::Int64
│      %4249  = Base.mul_int(8, %4248)::Int64
│      %4250  = Core.bitcast(Core.UInt, %4216)::UInt64
│      %4251  = Base.bitcast(UInt64, %4249)::UInt64
│      %4252  = Base.add_ptr(%4250, %4251)::UInt64
│      %4253  = Core.bitcast(Ptr{Float64}, %4252)::Ptr{Float64}
└─────          goto #1414
1414 ─ %4255  = Base.pointerref(%4253, 1, 1)::Float64
└─────          goto #1415
1415 ─          goto #1416
1416 ─          goto #1417
1417 ─          goto #1422 if not true
1418 ─ %4260  = Core.tuple(4, %4041, %3656, %3653, %3646)::NTuple{5, Int64}
│      %4261  = StrideArraysCore.getfield(%3551, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %4262  = Core.getfield(%4261, 5)::Int64
│      %4263  = Base.bitcast(UInt64, %4262)::UInt64
│      %4264  = Base.bitcast(Int64, %4263)::Int64
│      %4265  = Base.sle_int(1, %4041)::Bool
│      %4266  = Base.sle_int(%4041, 4)::Bool
│      %4267  = Base.and_int(%4265, %4266)::Bool
│      %4268  = Base.sle_int(1, %3656)::Bool
│      %4269  = Base.sle_int(%3656, 4)::Bool
│      %4270  = Base.and_int(%4268, %4269)::Bool
│      %4271  = Base.sle_int(1, %3653)::Bool
│      %4272  = Base.sle_int(%3653, 4)::Bool
│      %4273  = Base.and_int(%4271, %4272)::Bool
│      %4274  = Base.sub_int(%3646, 1)::Int64
│      %4275  = Base.bitcast(UInt64, %4274)::UInt64
│      %4276  = Base.bitcast(UInt64, %4264)::UInt64
│      %4277  = Base.ult_int(%4275, %4276)::Bool
│      %4278  = Base.and_int(%4277, true)::Bool
│      %4279  = Base.and_int(%4273, %4278)::Bool
│      %4280  = Base.and_int(%4270, %4279)::Bool
│      %4281  = Base.and_int(%4267, %4280)::Bool
│      %4282  = Base.and_int(true, %4281)::Bool
└─────          goto #1420 if not %4282
1419 ─          goto #1421
1420 ─          invoke Base.throw_boundserror(%3551::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %4260::NTuple{5, Int64})::Union{}
└─────          unreachable
1421 ─          nothing::Nothing
1422 ┄ %4288  = StrideArraysCore.getfield(%3551, :ptr)::Ptr{Float64}
│      %4289  = Base.sub_int(%4041, 1)::Int64
│      %4290  = Base.sub_int(%3656, 1)::Int64
│      %4291  = Base.sub_int(%3653, 1)::Int64
│      %4292  = Base.sub_int(%3646, 1)::Int64
└─────          goto #1431 if not true
1423 ┄ %4294  = φ (#1422 => 2, #1430 => %4306)::Int64
│      %4295  = Base.sle_int(1, %4294)::Bool
└─────          goto #1425 if not %4295
1424 ─ %4297  = Base.sle_int(%4294, 5)::Bool
└─────          goto #1426
1425 ─          nothing::Nothing
1426 ┄ %4300  = φ (#1424 => %4297, #1425 => false)::Bool
└─────          goto #1428 if not %4300
1427 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %4294, true)::Static.True
│      %4303  = Base.add_int(%4294, 1)::Int64
└─────          goto #1429
1428 ─          goto #1429
1429 ┄ %4306  = φ (#1427 => %4303)::Int64
│      %4307  = φ (#1427 => false, #1428 => true)::Bool
│      %4308  = Base.not_int(%4307)::Bool
└─────          goto #1431 if not %4308
1430 ─          goto #1423
1431 ┄          goto #1432
1432 ─          goto #1433
1433 ─ %4313  = Base.mul_int(%4292, 4)::Int64
│      %4314  = Base.add_int(%4291, %4313)::Int64
│      %4315  = Base.mul_int(%4314, 4)::Int64
│      %4316  = Base.add_int(%4290, %4315)::Int64
│      %4317  = Base.mul_int(%4316, 4)::Int64
│      %4318  = Base.add_int(%4289, %4317)::Int64
│      %4319  = Base.mul_int(%4318, 5)::Int64
│      %4320  = Base.add_int(3, %4319)::Int64
│      %4321  = Base.mul_int(8, %4320)::Int64
│      %4322  = Core.bitcast(Core.UInt, %4288)::UInt64
│      %4323  = Base.bitcast(UInt64, %4321)::UInt64
│      %4324  = Base.add_ptr(%4322, %4323)::UInt64
│      %4325  = Core.bitcast(Ptr{Float64}, %4324)::Ptr{Float64}
└─────          goto #1434
1434 ─ %4327  = Base.pointerref(%4325, 1, 1)::Float64
└─────          goto #1435
1435 ─          goto #1436
1436 ─          goto #1437
1437 ─          goto #1442 if not true
1438 ─ %4332  = Core.tuple(5, %4041, %3656, %3653, %3646)::NTuple{5, Int64}
│      %4333  = StrideArraysCore.getfield(%3551, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %4334  = Core.getfield(%4333, 5)::Int64
│      %4335  = Base.bitcast(UInt64, %4334)::UInt64
│      %4336  = Base.bitcast(Int64, %4335)::Int64
│      %4337  = Base.sle_int(1, %4041)::Bool
│      %4338  = Base.sle_int(%4041, 4)::Bool
│      %4339  = Base.and_int(%4337, %4338)::Bool
│      %4340  = Base.sle_int(1, %3656)::Bool
│      %4341  = Base.sle_int(%3656, 4)::Bool
│      %4342  = Base.and_int(%4340, %4341)::Bool
│      %4343  = Base.sle_int(1, %3653)::Bool
│      %4344  = Base.sle_int(%3653, 4)::Bool
│      %4345  = Base.and_int(%4343, %4344)::Bool
│      %4346  = Base.sub_int(%3646, 1)::Int64
│      %4347  = Base.bitcast(UInt64, %4346)::UInt64
│      %4348  = Base.bitcast(UInt64, %4336)::UInt64
│      %4349  = Base.ult_int(%4347, %4348)::Bool
│      %4350  = Base.and_int(%4349, true)::Bool
│      %4351  = Base.and_int(%4345, %4350)::Bool
│      %4352  = Base.and_int(%4342, %4351)::Bool
│      %4353  = Base.and_int(%4339, %4352)::Bool
│      %4354  = Base.and_int(true, %4353)::Bool
└─────          goto #1440 if not %4354
1439 ─          goto #1441
1440 ─          invoke Base.throw_boundserror(%3551::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %4332::NTuple{5, Int64})::Union{}
└─────          unreachable
1441 ─          nothing::Nothing
1442 ┄ %4360  = StrideArraysCore.getfield(%3551, :ptr)::Ptr{Float64}
│      %4361  = Base.sub_int(%4041, 1)::Int64
│      %4362  = Base.sub_int(%3656, 1)::Int64
│      %4363  = Base.sub_int(%3653, 1)::Int64
│      %4364  = Base.sub_int(%3646, 1)::Int64
└─────          goto #1451 if not true
1443 ┄ %4366  = φ (#1442 => 2, #1450 => %4378)::Int64
│      %4367  = Base.sle_int(1, %4366)::Bool
└─────          goto #1445 if not %4367
1444 ─ %4369  = Base.sle_int(%4366, 5)::Bool
└─────          goto #1446
1445 ─          nothing::Nothing
1446 ┄ %4372  = φ (#1444 => %4369, #1445 => false)::Bool
└─────          goto #1448 if not %4372
1447 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %4366, true)::Static.True
│      %4375  = Base.add_int(%4366, 1)::Int64
└─────          goto #1449
1448 ─          goto #1449
1449 ┄ %4378  = φ (#1447 => %4375)::Int64
│      %4379  = φ (#1447 => false, #1448 => true)::Bool
│      %4380  = Base.not_int(%4379)::Bool
└─────          goto #1451 if not %4380
1450 ─          goto #1443
1451 ┄          goto #1452
1452 ─          goto #1453
1453 ─ %4385  = Base.mul_int(%4364, 4)::Int64
│      %4386  = Base.add_int(%4363, %4385)::Int64
│      %4387  = Base.mul_int(%4386, 4)::Int64
│      %4388  = Base.add_int(%4362, %4387)::Int64
│      %4389  = Base.mul_int(%4388, 4)::Int64
│      %4390  = Base.add_int(%4361, %4389)::Int64
│      %4391  = Base.mul_int(%4390, 5)::Int64
│      %4392  = Base.add_int(4, %4391)::Int64
│      %4393  = Base.mul_int(8, %4392)::Int64
│      %4394  = Core.bitcast(Core.UInt, %4360)::UInt64
│      %4395  = Base.bitcast(UInt64, %4393)::UInt64
│      %4396  = Base.add_ptr(%4394, %4395)::UInt64
│      %4397  = Core.bitcast(Ptr{Float64}, %4396)::Ptr{Float64}
└─────          goto #1454
1454 ─ %4399  = Base.pointerref(%4397, 1, 1)::Float64
└─────          goto #1455
1455 ─          goto #1456
1456 ─          goto #1457
1457 ─          goto #1458
1458 ─          goto #1459
1459 ─          goto #1461
1460 ─          nothing::Nothing
1461 ┄          goto #1463
1462 ─          nothing::Nothing
1463 ┄          goto #1464
1464 ─          goto #1466
1465 ─          nothing::Nothing
1466 ┄          goto #1467
1467 ─          goto #1469
1468 ─          nothing::Nothing
1469 ┄          goto #1471
1470 ─          nothing::Nothing
1471 ┄          goto #1472
1472 ─          goto #1474
1473 ─          nothing::Nothing
1474 ┄          goto #1475
1475 ─          goto #1477
1476 ─          nothing::Nothing
1477 ┄          goto #1479
1478 ─          nothing::Nothing
1479 ┄          goto #1480
1480 ─          goto #1482
1481 ─          nothing::Nothing
1482 ┄          goto #1483
1483 ─          goto #1485
1484 ─          nothing::Nothing
1485 ┄          goto #1487
1486 ─          nothing::Nothing
1487 ┄          goto #1488
1488 ─          goto #1490
1489 ─          nothing::Nothing
1490 ┄          goto #1491
1491 ─ %4437  = Base.div_float(%3801, %3729)::Float64
│      %4438  = Base.div_float(%3873, %3729)::Float64
│      %4439  = Base.div_float(%3945, %3729)::Float64
│      %4440  = Base.getfield(equations, :gamma)::Float64
│      %4441  = Base.sub_float(%4440, 1.0)::Float64
│      %4442  = Base.mul_float(%3801, %4437)::Float64
│      %4443  = Base.muladd_float(%3873, %4438, %4442)::Float64
│      %4444  = Base.muladd_float(%3945, %4439, %4443)::Float64
│      %4445  = Base.muladd_float(-0.5, %4444, %4017)::Float64
│      %4446  = Base.mul_float(%4441, %4445)::Float64
└─────          goto #1492
1492 ─          goto #1494
1493 ─          nothing::Nothing
1494 ┄          goto #1496
1495 ─          nothing::Nothing
1496 ┄          goto #1497
1497 ─          goto #1499
1498 ─          nothing::Nothing
1499 ┄          goto #1500
1500 ─          goto #1502
1501 ─          nothing::Nothing
1502 ┄          goto #1504
1503 ─          nothing::Nothing
1504 ┄          goto #1505
1505 ─          goto #1507
1506 ─          nothing::Nothing
1507 ┄          goto #1508
1508 ─          goto #1510
1509 ─          nothing::Nothing
1510 ┄          goto #1512
1511 ─          nothing::Nothing
1512 ┄          goto #1513
1513 ─          goto #1515
1514 ─          nothing::Nothing
1515 ┄          goto #1516
1516 ─          goto #1518
1517 ─          nothing::Nothing
1518 ┄          goto #1520
1519 ─          nothing::Nothing
1520 ┄          goto #1521
1521 ─          goto #1523
1522 ─          nothing::Nothing
1523 ┄          goto #1524
1524 ─          goto #1526
1525 ─          nothing::Nothing
1526 ┄          goto #1528
1527 ─          nothing::Nothing
1528 ┄          goto #1529
1529 ─          goto #1531
1530 ─          nothing::Nothing
1531 ┄          goto #1532
1532 ─          goto #1534
1533 ─          nothing::Nothing
1534 ┄          goto #1536
1535 ─          nothing::Nothing
1536 ┄          goto #1537
1537 ─          goto #1539
1538 ─          nothing::Nothing
1539 ┄          goto #1540
1540 ─          goto #1542
1541 ─          nothing::Nothing
1542 ┄          goto #1544
1543 ─          nothing::Nothing
1544 ┄          goto #1545
1545 ─          goto #1547
1546 ─          nothing::Nothing
1547 ┄          goto #1548
1548 ─          goto #1550
1549 ─          nothing::Nothing
1550 ┄          goto #1552
1551 ─          nothing::Nothing
1552 ┄          goto #1553
1553 ─          goto #1555
1554 ─          nothing::Nothing
1555 ┄          goto #1556
1556 ─ %4512  = Base.div_float(%4183, %4111)::Float64
│      %4513  = Base.div_float(%4255, %4111)::Float64
│      %4514  = Base.div_float(%4327, %4111)::Float64
│      %4515  = Base.getfield(equations, :gamma)::Float64
│      %4516  = Base.sub_float(%4515, 1.0)::Float64
│      %4517  = Base.mul_float(%4183, %4512)::Float64
│      %4518  = Base.muladd_float(%4255, %4513, %4517)::Float64
│      %4519  = Base.muladd_float(%4327, %4514, %4518)::Float64
│      %4520  = Base.muladd_float(-0.5, %4519, %4399)::Float64
│      %4521  = Base.mul_float(%4516, %4520)::Float64
└─────          goto #1557
1557 ─          goto #1559
1558 ─          nothing::Nothing
1559 ┄          goto #1561
1560 ─          nothing::Nothing
1561 ┄          goto #1562
1562 ─          goto #1564
1563 ─          nothing::Nothing
1564 ┄          goto #1565
1565 ─          goto #1567
1566 ─          nothing::Nothing
1567 ┄          goto #1569
1568 ─          nothing::Nothing
1569 ┄          goto #1570
1570 ─          goto #1572
1571 ─          nothing::Nothing
1572 ┄          goto #1573
1573 ─          goto #1575
1574 ─          nothing::Nothing
1575 ┄          goto #1577
1576 ─          nothing::Nothing
1577 ┄          goto #1578
1578 ─          goto #1580
1579 ─          nothing::Nothing
1580 ┄          goto #1581
1581 ─          goto #1583
1582 ─          nothing::Nothing
1583 ┄          goto #1585
1584 ─          nothing::Nothing
1585 ┄          goto #1586
1586 ─          goto #1588
1587 ─          nothing::Nothing
1588 ┄          goto #1589
1589 ─ %4555  = Base.muladd_float(-2.0, %4111, %3729)::Float64
│      %4556  = Base.mul_float(%3729, %4555)::Float64
│      %4557  = Base.muladd_float(%4111, %4111, %4556)::Float64
│      %4558  = Base.muladd_float(2.0, %4111, %3729)::Float64
│      %4559  = Base.mul_float(%3729, %4558)::Float64
│      %4560  = Base.muladd_float(%4111, %4111, %4559)::Float64
│      %4561  = Base.div_float(%4557, %4560)::Float64
│      %4562  = Base.lt_float(%4561, 0.0001)::Bool
└─────          goto #1591 if not %4562
1590 ─ %4564  = Base.add_float(%3729, %4111)::Float64
│      %4565  = Base.muladd_float(%4561, 0.2857142857142857, 0.4)::Float64
│      %4566  = Base.muladd_float(%4561, %4565, 0.6666666666666666)::Float64
│      %4567  = Base.muladd_float(%4561, %4566, 2.0)::Float64
│      %4568  = Base.div_float(%4564, %4567)::Float64
└─────          goto #1592
1591 ─ %4570  = Base.sub_float(%4111, %3729)::Float64
│      %4571  = Base.div_float(%4111, %3729)::Float64
│      %4572  = $(Expr(:foreigncall, "llvm.log.f64", Float64, svec(Float64), 0, :(:llvmcall), :(%4571), :(%4571)))::Float64
│      %4573  = Base.div_float(%4570, %4572)::Float64
└─────          goto #1592
1592 ┄ %4575  = φ (#1590 => %4568, #1591 => %4573)::Float64
│      %4576  = Base.mul_float(%3729, %4521)::Float64
│      %4577  = Base.mul_float(%4111, %4446)::Float64
│      %4578  = Base.muladd_float(-2.0, %4577, %4576)::Float64
│      %4579  = Base.mul_float(%4576, %4578)::Float64
│      %4580  = Base.muladd_float(%4577, %4577, %4579)::Float64
│      %4581  = Base.muladd_float(2.0, %4577, %4576)::Float64
│      %4582  = Base.mul_float(%4576, %4581)::Float64
│      %4583  = Base.muladd_float(%4577, %4577, %4582)::Float64
│      %4584  = Base.div_float(%4580, %4583)::Float64
│      %4585  = Base.lt_float(%4584, 0.0001)::Bool
└─────          goto #1594 if not %4585
1593 ─ %4587  = Base.muladd_float(%4584, 0.2857142857142857, 0.4)::Float64
│      %4588  = Base.muladd_float(%4584, %4587, 0.6666666666666666)::Float64
│      %4589  = Base.muladd_float(%4584, %4588, 2.0)::Float64
│      %4590  = Base.add_float(%4576, %4577)::Float64
│      %4591  = Base.div_float(%4589, %4590)::Float64
└─────          goto #1595
1594 ─ %4593  = Base.div_float(%4577, %4576)::Float64
│      %4594  = $(Expr(:foreigncall, "llvm.log.f64", Float64, svec(Float64), 0, :(:llvmcall), :(%4593), :(%4593)))::Float64
│      %4595  = Base.sub_float(%4577, %4576)::Float64
│      %4596  = Base.div_float(%4594, %4595)::Float64
└─────          goto #1595
1595 ┄ %4598  = φ (#1593 => %4591, #1594 => %4596)::Float64
│      %4599  = Base.mul_float(%4446, %4521)::Float64
│      %4600  = Base.mul_float(%4599, %4598)::Float64
│      %4601  = Base.add_float(%4437, %4512)::Float64
│      %4602  = Base.mul_float(0.5, %4601)::Float64
│      %4603  = Base.add_float(%4438, %4513)::Float64
│      %4604  = Base.mul_float(0.5, %4603)::Float64
│      %4605  = Base.add_float(%4439, %4514)::Float64
│      %4606  = Base.mul_float(0.5, %4605)::Float64
│      %4607  = Base.add_float(%4446, %4521)::Float64
│      %4608  = Base.mul_float(0.5, %4607)::Float64
│      %4609  = Base.mul_float(%4437, %4512)::Float64
│      %4610  = Base.muladd_float(%4438, %4513, %4609)::Float64
│      %4611  = Base.muladd_float(%4439, %4514, %4610)::Float64
│      %4612  = Base.mul_float(0.5, %4611)::Float64
│      %4613  = Base.mul_float(%4575, %4602)::Float64
│      %4614  = Base.muladd_float(%4613, %4602, %4608)::Float64
│      %4615  = Base.mul_float(%4613, %4604)::Float64
│      %4616  = Base.mul_float(%4613, %4606)::Float64
│      %4617  = Base.mul_float(%4446, %4512)::Float64
│      %4618  = Base.muladd_float(%4521, %4437, %4617)::Float64
│      %4619  = Base.getfield(equations, :inv_gamma_minus_one)::Float64
│      %4620  = Base.muladd_float(%4600, %4619, %4612)::Float64
│      %4621  = Base.mul_float(%4613, %4620)::Float64
│      %4622  = Base.muladd_float(0.5, %4618, %4621)::Float64
│      %4623  = Core.tuple(%4613, %4614, %4615, %4616, %4622)::NTuple{5, Float64}
└─────          goto #1596
1596 ─ %4625  = Base.arrayref(false, %3651, %3659, %4041)::Float64
│      %4626  = Base.copysign_float(0.0, %4625)::Float64
│      %4627  = Core.ifelse(true, %4625, %4626)::Float64
└─────          goto #1640 if not true
1597 ┄ %4629  = φ (#1596 => 1, #1639 => %4788)::Int64
│      %4630  = φ (#1596 => 1, #1639 => %4789)::Int64
│      %4631  = Base.getfield(%4623, %4629, true)::Float64
└─────          goto #1602 if not true
1598 ─ %4633  = Core.tuple(%4629, %3659, %3656, %3653, %3646)::NTuple{5, Int64}
│      %4634  = StrideArraysCore.getfield(%3549, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %4635  = Core.getfield(%4634, 5)::Int64
│      %4636  = Base.bitcast(UInt64, %4635)::UInt64
│      %4637  = Base.bitcast(Int64, %4636)::Int64
│      %4638  = Base.sle_int(1, %4629)::Bool
│      %4639  = Base.sle_int(%4629, 5)::Bool
│      %4640  = Base.and_int(%4638, %4639)::Bool
│      %4641  = Base.sle_int(1, %3659)::Bool
│      %4642  = Base.sle_int(%3659, 4)::Bool
│      %4643  = Base.and_int(%4641, %4642)::Bool
│      %4644  = Base.sle_int(1, %3656)::Bool
│      %4645  = Base.sle_int(%3656, 4)::Bool
│      %4646  = Base.and_int(%4644, %4645)::Bool
│      %4647  = Base.sle_int(1, %3653)::Bool
│      %4648  = Base.sle_int(%3653, 4)::Bool
│      %4649  = Base.and_int(%4647, %4648)::Bool
│      %4650  = Base.sub_int(%3646, 1)::Int64
│      %4651  = Base.bitcast(UInt64, %4650)::UInt64
│      %4652  = Base.bitcast(UInt64, %4637)::UInt64
│      %4653  = Base.ult_int(%4651, %4652)::Bool
│      %4654  = Base.and_int(%4653, true)::Bool
│      %4655  = Base.and_int(%4649, %4654)::Bool
│      %4656  = Base.and_int(%4646, %4655)::Bool
│      %4657  = Base.and_int(%4643, %4656)::Bool
│      %4658  = Base.and_int(%4640, %4657)::Bool
└─────          goto #1600 if not %4658
1599 ─          goto #1601
1600 ─          invoke Base.throw_boundserror(%3549::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %4633::NTuple{5, Int64})::Union{}
└─────          unreachable
1601 ─          nothing::Nothing
1602 ┄ %4664  = StrideArraysCore.getfield(%3549, :ptr)::Ptr{Float64}
│      %4665  = Base.sub_int(%4629, 1)::Int64
│      %4666  = Base.sub_int(%3659, 1)::Int64
│      %4667  = Base.sub_int(%3656, 1)::Int64
│      %4668  = Base.sub_int(%3653, 1)::Int64
│      %4669  = Base.sub_int(%3646, 1)::Int64
└─────          goto #1611 if not true
1603 ┄ %4671  = φ (#1602 => 2, #1610 => %4683)::Int64
│      %4672  = Base.sle_int(1, %4671)::Bool
└─────          goto #1605 if not %4672
1604 ─ %4674  = Base.sle_int(%4671, 5)::Bool
└─────          goto #1606
1605 ─          nothing::Nothing
1606 ┄ %4677  = φ (#1604 => %4674, #1605 => false)::Bool
└─────          goto #1608 if not %4677
1607 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %4671, true)::Static.True
│      %4680  = Base.add_int(%4671, 1)::Int64
└─────          goto #1609
1608 ─          goto #1609
1609 ┄ %4683  = φ (#1607 => %4680)::Int64
│      %4684  = φ (#1607 => false, #1608 => true)::Bool
│      %4685  = Base.not_int(%4684)::Bool
└─────          goto #1611 if not %4685
1610 ─          goto #1603
1611 ┄          goto #1612
1612 ─          goto #1613
1613 ─ %4690  = Base.mul_int(%4669, 4)::Int64
│      %4691  = Base.add_int(%4668, %4690)::Int64
│      %4692  = Base.mul_int(%4691, 4)::Int64
│      %4693  = Base.add_int(%4667, %4692)::Int64
│      %4694  = Base.mul_int(%4693, 4)::Int64
│      %4695  = Base.add_int(%4666, %4694)::Int64
│      %4696  = Base.mul_int(%4695, 5)::Int64
│      %4697  = Base.add_int(%4665, %4696)::Int64
│      %4698  = Base.mul_int(8, %4697)::Int64
│      %4699  = Core.bitcast(Core.UInt, %4664)::UInt64
│      %4700  = Base.bitcast(UInt64, %4698)::UInt64
│      %4701  = Base.add_ptr(%4699, %4700)::UInt64
│      %4702  = Core.bitcast(Ptr{Float64}, %4701)::Ptr{Float64}
└─────          goto #1614
1614 ─ %4704  = Base.pointerref(%4702, 1, 1)::Float64
└─────          goto #1615
1615 ─          goto #1616
1616 ─ %4707  = Base.muladd_float(%4627, %4631, %4704)::Float64
└─────          goto #1621 if not true
1617 ─ %4709  = Core.tuple(%4629, %3659, %3656, %3653, %3646)::NTuple{5, Int64}
│      %4710  = StrideArraysCore.getfield(%3549, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %4711  = Core.getfield(%4710, 5)::Int64
│      %4712  = Base.bitcast(UInt64, %4711)::UInt64
│      %4713  = Base.bitcast(Int64, %4712)::Int64
│      %4714  = Base.sle_int(1, %4629)::Bool
│      %4715  = Base.sle_int(%4629, 5)::Bool
│      %4716  = Base.and_int(%4714, %4715)::Bool
│      %4717  = Base.sle_int(1, %3659)::Bool
│      %4718  = Base.sle_int(%3659, 4)::Bool
│      %4719  = Base.and_int(%4717, %4718)::Bool
│      %4720  = Base.sle_int(1, %3656)::Bool
│      %4721  = Base.sle_int(%3656, 4)::Bool
│      %4722  = Base.and_int(%4720, %4721)::Bool
│      %4723  = Base.sle_int(1, %3653)::Bool
│      %4724  = Base.sle_int(%3653, 4)::Bool
│      %4725  = Base.and_int(%4723, %4724)::Bool
│      %4726  = Base.sub_int(%3646, 1)::Int64
│      %4727  = Base.bitcast(UInt64, %4726)::UInt64
│      %4728  = Base.bitcast(UInt64, %4713)::UInt64
│      %4729  = Base.ult_int(%4727, %4728)::Bool
│      %4730  = Base.and_int(%4729, true)::Bool
│      %4731  = Base.and_int(%4725, %4730)::Bool
│      %4732  = Base.and_int(%4722, %4731)::Bool
│      %4733  = Base.and_int(%4719, %4732)::Bool
│      %4734  = Base.and_int(%4716, %4733)::Bool
└─────          goto #1619 if not %4734
1618 ─          goto #1620
1619 ─          invoke Base.throw_boundserror(%3549::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %4709::NTuple{5, Int64})::Union{}
└─────          unreachable
1620 ─          nothing::Nothing
1621 ┄ %4740  = StrideArraysCore.getfield(%3549, :ptr)::Ptr{Float64}
│      %4741  = Base.sub_int(%4629, 1)::Int64
│      %4742  = Base.sub_int(%3659, 1)::Int64
│      %4743  = Base.sub_int(%3656, 1)::Int64
│      %4744  = Base.sub_int(%3653, 1)::Int64
│      %4745  = Base.sub_int(%3646, 1)::Int64
└─────          goto #1630 if not true
1622 ┄ %4747  = φ (#1621 => 2, #1629 => %4759)::Int64
│      %4748  = Base.sle_int(1, %4747)::Bool
└─────          goto #1624 if not %4748
1623 ─ %4750  = Base.sle_int(%4747, 5)::Bool
└─────          goto #1625
1624 ─          nothing::Nothing
1625 ┄ %4753  = φ (#1623 => %4750, #1624 => false)::Bool
└─────          goto #1627 if not %4753
1626 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %4747, true)::Static.True
│      %4756  = Base.add_int(%4747, 1)::Int64
└─────          goto #1628
1627 ─          goto #1628
1628 ┄ %4759  = φ (#1626 => %4756)::Int64
│      %4760  = φ (#1626 => false, #1627 => true)::Bool
│      %4761  = Base.not_int(%4760)::Bool
└─────          goto #1630 if not %4761
1629 ─          goto #1622
1630 ┄          goto #1631
1631 ─          goto #1632
1632 ─ %4766  = Base.mul_int(%4745, 4)::Int64
│      %4767  = Base.add_int(%4744, %4766)::Int64
│      %4768  = Base.mul_int(%4767, 4)::Int64
│      %4769  = Base.add_int(%4743, %4768)::Int64
│      %4770  = Base.mul_int(%4769, 4)::Int64
│      %4771  = Base.add_int(%4742, %4770)::Int64
│      %4772  = Base.mul_int(%4771, 5)::Int64
│      %4773  = Base.add_int(%4741, %4772)::Int64
│      %4774  = Base.mul_int(8, %4773)::Int64
│      %4775  = Core.bitcast(Core.UInt, %4740)::UInt64
│      %4776  = Base.bitcast(UInt64, %4774)::UInt64
│      %4777  = Base.add_ptr(%4775, %4776)::UInt64
│      %4778  = Core.bitcast(Ptr{Float64}, %4777)::Ptr{Float64}
└─────          goto #1633
1633 ─          Base.pointerset(%4778, %4707, 1, 1)::Ptr{Float64}
└─────          goto #1634
1634 ─          goto #1635
1635 ─ %4783  = (%4630 === 5)::Bool
└─────          goto #1637 if not %4783
1636 ─          goto #1638
1637 ─ %4786  = Base.add_int(%4630, 1)::Int64
└─────          goto #1638
1638 ┄ %4788  = φ (#1637 => %4786)::Int64
│      %4789  = φ (#1637 => %4786)::Int64
│      %4790  = φ (#1636 => true, #1637 => false)::Bool
│      %4791  = Base.not_int(%4790)::Bool
└─────          goto #1640 if not %4791
1639 ─          goto #1597
1640 ┄          goto #1641
1641 ─ %4795  = Base.arrayref(false, %3651, %4041, %3659)::Float64
│      %4796  = Base.copysign_float(0.0, %4795)::Float64
│      %4797  = Core.ifelse(true, %4795, %4796)::Float64
└─────          goto #1685 if not true
1642 ┄ %4799  = φ (#1641 => 1, #1684 => %4958)::Int64
│      %4800  = φ (#1641 => 1, #1684 => %4959)::Int64
│      %4801  = Base.getfield(%4623, %4799, true)::Float64
└─────          goto #1647 if not true
1643 ─ %4803  = Core.tuple(%4799, %4041, %3656, %3653, %3646)::NTuple{5, Int64}
│      %4804  = StrideArraysCore.getfield(%3549, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %4805  = Core.getfield(%4804, 5)::Int64
│      %4806  = Base.bitcast(UInt64, %4805)::UInt64
│      %4807  = Base.bitcast(Int64, %4806)::Int64
│      %4808  = Base.sle_int(1, %4799)::Bool
│      %4809  = Base.sle_int(%4799, 5)::Bool
│      %4810  = Base.and_int(%4808, %4809)::Bool
│      %4811  = Base.sle_int(1, %4041)::Bool
│      %4812  = Base.sle_int(%4041, 4)::Bool
│      %4813  = Base.and_int(%4811, %4812)::Bool
│      %4814  = Base.sle_int(1, %3656)::Bool
│      %4815  = Base.sle_int(%3656, 4)::Bool
│      %4816  = Base.and_int(%4814, %4815)::Bool
│      %4817  = Base.sle_int(1, %3653)::Bool
│      %4818  = Base.sle_int(%3653, 4)::Bool
│      %4819  = Base.and_int(%4817, %4818)::Bool
│      %4820  = Base.sub_int(%3646, 1)::Int64
│      %4821  = Base.bitcast(UInt64, %4820)::UInt64
│      %4822  = Base.bitcast(UInt64, %4807)::UInt64
│      %4823  = Base.ult_int(%4821, %4822)::Bool
│      %4824  = Base.and_int(%4823, true)::Bool
│      %4825  = Base.and_int(%4819, %4824)::Bool
│      %4826  = Base.and_int(%4816, %4825)::Bool
│      %4827  = Base.and_int(%4813, %4826)::Bool
│      %4828  = Base.and_int(%4810, %4827)::Bool
└─────          goto #1645 if not %4828
1644 ─          goto #1646
1645 ─          invoke Base.throw_boundserror(%3549::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %4803::NTuple{5, Int64})::Union{}
└─────          unreachable
1646 ─          nothing::Nothing
1647 ┄ %4834  = StrideArraysCore.getfield(%3549, :ptr)::Ptr{Float64}
│      %4835  = Base.sub_int(%4799, 1)::Int64
│      %4836  = Base.sub_int(%4041, 1)::Int64
│      %4837  = Base.sub_int(%3656, 1)::Int64
│      %4838  = Base.sub_int(%3653, 1)::Int64
│      %4839  = Base.sub_int(%3646, 1)::Int64
└─────          goto #1656 if not true
1648 ┄ %4841  = φ (#1647 => 2, #1655 => %4853)::Int64
│      %4842  = Base.sle_int(1, %4841)::Bool
└─────          goto #1650 if not %4842
1649 ─ %4844  = Base.sle_int(%4841, 5)::Bool
└─────          goto #1651
1650 ─          nothing::Nothing
1651 ┄ %4847  = φ (#1649 => %4844, #1650 => false)::Bool
└─────          goto #1653 if not %4847
1652 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %4841, true)::Static.True
│      %4850  = Base.add_int(%4841, 1)::Int64
└─────          goto #1654
1653 ─          goto #1654
1654 ┄ %4853  = φ (#1652 => %4850)::Int64
│      %4854  = φ (#1652 => false, #1653 => true)::Bool
│      %4855  = Base.not_int(%4854)::Bool
└─────          goto #1656 if not %4855
1655 ─          goto #1648
1656 ┄          goto #1657
1657 ─          goto #1658
1658 ─ %4860  = Base.mul_int(%4839, 4)::Int64
│      %4861  = Base.add_int(%4838, %4860)::Int64
│      %4862  = Base.mul_int(%4861, 4)::Int64
│      %4863  = Base.add_int(%4837, %4862)::Int64
│      %4864  = Base.mul_int(%4863, 4)::Int64
│      %4865  = Base.add_int(%4836, %4864)::Int64
│      %4866  = Base.mul_int(%4865, 5)::Int64
│      %4867  = Base.add_int(%4835, %4866)::Int64
│      %4868  = Base.mul_int(8, %4867)::Int64
│      %4869  = Core.bitcast(Core.UInt, %4834)::UInt64
│      %4870  = Base.bitcast(UInt64, %4868)::UInt64
│      %4871  = Base.add_ptr(%4869, %4870)::UInt64
│      %4872  = Core.bitcast(Ptr{Float64}, %4871)::Ptr{Float64}
└─────          goto #1659
1659 ─ %4874  = Base.pointerref(%4872, 1, 1)::Float64
└─────          goto #1660
1660 ─          goto #1661
1661 ─ %4877  = Base.muladd_float(%4797, %4801, %4874)::Float64
└─────          goto #1666 if not true
1662 ─ %4879  = Core.tuple(%4799, %4041, %3656, %3653, %3646)::NTuple{5, Int64}
│      %4880  = StrideArraysCore.getfield(%3549, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %4881  = Core.getfield(%4880, 5)::Int64
│      %4882  = Base.bitcast(UInt64, %4881)::UInt64
│      %4883  = Base.bitcast(Int64, %4882)::Int64
│      %4884  = Base.sle_int(1, %4799)::Bool
│      %4885  = Base.sle_int(%4799, 5)::Bool
│      %4886  = Base.and_int(%4884, %4885)::Bool
│      %4887  = Base.sle_int(1, %4041)::Bool
│      %4888  = Base.sle_int(%4041, 4)::Bool
│      %4889  = Base.and_int(%4887, %4888)::Bool
│      %4890  = Base.sle_int(1, %3656)::Bool
│      %4891  = Base.sle_int(%3656, 4)::Bool
│      %4892  = Base.and_int(%4890, %4891)::Bool
│      %4893  = Base.sle_int(1, %3653)::Bool
│      %4894  = Base.sle_int(%3653, 4)::Bool
│      %4895  = Base.and_int(%4893, %4894)::Bool
│      %4896  = Base.sub_int(%3646, 1)::Int64
│      %4897  = Base.bitcast(UInt64, %4896)::UInt64
│      %4898  = Base.bitcast(UInt64, %4883)::UInt64
│      %4899  = Base.ult_int(%4897, %4898)::Bool
│      %4900  = Base.and_int(%4899, true)::Bool
│      %4901  = Base.and_int(%4895, %4900)::Bool
│      %4902  = Base.and_int(%4892, %4901)::Bool
│      %4903  = Base.and_int(%4889, %4902)::Bool
│      %4904  = Base.and_int(%4886, %4903)::Bool
└─────          goto #1664 if not %4904
1663 ─          goto #1665
1664 ─          invoke Base.throw_boundserror(%3549::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %4879::NTuple{5, Int64})::Union{}
└─────          unreachable
1665 ─          nothing::Nothing
1666 ┄ %4910  = StrideArraysCore.getfield(%3549, :ptr)::Ptr{Float64}
│      %4911  = Base.sub_int(%4799, 1)::Int64
│      %4912  = Base.sub_int(%4041, 1)::Int64
│      %4913  = Base.sub_int(%3656, 1)::Int64
│      %4914  = Base.sub_int(%3653, 1)::Int64
│      %4915  = Base.sub_int(%3646, 1)::Int64
└─────          goto #1675 if not true
1667 ┄ %4917  = φ (#1666 => 2, #1674 => %4929)::Int64
│      %4918  = Base.sle_int(1, %4917)::Bool
└─────          goto #1669 if not %4918
1668 ─ %4920  = Base.sle_int(%4917, 5)::Bool
└─────          goto #1670
1669 ─          nothing::Nothing
1670 ┄ %4923  = φ (#1668 => %4920, #1669 => false)::Bool
└─────          goto #1672 if not %4923
1671 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %4917, true)::Static.True
│      %4926  = Base.add_int(%4917, 1)::Int64
└─────          goto #1673
1672 ─          goto #1673
1673 ┄ %4929  = φ (#1671 => %4926)::Int64
│      %4930  = φ (#1671 => false, #1672 => true)::Bool
│      %4931  = Base.not_int(%4930)::Bool
└─────          goto #1675 if not %4931
1674 ─          goto #1667
1675 ┄          goto #1676
1676 ─          goto #1677
1677 ─ %4936  = Base.mul_int(%4915, 4)::Int64
│      %4937  = Base.add_int(%4914, %4936)::Int64
│      %4938  = Base.mul_int(%4937, 4)::Int64
│      %4939  = Base.add_int(%4913, %4938)::Int64
│      %4940  = Base.mul_int(%4939, 4)::Int64
│      %4941  = Base.add_int(%4912, %4940)::Int64
│      %4942  = Base.mul_int(%4941, 5)::Int64
│      %4943  = Base.add_int(%4911, %4942)::Int64
│      %4944  = Base.mul_int(8, %4943)::Int64
│      %4945  = Core.bitcast(Core.UInt, %4910)::UInt64
│      %4946  = Base.bitcast(UInt64, %4944)::UInt64
│      %4947  = Base.add_ptr(%4945, %4946)::UInt64
│      %4948  = Core.bitcast(Ptr{Float64}, %4947)::Ptr{Float64}
└─────          goto #1678
1678 ─          Base.pointerset(%4948, %4877, 1, 1)::Ptr{Float64}
└─────          goto #1679
1679 ─          goto #1680
1680 ─ %4953  = (%4800 === 5)::Bool
└─────          goto #1682 if not %4953
1681 ─          goto #1683
1682 ─ %4956  = Base.add_int(%4800, 1)::Int64
└─────          goto #1683
1683 ┄ %4958  = φ (#1682 => %4956)::Int64
│      %4959  = φ (#1682 => %4956)::Int64
│      %4960  = φ (#1681 => true, #1682 => false)::Bool
│      %4961  = Base.not_int(%4960)::Bool
└─────          goto #1685 if not %4961
1684 ─          goto #1642
1685 ┄          goto #1686
1686 ─ %4965  = (%4042 === %4029)::Bool
└─────          goto #1688 if not %4965
1687 ─          goto #1689
1688 ─ %4968  = Base.add_int(%4042, 1)::Int64
└─────          goto #1689
1689 ┄ %4970  = φ (#1688 => %4968)::Int64
│      %4971  = φ (#1688 => %4968)::Int64
│      %4972  = φ (#1687 => true, #1688 => false)::Bool
│      %4973  = Base.not_int(%4972)::Bool
└─────          goto #1691 if not %4973
1690 ─          goto #1357
1691 ┄ %4976  = Base.add_int(%3656, 1)::Int64
│      %4977  = Base.sle_int(%4976, 4)::Bool
└─────          goto #1693 if not %4977
1692 ─          goto #1694
1693 ─ %4980  = Base.sub_int(%4976, 1)::Int64
└─────          goto #1694
1694 ┄ %4982  = φ (#1692 => 4, #1693 => %4980)::Int64
└─────          goto #1695
1695 ─          goto #1696
1696 ─ %4985  = Base.slt_int(%4982, %4976)::Bool
└─────          goto #1698 if not %4985
1697 ─          goto #1699
1698 ─          goto #1699
1699 ┄ %4989  = φ (#1697 => true, #1698 => false)::Bool
│      %4990  = φ (#1698 => %4976)::Int64
│      %4991  = φ (#1698 => %4976)::Int64
│      %4992  = Base.not_int(%4989)::Bool
└─────          goto #2034 if not %4992
1700 ┄ %4994  = φ (#1699 => %4990, #2033 => %5923)::Int64
│      %4995  = φ (#1699 => %4991, #2033 => %5924)::Int64
└─────          goto #1705 if not true
1701 ─ %4997  = Core.tuple(1, %3659, %4994, %3653, %3646)::NTuple{5, Int64}
│      %4998  = StrideArraysCore.getfield(%3551, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %4999  = Core.getfield(%4998, 5)::Int64
│      %5000  = Base.bitcast(UInt64, %4999)::UInt64
│      %5001  = Base.bitcast(Int64, %5000)::Int64
│      %5002  = Base.sle_int(1, %3659)::Bool
│      %5003  = Base.sle_int(%3659, 4)::Bool
│      %5004  = Base.and_int(%5002, %5003)::Bool
│      %5005  = Base.sle_int(1, %4994)::Bool
│      %5006  = Base.sle_int(%4994, 4)::Bool
│      %5007  = Base.and_int(%5005, %5006)::Bool
│      %5008  = Base.sle_int(1, %3653)::Bool
│      %5009  = Base.sle_int(%3653, 4)::Bool
│      %5010  = Base.and_int(%5008, %5009)::Bool
│      %5011  = Base.sub_int(%3646, 1)::Int64
│      %5012  = Base.bitcast(UInt64, %5011)::UInt64
│      %5013  = Base.bitcast(UInt64, %5001)::UInt64
│      %5014  = Base.ult_int(%5012, %5013)::Bool
│      %5015  = Base.and_int(%5014, true)::Bool
│      %5016  = Base.and_int(%5010, %5015)::Bool
│      %5017  = Base.and_int(%5007, %5016)::Bool
│      %5018  = Base.and_int(%5004, %5017)::Bool
│      %5019  = Base.and_int(true, %5018)::Bool
└─────          goto #1703 if not %5019
1702 ─          goto #1704
1703 ─          invoke Base.throw_boundserror(%3551::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %4997::NTuple{5, Int64})::Union{}
└─────          unreachable
1704 ─          nothing::Nothing
1705 ┄ %5025  = StrideArraysCore.getfield(%3551, :ptr)::Ptr{Float64}
│      %5026  = Base.sub_int(%3659, 1)::Int64
│      %5027  = Base.sub_int(%4994, 1)::Int64
│      %5028  = Base.sub_int(%3653, 1)::Int64
│      %5029  = Base.sub_int(%3646, 1)::Int64
└─────          goto #1714 if not true
1706 ┄ %5031  = φ (#1705 => 2, #1713 => %5043)::Int64
│      %5032  = Base.sle_int(1, %5031)::Bool
└─────          goto #1708 if not %5032
1707 ─ %5034  = Base.sle_int(%5031, 5)::Bool
└─────          goto #1709
1708 ─          nothing::Nothing
1709 ┄ %5037  = φ (#1707 => %5034, #1708 => false)::Bool
└─────          goto #1711 if not %5037
1710 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %5031, true)::Static.True
│      %5040  = Base.add_int(%5031, 1)::Int64
└─────          goto #1712
1711 ─          goto #1712
1712 ┄ %5043  = φ (#1710 => %5040)::Int64
│      %5044  = φ (#1710 => false, #1711 => true)::Bool
│      %5045  = Base.not_int(%5044)::Bool
└─────          goto #1714 if not %5045
1713 ─          goto #1706
1714 ┄          goto #1715
1715 ─          goto #1716
1716 ─ %5050  = Base.mul_int(%5029, 4)::Int64
│      %5051  = Base.add_int(%5028, %5050)::Int64
│      %5052  = Base.mul_int(%5051, 4)::Int64
│      %5053  = Base.add_int(%5027, %5052)::Int64
│      %5054  = Base.mul_int(%5053, 4)::Int64
│      %5055  = Base.add_int(%5026, %5054)::Int64
│      %5056  = Base.mul_int(%5055, 5)::Int64
│      %5057  = Base.add_int(0, %5056)::Int64
│      %5058  = Base.mul_int(8, %5057)::Int64
│      %5059  = Core.bitcast(Core.UInt, %5025)::UInt64
│      %5060  = Base.bitcast(UInt64, %5058)::UInt64
│      %5061  = Base.add_ptr(%5059, %5060)::UInt64
│      %5062  = Core.bitcast(Ptr{Float64}, %5061)::Ptr{Float64}
└─────          goto #1717
1717 ─ %5064  = Base.pointerref(%5062, 1, 1)::Float64
└─────          goto #1718
1718 ─          goto #1719
1719 ─          goto #1720
1720 ─          goto #1725 if not true
1721 ─ %5069  = Core.tuple(2, %3659, %4994, %3653, %3646)::NTuple{5, Int64}
│      %5070  = StrideArraysCore.getfield(%3551, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %5071  = Core.getfield(%5070, 5)::Int64
│      %5072  = Base.bitcast(UInt64, %5071)::UInt64
│      %5073  = Base.bitcast(Int64, %5072)::Int64
│      %5074  = Base.sle_int(1, %3659)::Bool
│      %5075  = Base.sle_int(%3659, 4)::Bool
│      %5076  = Base.and_int(%5074, %5075)::Bool
│      %5077  = Base.sle_int(1, %4994)::Bool
│      %5078  = Base.sle_int(%4994, 4)::Bool
│      %5079  = Base.and_int(%5077, %5078)::Bool
│      %5080  = Base.sle_int(1, %3653)::Bool
│      %5081  = Base.sle_int(%3653, 4)::Bool
│      %5082  = Base.and_int(%5080, %5081)::Bool
│      %5083  = Base.sub_int(%3646, 1)::Int64
│      %5084  = Base.bitcast(UInt64, %5083)::UInt64
│      %5085  = Base.bitcast(UInt64, %5073)::UInt64
│      %5086  = Base.ult_int(%5084, %5085)::Bool
│      %5087  = Base.and_int(%5086, true)::Bool
│      %5088  = Base.and_int(%5082, %5087)::Bool
│      %5089  = Base.and_int(%5079, %5088)::Bool
│      %5090  = Base.and_int(%5076, %5089)::Bool
│      %5091  = Base.and_int(true, %5090)::Bool
└─────          goto #1723 if not %5091
1722 ─          goto #1724
1723 ─          invoke Base.throw_boundserror(%3551::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %5069::NTuple{5, Int64})::Union{}
└─────          unreachable
1724 ─          nothing::Nothing
1725 ┄ %5097  = StrideArraysCore.getfield(%3551, :ptr)::Ptr{Float64}
│      %5098  = Base.sub_int(%3659, 1)::Int64
│      %5099  = Base.sub_int(%4994, 1)::Int64
│      %5100  = Base.sub_int(%3653, 1)::Int64
│      %5101  = Base.sub_int(%3646, 1)::Int64
└─────          goto #1734 if not true
1726 ┄ %5103  = φ (#1725 => 2, #1733 => %5115)::Int64
│      %5104  = Base.sle_int(1, %5103)::Bool
└─────          goto #1728 if not %5104
1727 ─ %5106  = Base.sle_int(%5103, 5)::Bool
└─────          goto #1729
1728 ─          nothing::Nothing
1729 ┄ %5109  = φ (#1727 => %5106, #1728 => false)::Bool
└─────          goto #1731 if not %5109
1730 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %5103, true)::Static.True
│      %5112  = Base.add_int(%5103, 1)::Int64
└─────          goto #1732
1731 ─          goto #1732
1732 ┄ %5115  = φ (#1730 => %5112)::Int64
│      %5116  = φ (#1730 => false, #1731 => true)::Bool
│      %5117  = Base.not_int(%5116)::Bool
└─────          goto #1734 if not %5117
1733 ─          goto #1726
1734 ┄          goto #1735
1735 ─          goto #1736
1736 ─ %5122  = Base.mul_int(%5101, 4)::Int64
│      %5123  = Base.add_int(%5100, %5122)::Int64
│      %5124  = Base.mul_int(%5123, 4)::Int64
│      %5125  = Base.add_int(%5099, %5124)::Int64
│      %5126  = Base.mul_int(%5125, 4)::Int64
│      %5127  = Base.add_int(%5098, %5126)::Int64
│      %5128  = Base.mul_int(%5127, 5)::Int64
│      %5129  = Base.add_int(1, %5128)::Int64
│      %5130  = Base.mul_int(8, %5129)::Int64
│      %5131  = Core.bitcast(Core.UInt, %5097)::UInt64
│      %5132  = Base.bitcast(UInt64, %5130)::UInt64
│      %5133  = Base.add_ptr(%5131, %5132)::UInt64
│      %5134  = Core.bitcast(Ptr{Float64}, %5133)::Ptr{Float64}
└─────          goto #1737
1737 ─ %5136  = Base.pointerref(%5134, 1, 1)::Float64
└─────          goto #1738
1738 ─          goto #1739
1739 ─          goto #1740
1740 ─          goto #1745 if not true
1741 ─ %5141  = Core.tuple(3, %3659, %4994, %3653, %3646)::NTuple{5, Int64}
│      %5142  = StrideArraysCore.getfield(%3551, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %5143  = Core.getfield(%5142, 5)::Int64
│      %5144  = Base.bitcast(UInt64, %5143)::UInt64
│      %5145  = Base.bitcast(Int64, %5144)::Int64
│      %5146  = Base.sle_int(1, %3659)::Bool
│      %5147  = Base.sle_int(%3659, 4)::Bool
│      %5148  = Base.and_int(%5146, %5147)::Bool
│      %5149  = Base.sle_int(1, %4994)::Bool
│      %5150  = Base.sle_int(%4994, 4)::Bool
│      %5151  = Base.and_int(%5149, %5150)::Bool
│      %5152  = Base.sle_int(1, %3653)::Bool
│      %5153  = Base.sle_int(%3653, 4)::Bool
│      %5154  = Base.and_int(%5152, %5153)::Bool
│      %5155  = Base.sub_int(%3646, 1)::Int64
│      %5156  = Base.bitcast(UInt64, %5155)::UInt64
│      %5157  = Base.bitcast(UInt64, %5145)::UInt64
│      %5158  = Base.ult_int(%5156, %5157)::Bool
│      %5159  = Base.and_int(%5158, true)::Bool
│      %5160  = Base.and_int(%5154, %5159)::Bool
│      %5161  = Base.and_int(%5151, %5160)::Bool
│      %5162  = Base.and_int(%5148, %5161)::Bool
│      %5163  = Base.and_int(true, %5162)::Bool
└─────          goto #1743 if not %5163
1742 ─          goto #1744
1743 ─          invoke Base.throw_boundserror(%3551::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %5141::NTuple{5, Int64})::Union{}
└─────          unreachable
1744 ─          nothing::Nothing
1745 ┄ %5169  = StrideArraysCore.getfield(%3551, :ptr)::Ptr{Float64}
│      %5170  = Base.sub_int(%3659, 1)::Int64
│      %5171  = Base.sub_int(%4994, 1)::Int64
│      %5172  = Base.sub_int(%3653, 1)::Int64
│      %5173  = Base.sub_int(%3646, 1)::Int64
└─────          goto #1754 if not true
1746 ┄ %5175  = φ (#1745 => 2, #1753 => %5187)::Int64
│      %5176  = Base.sle_int(1, %5175)::Bool
└─────          goto #1748 if not %5176
1747 ─ %5178  = Base.sle_int(%5175, 5)::Bool
└─────          goto #1749
1748 ─          nothing::Nothing
1749 ┄ %5181  = φ (#1747 => %5178, #1748 => false)::Bool
└─────          goto #1751 if not %5181
1750 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %5175, true)::Static.True
│      %5184  = Base.add_int(%5175, 1)::Int64
└─────          goto #1752
1751 ─          goto #1752
1752 ┄ %5187  = φ (#1750 => %5184)::Int64
│      %5188  = φ (#1750 => false, #1751 => true)::Bool
│      %5189  = Base.not_int(%5188)::Bool
└─────          goto #1754 if not %5189
1753 ─          goto #1746
1754 ┄          goto #1755
1755 ─          goto #1756
1756 ─ %5194  = Base.mul_int(%5173, 4)::Int64
│      %5195  = Base.add_int(%5172, %5194)::Int64
│      %5196  = Base.mul_int(%5195, 4)::Int64
│      %5197  = Base.add_int(%5171, %5196)::Int64
│      %5198  = Base.mul_int(%5197, 4)::Int64
│      %5199  = Base.add_int(%5170, %5198)::Int64
│      %5200  = Base.mul_int(%5199, 5)::Int64
│      %5201  = Base.add_int(2, %5200)::Int64
│      %5202  = Base.mul_int(8, %5201)::Int64
│      %5203  = Core.bitcast(Core.UInt, %5169)::UInt64
│      %5204  = Base.bitcast(UInt64, %5202)::UInt64
│      %5205  = Base.add_ptr(%5203, %5204)::UInt64
│      %5206  = Core.bitcast(Ptr{Float64}, %5205)::Ptr{Float64}
└─────          goto #1757
1757 ─ %5208  = Base.pointerref(%5206, 1, 1)::Float64
└─────          goto #1758
1758 ─          goto #1759
1759 ─          goto #1760
1760 ─          goto #1765 if not true
1761 ─ %5213  = Core.tuple(4, %3659, %4994, %3653, %3646)::NTuple{5, Int64}
│      %5214  = StrideArraysCore.getfield(%3551, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %5215  = Core.getfield(%5214, 5)::Int64
│      %5216  = Base.bitcast(UInt64, %5215)::UInt64
│      %5217  = Base.bitcast(Int64, %5216)::Int64
│      %5218  = Base.sle_int(1, %3659)::Bool
│      %5219  = Base.sle_int(%3659, 4)::Bool
│      %5220  = Base.and_int(%5218, %5219)::Bool
│      %5221  = Base.sle_int(1, %4994)::Bool
│      %5222  = Base.sle_int(%4994, 4)::Bool
│      %5223  = Base.and_int(%5221, %5222)::Bool
│      %5224  = Base.sle_int(1, %3653)::Bool
│      %5225  = Base.sle_int(%3653, 4)::Bool
│      %5226  = Base.and_int(%5224, %5225)::Bool
│      %5227  = Base.sub_int(%3646, 1)::Int64
│      %5228  = Base.bitcast(UInt64, %5227)::UInt64
│      %5229  = Base.bitcast(UInt64, %5217)::UInt64
│      %5230  = Base.ult_int(%5228, %5229)::Bool
│      %5231  = Base.and_int(%5230, true)::Bool
│      %5232  = Base.and_int(%5226, %5231)::Bool
│      %5233  = Base.and_int(%5223, %5232)::Bool
│      %5234  = Base.and_int(%5220, %5233)::Bool
│      %5235  = Base.and_int(true, %5234)::Bool
└─────          goto #1763 if not %5235
1762 ─          goto #1764
1763 ─          invoke Base.throw_boundserror(%3551::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %5213::NTuple{5, Int64})::Union{}
└─────          unreachable
1764 ─          nothing::Nothing
1765 ┄ %5241  = StrideArraysCore.getfield(%3551, :ptr)::Ptr{Float64}
│      %5242  = Base.sub_int(%3659, 1)::Int64
│      %5243  = Base.sub_int(%4994, 1)::Int64
│      %5244  = Base.sub_int(%3653, 1)::Int64
│      %5245  = Base.sub_int(%3646, 1)::Int64
└─────          goto #1774 if not true
1766 ┄ %5247  = φ (#1765 => 2, #1773 => %5259)::Int64
│      %5248  = Base.sle_int(1, %5247)::Bool
└─────          goto #1768 if not %5248
1767 ─ %5250  = Base.sle_int(%5247, 5)::Bool
└─────          goto #1769
1768 ─          nothing::Nothing
1769 ┄ %5253  = φ (#1767 => %5250, #1768 => false)::Bool
└─────          goto #1771 if not %5253
1770 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %5247, true)::Static.True
│      %5256  = Base.add_int(%5247, 1)::Int64
└─────          goto #1772
1771 ─          goto #1772
1772 ┄ %5259  = φ (#1770 => %5256)::Int64
│      %5260  = φ (#1770 => false, #1771 => true)::Bool
│      %5261  = Base.not_int(%5260)::Bool
└─────          goto #1774 if not %5261
1773 ─          goto #1766
1774 ┄          goto #1775
1775 ─          goto #1776
1776 ─ %5266  = Base.mul_int(%5245, 4)::Int64
│      %5267  = Base.add_int(%5244, %5266)::Int64
│      %5268  = Base.mul_int(%5267, 4)::Int64
│      %5269  = Base.add_int(%5243, %5268)::Int64
│      %5270  = Base.mul_int(%5269, 4)::Int64
│      %5271  = Base.add_int(%5242, %5270)::Int64
│      %5272  = Base.mul_int(%5271, 5)::Int64
│      %5273  = Base.add_int(3, %5272)::Int64
│      %5274  = Base.mul_int(8, %5273)::Int64
│      %5275  = Core.bitcast(Core.UInt, %5241)::UInt64
│      %5276  = Base.bitcast(UInt64, %5274)::UInt64
│      %5277  = Base.add_ptr(%5275, %5276)::UInt64
│      %5278  = Core.bitcast(Ptr{Float64}, %5277)::Ptr{Float64}
└─────          goto #1777
1777 ─ %5280  = Base.pointerref(%5278, 1, 1)::Float64
└─────          goto #1778
1778 ─          goto #1779
1779 ─          goto #1780
1780 ─          goto #1785 if not true
1781 ─ %5285  = Core.tuple(5, %3659, %4994, %3653, %3646)::NTuple{5, Int64}
│      %5286  = StrideArraysCore.getfield(%3551, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %5287  = Core.getfield(%5286, 5)::Int64
│      %5288  = Base.bitcast(UInt64, %5287)::UInt64
│      %5289  = Base.bitcast(Int64, %5288)::Int64
│      %5290  = Base.sle_int(1, %3659)::Bool
│      %5291  = Base.sle_int(%3659, 4)::Bool
│      %5292  = Base.and_int(%5290, %5291)::Bool
│      %5293  = Base.sle_int(1, %4994)::Bool
│      %5294  = Base.sle_int(%4994, 4)::Bool
│      %5295  = Base.and_int(%5293, %5294)::Bool
│      %5296  = Base.sle_int(1, %3653)::Bool
│      %5297  = Base.sle_int(%3653, 4)::Bool
│      %5298  = Base.and_int(%5296, %5297)::Bool
│      %5299  = Base.sub_int(%3646, 1)::Int64
│      %5300  = Base.bitcast(UInt64, %5299)::UInt64
│      %5301  = Base.bitcast(UInt64, %5289)::UInt64
│      %5302  = Base.ult_int(%5300, %5301)::Bool
│      %5303  = Base.and_int(%5302, true)::Bool
│      %5304  = Base.and_int(%5298, %5303)::Bool
│      %5305  = Base.and_int(%5295, %5304)::Bool
│      %5306  = Base.and_int(%5292, %5305)::Bool
│      %5307  = Base.and_int(true, %5306)::Bool
└─────          goto #1783 if not %5307
1782 ─          goto #1784
1783 ─          invoke Base.throw_boundserror(%3551::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %5285::NTuple{5, Int64})::Union{}
└─────          unreachable
1784 ─          nothing::Nothing
1785 ┄ %5313  = StrideArraysCore.getfield(%3551, :ptr)::Ptr{Float64}
│      %5314  = Base.sub_int(%3659, 1)::Int64
│      %5315  = Base.sub_int(%4994, 1)::Int64
│      %5316  = Base.sub_int(%3653, 1)::Int64
│      %5317  = Base.sub_int(%3646, 1)::Int64
└─────          goto #1794 if not true
1786 ┄ %5319  = φ (#1785 => 2, #1793 => %5331)::Int64
│      %5320  = Base.sle_int(1, %5319)::Bool
└─────          goto #1788 if not %5320
1787 ─ %5322  = Base.sle_int(%5319, 5)::Bool
└─────          goto #1789
1788 ─          nothing::Nothing
1789 ┄ %5325  = φ (#1787 => %5322, #1788 => false)::Bool
└─────          goto #1791 if not %5325
1790 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %5319, true)::Static.True
│      %5328  = Base.add_int(%5319, 1)::Int64
└─────          goto #1792
1791 ─          goto #1792
1792 ┄ %5331  = φ (#1790 => %5328)::Int64
│      %5332  = φ (#1790 => false, #1791 => true)::Bool
│      %5333  = Base.not_int(%5332)::Bool
└─────          goto #1794 if not %5333
1793 ─          goto #1786
1794 ┄          goto #1795
1795 ─          goto #1796
1796 ─ %5338  = Base.mul_int(%5317, 4)::Int64
│      %5339  = Base.add_int(%5316, %5338)::Int64
│      %5340  = Base.mul_int(%5339, 4)::Int64
│      %5341  = Base.add_int(%5315, %5340)::Int64
│      %5342  = Base.mul_int(%5341, 4)::Int64
│      %5343  = Base.add_int(%5314, %5342)::Int64
│      %5344  = Base.mul_int(%5343, 5)::Int64
│      %5345  = Base.add_int(4, %5344)::Int64
│      %5346  = Base.mul_int(8, %5345)::Int64
│      %5347  = Core.bitcast(Core.UInt, %5313)::UInt64
│      %5348  = Base.bitcast(UInt64, %5346)::UInt64
│      %5349  = Base.add_ptr(%5347, %5348)::UInt64
│      %5350  = Core.bitcast(Ptr{Float64}, %5349)::Ptr{Float64}
└─────          goto #1797
1797 ─ %5352  = Base.pointerref(%5350, 1, 1)::Float64
└─────          goto #1798
1798 ─          goto #1799
1799 ─          goto #1800
1800 ─          goto #1801
1801 ─          goto #1802
1802 ─          goto #1804
1803 ─          nothing::Nothing
1804 ┄          goto #1806
1805 ─          nothing::Nothing
1806 ┄          goto #1807
1807 ─          goto #1809
1808 ─          nothing::Nothing
1809 ┄          goto #1810
1810 ─          goto #1812
1811 ─          nothing::Nothing
1812 ┄          goto #1814
1813 ─          nothing::Nothing
1814 ┄          goto #1815
1815 ─          goto #1817
1816 ─          nothing::Nothing
1817 ┄          goto #1818
1818 ─          goto #1820
1819 ─          nothing::Nothing
1820 ┄          goto #1822
1821 ─          nothing::Nothing
1822 ┄          goto #1823
1823 ─          goto #1825
1824 ─          nothing::Nothing
1825 ┄          goto #1826
1826 ─          goto #1828
1827 ─          nothing::Nothing
1828 ┄          goto #1830
1829 ─          nothing::Nothing
1830 ┄          goto #1831
1831 ─          goto #1833
1832 ─          nothing::Nothing
1833 ┄          goto #1834
1834 ─ %5390  = Base.div_float(%3801, %3729)::Float64
│      %5391  = Base.div_float(%3873, %3729)::Float64
│      %5392  = Base.div_float(%3945, %3729)::Float64
│      %5393  = Base.getfield(equations, :gamma)::Float64
│      %5394  = Base.sub_float(%5393, 1.0)::Float64
│      %5395  = Base.mul_float(%3801, %5390)::Float64
│      %5396  = Base.muladd_float(%3873, %5391, %5395)::Float64
│      %5397  = Base.muladd_float(%3945, %5392, %5396)::Float64
│      %5398  = Base.muladd_float(-0.5, %5397, %4017)::Float64
│      %5399  = Base.mul_float(%5394, %5398)::Float64
└─────          goto #1835
1835 ─          goto #1837
1836 ─          nothing::Nothing
1837 ┄          goto #1839
1838 ─          nothing::Nothing
1839 ┄          goto #1840
1840 ─          goto #1842
1841 ─          nothing::Nothing
1842 ┄          goto #1843
1843 ─          goto #1845
1844 ─          nothing::Nothing
1845 ┄          goto #1847
1846 ─          nothing::Nothing
1847 ┄          goto #1848
1848 ─          goto #1850
1849 ─          nothing::Nothing
1850 ┄          goto #1851
1851 ─          goto #1853
1852 ─          nothing::Nothing
1853 ┄          goto #1855
1854 ─          nothing::Nothing
1855 ┄          goto #1856
1856 ─          goto #1858
1857 ─          nothing::Nothing
1858 ┄          goto #1859
1859 ─          goto #1861
1860 ─          nothing::Nothing
1861 ┄          goto #1863
1862 ─          nothing::Nothing
1863 ┄          goto #1864
1864 ─          goto #1866
1865 ─          nothing::Nothing
1866 ┄          goto #1867
1867 ─          goto #1869
1868 ─          nothing::Nothing
1869 ┄          goto #1871
1870 ─          nothing::Nothing
1871 ┄          goto #1872
1872 ─          goto #1874
1873 ─          nothing::Nothing
1874 ┄          goto #1875
1875 ─          goto #1877
1876 ─          nothing::Nothing
1877 ┄          goto #1879
1878 ─          nothing::Nothing
1879 ┄          goto #1880
1880 ─          goto #1882
1881 ─          nothing::Nothing
1882 ┄          goto #1883
1883 ─          goto #1885
1884 ─          nothing::Nothing
1885 ┄          goto #1887
1886 ─          nothing::Nothing
1887 ┄          goto #1888
1888 ─          goto #1890
1889 ─          nothing::Nothing
1890 ┄          goto #1891
1891 ─          goto #1893
1892 ─          nothing::Nothing
1893 ┄          goto #1895
1894 ─          nothing::Nothing
1895 ┄          goto #1896
1896 ─          goto #1898
1897 ─          nothing::Nothing
1898 ┄          goto #1899
1899 ─ %5465  = Base.div_float(%5136, %5064)::Float64
│      %5466  = Base.div_float(%5208, %5064)::Float64
│      %5467  = Base.div_float(%5280, %5064)::Float64
│      %5468  = Base.getfield(equations, :gamma)::Float64
│      %5469  = Base.sub_float(%5468, 1.0)::Float64
│      %5470  = Base.mul_float(%5136, %5465)::Float64
│      %5471  = Base.muladd_float(%5208, %5466, %5470)::Float64
│      %5472  = Base.muladd_float(%5280, %5467, %5471)::Float64
│      %5473  = Base.muladd_float(-0.5, %5472, %5352)::Float64
│      %5474  = Base.mul_float(%5469, %5473)::Float64
└─────          goto #1900
1900 ─          goto #1902
1901 ─          nothing::Nothing
1902 ┄          goto #1904
1903 ─          nothing::Nothing
1904 ┄          goto #1905
1905 ─          goto #1907
1906 ─          nothing::Nothing
1907 ┄          goto #1908
1908 ─          goto #1910
1909 ─          nothing::Nothing
1910 ┄          goto #1912
1911 ─          nothing::Nothing
1912 ┄          goto #1913
1913 ─          goto #1915
1914 ─          nothing::Nothing
1915 ┄          goto #1916
1916 ─          goto #1918
1917 ─          nothing::Nothing
1918 ┄          goto #1920
1919 ─          nothing::Nothing
1920 ┄          goto #1921
1921 ─          goto #1923
1922 ─          nothing::Nothing
1923 ┄          goto #1924
1924 ─          goto #1926
1925 ─          nothing::Nothing
1926 ┄          goto #1928
1927 ─          nothing::Nothing
1928 ┄          goto #1929
1929 ─          goto #1931
1930 ─          nothing::Nothing
1931 ┄          goto #1932
1932 ─ %5508  = Base.muladd_float(-2.0, %5064, %3729)::Float64
│      %5509  = Base.mul_float(%3729, %5508)::Float64
│      %5510  = Base.muladd_float(%5064, %5064, %5509)::Float64
│      %5511  = Base.muladd_float(2.0, %5064, %3729)::Float64
│      %5512  = Base.mul_float(%3729, %5511)::Float64
│      %5513  = Base.muladd_float(%5064, %5064, %5512)::Float64
│      %5514  = Base.div_float(%5510, %5513)::Float64
│      %5515  = Base.lt_float(%5514, 0.0001)::Bool
└─────          goto #1934 if not %5515
1933 ─ %5517  = Base.add_float(%3729, %5064)::Float64
│      %5518  = Base.muladd_float(%5514, 0.2857142857142857, 0.4)::Float64
│      %5519  = Base.muladd_float(%5514, %5518, 0.6666666666666666)::Float64
│      %5520  = Base.muladd_float(%5514, %5519, 2.0)::Float64
│      %5521  = Base.div_float(%5517, %5520)::Float64
└─────          goto #1935
1934 ─ %5523  = Base.sub_float(%5064, %3729)::Float64
│      %5524  = Base.div_float(%5064, %3729)::Float64
│      %5525  = $(Expr(:foreigncall, "llvm.log.f64", Float64, svec(Float64), 0, :(:llvmcall), :(%5524), :(%5524)))::Float64
│      %5526  = Base.div_float(%5523, %5525)::Float64
└─────          goto #1935
1935 ┄ %5528  = φ (#1933 => %5521, #1934 => %5526)::Float64
│      %5529  = Base.mul_float(%3729, %5474)::Float64
│      %5530  = Base.mul_float(%5064, %5399)::Float64
│      %5531  = Base.muladd_float(-2.0, %5530, %5529)::Float64
│      %5532  = Base.mul_float(%5529, %5531)::Float64
│      %5533  = Base.muladd_float(%5530, %5530, %5532)::Float64
│      %5534  = Base.muladd_float(2.0, %5530, %5529)::Float64
│      %5535  = Base.mul_float(%5529, %5534)::Float64
│      %5536  = Base.muladd_float(%5530, %5530, %5535)::Float64
│      %5537  = Base.div_float(%5533, %5536)::Float64
│      %5538  = Base.lt_float(%5537, 0.0001)::Bool
└─────          goto #1937 if not %5538
1936 ─ %5540  = Base.muladd_float(%5537, 0.2857142857142857, 0.4)::Float64
│      %5541  = Base.muladd_float(%5537, %5540, 0.6666666666666666)::Float64
│      %5542  = Base.muladd_float(%5537, %5541, 2.0)::Float64
│      %5543  = Base.add_float(%5529, %5530)::Float64
│      %5544  = Base.div_float(%5542, %5543)::Float64
└─────          goto #1938
1937 ─ %5546  = Base.div_float(%5530, %5529)::Float64
│      %5547  = $(Expr(:foreigncall, "llvm.log.f64", Float64, svec(Float64), 0, :(:llvmcall), :(%5546), :(%5546)))::Float64
│      %5548  = Base.sub_float(%5530, %5529)::Float64
│      %5549  = Base.div_float(%5547, %5548)::Float64
└─────          goto #1938
1938 ┄ %5551  = φ (#1936 => %5544, #1937 => %5549)::Float64
│      %5552  = Base.mul_float(%5399, %5474)::Float64
│      %5553  = Base.mul_float(%5552, %5551)::Float64
│      %5554  = Base.add_float(%5390, %5465)::Float64
│      %5555  = Base.mul_float(0.5, %5554)::Float64
│      %5556  = Base.add_float(%5391, %5466)::Float64
│      %5557  = Base.mul_float(0.5, %5556)::Float64
│      %5558  = Base.add_float(%5392, %5467)::Float64
│      %5559  = Base.mul_float(0.5, %5558)::Float64
│      %5560  = Base.add_float(%5399, %5474)::Float64
│      %5561  = Base.mul_float(0.5, %5560)::Float64
│      %5562  = Base.mul_float(%5390, %5465)::Float64
│      %5563  = Base.muladd_float(%5391, %5466, %5562)::Float64
│      %5564  = Base.muladd_float(%5392, %5467, %5563)::Float64
│      %5565  = Base.mul_float(0.5, %5564)::Float64
│      %5566  = Base.mul_float(%5528, %5557)::Float64
│      %5567  = Base.mul_float(%5566, %5555)::Float64
│      %5568  = Base.muladd_float(%5566, %5557, %5561)::Float64
│      %5569  = Base.mul_float(%5566, %5559)::Float64
│      %5570  = Base.mul_float(%5399, %5466)::Float64
│      %5571  = Base.muladd_float(%5474, %5391, %5570)::Float64
│      %5572  = Base.getfield(equations, :inv_gamma_minus_one)::Float64
│      %5573  = Base.muladd_float(%5553, %5572, %5565)::Float64
│      %5574  = Base.mul_float(%5566, %5573)::Float64
│      %5575  = Base.muladd_float(0.5, %5571, %5574)::Float64
│      %5576  = Core.tuple(%5566, %5567, %5568, %5569, %5575)::NTuple{5, Float64}
└─────          goto #1939
1939 ─ %5578  = Base.arrayref(false, %3651, %3656, %4994)::Float64
│      %5579  = Base.copysign_float(0.0, %5578)::Float64
│      %5580  = Core.ifelse(true, %5578, %5579)::Float64
└─────          goto #1983 if not true
1940 ┄ %5582  = φ (#1939 => 1, #1982 => %5741)::Int64
│      %5583  = φ (#1939 => 1, #1982 => %5742)::Int64
│      %5584  = Base.getfield(%5576, %5582, true)::Float64
└─────          goto #1945 if not true
1941 ─ %5586  = Core.tuple(%5582, %3659, %3656, %3653, %3646)::NTuple{5, Int64}
│      %5587  = StrideArraysCore.getfield(%3549, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %5588  = Core.getfield(%5587, 5)::Int64
│      %5589  = Base.bitcast(UInt64, %5588)::UInt64
│      %5590  = Base.bitcast(Int64, %5589)::Int64
│      %5591  = Base.sle_int(1, %5582)::Bool
│      %5592  = Base.sle_int(%5582, 5)::Bool
│      %5593  = Base.and_int(%5591, %5592)::Bool
│      %5594  = Base.sle_int(1, %3659)::Bool
│      %5595  = Base.sle_int(%3659, 4)::Bool
│      %5596  = Base.and_int(%5594, %5595)::Bool
│      %5597  = Base.sle_int(1, %3656)::Bool
│      %5598  = Base.sle_int(%3656, 4)::Bool
│      %5599  = Base.and_int(%5597, %5598)::Bool
│      %5600  = Base.sle_int(1, %3653)::Bool
│      %5601  = Base.sle_int(%3653, 4)::Bool
│      %5602  = Base.and_int(%5600, %5601)::Bool
│      %5603  = Base.sub_int(%3646, 1)::Int64
│      %5604  = Base.bitcast(UInt64, %5603)::UInt64
│      %5605  = Base.bitcast(UInt64, %5590)::UInt64
│      %5606  = Base.ult_int(%5604, %5605)::Bool
│      %5607  = Base.and_int(%5606, true)::Bool
│      %5608  = Base.and_int(%5602, %5607)::Bool
│      %5609  = Base.and_int(%5599, %5608)::Bool
│      %5610  = Base.and_int(%5596, %5609)::Bool
│      %5611  = Base.and_int(%5593, %5610)::Bool
└─────          goto #1943 if not %5611
1942 ─          goto #1944
1943 ─          invoke Base.throw_boundserror(%3549::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %5586::NTuple{5, Int64})::Union{}
└─────          unreachable
1944 ─          nothing::Nothing
1945 ┄ %5617  = StrideArraysCore.getfield(%3549, :ptr)::Ptr{Float64}
│      %5618  = Base.sub_int(%5582, 1)::Int64
│      %5619  = Base.sub_int(%3659, 1)::Int64
│      %5620  = Base.sub_int(%3656, 1)::Int64
│      %5621  = Base.sub_int(%3653, 1)::Int64
│      %5622  = Base.sub_int(%3646, 1)::Int64
└─────          goto #1954 if not true
1946 ┄ %5624  = φ (#1945 => 2, #1953 => %5636)::Int64
│      %5625  = Base.sle_int(1, %5624)::Bool
└─────          goto #1948 if not %5625
1947 ─ %5627  = Base.sle_int(%5624, 5)::Bool
└─────          goto #1949
1948 ─          nothing::Nothing
1949 ┄ %5630  = φ (#1947 => %5627, #1948 => false)::Bool
└─────          goto #1951 if not %5630
1950 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %5624, true)::Static.True
│      %5633  = Base.add_int(%5624, 1)::Int64
└─────          goto #1952
1951 ─          goto #1952
1952 ┄ %5636  = φ (#1950 => %5633)::Int64
│      %5637  = φ (#1950 => false, #1951 => true)::Bool
│      %5638  = Base.not_int(%5637)::Bool
└─────          goto #1954 if not %5638
1953 ─          goto #1946
1954 ┄          goto #1955
1955 ─          goto #1956
1956 ─ %5643  = Base.mul_int(%5622, 4)::Int64
│      %5644  = Base.add_int(%5621, %5643)::Int64
│      %5645  = Base.mul_int(%5644, 4)::Int64
│      %5646  = Base.add_int(%5620, %5645)::Int64
│      %5647  = Base.mul_int(%5646, 4)::Int64
│      %5648  = Base.add_int(%5619, %5647)::Int64
│      %5649  = Base.mul_int(%5648, 5)::Int64
│      %5650  = Base.add_int(%5618, %5649)::Int64
│      %5651  = Base.mul_int(8, %5650)::Int64
│      %5652  = Core.bitcast(Core.UInt, %5617)::UInt64
│      %5653  = Base.bitcast(UInt64, %5651)::UInt64
│      %5654  = Base.add_ptr(%5652, %5653)::UInt64
│      %5655  = Core.bitcast(Ptr{Float64}, %5654)::Ptr{Float64}
└─────          goto #1957
1957 ─ %5657  = Base.pointerref(%5655, 1, 1)::Float64
└─────          goto #1958
1958 ─          goto #1959
1959 ─ %5660  = Base.muladd_float(%5580, %5584, %5657)::Float64
└─────          goto #1964 if not true
1960 ─ %5662  = Core.tuple(%5582, %3659, %3656, %3653, %3646)::NTuple{5, Int64}
│      %5663  = StrideArraysCore.getfield(%3549, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %5664  = Core.getfield(%5663, 5)::Int64
│      %5665  = Base.bitcast(UInt64, %5664)::UInt64
│      %5666  = Base.bitcast(Int64, %5665)::Int64
│      %5667  = Base.sle_int(1, %5582)::Bool
│      %5668  = Base.sle_int(%5582, 5)::Bool
│      %5669  = Base.and_int(%5667, %5668)::Bool
│      %5670  = Base.sle_int(1, %3659)::Bool
│      %5671  = Base.sle_int(%3659, 4)::Bool
│      %5672  = Base.and_int(%5670, %5671)::Bool
│      %5673  = Base.sle_int(1, %3656)::Bool
│      %5674  = Base.sle_int(%3656, 4)::Bool
│      %5675  = Base.and_int(%5673, %5674)::Bool
│      %5676  = Base.sle_int(1, %3653)::Bool
│      %5677  = Base.sle_int(%3653, 4)::Bool
│      %5678  = Base.and_int(%5676, %5677)::Bool
│      %5679  = Base.sub_int(%3646, 1)::Int64
│      %5680  = Base.bitcast(UInt64, %5679)::UInt64
│      %5681  = Base.bitcast(UInt64, %5666)::UInt64
│      %5682  = Base.ult_int(%5680, %5681)::Bool
│      %5683  = Base.and_int(%5682, true)::Bool
│      %5684  = Base.and_int(%5678, %5683)::Bool
│      %5685  = Base.and_int(%5675, %5684)::Bool
│      %5686  = Base.and_int(%5672, %5685)::Bool
│      %5687  = Base.and_int(%5669, %5686)::Bool
└─────          goto #1962 if not %5687
1961 ─          goto #1963
1962 ─          invoke Base.throw_boundserror(%3549::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %5662::NTuple{5, Int64})::Union{}
└─────          unreachable
1963 ─          nothing::Nothing
1964 ┄ %5693  = StrideArraysCore.getfield(%3549, :ptr)::Ptr{Float64}
│      %5694  = Base.sub_int(%5582, 1)::Int64
│      %5695  = Base.sub_int(%3659, 1)::Int64
│      %5696  = Base.sub_int(%3656, 1)::Int64
│      %5697  = Base.sub_int(%3653, 1)::Int64
│      %5698  = Base.sub_int(%3646, 1)::Int64
└─────          goto #1973 if not true
1965 ┄ %5700  = φ (#1964 => 2, #1972 => %5712)::Int64
│      %5701  = Base.sle_int(1, %5700)::Bool
└─────          goto #1967 if not %5701
1966 ─ %5703  = Base.sle_int(%5700, 5)::Bool
└─────          goto #1968
1967 ─          nothing::Nothing
1968 ┄ %5706  = φ (#1966 => %5703, #1967 => false)::Bool
└─────          goto #1970 if not %5706
1969 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %5700, true)::Static.True
│      %5709  = Base.add_int(%5700, 1)::Int64
└─────          goto #1971
1970 ─          goto #1971
1971 ┄ %5712  = φ (#1969 => %5709)::Int64
│      %5713  = φ (#1969 => false, #1970 => true)::Bool
│      %5714  = Base.not_int(%5713)::Bool
└─────          goto #1973 if not %5714
1972 ─          goto #1965
1973 ┄          goto #1974
1974 ─          goto #1975
1975 ─ %5719  = Base.mul_int(%5698, 4)::Int64
│      %5720  = Base.add_int(%5697, %5719)::Int64
│      %5721  = Base.mul_int(%5720, 4)::Int64
│      %5722  = Base.add_int(%5696, %5721)::Int64
│      %5723  = Base.mul_int(%5722, 4)::Int64
│      %5724  = Base.add_int(%5695, %5723)::Int64
│      %5725  = Base.mul_int(%5724, 5)::Int64
│      %5726  = Base.add_int(%5694, %5725)::Int64
│      %5727  = Base.mul_int(8, %5726)::Int64
│      %5728  = Core.bitcast(Core.UInt, %5693)::UInt64
│      %5729  = Base.bitcast(UInt64, %5727)::UInt64
│      %5730  = Base.add_ptr(%5728, %5729)::UInt64
│      %5731  = Core.bitcast(Ptr{Float64}, %5730)::Ptr{Float64}
└─────          goto #1976
1976 ─          Base.pointerset(%5731, %5660, 1, 1)::Ptr{Float64}
└─────          goto #1977
1977 ─          goto #1978
1978 ─ %5736  = (%5583 === 5)::Bool
└─────          goto #1980 if not %5736
1979 ─          goto #1981
1980 ─ %5739  = Base.add_int(%5583, 1)::Int64
└─────          goto #1981
1981 ┄ %5741  = φ (#1980 => %5739)::Int64
│      %5742  = φ (#1980 => %5739)::Int64
│      %5743  = φ (#1979 => true, #1980 => false)::Bool
│      %5744  = Base.not_int(%5743)::Bool
└─────          goto #1983 if not %5744
1982 ─          goto #1940
1983 ┄          goto #1984
1984 ─ %5748  = Base.arrayref(false, %3651, %4994, %3656)::Float64
│      %5749  = Base.copysign_float(0.0, %5748)::Float64
│      %5750  = Core.ifelse(true, %5748, %5749)::Float64
└─────          goto #2028 if not true
1985 ┄ %5752  = φ (#1984 => 1, #2027 => %5911)::Int64
│      %5753  = φ (#1984 => 1, #2027 => %5912)::Int64
│      %5754  = Base.getfield(%5576, %5752, true)::Float64
└─────          goto #1990 if not true
1986 ─ %5756  = Core.tuple(%5752, %3659, %4994, %3653, %3646)::NTuple{5, Int64}
│      %5757  = StrideArraysCore.getfield(%3549, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %5758  = Core.getfield(%5757, 5)::Int64
│      %5759  = Base.bitcast(UInt64, %5758)::UInt64
│      %5760  = Base.bitcast(Int64, %5759)::Int64
│      %5761  = Base.sle_int(1, %5752)::Bool
│      %5762  = Base.sle_int(%5752, 5)::Bool
│      %5763  = Base.and_int(%5761, %5762)::Bool
│      %5764  = Base.sle_int(1, %3659)::Bool
│      %5765  = Base.sle_int(%3659, 4)::Bool
│      %5766  = Base.and_int(%5764, %5765)::Bool
│      %5767  = Base.sle_int(1, %4994)::Bool
│      %5768  = Base.sle_int(%4994, 4)::Bool
│      %5769  = Base.and_int(%5767, %5768)::Bool
│      %5770  = Base.sle_int(1, %3653)::Bool
│      %5771  = Base.sle_int(%3653, 4)::Bool
│      %5772  = Base.and_int(%5770, %5771)::Bool
│      %5773  = Base.sub_int(%3646, 1)::Int64
│      %5774  = Base.bitcast(UInt64, %5773)::UInt64
│      %5775  = Base.bitcast(UInt64, %5760)::UInt64
│      %5776  = Base.ult_int(%5774, %5775)::Bool
│      %5777  = Base.and_int(%5776, true)::Bool
│      %5778  = Base.and_int(%5772, %5777)::Bool
│      %5779  = Base.and_int(%5769, %5778)::Bool
│      %5780  = Base.and_int(%5766, %5779)::Bool
│      %5781  = Base.and_int(%5763, %5780)::Bool
└─────          goto #1988 if not %5781
1987 ─          goto #1989
1988 ─          invoke Base.throw_boundserror(%3549::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %5756::NTuple{5, Int64})::Union{}
└─────          unreachable
1989 ─          nothing::Nothing
1990 ┄ %5787  = StrideArraysCore.getfield(%3549, :ptr)::Ptr{Float64}
│      %5788  = Base.sub_int(%5752, 1)::Int64
│      %5789  = Base.sub_int(%3659, 1)::Int64
│      %5790  = Base.sub_int(%4994, 1)::Int64
│      %5791  = Base.sub_int(%3653, 1)::Int64
│      %5792  = Base.sub_int(%3646, 1)::Int64
└─────          goto #1999 if not true
1991 ┄ %5794  = φ (#1990 => 2, #1998 => %5806)::Int64
│      %5795  = Base.sle_int(1, %5794)::Bool
└─────          goto #1993 if not %5795
1992 ─ %5797  = Base.sle_int(%5794, 5)::Bool
└─────          goto #1994
1993 ─          nothing::Nothing
1994 ┄ %5800  = φ (#1992 => %5797, #1993 => false)::Bool
└─────          goto #1996 if not %5800
1995 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %5794, true)::Static.True
│      %5803  = Base.add_int(%5794, 1)::Int64
└─────          goto #1997
1996 ─          goto #1997
1997 ┄ %5806  = φ (#1995 => %5803)::Int64
│      %5807  = φ (#1995 => false, #1996 => true)::Bool
│      %5808  = Base.not_int(%5807)::Bool
└─────          goto #1999 if not %5808
1998 ─          goto #1991
1999 ┄          goto #2000
2000 ─          goto #2001
2001 ─ %5813  = Base.mul_int(%5792, 4)::Int64
│      %5814  = Base.add_int(%5791, %5813)::Int64
│      %5815  = Base.mul_int(%5814, 4)::Int64
│      %5816  = Base.add_int(%5790, %5815)::Int64
│      %5817  = Base.mul_int(%5816, 4)::Int64
│      %5818  = Base.add_int(%5789, %5817)::Int64
│      %5819  = Base.mul_int(%5818, 5)::Int64
│      %5820  = Base.add_int(%5788, %5819)::Int64
│      %5821  = Base.mul_int(8, %5820)::Int64
│      %5822  = Core.bitcast(Core.UInt, %5787)::UInt64
│      %5823  = Base.bitcast(UInt64, %5821)::UInt64
│      %5824  = Base.add_ptr(%5822, %5823)::UInt64
│      %5825  = Core.bitcast(Ptr{Float64}, %5824)::Ptr{Float64}
└─────          goto #2002
2002 ─ %5827  = Base.pointerref(%5825, 1, 1)::Float64
└─────          goto #2003
2003 ─          goto #2004
2004 ─ %5830  = Base.muladd_float(%5750, %5754, %5827)::Float64
└─────          goto #2009 if not true
2005 ─ %5832  = Core.tuple(%5752, %3659, %4994, %3653, %3646)::NTuple{5, Int64}
│      %5833  = StrideArraysCore.getfield(%3549, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %5834  = Core.getfield(%5833, 5)::Int64
│      %5835  = Base.bitcast(UInt64, %5834)::UInt64
│      %5836  = Base.bitcast(Int64, %5835)::Int64
│      %5837  = Base.sle_int(1, %5752)::Bool
│      %5838  = Base.sle_int(%5752, 5)::Bool
│      %5839  = Base.and_int(%5837, %5838)::Bool
│      %5840  = Base.sle_int(1, %3659)::Bool
│      %5841  = Base.sle_int(%3659, 4)::Bool
│      %5842  = Base.and_int(%5840, %5841)::Bool
│      %5843  = Base.sle_int(1, %4994)::Bool
│      %5844  = Base.sle_int(%4994, 4)::Bool
│      %5845  = Base.and_int(%5843, %5844)::Bool
│      %5846  = Base.sle_int(1, %3653)::Bool
│      %5847  = Base.sle_int(%3653, 4)::Bool
│      %5848  = Base.and_int(%5846, %5847)::Bool
│      %5849  = Base.sub_int(%3646, 1)::Int64
│      %5850  = Base.bitcast(UInt64, %5849)::UInt64
│      %5851  = Base.bitcast(UInt64, %5836)::UInt64
│      %5852  = Base.ult_int(%5850, %5851)::Bool
│      %5853  = Base.and_int(%5852, true)::Bool
│      %5854  = Base.and_int(%5848, %5853)::Bool
│      %5855  = Base.and_int(%5845, %5854)::Bool
│      %5856  = Base.and_int(%5842, %5855)::Bool
│      %5857  = Base.and_int(%5839, %5856)::Bool
└─────          goto #2007 if not %5857
2006 ─          goto #2008
2007 ─          invoke Base.throw_boundserror(%3549::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %5832::NTuple{5, Int64})::Union{}
└─────          unreachable
2008 ─          nothing::Nothing
2009 ┄ %5863  = StrideArraysCore.getfield(%3549, :ptr)::Ptr{Float64}
│      %5864  = Base.sub_int(%5752, 1)::Int64
│      %5865  = Base.sub_int(%3659, 1)::Int64
│      %5866  = Base.sub_int(%4994, 1)::Int64
│      %5867  = Base.sub_int(%3653, 1)::Int64
│      %5868  = Base.sub_int(%3646, 1)::Int64
└─────          goto #2018 if not true
2010 ┄ %5870  = φ (#2009 => 2, #2017 => %5882)::Int64
│      %5871  = Base.sle_int(1, %5870)::Bool
└─────          goto #2012 if not %5871
2011 ─ %5873  = Base.sle_int(%5870, 5)::Bool
└─────          goto #2013
2012 ─          nothing::Nothing
2013 ┄ %5876  = φ (#2011 => %5873, #2012 => false)::Bool
└─────          goto #2015 if not %5876
2014 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %5870, true)::Static.True
│      %5879  = Base.add_int(%5870, 1)::Int64
└─────          goto #2016
2015 ─          goto #2016
2016 ┄ %5882  = φ (#2014 => %5879)::Int64
│      %5883  = φ (#2014 => false, #2015 => true)::Bool
│      %5884  = Base.not_int(%5883)::Bool
└─────          goto #2018 if not %5884
2017 ─          goto #2010
2018 ┄          goto #2019
2019 ─          goto #2020
2020 ─ %5889  = Base.mul_int(%5868, 4)::Int64
│      %5890  = Base.add_int(%5867, %5889)::Int64
│      %5891  = Base.mul_int(%5890, 4)::Int64
│      %5892  = Base.add_int(%5866, %5891)::Int64
│      %5893  = Base.mul_int(%5892, 4)::Int64
│      %5894  = Base.add_int(%5865, %5893)::Int64
│      %5895  = Base.mul_int(%5894, 5)::Int64
│      %5896  = Base.add_int(%5864, %5895)::Int64
│      %5897  = Base.mul_int(8, %5896)::Int64
│      %5898  = Core.bitcast(Core.UInt, %5863)::UInt64
│      %5899  = Base.bitcast(UInt64, %5897)::UInt64
│      %5900  = Base.add_ptr(%5898, %5899)::UInt64
│      %5901  = Core.bitcast(Ptr{Float64}, %5900)::Ptr{Float64}
└─────          goto #2021
2021 ─          Base.pointerset(%5901, %5830, 1, 1)::Ptr{Float64}
└─────          goto #2022
2022 ─          goto #2023
2023 ─ %5906  = (%5753 === 5)::Bool
└─────          goto #2025 if not %5906
2024 ─          goto #2026
2025 ─ %5909  = Base.add_int(%5753, 1)::Int64
└─────          goto #2026
2026 ┄ %5911  = φ (#2025 => %5909)::Int64
│      %5912  = φ (#2025 => %5909)::Int64
│      %5913  = φ (#2024 => true, #2025 => false)::Bool
│      %5914  = Base.not_int(%5913)::Bool
└─────          goto #2028 if not %5914
2027 ─          goto #1985
2028 ┄          goto #2029
2029 ─ %5918  = (%4995 === %4982)::Bool
└─────          goto #2031 if not %5918
2030 ─          goto #2032
2031 ─ %5921  = Base.add_int(%4995, 1)::Int64
└─────          goto #2032
2032 ┄ %5923  = φ (#2031 => %5921)::Int64
│      %5924  = φ (#2031 => %5921)::Int64
│      %5925  = φ (#2030 => true, #2031 => false)::Bool
│      %5926  = Base.not_int(%5925)::Bool
└─────          goto #2034 if not %5926
2033 ─          goto #1700
2034 ┄ %5929  = Base.add_int(%3653, 1)::Int64
│      %5930  = Base.sle_int(%5929, 4)::Bool
└─────          goto #2036 if not %5930
2035 ─          goto #2037
2036 ─ %5933  = Base.sub_int(%5929, 1)::Int64
└─────          goto #2037
2037 ┄ %5935  = φ (#2035 => 4, #2036 => %5933)::Int64
└─────          goto #2038
2038 ─          goto #2039
2039 ─ %5938  = Base.slt_int(%5935, %5929)::Bool
└─────          goto #2041 if not %5938
2040 ─          goto #2042
2041 ─          goto #2042
2042 ┄ %5942  = φ (#2040 => true, #2041 => false)::Bool
│      %5943  = φ (#2041 => %5929)::Int64
│      %5944  = φ (#2041 => %5929)::Int64
│      %5945  = Base.not_int(%5942)::Bool
└─────          goto #2377 if not %5945
2043 ┄ %5947  = φ (#2042 => %5943, #2376 => %6876)::Int64
│      %5948  = φ (#2042 => %5944, #2376 => %6877)::Int64
└─────          goto #2048 if not true
2044 ─ %5950  = Core.tuple(1, %3659, %3656, %5947, %3646)::NTuple{5, Int64}
│      %5951  = StrideArraysCore.getfield(%3551, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %5952  = Core.getfield(%5951, 5)::Int64
│      %5953  = Base.bitcast(UInt64, %5952)::UInt64
│      %5954  = Base.bitcast(Int64, %5953)::Int64
│      %5955  = Base.sle_int(1, %3659)::Bool
│      %5956  = Base.sle_int(%3659, 4)::Bool
│      %5957  = Base.and_int(%5955, %5956)::Bool
│      %5958  = Base.sle_int(1, %3656)::Bool
│      %5959  = Base.sle_int(%3656, 4)::Bool
│      %5960  = Base.and_int(%5958, %5959)::Bool
│      %5961  = Base.sle_int(1, %5947)::Bool
│      %5962  = Base.sle_int(%5947, 4)::Bool
│      %5963  = Base.and_int(%5961, %5962)::Bool
│      %5964  = Base.sub_int(%3646, 1)::Int64
│      %5965  = Base.bitcast(UInt64, %5964)::UInt64
│      %5966  = Base.bitcast(UInt64, %5954)::UInt64
│      %5967  = Base.ult_int(%5965, %5966)::Bool
│      %5968  = Base.and_int(%5967, true)::Bool
│      %5969  = Base.and_int(%5963, %5968)::Bool
│      %5970  = Base.and_int(%5960, %5969)::Bool
│      %5971  = Base.and_int(%5957, %5970)::Bool
│      %5972  = Base.and_int(true, %5971)::Bool
└─────          goto #2046 if not %5972
2045 ─          goto #2047
2046 ─          invoke Base.throw_boundserror(%3551::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %5950::NTuple{5, Int64})::Union{}
└─────          unreachable
2047 ─          nothing::Nothing
2048 ┄ %5978  = StrideArraysCore.getfield(%3551, :ptr)::Ptr{Float64}
│      %5979  = Base.sub_int(%3659, 1)::Int64
│      %5980  = Base.sub_int(%3656, 1)::Int64
│      %5981  = Base.sub_int(%5947, 1)::Int64
│      %5982  = Base.sub_int(%3646, 1)::Int64
└─────          goto #2057 if not true
2049 ┄ %5984  = φ (#2048 => 2, #2056 => %5996)::Int64
│      %5985  = Base.sle_int(1, %5984)::Bool
└─────          goto #2051 if not %5985
2050 ─ %5987  = Base.sle_int(%5984, 5)::Bool
└─────          goto #2052
2051 ─          nothing::Nothing
2052 ┄ %5990  = φ (#2050 => %5987, #2051 => false)::Bool
└─────          goto #2054 if not %5990
2053 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %5984, true)::Static.True
│      %5993  = Base.add_int(%5984, 1)::Int64
└─────          goto #2055
2054 ─          goto #2055
2055 ┄ %5996  = φ (#2053 => %5993)::Int64
│      %5997  = φ (#2053 => false, #2054 => true)::Bool
│      %5998  = Base.not_int(%5997)::Bool
└─────          goto #2057 if not %5998
2056 ─          goto #2049
2057 ┄          goto #2058
2058 ─          goto #2059
2059 ─ %6003  = Base.mul_int(%5982, 4)::Int64
│      %6004  = Base.add_int(%5981, %6003)::Int64
│      %6005  = Base.mul_int(%6004, 4)::Int64
│      %6006  = Base.add_int(%5980, %6005)::Int64
│      %6007  = Base.mul_int(%6006, 4)::Int64
│      %6008  = Base.add_int(%5979, %6007)::Int64
│      %6009  = Base.mul_int(%6008, 5)::Int64
│      %6010  = Base.add_int(0, %6009)::Int64
│      %6011  = Base.mul_int(8, %6010)::Int64
│      %6012  = Core.bitcast(Core.UInt, %5978)::UInt64
│      %6013  = Base.bitcast(UInt64, %6011)::UInt64
│      %6014  = Base.add_ptr(%6012, %6013)::UInt64
│      %6015  = Core.bitcast(Ptr{Float64}, %6014)::Ptr{Float64}
└─────          goto #2060
2060 ─ %6017  = Base.pointerref(%6015, 1, 1)::Float64
└─────          goto #2061
2061 ─          goto #2062
2062 ─          goto #2063
2063 ─          goto #2068 if not true
2064 ─ %6022  = Core.tuple(2, %3659, %3656, %5947, %3646)::NTuple{5, Int64}
│      %6023  = StrideArraysCore.getfield(%3551, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %6024  = Core.getfield(%6023, 5)::Int64
│      %6025  = Base.bitcast(UInt64, %6024)::UInt64
│      %6026  = Base.bitcast(Int64, %6025)::Int64
│      %6027  = Base.sle_int(1, %3659)::Bool
│      %6028  = Base.sle_int(%3659, 4)::Bool
│      %6029  = Base.and_int(%6027, %6028)::Bool
│      %6030  = Base.sle_int(1, %3656)::Bool
│      %6031  = Base.sle_int(%3656, 4)::Bool
│      %6032  = Base.and_int(%6030, %6031)::Bool
│      %6033  = Base.sle_int(1, %5947)::Bool
│      %6034  = Base.sle_int(%5947, 4)::Bool
│      %6035  = Base.and_int(%6033, %6034)::Bool
│      %6036  = Base.sub_int(%3646, 1)::Int64
│      %6037  = Base.bitcast(UInt64, %6036)::UInt64
│      %6038  = Base.bitcast(UInt64, %6026)::UInt64
│      %6039  = Base.ult_int(%6037, %6038)::Bool
│      %6040  = Base.and_int(%6039, true)::Bool
│      %6041  = Base.and_int(%6035, %6040)::Bool
│      %6042  = Base.and_int(%6032, %6041)::Bool
│      %6043  = Base.and_int(%6029, %6042)::Bool
│      %6044  = Base.and_int(true, %6043)::Bool
└─────          goto #2066 if not %6044
2065 ─          goto #2067
2066 ─          invoke Base.throw_boundserror(%3551::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %6022::NTuple{5, Int64})::Union{}
└─────          unreachable
2067 ─          nothing::Nothing
2068 ┄ %6050  = StrideArraysCore.getfield(%3551, :ptr)::Ptr{Float64}
│      %6051  = Base.sub_int(%3659, 1)::Int64
│      %6052  = Base.sub_int(%3656, 1)::Int64
│      %6053  = Base.sub_int(%5947, 1)::Int64
│      %6054  = Base.sub_int(%3646, 1)::Int64
└─────          goto #2077 if not true
2069 ┄ %6056  = φ (#2068 => 2, #2076 => %6068)::Int64
│      %6057  = Base.sle_int(1, %6056)::Bool
└─────          goto #2071 if not %6057
2070 ─ %6059  = Base.sle_int(%6056, 5)::Bool
└─────          goto #2072
2071 ─          nothing::Nothing
2072 ┄ %6062  = φ (#2070 => %6059, #2071 => false)::Bool
└─────          goto #2074 if not %6062
2073 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %6056, true)::Static.True
│      %6065  = Base.add_int(%6056, 1)::Int64
└─────          goto #2075
2074 ─          goto #2075
2075 ┄ %6068  = φ (#2073 => %6065)::Int64
│      %6069  = φ (#2073 => false, #2074 => true)::Bool
│      %6070  = Base.not_int(%6069)::Bool
└─────          goto #2077 if not %6070
2076 ─          goto #2069
2077 ┄          goto #2078
2078 ─          goto #2079
2079 ─ %6075  = Base.mul_int(%6054, 4)::Int64
│      %6076  = Base.add_int(%6053, %6075)::Int64
│      %6077  = Base.mul_int(%6076, 4)::Int64
│      %6078  = Base.add_int(%6052, %6077)::Int64
│      %6079  = Base.mul_int(%6078, 4)::Int64
│      %6080  = Base.add_int(%6051, %6079)::Int64
│      %6081  = Base.mul_int(%6080, 5)::Int64
│      %6082  = Base.add_int(1, %6081)::Int64
│      %6083  = Base.mul_int(8, %6082)::Int64
│      %6084  = Core.bitcast(Core.UInt, %6050)::UInt64
│      %6085  = Base.bitcast(UInt64, %6083)::UInt64
│      %6086  = Base.add_ptr(%6084, %6085)::UInt64
│      %6087  = Core.bitcast(Ptr{Float64}, %6086)::Ptr{Float64}
└─────          goto #2080
2080 ─ %6089  = Base.pointerref(%6087, 1, 1)::Float64
└─────          goto #2081
2081 ─          goto #2082
2082 ─          goto #2083
2083 ─          goto #2088 if not true
2084 ─ %6094  = Core.tuple(3, %3659, %3656, %5947, %3646)::NTuple{5, Int64}
│      %6095  = StrideArraysCore.getfield(%3551, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %6096  = Core.getfield(%6095, 5)::Int64
│      %6097  = Base.bitcast(UInt64, %6096)::UInt64
│      %6098  = Base.bitcast(Int64, %6097)::Int64
│      %6099  = Base.sle_int(1, %3659)::Bool
│      %6100  = Base.sle_int(%3659, 4)::Bool
│      %6101  = Base.and_int(%6099, %6100)::Bool
│      %6102  = Base.sle_int(1, %3656)::Bool
│      %6103  = Base.sle_int(%3656, 4)::Bool
│      %6104  = Base.and_int(%6102, %6103)::Bool
│      %6105  = Base.sle_int(1, %5947)::Bool
│      %6106  = Base.sle_int(%5947, 4)::Bool
│      %6107  = Base.and_int(%6105, %6106)::Bool
│      %6108  = Base.sub_int(%3646, 1)::Int64
│      %6109  = Base.bitcast(UInt64, %6108)::UInt64
│      %6110  = Base.bitcast(UInt64, %6098)::UInt64
│      %6111  = Base.ult_int(%6109, %6110)::Bool
│      %6112  = Base.and_int(%6111, true)::Bool
│      %6113  = Base.and_int(%6107, %6112)::Bool
│      %6114  = Base.and_int(%6104, %6113)::Bool
│      %6115  = Base.and_int(%6101, %6114)::Bool
│      %6116  = Base.and_int(true, %6115)::Bool
└─────          goto #2086 if not %6116
2085 ─          goto #2087
2086 ─          invoke Base.throw_boundserror(%3551::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %6094::NTuple{5, Int64})::Union{}
└─────          unreachable
2087 ─          nothing::Nothing
2088 ┄ %6122  = StrideArraysCore.getfield(%3551, :ptr)::Ptr{Float64}
│      %6123  = Base.sub_int(%3659, 1)::Int64
│      %6124  = Base.sub_int(%3656, 1)::Int64
│      %6125  = Base.sub_int(%5947, 1)::Int64
│      %6126  = Base.sub_int(%3646, 1)::Int64
└─────          goto #2097 if not true
2089 ┄ %6128  = φ (#2088 => 2, #2096 => %6140)::Int64
│      %6129  = Base.sle_int(1, %6128)::Bool
└─────          goto #2091 if not %6129
2090 ─ %6131  = Base.sle_int(%6128, 5)::Bool
└─────          goto #2092
2091 ─          nothing::Nothing
2092 ┄ %6134  = φ (#2090 => %6131, #2091 => false)::Bool
└─────          goto #2094 if not %6134
2093 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %6128, true)::Static.True
│      %6137  = Base.add_int(%6128, 1)::Int64
└─────          goto #2095
2094 ─          goto #2095
2095 ┄ %6140  = φ (#2093 => %6137)::Int64
│      %6141  = φ (#2093 => false, #2094 => true)::Bool
│      %6142  = Base.not_int(%6141)::Bool
└─────          goto #2097 if not %6142
2096 ─          goto #2089
2097 ┄          goto #2098
2098 ─          goto #2099
2099 ─ %6147  = Base.mul_int(%6126, 4)::Int64
│      %6148  = Base.add_int(%6125, %6147)::Int64
│      %6149  = Base.mul_int(%6148, 4)::Int64
│      %6150  = Base.add_int(%6124, %6149)::Int64
│      %6151  = Base.mul_int(%6150, 4)::Int64
│      %6152  = Base.add_int(%6123, %6151)::Int64
│      %6153  = Base.mul_int(%6152, 5)::Int64
│      %6154  = Base.add_int(2, %6153)::Int64
│      %6155  = Base.mul_int(8, %6154)::Int64
│      %6156  = Core.bitcast(Core.UInt, %6122)::UInt64
│      %6157  = Base.bitcast(UInt64, %6155)::UInt64
│      %6158  = Base.add_ptr(%6156, %6157)::UInt64
│      %6159  = Core.bitcast(Ptr{Float64}, %6158)::Ptr{Float64}
└─────          goto #2100
2100 ─ %6161  = Base.pointerref(%6159, 1, 1)::Float64
└─────          goto #2101
2101 ─          goto #2102
2102 ─          goto #2103
2103 ─          goto #2108 if not true
2104 ─ %6166  = Core.tuple(4, %3659, %3656, %5947, %3646)::NTuple{5, Int64}
│      %6167  = StrideArraysCore.getfield(%3551, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %6168  = Core.getfield(%6167, 5)::Int64
│      %6169  = Base.bitcast(UInt64, %6168)::UInt64
│      %6170  = Base.bitcast(Int64, %6169)::Int64
│      %6171  = Base.sle_int(1, %3659)::Bool
│      %6172  = Base.sle_int(%3659, 4)::Bool
│      %6173  = Base.and_int(%6171, %6172)::Bool
│      %6174  = Base.sle_int(1, %3656)::Bool
│      %6175  = Base.sle_int(%3656, 4)::Bool
│      %6176  = Base.and_int(%6174, %6175)::Bool
│      %6177  = Base.sle_int(1, %5947)::Bool
│      %6178  = Base.sle_int(%5947, 4)::Bool
│      %6179  = Base.and_int(%6177, %6178)::Bool
│      %6180  = Base.sub_int(%3646, 1)::Int64
│      %6181  = Base.bitcast(UInt64, %6180)::UInt64
│      %6182  = Base.bitcast(UInt64, %6170)::UInt64
│      %6183  = Base.ult_int(%6181, %6182)::Bool
│      %6184  = Base.and_int(%6183, true)::Bool
│      %6185  = Base.and_int(%6179, %6184)::Bool
│      %6186  = Base.and_int(%6176, %6185)::Bool
│      %6187  = Base.and_int(%6173, %6186)::Bool
│      %6188  = Base.and_int(true, %6187)::Bool
└─────          goto #2106 if not %6188
2105 ─          goto #2107
2106 ─          invoke Base.throw_boundserror(%3551::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %6166::NTuple{5, Int64})::Union{}
└─────          unreachable
2107 ─          nothing::Nothing
2108 ┄ %6194  = StrideArraysCore.getfield(%3551, :ptr)::Ptr{Float64}
│      %6195  = Base.sub_int(%3659, 1)::Int64
│      %6196  = Base.sub_int(%3656, 1)::Int64
│      %6197  = Base.sub_int(%5947, 1)::Int64
│      %6198  = Base.sub_int(%3646, 1)::Int64
└─────          goto #2117 if not true
2109 ┄ %6200  = φ (#2108 => 2, #2116 => %6212)::Int64
│      %6201  = Base.sle_int(1, %6200)::Bool
└─────          goto #2111 if not %6201
2110 ─ %6203  = Base.sle_int(%6200, 5)::Bool
└─────          goto #2112
2111 ─          nothing::Nothing
2112 ┄ %6206  = φ (#2110 => %6203, #2111 => false)::Bool
└─────          goto #2114 if not %6206
2113 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %6200, true)::Static.True
│      %6209  = Base.add_int(%6200, 1)::Int64
└─────          goto #2115
2114 ─          goto #2115
2115 ┄ %6212  = φ (#2113 => %6209)::Int64
│      %6213  = φ (#2113 => false, #2114 => true)::Bool
│      %6214  = Base.not_int(%6213)::Bool
└─────          goto #2117 if not %6214
2116 ─          goto #2109
2117 ┄          goto #2118
2118 ─          goto #2119
2119 ─ %6219  = Base.mul_int(%6198, 4)::Int64
│      %6220  = Base.add_int(%6197, %6219)::Int64
│      %6221  = Base.mul_int(%6220, 4)::Int64
│      %6222  = Base.add_int(%6196, %6221)::Int64
│      %6223  = Base.mul_int(%6222, 4)::Int64
│      %6224  = Base.add_int(%6195, %6223)::Int64
│      %6225  = Base.mul_int(%6224, 5)::Int64
│      %6226  = Base.add_int(3, %6225)::Int64
│      %6227  = Base.mul_int(8, %6226)::Int64
│      %6228  = Core.bitcast(Core.UInt, %6194)::UInt64
│      %6229  = Base.bitcast(UInt64, %6227)::UInt64
│      %6230  = Base.add_ptr(%6228, %6229)::UInt64
│      %6231  = Core.bitcast(Ptr{Float64}, %6230)::Ptr{Float64}
└─────          goto #2120
2120 ─ %6233  = Base.pointerref(%6231, 1, 1)::Float64
└─────          goto #2121
2121 ─          goto #2122
2122 ─          goto #2123
2123 ─          goto #2128 if not true
2124 ─ %6238  = Core.tuple(5, %3659, %3656, %5947, %3646)::NTuple{5, Int64}
│      %6239  = StrideArraysCore.getfield(%3551, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %6240  = Core.getfield(%6239, 5)::Int64
│      %6241  = Base.bitcast(UInt64, %6240)::UInt64
│      %6242  = Base.bitcast(Int64, %6241)::Int64
│      %6243  = Base.sle_int(1, %3659)::Bool
│      %6244  = Base.sle_int(%3659, 4)::Bool
│      %6245  = Base.and_int(%6243, %6244)::Bool
│      %6246  = Base.sle_int(1, %3656)::Bool
│      %6247  = Base.sle_int(%3656, 4)::Bool
│      %6248  = Base.and_int(%6246, %6247)::Bool
│      %6249  = Base.sle_int(1, %5947)::Bool
│      %6250  = Base.sle_int(%5947, 4)::Bool
│      %6251  = Base.and_int(%6249, %6250)::Bool
│      %6252  = Base.sub_int(%3646, 1)::Int64
│      %6253  = Base.bitcast(UInt64, %6252)::UInt64
│      %6254  = Base.bitcast(UInt64, %6242)::UInt64
│      %6255  = Base.ult_int(%6253, %6254)::Bool
│      %6256  = Base.and_int(%6255, true)::Bool
│      %6257  = Base.and_int(%6251, %6256)::Bool
│      %6258  = Base.and_int(%6248, %6257)::Bool
│      %6259  = Base.and_int(%6245, %6258)::Bool
│      %6260  = Base.and_int(true, %6259)::Bool
└─────          goto #2126 if not %6260
2125 ─          goto #2127
2126 ─          invoke Base.throw_boundserror(%3551::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %6238::NTuple{5, Int64})::Union{}
└─────          unreachable
2127 ─          nothing::Nothing
2128 ┄ %6266  = StrideArraysCore.getfield(%3551, :ptr)::Ptr{Float64}
│      %6267  = Base.sub_int(%3659, 1)::Int64
│      %6268  = Base.sub_int(%3656, 1)::Int64
│      %6269  = Base.sub_int(%5947, 1)::Int64
│      %6270  = Base.sub_int(%3646, 1)::Int64
└─────          goto #2137 if not true
2129 ┄ %6272  = φ (#2128 => 2, #2136 => %6284)::Int64
│      %6273  = Base.sle_int(1, %6272)::Bool
└─────          goto #2131 if not %6273
2130 ─ %6275  = Base.sle_int(%6272, 5)::Bool
└─────          goto #2132
2131 ─          nothing::Nothing
2132 ┄ %6278  = φ (#2130 => %6275, #2131 => false)::Bool
└─────          goto #2134 if not %6278
2133 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %6272, true)::Static.True
│      %6281  = Base.add_int(%6272, 1)::Int64
└─────          goto #2135
2134 ─          goto #2135
2135 ┄ %6284  = φ (#2133 => %6281)::Int64
│      %6285  = φ (#2133 => false, #2134 => true)::Bool
│      %6286  = Base.not_int(%6285)::Bool
└─────          goto #2137 if not %6286
2136 ─          goto #2129
2137 ┄          goto #2138
2138 ─          goto #2139
2139 ─ %6291  = Base.mul_int(%6270, 4)::Int64
│      %6292  = Base.add_int(%6269, %6291)::Int64
│      %6293  = Base.mul_int(%6292, 4)::Int64
│      %6294  = Base.add_int(%6268, %6293)::Int64
│      %6295  = Base.mul_int(%6294, 4)::Int64
│      %6296  = Base.add_int(%6267, %6295)::Int64
│      %6297  = Base.mul_int(%6296, 5)::Int64
│      %6298  = Base.add_int(4, %6297)::Int64
│      %6299  = Base.mul_int(8, %6298)::Int64
│      %6300  = Core.bitcast(Core.UInt, %6266)::UInt64
│      %6301  = Base.bitcast(UInt64, %6299)::UInt64
│      %6302  = Base.add_ptr(%6300, %6301)::UInt64
│      %6303  = Core.bitcast(Ptr{Float64}, %6302)::Ptr{Float64}
└─────          goto #2140
2140 ─ %6305  = Base.pointerref(%6303, 1, 1)::Float64
└─────          goto #2141
2141 ─          goto #2142
2142 ─          goto #2143
2143 ─          goto #2144
2144 ─          goto #2145
2145 ─          goto #2147
2146 ─          nothing::Nothing
2147 ┄          goto #2149
2148 ─          nothing::Nothing
2149 ┄          goto #2150
2150 ─          goto #2152
2151 ─          nothing::Nothing
2152 ┄          goto #2153
2153 ─          goto #2155
2154 ─          nothing::Nothing
2155 ┄          goto #2157
2156 ─          nothing::Nothing
2157 ┄          goto #2158
2158 ─          goto #2160
2159 ─          nothing::Nothing
2160 ┄          goto #2161
2161 ─          goto #2163
2162 ─          nothing::Nothing
2163 ┄          goto #2165
2164 ─          nothing::Nothing
2165 ┄          goto #2166
2166 ─          goto #2168
2167 ─          nothing::Nothing
2168 ┄          goto #2169
2169 ─          goto #2171
2170 ─          nothing::Nothing
2171 ┄          goto #2173
2172 ─          nothing::Nothing
2173 ┄          goto #2174
2174 ─          goto #2176
2175 ─          nothing::Nothing
2176 ┄          goto #2177
2177 ─ %6343  = Base.div_float(%3801, %3729)::Float64
│      %6344  = Base.div_float(%3873, %3729)::Float64
│      %6345  = Base.div_float(%3945, %3729)::Float64
│      %6346  = Base.getfield(equations, :gamma)::Float64
│      %6347  = Base.sub_float(%6346, 1.0)::Float64
│      %6348  = Base.mul_float(%3801, %6343)::Float64
│      %6349  = Base.muladd_float(%3873, %6344, %6348)::Float64
│      %6350  = Base.muladd_float(%3945, %6345, %6349)::Float64
│      %6351  = Base.muladd_float(-0.5, %6350, %4017)::Float64
│      %6352  = Base.mul_float(%6347, %6351)::Float64
└─────          goto #2178
2178 ─          goto #2180
2179 ─          nothing::Nothing
2180 ┄          goto #2182
2181 ─          nothing::Nothing
2182 ┄          goto #2183
2183 ─          goto #2185
2184 ─          nothing::Nothing
2185 ┄          goto #2186
2186 ─          goto #2188
2187 ─          nothing::Nothing
2188 ┄          goto #2190
2189 ─          nothing::Nothing
2190 ┄          goto #2191
2191 ─          goto #2193
2192 ─          nothing::Nothing
2193 ┄          goto #2194
2194 ─          goto #2196
2195 ─          nothing::Nothing
2196 ┄          goto #2198
2197 ─          nothing::Nothing
2198 ┄          goto #2199
2199 ─          goto #2201
2200 ─          nothing::Nothing
2201 ┄          goto #2202
2202 ─          goto #2204
2203 ─          nothing::Nothing
2204 ┄          goto #2206
2205 ─          nothing::Nothing
2206 ┄          goto #2207
2207 ─          goto #2209
2208 ─          nothing::Nothing
2209 ┄          goto #2210
2210 ─          goto #2212
2211 ─          nothing::Nothing
2212 ┄          goto #2214
2213 ─          nothing::Nothing
2214 ┄          goto #2215
2215 ─          goto #2217
2216 ─          nothing::Nothing
2217 ┄          goto #2218
2218 ─          goto #2220
2219 ─          nothing::Nothing
2220 ┄          goto #2222
2221 ─          nothing::Nothing
2222 ┄          goto #2223
2223 ─          goto #2225
2224 ─          nothing::Nothing
2225 ┄          goto #2226
2226 ─          goto #2228
2227 ─          nothing::Nothing
2228 ┄          goto #2230
2229 ─          nothing::Nothing
2230 ┄          goto #2231
2231 ─          goto #2233
2232 ─          nothing::Nothing
2233 ┄          goto #2234
2234 ─          goto #2236
2235 ─          nothing::Nothing
2236 ┄          goto #2238
2237 ─          nothing::Nothing
2238 ┄          goto #2239
2239 ─          goto #2241
2240 ─          nothing::Nothing
2241 ┄          goto #2242
2242 ─ %6418  = Base.div_float(%6089, %6017)::Float64
│      %6419  = Base.div_float(%6161, %6017)::Float64
│      %6420  = Base.div_float(%6233, %6017)::Float64
│      %6421  = Base.getfield(equations, :gamma)::Float64
│      %6422  = Base.sub_float(%6421, 1.0)::Float64
│      %6423  = Base.mul_float(%6089, %6418)::Float64
│      %6424  = Base.muladd_float(%6161, %6419, %6423)::Float64
│      %6425  = Base.muladd_float(%6233, %6420, %6424)::Float64
│      %6426  = Base.muladd_float(-0.5, %6425, %6305)::Float64
│      %6427  = Base.mul_float(%6422, %6426)::Float64
└─────          goto #2243
2243 ─          goto #2245
2244 ─          nothing::Nothing
2245 ┄          goto #2247
2246 ─          nothing::Nothing
2247 ┄          goto #2248
2248 ─          goto #2250
2249 ─          nothing::Nothing
2250 ┄          goto #2251
2251 ─          goto #2253
2252 ─          nothing::Nothing
2253 ┄          goto #2255
2254 ─          nothing::Nothing
2255 ┄          goto #2256
2256 ─          goto #2258
2257 ─          nothing::Nothing
2258 ┄          goto #2259
2259 ─          goto #2261
2260 ─          nothing::Nothing
2261 ┄          goto #2263
2262 ─          nothing::Nothing
2263 ┄          goto #2264
2264 ─          goto #2266
2265 ─          nothing::Nothing
2266 ┄          goto #2267
2267 ─          goto #2269
2268 ─          nothing::Nothing
2269 ┄          goto #2271
2270 ─          nothing::Nothing
2271 ┄          goto #2272
2272 ─          goto #2274
2273 ─          nothing::Nothing
2274 ┄          goto #2275
2275 ─ %6461  = Base.muladd_float(-2.0, %6017, %3729)::Float64
│      %6462  = Base.mul_float(%3729, %6461)::Float64
│      %6463  = Base.muladd_float(%6017, %6017, %6462)::Float64
│      %6464  = Base.muladd_float(2.0, %6017, %3729)::Float64
│      %6465  = Base.mul_float(%3729, %6464)::Float64
│      %6466  = Base.muladd_float(%6017, %6017, %6465)::Float64
│      %6467  = Base.div_float(%6463, %6466)::Float64
│      %6468  = Base.lt_float(%6467, 0.0001)::Bool
└─────          goto #2277 if not %6468
2276 ─ %6470  = Base.add_float(%3729, %6017)::Float64
│      %6471  = Base.muladd_float(%6467, 0.2857142857142857, 0.4)::Float64
│      %6472  = Base.muladd_float(%6467, %6471, 0.6666666666666666)::Float64
│      %6473  = Base.muladd_float(%6467, %6472, 2.0)::Float64
│      %6474  = Base.div_float(%6470, %6473)::Float64
└─────          goto #2278
2277 ─ %6476  = Base.sub_float(%6017, %3729)::Float64
│      %6477  = Base.div_float(%6017, %3729)::Float64
│      %6478  = $(Expr(:foreigncall, "llvm.log.f64", Float64, svec(Float64), 0, :(:llvmcall), :(%6477), :(%6477)))::Float64
│      %6479  = Base.div_float(%6476, %6478)::Float64
└─────          goto #2278
2278 ┄ %6481  = φ (#2276 => %6474, #2277 => %6479)::Float64
│      %6482  = Base.mul_float(%3729, %6427)::Float64
│      %6483  = Base.mul_float(%6017, %6352)::Float64
│      %6484  = Base.muladd_float(-2.0, %6483, %6482)::Float64
│      %6485  = Base.mul_float(%6482, %6484)::Float64
│      %6486  = Base.muladd_float(%6483, %6483, %6485)::Float64
│      %6487  = Base.muladd_float(2.0, %6483, %6482)::Float64
│      %6488  = Base.mul_float(%6482, %6487)::Float64
│      %6489  = Base.muladd_float(%6483, %6483, %6488)::Float64
│      %6490  = Base.div_float(%6486, %6489)::Float64
│      %6491  = Base.lt_float(%6490, 0.0001)::Bool
└─────          goto #2280 if not %6491
2279 ─ %6493  = Base.muladd_float(%6490, 0.2857142857142857, 0.4)::Float64
│      %6494  = Base.muladd_float(%6490, %6493, 0.6666666666666666)::Float64
│      %6495  = Base.muladd_float(%6490, %6494, 2.0)::Float64
│      %6496  = Base.add_float(%6482, %6483)::Float64
│      %6497  = Base.div_float(%6495, %6496)::Float64
└─────          goto #2281
2280 ─ %6499  = Base.div_float(%6483, %6482)::Float64
│      %6500  = $(Expr(:foreigncall, "llvm.log.f64", Float64, svec(Float64), 0, :(:llvmcall), :(%6499), :(%6499)))::Float64
│      %6501  = Base.sub_float(%6483, %6482)::Float64
│      %6502  = Base.div_float(%6500, %6501)::Float64
└─────          goto #2281
2281 ┄ %6504  = φ (#2279 => %6497, #2280 => %6502)::Float64
│      %6505  = Base.mul_float(%6352, %6427)::Float64
│      %6506  = Base.mul_float(%6505, %6504)::Float64
│      %6507  = Base.add_float(%6343, %6418)::Float64
│      %6508  = Base.mul_float(0.5, %6507)::Float64
│      %6509  = Base.add_float(%6344, %6419)::Float64
│      %6510  = Base.mul_float(0.5, %6509)::Float64
│      %6511  = Base.add_float(%6345, %6420)::Float64
│      %6512  = Base.mul_float(0.5, %6511)::Float64
│      %6513  = Base.add_float(%6352, %6427)::Float64
│      %6514  = Base.mul_float(0.5, %6513)::Float64
│      %6515  = Base.mul_float(%6343, %6418)::Float64
│      %6516  = Base.muladd_float(%6344, %6419, %6515)::Float64
│      %6517  = Base.muladd_float(%6345, %6420, %6516)::Float64
│      %6518  = Base.mul_float(0.5, %6517)::Float64
│      %6519  = Base.mul_float(%6481, %6512)::Float64
│      %6520  = Base.mul_float(%6519, %6508)::Float64
│      %6521  = Base.mul_float(%6519, %6510)::Float64
│      %6522  = Base.muladd_float(%6519, %6512, %6514)::Float64
│      %6523  = Base.mul_float(%6352, %6420)::Float64
│      %6524  = Base.muladd_float(%6427, %6345, %6523)::Float64
│      %6525  = Base.getfield(equations, :inv_gamma_minus_one)::Float64
│      %6526  = Base.muladd_float(%6506, %6525, %6518)::Float64
│      %6527  = Base.mul_float(%6519, %6526)::Float64
│      %6528  = Base.muladd_float(0.5, %6524, %6527)::Float64
│      %6529  = Core.tuple(%6519, %6520, %6521, %6522, %6528)::NTuple{5, Float64}
└─────          goto #2282
2282 ─ %6531  = Base.arrayref(false, %3651, %3653, %5947)::Float64
│      %6532  = Base.copysign_float(0.0, %6531)::Float64
│      %6533  = Core.ifelse(true, %6531, %6532)::Float64
└─────          goto #2326 if not true
2283 ┄ %6535  = φ (#2282 => 1, #2325 => %6694)::Int64
│      %6536  = φ (#2282 => 1, #2325 => %6695)::Int64
│      %6537  = Base.getfield(%6529, %6535, true)::Float64
└─────          goto #2288 if not true
2284 ─ %6539  = Core.tuple(%6535, %3659, %3656, %3653, %3646)::NTuple{5, Int64}
│      %6540  = StrideArraysCore.getfield(%3549, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %6541  = Core.getfield(%6540, 5)::Int64
│      %6542  = Base.bitcast(UInt64, %6541)::UInt64
│      %6543  = Base.bitcast(Int64, %6542)::Int64
│      %6544  = Base.sle_int(1, %6535)::Bool
│      %6545  = Base.sle_int(%6535, 5)::Bool
│      %6546  = Base.and_int(%6544, %6545)::Bool
│      %6547  = Base.sle_int(1, %3659)::Bool
│      %6548  = Base.sle_int(%3659, 4)::Bool
│      %6549  = Base.and_int(%6547, %6548)::Bool
│      %6550  = Base.sle_int(1, %3656)::Bool
│      %6551  = Base.sle_int(%3656, 4)::Bool
│      %6552  = Base.and_int(%6550, %6551)::Bool
│      %6553  = Base.sle_int(1, %3653)::Bool
│      %6554  = Base.sle_int(%3653, 4)::Bool
│      %6555  = Base.and_int(%6553, %6554)::Bool
│      %6556  = Base.sub_int(%3646, 1)::Int64
│      %6557  = Base.bitcast(UInt64, %6556)::UInt64
│      %6558  = Base.bitcast(UInt64, %6543)::UInt64
│      %6559  = Base.ult_int(%6557, %6558)::Bool
│      %6560  = Base.and_int(%6559, true)::Bool
│      %6561  = Base.and_int(%6555, %6560)::Bool
│      %6562  = Base.and_int(%6552, %6561)::Bool
│      %6563  = Base.and_int(%6549, %6562)::Bool
│      %6564  = Base.and_int(%6546, %6563)::Bool
└─────          goto #2286 if not %6564
2285 ─          goto #2287
2286 ─          invoke Base.throw_boundserror(%3549::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %6539::NTuple{5, Int64})::Union{}
└─────          unreachable
2287 ─          nothing::Nothing
2288 ┄ %6570  = StrideArraysCore.getfield(%3549, :ptr)::Ptr{Float64}
│      %6571  = Base.sub_int(%6535, 1)::Int64
│      %6572  = Base.sub_int(%3659, 1)::Int64
│      %6573  = Base.sub_int(%3656, 1)::Int64
│      %6574  = Base.sub_int(%3653, 1)::Int64
│      %6575  = Base.sub_int(%3646, 1)::Int64
└─────          goto #2297 if not true
2289 ┄ %6577  = φ (#2288 => 2, #2296 => %6589)::Int64
│      %6578  = Base.sle_int(1, %6577)::Bool
└─────          goto #2291 if not %6578
2290 ─ %6580  = Base.sle_int(%6577, 5)::Bool
└─────          goto #2292
2291 ─          nothing::Nothing
2292 ┄ %6583  = φ (#2290 => %6580, #2291 => false)::Bool
└─────          goto #2294 if not %6583
2293 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %6577, true)::Static.True
│      %6586  = Base.add_int(%6577, 1)::Int64
└─────          goto #2295
2294 ─          goto #2295
2295 ┄ %6589  = φ (#2293 => %6586)::Int64
│      %6590  = φ (#2293 => false, #2294 => true)::Bool
│      %6591  = Base.not_int(%6590)::Bool
└─────          goto #2297 if not %6591
2296 ─          goto #2289
2297 ┄          goto #2298
2298 ─          goto #2299
2299 ─ %6596  = Base.mul_int(%6575, 4)::Int64
│      %6597  = Base.add_int(%6574, %6596)::Int64
│      %6598  = Base.mul_int(%6597, 4)::Int64
│      %6599  = Base.add_int(%6573, %6598)::Int64
│      %6600  = Base.mul_int(%6599, 4)::Int64
│      %6601  = Base.add_int(%6572, %6600)::Int64
│      %6602  = Base.mul_int(%6601, 5)::Int64
│      %6603  = Base.add_int(%6571, %6602)::Int64
│      %6604  = Base.mul_int(8, %6603)::Int64
│      %6605  = Core.bitcast(Core.UInt, %6570)::UInt64
│      %6606  = Base.bitcast(UInt64, %6604)::UInt64
│      %6607  = Base.add_ptr(%6605, %6606)::UInt64
│      %6608  = Core.bitcast(Ptr{Float64}, %6607)::Ptr{Float64}
└─────          goto #2300
2300 ─ %6610  = Base.pointerref(%6608, 1, 1)::Float64
└─────          goto #2301
2301 ─          goto #2302
2302 ─ %6613  = Base.muladd_float(%6533, %6537, %6610)::Float64
└─────          goto #2307 if not true
2303 ─ %6615  = Core.tuple(%6535, %3659, %3656, %3653, %3646)::NTuple{5, Int64}
│      %6616  = StrideArraysCore.getfield(%3549, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %6617  = Core.getfield(%6616, 5)::Int64
│      %6618  = Base.bitcast(UInt64, %6617)::UInt64
│      %6619  = Base.bitcast(Int64, %6618)::Int64
│      %6620  = Base.sle_int(1, %6535)::Bool
│      %6621  = Base.sle_int(%6535, 5)::Bool
│      %6622  = Base.and_int(%6620, %6621)::Bool
│      %6623  = Base.sle_int(1, %3659)::Bool
│      %6624  = Base.sle_int(%3659, 4)::Bool
│      %6625  = Base.and_int(%6623, %6624)::Bool
│      %6626  = Base.sle_int(1, %3656)::Bool
│      %6627  = Base.sle_int(%3656, 4)::Bool
│      %6628  = Base.and_int(%6626, %6627)::Bool
│      %6629  = Base.sle_int(1, %3653)::Bool
│      %6630  = Base.sle_int(%3653, 4)::Bool
│      %6631  = Base.and_int(%6629, %6630)::Bool
│      %6632  = Base.sub_int(%3646, 1)::Int64
│      %6633  = Base.bitcast(UInt64, %6632)::UInt64
│      %6634  = Base.bitcast(UInt64, %6619)::UInt64
│      %6635  = Base.ult_int(%6633, %6634)::Bool
│      %6636  = Base.and_int(%6635, true)::Bool
│      %6637  = Base.and_int(%6631, %6636)::Bool
│      %6638  = Base.and_int(%6628, %6637)::Bool
│      %6639  = Base.and_int(%6625, %6638)::Bool
│      %6640  = Base.and_int(%6622, %6639)::Bool
└─────          goto #2305 if not %6640
2304 ─          goto #2306
2305 ─          invoke Base.throw_boundserror(%3549::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %6615::NTuple{5, Int64})::Union{}
└─────          unreachable
2306 ─          nothing::Nothing
2307 ┄ %6646  = StrideArraysCore.getfield(%3549, :ptr)::Ptr{Float64}
│      %6647  = Base.sub_int(%6535, 1)::Int64
│      %6648  = Base.sub_int(%3659, 1)::Int64
│      %6649  = Base.sub_int(%3656, 1)::Int64
│      %6650  = Base.sub_int(%3653, 1)::Int64
│      %6651  = Base.sub_int(%3646, 1)::Int64
└─────          goto #2316 if not true
2308 ┄ %6653  = φ (#2307 => 2, #2315 => %6665)::Int64
│      %6654  = Base.sle_int(1, %6653)::Bool
└─────          goto #2310 if not %6654
2309 ─ %6656  = Base.sle_int(%6653, 5)::Bool
└─────          goto #2311
2310 ─          nothing::Nothing
2311 ┄ %6659  = φ (#2309 => %6656, #2310 => false)::Bool
└─────          goto #2313 if not %6659
2312 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %6653, true)::Static.True
│      %6662  = Base.add_int(%6653, 1)::Int64
└─────          goto #2314
2313 ─          goto #2314
2314 ┄ %6665  = φ (#2312 => %6662)::Int64
│      %6666  = φ (#2312 => false, #2313 => true)::Bool
│      %6667  = Base.not_int(%6666)::Bool
└─────          goto #2316 if not %6667
2315 ─          goto #2308
2316 ┄          goto #2317
2317 ─          goto #2318
2318 ─ %6672  = Base.mul_int(%6651, 4)::Int64
│      %6673  = Base.add_int(%6650, %6672)::Int64
│      %6674  = Base.mul_int(%6673, 4)::Int64
│      %6675  = Base.add_int(%6649, %6674)::Int64
│      %6676  = Base.mul_int(%6675, 4)::Int64
│      %6677  = Base.add_int(%6648, %6676)::Int64
│      %6678  = Base.mul_int(%6677, 5)::Int64
│      %6679  = Base.add_int(%6647, %6678)::Int64
│      %6680  = Base.mul_int(8, %6679)::Int64
│      %6681  = Core.bitcast(Core.UInt, %6646)::UInt64
│      %6682  = Base.bitcast(UInt64, %6680)::UInt64
│      %6683  = Base.add_ptr(%6681, %6682)::UInt64
│      %6684  = Core.bitcast(Ptr{Float64}, %6683)::Ptr{Float64}
└─────          goto #2319
2319 ─          Base.pointerset(%6684, %6613, 1, 1)::Ptr{Float64}
└─────          goto #2320
2320 ─          goto #2321
2321 ─ %6689  = (%6536 === 5)::Bool
└─────          goto #2323 if not %6689
2322 ─          goto #2324
2323 ─ %6692  = Base.add_int(%6536, 1)::Int64
└─────          goto #2324
2324 ┄ %6694  = φ (#2323 => %6692)::Int64
│      %6695  = φ (#2323 => %6692)::Int64
│      %6696  = φ (#2322 => true, #2323 => false)::Bool
│      %6697  = Base.not_int(%6696)::Bool
└─────          goto #2326 if not %6697
2325 ─          goto #2283
2326 ┄          goto #2327
2327 ─ %6701  = Base.arrayref(false, %3651, %5947, %3653)::Float64
│      %6702  = Base.copysign_float(0.0, %6701)::Float64
│      %6703  = Core.ifelse(true, %6701, %6702)::Float64
└─────          goto #2371 if not true
2328 ┄ %6705  = φ (#2327 => 1, #2370 => %6864)::Int64
│      %6706  = φ (#2327 => 1, #2370 => %6865)::Int64
│      %6707  = Base.getfield(%6529, %6705, true)::Float64
└─────          goto #2333 if not true
2329 ─ %6709  = Core.tuple(%6705, %3659, %3656, %5947, %3646)::NTuple{5, Int64}
│      %6710  = StrideArraysCore.getfield(%3549, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %6711  = Core.getfield(%6710, 5)::Int64
│      %6712  = Base.bitcast(UInt64, %6711)::UInt64
│      %6713  = Base.bitcast(Int64, %6712)::Int64
│      %6714  = Base.sle_int(1, %6705)::Bool
│      %6715  = Base.sle_int(%6705, 5)::Bool
│      %6716  = Base.and_int(%6714, %6715)::Bool
│      %6717  = Base.sle_int(1, %3659)::Bool
│      %6718  = Base.sle_int(%3659, 4)::Bool
│      %6719  = Base.and_int(%6717, %6718)::Bool
│      %6720  = Base.sle_int(1, %3656)::Bool
│      %6721  = Base.sle_int(%3656, 4)::Bool
│      %6722  = Base.and_int(%6720, %6721)::Bool
│      %6723  = Base.sle_int(1, %5947)::Bool
│      %6724  = Base.sle_int(%5947, 4)::Bool
│      %6725  = Base.and_int(%6723, %6724)::Bool
│      %6726  = Base.sub_int(%3646, 1)::Int64
│      %6727  = Base.bitcast(UInt64, %6726)::UInt64
│      %6728  = Base.bitcast(UInt64, %6713)::UInt64
│      %6729  = Base.ult_int(%6727, %6728)::Bool
│      %6730  = Base.and_int(%6729, true)::Bool
│      %6731  = Base.and_int(%6725, %6730)::Bool
│      %6732  = Base.and_int(%6722, %6731)::Bool
│      %6733  = Base.and_int(%6719, %6732)::Bool
│      %6734  = Base.and_int(%6716, %6733)::Bool
└─────          goto #2331 if not %6734
2330 ─          goto #2332
2331 ─          invoke Base.throw_boundserror(%3549::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %6709::NTuple{5, Int64})::Union{}
└─────          unreachable
2332 ─          nothing::Nothing
2333 ┄ %6740  = StrideArraysCore.getfield(%3549, :ptr)::Ptr{Float64}
│      %6741  = Base.sub_int(%6705, 1)::Int64
│      %6742  = Base.sub_int(%3659, 1)::Int64
│      %6743  = Base.sub_int(%3656, 1)::Int64
│      %6744  = Base.sub_int(%5947, 1)::Int64
│      %6745  = Base.sub_int(%3646, 1)::Int64
└─────          goto #2342 if not true
2334 ┄ %6747  = φ (#2333 => 2, #2341 => %6759)::Int64
│      %6748  = Base.sle_int(1, %6747)::Bool
└─────          goto #2336 if not %6748
2335 ─ %6750  = Base.sle_int(%6747, 5)::Bool
└─────          goto #2337
2336 ─          nothing::Nothing
2337 ┄ %6753  = φ (#2335 => %6750, #2336 => false)::Bool
└─────          goto #2339 if not %6753
2338 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %6747, true)::Static.True
│      %6756  = Base.add_int(%6747, 1)::Int64
└─────          goto #2340
2339 ─          goto #2340
2340 ┄ %6759  = φ (#2338 => %6756)::Int64
│      %6760  = φ (#2338 => false, #2339 => true)::Bool
│      %6761  = Base.not_int(%6760)::Bool
└─────          goto #2342 if not %6761
2341 ─          goto #2334
2342 ┄          goto #2343
2343 ─          goto #2344
2344 ─ %6766  = Base.mul_int(%6745, 4)::Int64
│      %6767  = Base.add_int(%6744, %6766)::Int64
│      %6768  = Base.mul_int(%6767, 4)::Int64
│      %6769  = Base.add_int(%6743, %6768)::Int64
│      %6770  = Base.mul_int(%6769, 4)::Int64
│      %6771  = Base.add_int(%6742, %6770)::Int64
│      %6772  = Base.mul_int(%6771, 5)::Int64
│      %6773  = Base.add_int(%6741, %6772)::Int64
│      %6774  = Base.mul_int(8, %6773)::Int64
│      %6775  = Core.bitcast(Core.UInt, %6740)::UInt64
│      %6776  = Base.bitcast(UInt64, %6774)::UInt64
│      %6777  = Base.add_ptr(%6775, %6776)::UInt64
│      %6778  = Core.bitcast(Ptr{Float64}, %6777)::Ptr{Float64}
└─────          goto #2345
2345 ─ %6780  = Base.pointerref(%6778, 1, 1)::Float64
└─────          goto #2346
2346 ─          goto #2347
2347 ─ %6783  = Base.muladd_float(%6703, %6707, %6780)::Float64
└─────          goto #2352 if not true
2348 ─ %6785  = Core.tuple(%6705, %3659, %3656, %5947, %3646)::NTuple{5, Int64}
│      %6786  = StrideArraysCore.getfield(%3549, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %6787  = Core.getfield(%6786, 5)::Int64
│      %6788  = Base.bitcast(UInt64, %6787)::UInt64
│      %6789  = Base.bitcast(Int64, %6788)::Int64
│      %6790  = Base.sle_int(1, %6705)::Bool
│      %6791  = Base.sle_int(%6705, 5)::Bool
│      %6792  = Base.and_int(%6790, %6791)::Bool
│      %6793  = Base.sle_int(1, %3659)::Bool
│      %6794  = Base.sle_int(%3659, 4)::Bool
│      %6795  = Base.and_int(%6793, %6794)::Bool
│      %6796  = Base.sle_int(1, %3656)::Bool
│      %6797  = Base.sle_int(%3656, 4)::Bool
│      %6798  = Base.and_int(%6796, %6797)::Bool
│      %6799  = Base.sle_int(1, %5947)::Bool
│      %6800  = Base.sle_int(%5947, 4)::Bool
│      %6801  = Base.and_int(%6799, %6800)::Bool
│      %6802  = Base.sub_int(%3646, 1)::Int64
│      %6803  = Base.bitcast(UInt64, %6802)::UInt64
│      %6804  = Base.bitcast(UInt64, %6789)::UInt64
│      %6805  = Base.ult_int(%6803, %6804)::Bool
│      %6806  = Base.and_int(%6805, true)::Bool
│      %6807  = Base.and_int(%6801, %6806)::Bool
│      %6808  = Base.and_int(%6798, %6807)::Bool
│      %6809  = Base.and_int(%6795, %6808)::Bool
│      %6810  = Base.and_int(%6792, %6809)::Bool
└─────          goto #2350 if not %6810
2349 ─          goto #2351
2350 ─          invoke Base.throw_boundserror(%3549::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %6785::NTuple{5, Int64})::Union{}
└─────          unreachable
2351 ─          nothing::Nothing
2352 ┄ %6816  = StrideArraysCore.getfield(%3549, :ptr)::Ptr{Float64}
│      %6817  = Base.sub_int(%6705, 1)::Int64
│      %6818  = Base.sub_int(%3659, 1)::Int64
│      %6819  = Base.sub_int(%3656, 1)::Int64
│      %6820  = Base.sub_int(%5947, 1)::Int64
│      %6821  = Base.sub_int(%3646, 1)::Int64
└─────          goto #2361 if not true
2353 ┄ %6823  = φ (#2352 => 2, #2360 => %6835)::Int64
│      %6824  = Base.sle_int(1, %6823)::Bool
└─────          goto #2355 if not %6824
2354 ─ %6826  = Base.sle_int(%6823, 5)::Bool
└─────          goto #2356
2355 ─          nothing::Nothing
2356 ┄ %6829  = φ (#2354 => %6826, #2355 => false)::Bool
└─────          goto #2358 if not %6829
2357 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %6823, true)::Static.True
│      %6832  = Base.add_int(%6823, 1)::Int64
└─────          goto #2359
2358 ─          goto #2359
2359 ┄ %6835  = φ (#2357 => %6832)::Int64
│      %6836  = φ (#2357 => false, #2358 => true)::Bool
│      %6837  = Base.not_int(%6836)::Bool
└─────          goto #2361 if not %6837
2360 ─          goto #2353
2361 ┄          goto #2362
2362 ─          goto #2363
2363 ─ %6842  = Base.mul_int(%6821, 4)::Int64
│      %6843  = Base.add_int(%6820, %6842)::Int64
│      %6844  = Base.mul_int(%6843, 4)::Int64
│      %6845  = Base.add_int(%6819, %6844)::Int64
│      %6846  = Base.mul_int(%6845, 4)::Int64
│      %6847  = Base.add_int(%6818, %6846)::Int64
│      %6848  = Base.mul_int(%6847, 5)::Int64
│      %6849  = Base.add_int(%6817, %6848)::Int64
│      %6850  = Base.mul_int(8, %6849)::Int64
│      %6851  = Core.bitcast(Core.UInt, %6816)::UInt64
│      %6852  = Base.bitcast(UInt64, %6850)::UInt64
│      %6853  = Base.add_ptr(%6851, %6852)::UInt64
│      %6854  = Core.bitcast(Ptr{Float64}, %6853)::Ptr{Float64}
└─────          goto #2364
2364 ─          Base.pointerset(%6854, %6783, 1, 1)::Ptr{Float64}
└─────          goto #2365
2365 ─          goto #2366
2366 ─ %6859  = (%6706 === 5)::Bool
└─────          goto #2368 if not %6859
2367 ─          goto #2369
2368 ─ %6862  = Base.add_int(%6706, 1)::Int64
└─────          goto #2369
2369 ┄ %6864  = φ (#2368 => %6862)::Int64
│      %6865  = φ (#2368 => %6862)::Int64
│      %6866  = φ (#2367 => true, #2368 => false)::Bool
│      %6867  = Base.not_int(%6866)::Bool
└─────          goto #2371 if not %6867
2370 ─          goto #2328
2371 ┄          goto #2372
2372 ─ %6871  = (%5948 === %5935)::Bool
└─────          goto #2374 if not %6871
2373 ─          goto #2375
2374 ─ %6874  = Base.add_int(%5948, 1)::Int64
└─────          goto #2375
2375 ┄ %6876  = φ (#2374 => %6874)::Int64
│      %6877  = φ (#2374 => %6874)::Int64
│      %6878  = φ (#2373 => true, #2374 => false)::Bool
│      %6879  = Base.not_int(%6878)::Bool
└─────          goto #2377 if not %6879
2376 ─          goto #2043
2377 ┄ %6882  = (%3660 === 4)::Bool
└─────          goto #2379 if not %6882
2378 ─          goto #2380
2379 ─ %6885  = Base.add_int(%3660, 1)::Int64
└─────          goto #2380
2380 ┄ %6887  = φ (#2379 => %6885)::Int64
│      %6888  = φ (#2379 => %6885)::Int64
│      %6889  = φ (#2378 => true, #2379 => false)::Bool
│      %6890  = Base.not_int(%6889)::Bool
└─────          goto #2382 if not %6890
2381 ─          goto #1246
2382 ┄ %6893  = (%3657 === 4)::Bool
└─────          goto #2384 if not %6893
2383 ─          goto #2385
2384 ─ %6896  = Base.add_int(%3657, 1)::Int64
└─────          goto #2385
2385 ┄ %6898  = φ (#2384 => %6896)::Int64
│      %6899  = φ (#2384 => %6896)::Int64
│      %6900  = φ (#2383 => true, #2384 => false)::Bool
│      %6901  = Base.not_int(%6900)::Bool
└─────          goto #2387 if not %6901
2386 ─          goto #1245
2387 ┄ %6904  = (%3654 === 4)::Bool
└─────          goto #2389 if not %6904
2388 ─          goto #2390
2389 ─ %6907  = Base.add_int(%3654, 1)::Int64
└─────          goto #2390
2390 ┄ %6909  = φ (#2389 => %6907)::Int64
│      %6910  = φ (#2389 => %6907)::Int64
│      %6911  = φ (#2388 => true, #2389 => false)::Bool
│      %6912  = Base.not_int(%6911)::Bool
└─────          goto #2392 if not %6912
2391 ─          goto #1244
2392 ┄          goto #2393
2393 ─          goto #2394
2394 ─          goto #2395
2395 ─          nothing::Nothing
2396 ┄ %6919  = Base.add_int(%3646, 1)::Int64
└─────          goto #1241
2397 ─          goto #2398
2398 ─          goto #2400
2399 ─          nothing::Nothing
2400 ┄          goto #2402
2401 ─          nothing::Nothing
2402 ┄          goto #2403
2403 ─          goto #2405
2404 ─          nothing::Nothing
2405 ┄          goto #2407
2406 ─          nothing::Nothing
2407 ┄          goto #2408
2408 ─          goto #2409
2409 ─          goto #2410
2410 ─          goto #2411
2411 ─          goto #2412
2412 ─          goto #2424 if not true
2413 ─          nothing::Nothing
2414 ┄ %6938  = φ (#2413 => 0x00000000, #2422 => %6948)::UInt32
│      %6939  = φ (#2413 => %3533, #2422 => %6947)::UInt64
│      %6940  = (%6939 === 0x0000000000000000)::Bool
│      %6941  = Base.not_int(%6940)::Bool
└─────          goto #2423 if not %6941
2415 ─ %6943  = Base.cttz_int(%6939)::UInt64
│      %6944  = Base.bitcast(Int64, %6943)::Int64
│      %6945  = Base.trunc_int(UInt32, %6944)::UInt32
│      %6946  = Base.add_int(%6945, 0x00000001)::UInt32
│      %6947  = Base.lshr_int(%6939, %6946)::UInt64
│      %6948  = Base.add_int(%6938, %6946)::UInt32
│      %6949  = ThreadingUtilities.THREADPOOLPTR::Base.RefValue{Ptr{UInt64}}
│      %6950  = Base.getfield(%6949, :x)::Ptr{UInt64}
│      %6951  = Base.mul_int(%6948, 0x00000200)::UInt32
│      %6952  = Core.bitcast(Core.UInt, %6950)::UInt64
│      %6953  = Core.zext_int(Core.UInt64, %6951)::UInt64
│      %6954  = Base.add_ptr(%6952, %6953)::UInt64
└───── %6955  = Core.bitcast(Ptr{UInt64}, %6954)::Ptr{UInt64}
2416 ┄ %6956  = φ (#2415 => 0x00000000, #2419 => %6964)::UInt32
│      %6957  = Base.bitcast(Ptr{UInt32}, %6955)::Ptr{UInt32}
│      %6958  = Base.llvmcall("%p = inttoptr i64 %0 to i32*\n%v = load atomic i32, i32* %p acquire, align 16\nret i32 %v\n", UInt32, Tuple{Ptr{UInt32}}, %6957)::UInt32
│      %6959  = Base.bitcast(ThreadingUtilities.ThreadState, %6958)::ThreadingUtilities.ThreadState
│      %6960  = ThreadingUtilities.TASK::ThreadingUtilities.ThreadState
│      %6961  = (%6959 === %6960)::Bool
└─────          goto #2420 if not %6961
2417 ─          $(Expr(:foreigncall, :(:jl_cpu_pause), Nothing, svec(), 0, :(:ccall)))::Nothing
│      %6964  = Base.add_int(%6956, 0x00000001)::UInt32
│      %6965  = Base.ult_int(0x00010000, %6964)::Bool
└─────          goto #2419 if not %6965
2418 ─          invoke ThreadingUtilities.checktask(%6948::UInt32)::Bool
2419 ┄          goto #2416
2420 ─          goto #2421
2421 ─          goto #2422
2422 ─          goto #2414
2423 ─          nothing::Nothing
2424 ┄ %6973  = PolyesterWeave.WORKERS::Base.RefValue{NTuple{8, UInt64}}
│      %6974  = $(Expr(:foreigncall, :(:jl_value_ptr), Ptr{Nothing}, svec(Any), 0, :(:ccall), :(%6973)))::Ptr{Nothing}
│      %6975  = Base.bitcast(Ptr{UInt64}, %6974)::Ptr{UInt64}
│               Base.llvmcall("%p = inttoptr i64 %0 to i64*\n%v = atomicrmw or i64* %p, i64 %1 acq_rel\nret i64 %v\n", UInt64, Tuple{Ptr{UInt64}, UInt64}, %6975, %3534)::UInt64
│               $(Expr(:gc_preserve_end, :(%3556)))
└─────          goto #2425
2425 ─          goto #3618
2426 ┄ %6980  = Base.bitcast(Int64, %3476)::Int64
│      %6981  = Base.mul_int(%6980, 1)::Int64
│      %6982  = Base.add_int(%6981, 1)::Int64
└───── %6983  = Base.sub_int(%6982, 1)::Int64
2427 ┄ %6984  = φ (#2426 => 1, #3614 => %10417)::Int64
│      %6985  = Base.sle_int(%6984, %6983)::Bool
└─────          goto #3615 if not %6985
2428 ─          goto #3614 if not true
2429 ─ %6988  = Base.getfield(dg, :basis)::LobattoLegendreBasis{Float64, 4, SVector{4, Float64}, Matrix{Float64}, Matrix{Float64}}
│      %6989  = Base.getfield(%6988, :derivative_split)::Matrix{Float64}
└─────          goto #3610 if not true
2430 ┄ %6991  = φ (#2429 => 1, #3609 => %10407)::Int64
│      %6992  = φ (#2429 => 1, #3609 => %10408)::Int64
└─────          goto #3605 if not true
2431 ┄ %6994  = φ (#2430 => 1, #3604 => %10397)::Int64
│      %6995  = φ (#2430 => 1, #3604 => %10396)::Int64
└─────          goto #3600 if not true
2432 ┄ %6997  = φ (#2431 => 1, #3599 => %10386)::Int64
│      %6998  = φ (#2431 => 1, #3599 => %10385)::Int64
│      %6999  = StrideArraysCore.getfield(u, :data)::Vector{Float64}
│      %7000  = $(Expr(:gc_preserve_begin, :(%6999)))
│      %7001  = StrideArraysCore.getfield(u, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #2437 if not true
2433 ─ %7003  = Core.tuple(1, %6997, %6994, %6991, %6984)::NTuple{5, Int64}
│      %7004  = StrideArraysCore.getfield(%7001, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %7005  = Core.getfield(%7004, 5)::Int64
│      %7006  = Base.bitcast(UInt64, %7005)::UInt64
│      %7007  = Base.bitcast(Int64, %7006)::Int64
│      %7008  = Base.sle_int(1, %6997)::Bool
│      %7009  = Base.sle_int(%6997, 4)::Bool
│      %7010  = Base.and_int(%7008, %7009)::Bool
│      %7011  = Base.sle_int(1, %6994)::Bool
│      %7012  = Base.sle_int(%6994, 4)::Bool
│      %7013  = Base.and_int(%7011, %7012)::Bool
│      %7014  = Base.sle_int(1, %6991)::Bool
│      %7015  = Base.sle_int(%6991, 4)::Bool
│      %7016  = Base.and_int(%7014, %7015)::Bool
│      %7017  = Base.sub_int(%6984, 1)::Int64
│      %7018  = Base.bitcast(UInt64, %7017)::UInt64
│      %7019  = Base.bitcast(UInt64, %7007)::UInt64
│      %7020  = Base.ult_int(%7018, %7019)::Bool
│      %7021  = Base.and_int(%7020, true)::Bool
│      %7022  = Base.and_int(%7016, %7021)::Bool
│      %7023  = Base.and_int(%7013, %7022)::Bool
│      %7024  = Base.and_int(%7010, %7023)::Bool
│      %7025  = Base.and_int(true, %7024)::Bool
└─────          goto #2435 if not %7025
2434 ─          goto #2436
2435 ─          invoke Base.throw_boundserror(%7001::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %7003::NTuple{5, Int64})::Union{}
└─────          unreachable
2436 ─          nothing::Nothing
2437 ┄ %7031  = StrideArraysCore.getfield(%7001, :ptr)::Ptr{Float64}
│      %7032  = Base.sub_int(%6997, 1)::Int64
│      %7033  = Base.sub_int(%6994, 1)::Int64
│      %7034  = Base.sub_int(%6991, 1)::Int64
│      %7035  = Base.sub_int(%6984, 1)::Int64
└─────          goto #2446 if not true
2438 ┄ %7037  = φ (#2437 => 2, #2445 => %7049)::Int64
│      %7038  = Base.sle_int(1, %7037)::Bool
└─────          goto #2440 if not %7038
2439 ─ %7040  = Base.sle_int(%7037, 5)::Bool
└─────          goto #2441
2440 ─          nothing::Nothing
2441 ┄ %7043  = φ (#2439 => %7040, #2440 => false)::Bool
└─────          goto #2443 if not %7043
2442 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %7037, true)::Static.True
│      %7046  = Base.add_int(%7037, 1)::Int64
└─────          goto #2444
2443 ─          goto #2444
2444 ┄ %7049  = φ (#2442 => %7046)::Int64
│      %7050  = φ (#2442 => false, #2443 => true)::Bool
│      %7051  = Base.not_int(%7050)::Bool
└─────          goto #2446 if not %7051
2445 ─          goto #2438
2446 ┄          goto #2447
2447 ─          goto #2448
2448 ─ %7056  = Base.mul_int(%7035, 4)::Int64
│      %7057  = Base.add_int(%7034, %7056)::Int64
│      %7058  = Base.mul_int(%7057, 4)::Int64
│      %7059  = Base.add_int(%7033, %7058)::Int64
│      %7060  = Base.mul_int(%7059, 4)::Int64
│      %7061  = Base.add_int(%7032, %7060)::Int64
│      %7062  = Base.mul_int(%7061, 5)::Int64
│      %7063  = Base.add_int(0, %7062)::Int64
│      %7064  = Base.mul_int(8, %7063)::Int64
│      %7065  = Core.bitcast(Core.UInt, %7031)::UInt64
│      %7066  = Base.bitcast(UInt64, %7064)::UInt64
│      %7067  = Base.add_ptr(%7065, %7066)::UInt64
│      %7068  = Core.bitcast(Ptr{Float64}, %7067)::Ptr{Float64}
└─────          goto #2449
2449 ─ %7070  = Base.pointerref(%7068, 1, 1)::Float64
└─────          goto #2450
2450 ─          goto #2451
2451 ─          $(Expr(:gc_preserve_end, :(%7000)))
└─────          goto #2452
2452 ─          goto #2453
2453 ─ %7076  = StrideArraysCore.getfield(u, :data)::Vector{Float64}
│      %7077  = $(Expr(:gc_preserve_begin, :(%7076)))
│      %7078  = StrideArraysCore.getfield(u, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #2458 if not true
2454 ─ %7080  = Core.tuple(2, %6997, %6994, %6991, %6984)::NTuple{5, Int64}
│      %7081  = StrideArraysCore.getfield(%7078, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %7082  = Core.getfield(%7081, 5)::Int64
│      %7083  = Base.bitcast(UInt64, %7082)::UInt64
│      %7084  = Base.bitcast(Int64, %7083)::Int64
│      %7085  = Base.sle_int(1, %6997)::Bool
│      %7086  = Base.sle_int(%6997, 4)::Bool
│      %7087  = Base.and_int(%7085, %7086)::Bool
│      %7088  = Base.sle_int(1, %6994)::Bool
│      %7089  = Base.sle_int(%6994, 4)::Bool
│      %7090  = Base.and_int(%7088, %7089)::Bool
│      %7091  = Base.sle_int(1, %6991)::Bool
│      %7092  = Base.sle_int(%6991, 4)::Bool
│      %7093  = Base.and_int(%7091, %7092)::Bool
│      %7094  = Base.sub_int(%6984, 1)::Int64
│      %7095  = Base.bitcast(UInt64, %7094)::UInt64
│      %7096  = Base.bitcast(UInt64, %7084)::UInt64
│      %7097  = Base.ult_int(%7095, %7096)::Bool
│      %7098  = Base.and_int(%7097, true)::Bool
│      %7099  = Base.and_int(%7093, %7098)::Bool
│      %7100  = Base.and_int(%7090, %7099)::Bool
│      %7101  = Base.and_int(%7087, %7100)::Bool
│      %7102  = Base.and_int(true, %7101)::Bool
└─────          goto #2456 if not %7102
2455 ─          goto #2457
2456 ─          invoke Base.throw_boundserror(%7078::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %7080::NTuple{5, Int64})::Union{}
└─────          unreachable
2457 ─          nothing::Nothing
2458 ┄ %7108  = StrideArraysCore.getfield(%7078, :ptr)::Ptr{Float64}
│      %7109  = Base.sub_int(%6997, 1)::Int64
│      %7110  = Base.sub_int(%6994, 1)::Int64
│      %7111  = Base.sub_int(%6991, 1)::Int64
│      %7112  = Base.sub_int(%6984, 1)::Int64
└─────          goto #2467 if not true
2459 ┄ %7114  = φ (#2458 => 2, #2466 => %7126)::Int64
│      %7115  = Base.sle_int(1, %7114)::Bool
└─────          goto #2461 if not %7115
2460 ─ %7117  = Base.sle_int(%7114, 5)::Bool
└─────          goto #2462
2461 ─          nothing::Nothing
2462 ┄ %7120  = φ (#2460 => %7117, #2461 => false)::Bool
└─────          goto #2464 if not %7120
2463 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %7114, true)::Static.True
│      %7123  = Base.add_int(%7114, 1)::Int64
└─────          goto #2465
2464 ─          goto #2465
2465 ┄ %7126  = φ (#2463 => %7123)::Int64
│      %7127  = φ (#2463 => false, #2464 => true)::Bool
│      %7128  = Base.not_int(%7127)::Bool
└─────          goto #2467 if not %7128
2466 ─          goto #2459
2467 ┄          goto #2468
2468 ─          goto #2469
2469 ─ %7133  = Base.mul_int(%7112, 4)::Int64
│      %7134  = Base.add_int(%7111, %7133)::Int64
│      %7135  = Base.mul_int(%7134, 4)::Int64
│      %7136  = Base.add_int(%7110, %7135)::Int64
│      %7137  = Base.mul_int(%7136, 4)::Int64
│      %7138  = Base.add_int(%7109, %7137)::Int64
│      %7139  = Base.mul_int(%7138, 5)::Int64
│      %7140  = Base.add_int(1, %7139)::Int64
│      %7141  = Base.mul_int(8, %7140)::Int64
│      %7142  = Core.bitcast(Core.UInt, %7108)::UInt64
│      %7143  = Base.bitcast(UInt64, %7141)::UInt64
│      %7144  = Base.add_ptr(%7142, %7143)::UInt64
│      %7145  = Core.bitcast(Ptr{Float64}, %7144)::Ptr{Float64}
└─────          goto #2470
2470 ─ %7147  = Base.pointerref(%7145, 1, 1)::Float64
└─────          goto #2471
2471 ─          goto #2472
2472 ─          $(Expr(:gc_preserve_end, :(%7077)))
└─────          goto #2473
2473 ─          goto #2474
2474 ─ %7153  = StrideArraysCore.getfield(u, :data)::Vector{Float64}
│      %7154  = $(Expr(:gc_preserve_begin, :(%7153)))
│      %7155  = StrideArraysCore.getfield(u, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #2479 if not true
2475 ─ %7157  = Core.tuple(3, %6997, %6994, %6991, %6984)::NTuple{5, Int64}
│      %7158  = StrideArraysCore.getfield(%7155, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %7159  = Core.getfield(%7158, 5)::Int64
│      %7160  = Base.bitcast(UInt64, %7159)::UInt64
│      %7161  = Base.bitcast(Int64, %7160)::Int64
│      %7162  = Base.sle_int(1, %6997)::Bool
│      %7163  = Base.sle_int(%6997, 4)::Bool
│      %7164  = Base.and_int(%7162, %7163)::Bool
│      %7165  = Base.sle_int(1, %6994)::Bool
│      %7166  = Base.sle_int(%6994, 4)::Bool
│      %7167  = Base.and_int(%7165, %7166)::Bool
│      %7168  = Base.sle_int(1, %6991)::Bool
│      %7169  = Base.sle_int(%6991, 4)::Bool
│      %7170  = Base.and_int(%7168, %7169)::Bool
│      %7171  = Base.sub_int(%6984, 1)::Int64
│      %7172  = Base.bitcast(UInt64, %7171)::UInt64
│      %7173  = Base.bitcast(UInt64, %7161)::UInt64
│      %7174  = Base.ult_int(%7172, %7173)::Bool
│      %7175  = Base.and_int(%7174, true)::Bool
│      %7176  = Base.and_int(%7170, %7175)::Bool
│      %7177  = Base.and_int(%7167, %7176)::Bool
│      %7178  = Base.and_int(%7164, %7177)::Bool
│      %7179  = Base.and_int(true, %7178)::Bool
└─────          goto #2477 if not %7179
2476 ─          goto #2478
2477 ─          invoke Base.throw_boundserror(%7155::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %7157::NTuple{5, Int64})::Union{}
└─────          unreachable
2478 ─          nothing::Nothing
2479 ┄ %7185  = StrideArraysCore.getfield(%7155, :ptr)::Ptr{Float64}
│      %7186  = Base.sub_int(%6997, 1)::Int64
│      %7187  = Base.sub_int(%6994, 1)::Int64
│      %7188  = Base.sub_int(%6991, 1)::Int64
│      %7189  = Base.sub_int(%6984, 1)::Int64
└─────          goto #2488 if not true
2480 ┄ %7191  = φ (#2479 => 2, #2487 => %7203)::Int64
│      %7192  = Base.sle_int(1, %7191)::Bool
└─────          goto #2482 if not %7192
2481 ─ %7194  = Base.sle_int(%7191, 5)::Bool
└─────          goto #2483
2482 ─          nothing::Nothing
2483 ┄ %7197  = φ (#2481 => %7194, #2482 => false)::Bool
└─────          goto #2485 if not %7197
2484 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %7191, true)::Static.True
│      %7200  = Base.add_int(%7191, 1)::Int64
└─────          goto #2486
2485 ─          goto #2486
2486 ┄ %7203  = φ (#2484 => %7200)::Int64
│      %7204  = φ (#2484 => false, #2485 => true)::Bool
│      %7205  = Base.not_int(%7204)::Bool
└─────          goto #2488 if not %7205
2487 ─          goto #2480
2488 ┄          goto #2489
2489 ─          goto #2490
2490 ─ %7210  = Base.mul_int(%7189, 4)::Int64
│      %7211  = Base.add_int(%7188, %7210)::Int64
│      %7212  = Base.mul_int(%7211, 4)::Int64
│      %7213  = Base.add_int(%7187, %7212)::Int64
│      %7214  = Base.mul_int(%7213, 4)::Int64
│      %7215  = Base.add_int(%7186, %7214)::Int64
│      %7216  = Base.mul_int(%7215, 5)::Int64
│      %7217  = Base.add_int(2, %7216)::Int64
│      %7218  = Base.mul_int(8, %7217)::Int64
│      %7219  = Core.bitcast(Core.UInt, %7185)::UInt64
│      %7220  = Base.bitcast(UInt64, %7218)::UInt64
│      %7221  = Base.add_ptr(%7219, %7220)::UInt64
│      %7222  = Core.bitcast(Ptr{Float64}, %7221)::Ptr{Float64}
└─────          goto #2491
2491 ─ %7224  = Base.pointerref(%7222, 1, 1)::Float64
└─────          goto #2492
2492 ─          goto #2493
2493 ─          $(Expr(:gc_preserve_end, :(%7154)))
└─────          goto #2494
2494 ─          goto #2495
2495 ─ %7230  = StrideArraysCore.getfield(u, :data)::Vector{Float64}
│      %7231  = $(Expr(:gc_preserve_begin, :(%7230)))
│      %7232  = StrideArraysCore.getfield(u, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #2500 if not true
2496 ─ %7234  = Core.tuple(4, %6997, %6994, %6991, %6984)::NTuple{5, Int64}
│      %7235  = StrideArraysCore.getfield(%7232, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %7236  = Core.getfield(%7235, 5)::Int64
│      %7237  = Base.bitcast(UInt64, %7236)::UInt64
│      %7238  = Base.bitcast(Int64, %7237)::Int64
│      %7239  = Base.sle_int(1, %6997)::Bool
│      %7240  = Base.sle_int(%6997, 4)::Bool
│      %7241  = Base.and_int(%7239, %7240)::Bool
│      %7242  = Base.sle_int(1, %6994)::Bool
│      %7243  = Base.sle_int(%6994, 4)::Bool
│      %7244  = Base.and_int(%7242, %7243)::Bool
│      %7245  = Base.sle_int(1, %6991)::Bool
│      %7246  = Base.sle_int(%6991, 4)::Bool
│      %7247  = Base.and_int(%7245, %7246)::Bool
│      %7248  = Base.sub_int(%6984, 1)::Int64
│      %7249  = Base.bitcast(UInt64, %7248)::UInt64
│      %7250  = Base.bitcast(UInt64, %7238)::UInt64
│      %7251  = Base.ult_int(%7249, %7250)::Bool
│      %7252  = Base.and_int(%7251, true)::Bool
│      %7253  = Base.and_int(%7247, %7252)::Bool
│      %7254  = Base.and_int(%7244, %7253)::Bool
│      %7255  = Base.and_int(%7241, %7254)::Bool
│      %7256  = Base.and_int(true, %7255)::Bool
└─────          goto #2498 if not %7256
2497 ─          goto #2499
2498 ─          invoke Base.throw_boundserror(%7232::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %7234::NTuple{5, Int64})::Union{}
└─────          unreachable
2499 ─          nothing::Nothing
2500 ┄ %7262  = StrideArraysCore.getfield(%7232, :ptr)::Ptr{Float64}
│      %7263  = Base.sub_int(%6997, 1)::Int64
│      %7264  = Base.sub_int(%6994, 1)::Int64
│      %7265  = Base.sub_int(%6991, 1)::Int64
│      %7266  = Base.sub_int(%6984, 1)::Int64
└─────          goto #2509 if not true
2501 ┄ %7268  = φ (#2500 => 2, #2508 => %7280)::Int64
│      %7269  = Base.sle_int(1, %7268)::Bool
└─────          goto #2503 if not %7269
2502 ─ %7271  = Base.sle_int(%7268, 5)::Bool
└─────          goto #2504
2503 ─          nothing::Nothing
2504 ┄ %7274  = φ (#2502 => %7271, #2503 => false)::Bool
└─────          goto #2506 if not %7274
2505 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %7268, true)::Static.True
│      %7277  = Base.add_int(%7268, 1)::Int64
└─────          goto #2507
2506 ─          goto #2507
2507 ┄ %7280  = φ (#2505 => %7277)::Int64
│      %7281  = φ (#2505 => false, #2506 => true)::Bool
│      %7282  = Base.not_int(%7281)::Bool
└─────          goto #2509 if not %7282
2508 ─          goto #2501
2509 ┄          goto #2510
2510 ─          goto #2511
2511 ─ %7287  = Base.mul_int(%7266, 4)::Int64
│      %7288  = Base.add_int(%7265, %7287)::Int64
│      %7289  = Base.mul_int(%7288, 4)::Int64
│      %7290  = Base.add_int(%7264, %7289)::Int64
│      %7291  = Base.mul_int(%7290, 4)::Int64
│      %7292  = Base.add_int(%7263, %7291)::Int64
│      %7293  = Base.mul_int(%7292, 5)::Int64
│      %7294  = Base.add_int(3, %7293)::Int64
│      %7295  = Base.mul_int(8, %7294)::Int64
│      %7296  = Core.bitcast(Core.UInt, %7262)::UInt64
│      %7297  = Base.bitcast(UInt64, %7295)::UInt64
│      %7298  = Base.add_ptr(%7296, %7297)::UInt64
│      %7299  = Core.bitcast(Ptr{Float64}, %7298)::Ptr{Float64}
└─────          goto #2512
2512 ─ %7301  = Base.pointerref(%7299, 1, 1)::Float64
└─────          goto #2513
2513 ─          goto #2514
2514 ─          $(Expr(:gc_preserve_end, :(%7231)))
└─────          goto #2515
2515 ─          goto #2516
2516 ─ %7307  = StrideArraysCore.getfield(u, :data)::Vector{Float64}
│      %7308  = $(Expr(:gc_preserve_begin, :(%7307)))
│      %7309  = StrideArraysCore.getfield(u, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #2521 if not true
2517 ─ %7311  = Core.tuple(5, %6997, %6994, %6991, %6984)::NTuple{5, Int64}
│      %7312  = StrideArraysCore.getfield(%7309, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %7313  = Core.getfield(%7312, 5)::Int64
│      %7314  = Base.bitcast(UInt64, %7313)::UInt64
│      %7315  = Base.bitcast(Int64, %7314)::Int64
│      %7316  = Base.sle_int(1, %6997)::Bool
│      %7317  = Base.sle_int(%6997, 4)::Bool
│      %7318  = Base.and_int(%7316, %7317)::Bool
│      %7319  = Base.sle_int(1, %6994)::Bool
│      %7320  = Base.sle_int(%6994, 4)::Bool
│      %7321  = Base.and_int(%7319, %7320)::Bool
│      %7322  = Base.sle_int(1, %6991)::Bool
│      %7323  = Base.sle_int(%6991, 4)::Bool
│      %7324  = Base.and_int(%7322, %7323)::Bool
│      %7325  = Base.sub_int(%6984, 1)::Int64
│      %7326  = Base.bitcast(UInt64, %7325)::UInt64
│      %7327  = Base.bitcast(UInt64, %7315)::UInt64
│      %7328  = Base.ult_int(%7326, %7327)::Bool
│      %7329  = Base.and_int(%7328, true)::Bool
│      %7330  = Base.and_int(%7324, %7329)::Bool
│      %7331  = Base.and_int(%7321, %7330)::Bool
│      %7332  = Base.and_int(%7318, %7331)::Bool
│      %7333  = Base.and_int(true, %7332)::Bool
└─────          goto #2519 if not %7333
2518 ─          goto #2520
2519 ─          invoke Base.throw_boundserror(%7309::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %7311::NTuple{5, Int64})::Union{}
└─────          unreachable
2520 ─          nothing::Nothing
2521 ┄ %7339  = StrideArraysCore.getfield(%7309, :ptr)::Ptr{Float64}
│      %7340  = Base.sub_int(%6997, 1)::Int64
│      %7341  = Base.sub_int(%6994, 1)::Int64
│      %7342  = Base.sub_int(%6991, 1)::Int64
│      %7343  = Base.sub_int(%6984, 1)::Int64
└─────          goto #2530 if not true
2522 ┄ %7345  = φ (#2521 => 2, #2529 => %7357)::Int64
│      %7346  = Base.sle_int(1, %7345)::Bool
└─────          goto #2524 if not %7346
2523 ─ %7348  = Base.sle_int(%7345, 5)::Bool
└─────          goto #2525
2524 ─          nothing::Nothing
2525 ┄ %7351  = φ (#2523 => %7348, #2524 => false)::Bool
└─────          goto #2527 if not %7351
2526 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %7345, true)::Static.True
│      %7354  = Base.add_int(%7345, 1)::Int64
└─────          goto #2528
2527 ─          goto #2528
2528 ┄ %7357  = φ (#2526 => %7354)::Int64
│      %7358  = φ (#2526 => false, #2527 => true)::Bool
│      %7359  = Base.not_int(%7358)::Bool
└─────          goto #2530 if not %7359
2529 ─          goto #2522
2530 ┄          goto #2531
2531 ─          goto #2532
2532 ─ %7364  = Base.mul_int(%7343, 4)::Int64
│      %7365  = Base.add_int(%7342, %7364)::Int64
│      %7366  = Base.mul_int(%7365, 4)::Int64
│      %7367  = Base.add_int(%7341, %7366)::Int64
│      %7368  = Base.mul_int(%7367, 4)::Int64
│      %7369  = Base.add_int(%7340, %7368)::Int64
│      %7370  = Base.mul_int(%7369, 5)::Int64
│      %7371  = Base.add_int(4, %7370)::Int64
│      %7372  = Base.mul_int(8, %7371)::Int64
│      %7373  = Core.bitcast(Core.UInt, %7339)::UInt64
│      %7374  = Base.bitcast(UInt64, %7372)::UInt64
│      %7375  = Base.add_ptr(%7373, %7374)::UInt64
│      %7376  = Core.bitcast(Ptr{Float64}, %7375)::Ptr{Float64}
└─────          goto #2533
2533 ─ %7378  = Base.pointerref(%7376, 1, 1)::Float64
└─────          goto #2534
2534 ─          goto #2535
2535 ─          $(Expr(:gc_preserve_end, :(%7308)))
└─────          goto #2536
2536 ─          goto #2537
2537 ─          goto #2538
2538 ─          goto #2539
2539 ─ %7386  = Base.add_int(%6997, 1)::Int64
│      %7387  = Base.sle_int(%7386, 4)::Bool
└─────          goto #2541 if not %7387
2540 ─          goto #2542
2541 ─ %7390  = Base.sub_int(%7386, 1)::Int64
└─────          goto #2542
2542 ┄ %7392  = φ (#2540 => 4, #2541 => %7390)::Int64
└─────          goto #2543
2543 ─          goto #2544
2544 ─ %7395  = Base.slt_int(%7392, %7386)::Bool
└─────          goto #2546 if not %7395
2545 ─          goto #2547
2546 ─          goto #2547
2547 ┄ %7399  = φ (#2545 => true, #2546 => false)::Bool
│      %7400  = φ (#2546 => %7386)::Int64
│      %7401  = φ (#2546 => %7386)::Int64
│      %7402  = Base.not_int(%7399)::Bool
└─────          goto #2891 if not %7402
2548 ┄ %7404  = φ (#2547 => %7400, #2890 => %8378)::Int64
│      %7405  = φ (#2547 => %7401, #2890 => %8379)::Int64
│      %7406  = StrideArraysCore.getfield(u, :data)::Vector{Float64}
│      %7407  = $(Expr(:gc_preserve_begin, :(%7406)))
│      %7408  = StrideArraysCore.getfield(u, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #2553 if not true
2549 ─ %7410  = Core.tuple(1, %7404, %6994, %6991, %6984)::NTuple{5, Int64}
│      %7411  = StrideArraysCore.getfield(%7408, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %7412  = Core.getfield(%7411, 5)::Int64
│      %7413  = Base.bitcast(UInt64, %7412)::UInt64
│      %7414  = Base.bitcast(Int64, %7413)::Int64
│      %7415  = Base.sle_int(1, %7404)::Bool
│      %7416  = Base.sle_int(%7404, 4)::Bool
│      %7417  = Base.and_int(%7415, %7416)::Bool
│      %7418  = Base.sle_int(1, %6994)::Bool
│      %7419  = Base.sle_int(%6994, 4)::Bool
│      %7420  = Base.and_int(%7418, %7419)::Bool
│      %7421  = Base.sle_int(1, %6991)::Bool
│      %7422  = Base.sle_int(%6991, 4)::Bool
│      %7423  = Base.and_int(%7421, %7422)::Bool
│      %7424  = Base.sub_int(%6984, 1)::Int64
│      %7425  = Base.bitcast(UInt64, %7424)::UInt64
│      %7426  = Base.bitcast(UInt64, %7414)::UInt64
│      %7427  = Base.ult_int(%7425, %7426)::Bool
│      %7428  = Base.and_int(%7427, true)::Bool
│      %7429  = Base.and_int(%7423, %7428)::Bool
│      %7430  = Base.and_int(%7420, %7429)::Bool
│      %7431  = Base.and_int(%7417, %7430)::Bool
│      %7432  = Base.and_int(true, %7431)::Bool
└─────          goto #2551 if not %7432
2550 ─          goto #2552
2551 ─          invoke Base.throw_boundserror(%7408::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %7410::NTuple{5, Int64})::Union{}
└─────          unreachable
2552 ─          nothing::Nothing
2553 ┄ %7438  = StrideArraysCore.getfield(%7408, :ptr)::Ptr{Float64}
│      %7439  = Base.sub_int(%7404, 1)::Int64
│      %7440  = Base.sub_int(%6994, 1)::Int64
│      %7441  = Base.sub_int(%6991, 1)::Int64
│      %7442  = Base.sub_int(%6984, 1)::Int64
└─────          goto #2562 if not true
2554 ┄ %7444  = φ (#2553 => 2, #2561 => %7456)::Int64
│      %7445  = Base.sle_int(1, %7444)::Bool
└─────          goto #2556 if not %7445
2555 ─ %7447  = Base.sle_int(%7444, 5)::Bool
└─────          goto #2557
2556 ─          nothing::Nothing
2557 ┄ %7450  = φ (#2555 => %7447, #2556 => false)::Bool
└─────          goto #2559 if not %7450
2558 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %7444, true)::Static.True
│      %7453  = Base.add_int(%7444, 1)::Int64
└─────          goto #2560
2559 ─          goto #2560
2560 ┄ %7456  = φ (#2558 => %7453)::Int64
│      %7457  = φ (#2558 => false, #2559 => true)::Bool
│      %7458  = Base.not_int(%7457)::Bool
└─────          goto #2562 if not %7458
2561 ─          goto #2554
2562 ┄          goto #2563
2563 ─          goto #2564
2564 ─ %7463  = Base.mul_int(%7442, 4)::Int64
│      %7464  = Base.add_int(%7441, %7463)::Int64
│      %7465  = Base.mul_int(%7464, 4)::Int64
│      %7466  = Base.add_int(%7440, %7465)::Int64
│      %7467  = Base.mul_int(%7466, 4)::Int64
│      %7468  = Base.add_int(%7439, %7467)::Int64
│      %7469  = Base.mul_int(%7468, 5)::Int64
│      %7470  = Base.add_int(0, %7469)::Int64
│      %7471  = Base.mul_int(8, %7470)::Int64
│      %7472  = Core.bitcast(Core.UInt, %7438)::UInt64
│      %7473  = Base.bitcast(UInt64, %7471)::UInt64
│      %7474  = Base.add_ptr(%7472, %7473)::UInt64
│      %7475  = Core.bitcast(Ptr{Float64}, %7474)::Ptr{Float64}
└─────          goto #2565
2565 ─ %7477  = Base.pointerref(%7475, 1, 1)::Float64
└─────          goto #2566
2566 ─          goto #2567
2567 ─          $(Expr(:gc_preserve_end, :(%7407)))
└─────          goto #2568
2568 ─          goto #2569
2569 ─ %7483  = StrideArraysCore.getfield(u, :data)::Vector{Float64}
│      %7484  = $(Expr(:gc_preserve_begin, :(%7483)))
│      %7485  = StrideArraysCore.getfield(u, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #2574 if not true
2570 ─ %7487  = Core.tuple(2, %7404, %6994, %6991, %6984)::NTuple{5, Int64}
│      %7488  = StrideArraysCore.getfield(%7485, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %7489  = Core.getfield(%7488, 5)::Int64
│      %7490  = Base.bitcast(UInt64, %7489)::UInt64
│      %7491  = Base.bitcast(Int64, %7490)::Int64
│      %7492  = Base.sle_int(1, %7404)::Bool
│      %7493  = Base.sle_int(%7404, 4)::Bool
│      %7494  = Base.and_int(%7492, %7493)::Bool
│      %7495  = Base.sle_int(1, %6994)::Bool
│      %7496  = Base.sle_int(%6994, 4)::Bool
│      %7497  = Base.and_int(%7495, %7496)::Bool
│      %7498  = Base.sle_int(1, %6991)::Bool
│      %7499  = Base.sle_int(%6991, 4)::Bool
│      %7500  = Base.and_int(%7498, %7499)::Bool
│      %7501  = Base.sub_int(%6984, 1)::Int64
│      %7502  = Base.bitcast(UInt64, %7501)::UInt64
│      %7503  = Base.bitcast(UInt64, %7491)::UInt64
│      %7504  = Base.ult_int(%7502, %7503)::Bool
│      %7505  = Base.and_int(%7504, true)::Bool
│      %7506  = Base.and_int(%7500, %7505)::Bool
│      %7507  = Base.and_int(%7497, %7506)::Bool
│      %7508  = Base.and_int(%7494, %7507)::Bool
│      %7509  = Base.and_int(true, %7508)::Bool
└─────          goto #2572 if not %7509
2571 ─          goto #2573
2572 ─          invoke Base.throw_boundserror(%7485::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %7487::NTuple{5, Int64})::Union{}
└─────          unreachable
2573 ─          nothing::Nothing
2574 ┄ %7515  = StrideArraysCore.getfield(%7485, :ptr)::Ptr{Float64}
│      %7516  = Base.sub_int(%7404, 1)::Int64
│      %7517  = Base.sub_int(%6994, 1)::Int64
│      %7518  = Base.sub_int(%6991, 1)::Int64
│      %7519  = Base.sub_int(%6984, 1)::Int64
└─────          goto #2583 if not true
2575 ┄ %7521  = φ (#2574 => 2, #2582 => %7533)::Int64
│      %7522  = Base.sle_int(1, %7521)::Bool
└─────          goto #2577 if not %7522
2576 ─ %7524  = Base.sle_int(%7521, 5)::Bool
└─────          goto #2578
2577 ─          nothing::Nothing
2578 ┄ %7527  = φ (#2576 => %7524, #2577 => false)::Bool
└─────          goto #2580 if not %7527
2579 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %7521, true)::Static.True
│      %7530  = Base.add_int(%7521, 1)::Int64
└─────          goto #2581
2580 ─          goto #2581
2581 ┄ %7533  = φ (#2579 => %7530)::Int64
│      %7534  = φ (#2579 => false, #2580 => true)::Bool
│      %7535  = Base.not_int(%7534)::Bool
└─────          goto #2583 if not %7535
2582 ─          goto #2575
2583 ┄          goto #2584
2584 ─          goto #2585
2585 ─ %7540  = Base.mul_int(%7519, 4)::Int64
│      %7541  = Base.add_int(%7518, %7540)::Int64
│      %7542  = Base.mul_int(%7541, 4)::Int64
│      %7543  = Base.add_int(%7517, %7542)::Int64
│      %7544  = Base.mul_int(%7543, 4)::Int64
│      %7545  = Base.add_int(%7516, %7544)::Int64
│      %7546  = Base.mul_int(%7545, 5)::Int64
│      %7547  = Base.add_int(1, %7546)::Int64
│      %7548  = Base.mul_int(8, %7547)::Int64
│      %7549  = Core.bitcast(Core.UInt, %7515)::UInt64
│      %7550  = Base.bitcast(UInt64, %7548)::UInt64
│      %7551  = Base.add_ptr(%7549, %7550)::UInt64
│      %7552  = Core.bitcast(Ptr{Float64}, %7551)::Ptr{Float64}
└─────          goto #2586
2586 ─ %7554  = Base.pointerref(%7552, 1, 1)::Float64
└─────          goto #2587
2587 ─          goto #2588
2588 ─          $(Expr(:gc_preserve_end, :(%7484)))
└─────          goto #2589
2589 ─          goto #2590
2590 ─ %7560  = StrideArraysCore.getfield(u, :data)::Vector{Float64}
│      %7561  = $(Expr(:gc_preserve_begin, :(%7560)))
│      %7562  = StrideArraysCore.getfield(u, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #2595 if not true
2591 ─ %7564  = Core.tuple(3, %7404, %6994, %6991, %6984)::NTuple{5, Int64}
│      %7565  = StrideArraysCore.getfield(%7562, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %7566  = Core.getfield(%7565, 5)::Int64
│      %7567  = Base.bitcast(UInt64, %7566)::UInt64
│      %7568  = Base.bitcast(Int64, %7567)::Int64
│      %7569  = Base.sle_int(1, %7404)::Bool
│      %7570  = Base.sle_int(%7404, 4)::Bool
│      %7571  = Base.and_int(%7569, %7570)::Bool
│      %7572  = Base.sle_int(1, %6994)::Bool
│      %7573  = Base.sle_int(%6994, 4)::Bool
│      %7574  = Base.and_int(%7572, %7573)::Bool
│      %7575  = Base.sle_int(1, %6991)::Bool
│      %7576  = Base.sle_int(%6991, 4)::Bool
│      %7577  = Base.and_int(%7575, %7576)::Bool
│      %7578  = Base.sub_int(%6984, 1)::Int64
│      %7579  = Base.bitcast(UInt64, %7578)::UInt64
│      %7580  = Base.bitcast(UInt64, %7568)::UInt64
│      %7581  = Base.ult_int(%7579, %7580)::Bool
│      %7582  = Base.and_int(%7581, true)::Bool
│      %7583  = Base.and_int(%7577, %7582)::Bool
│      %7584  = Base.and_int(%7574, %7583)::Bool
│      %7585  = Base.and_int(%7571, %7584)::Bool
│      %7586  = Base.and_int(true, %7585)::Bool
└─────          goto #2593 if not %7586
2592 ─          goto #2594
2593 ─          invoke Base.throw_boundserror(%7562::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %7564::NTuple{5, Int64})::Union{}
└─────          unreachable
2594 ─          nothing::Nothing
2595 ┄ %7592  = StrideArraysCore.getfield(%7562, :ptr)::Ptr{Float64}
│      %7593  = Base.sub_int(%7404, 1)::Int64
│      %7594  = Base.sub_int(%6994, 1)::Int64
│      %7595  = Base.sub_int(%6991, 1)::Int64
│      %7596  = Base.sub_int(%6984, 1)::Int64
└─────          goto #2604 if not true
2596 ┄ %7598  = φ (#2595 => 2, #2603 => %7610)::Int64
│      %7599  = Base.sle_int(1, %7598)::Bool
└─────          goto #2598 if not %7599
2597 ─ %7601  = Base.sle_int(%7598, 5)::Bool
└─────          goto #2599
2598 ─          nothing::Nothing
2599 ┄ %7604  = φ (#2597 => %7601, #2598 => false)::Bool
└─────          goto #2601 if not %7604
2600 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %7598, true)::Static.True
│      %7607  = Base.add_int(%7598, 1)::Int64
└─────          goto #2602
2601 ─          goto #2602
2602 ┄ %7610  = φ (#2600 => %7607)::Int64
│      %7611  = φ (#2600 => false, #2601 => true)::Bool
│      %7612  = Base.not_int(%7611)::Bool
└─────          goto #2604 if not %7612
2603 ─          goto #2596
2604 ┄          goto #2605
2605 ─          goto #2606
2606 ─ %7617  = Base.mul_int(%7596, 4)::Int64
│      %7618  = Base.add_int(%7595, %7617)::Int64
│      %7619  = Base.mul_int(%7618, 4)::Int64
│      %7620  = Base.add_int(%7594, %7619)::Int64
│      %7621  = Base.mul_int(%7620, 4)::Int64
│      %7622  = Base.add_int(%7593, %7621)::Int64
│      %7623  = Base.mul_int(%7622, 5)::Int64
│      %7624  = Base.add_int(2, %7623)::Int64
│      %7625  = Base.mul_int(8, %7624)::Int64
│      %7626  = Core.bitcast(Core.UInt, %7592)::UInt64
│      %7627  = Base.bitcast(UInt64, %7625)::UInt64
│      %7628  = Base.add_ptr(%7626, %7627)::UInt64
│      %7629  = Core.bitcast(Ptr{Float64}, %7628)::Ptr{Float64}
└─────          goto #2607
2607 ─ %7631  = Base.pointerref(%7629, 1, 1)::Float64
└─────          goto #2608
2608 ─          goto #2609
2609 ─          $(Expr(:gc_preserve_end, :(%7561)))
└─────          goto #2610
2610 ─          goto #2611
2611 ─ %7637  = StrideArraysCore.getfield(u, :data)::Vector{Float64}
│      %7638  = $(Expr(:gc_preserve_begin, :(%7637)))
│      %7639  = StrideArraysCore.getfield(u, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #2616 if not true
2612 ─ %7641  = Core.tuple(4, %7404, %6994, %6991, %6984)::NTuple{5, Int64}
│      %7642  = StrideArraysCore.getfield(%7639, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %7643  = Core.getfield(%7642, 5)::Int64
│      %7644  = Base.bitcast(UInt64, %7643)::UInt64
│      %7645  = Base.bitcast(Int64, %7644)::Int64
│      %7646  = Base.sle_int(1, %7404)::Bool
│      %7647  = Base.sle_int(%7404, 4)::Bool
│      %7648  = Base.and_int(%7646, %7647)::Bool
│      %7649  = Base.sle_int(1, %6994)::Bool
│      %7650  = Base.sle_int(%6994, 4)::Bool
│      %7651  = Base.and_int(%7649, %7650)::Bool
│      %7652  = Base.sle_int(1, %6991)::Bool
│      %7653  = Base.sle_int(%6991, 4)::Bool
│      %7654  = Base.and_int(%7652, %7653)::Bool
│      %7655  = Base.sub_int(%6984, 1)::Int64
│      %7656  = Base.bitcast(UInt64, %7655)::UInt64
│      %7657  = Base.bitcast(UInt64, %7645)::UInt64
│      %7658  = Base.ult_int(%7656, %7657)::Bool
│      %7659  = Base.and_int(%7658, true)::Bool
│      %7660  = Base.and_int(%7654, %7659)::Bool
│      %7661  = Base.and_int(%7651, %7660)::Bool
│      %7662  = Base.and_int(%7648, %7661)::Bool
│      %7663  = Base.and_int(true, %7662)::Bool
└─────          goto #2614 if not %7663
2613 ─          goto #2615
2614 ─          invoke Base.throw_boundserror(%7639::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %7641::NTuple{5, Int64})::Union{}
└─────          unreachable
2615 ─          nothing::Nothing
2616 ┄ %7669  = StrideArraysCore.getfield(%7639, :ptr)::Ptr{Float64}
│      %7670  = Base.sub_int(%7404, 1)::Int64
│      %7671  = Base.sub_int(%6994, 1)::Int64
│      %7672  = Base.sub_int(%6991, 1)::Int64
│      %7673  = Base.sub_int(%6984, 1)::Int64
└─────          goto #2625 if not true
2617 ┄ %7675  = φ (#2616 => 2, #2624 => %7687)::Int64
│      %7676  = Base.sle_int(1, %7675)::Bool
└─────          goto #2619 if not %7676
2618 ─ %7678  = Base.sle_int(%7675, 5)::Bool
└─────          goto #2620
2619 ─          nothing::Nothing
2620 ┄ %7681  = φ (#2618 => %7678, #2619 => false)::Bool
└─────          goto #2622 if not %7681
2621 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %7675, true)::Static.True
│      %7684  = Base.add_int(%7675, 1)::Int64
└─────          goto #2623
2622 ─          goto #2623
2623 ┄ %7687  = φ (#2621 => %7684)::Int64
│      %7688  = φ (#2621 => false, #2622 => true)::Bool
│      %7689  = Base.not_int(%7688)::Bool
└─────          goto #2625 if not %7689
2624 ─          goto #2617
2625 ┄          goto #2626
2626 ─          goto #2627
2627 ─ %7694  = Base.mul_int(%7673, 4)::Int64
│      %7695  = Base.add_int(%7672, %7694)::Int64
│      %7696  = Base.mul_int(%7695, 4)::Int64
│      %7697  = Base.add_int(%7671, %7696)::Int64
│      %7698  = Base.mul_int(%7697, 4)::Int64
│      %7699  = Base.add_int(%7670, %7698)::Int64
│      %7700  = Base.mul_int(%7699, 5)::Int64
│      %7701  = Base.add_int(3, %7700)::Int64
│      %7702  = Base.mul_int(8, %7701)::Int64
│      %7703  = Core.bitcast(Core.UInt, %7669)::UInt64
│      %7704  = Base.bitcast(UInt64, %7702)::UInt64
│      %7705  = Base.add_ptr(%7703, %7704)::UInt64
│      %7706  = Core.bitcast(Ptr{Float64}, %7705)::Ptr{Float64}
└─────          goto #2628
2628 ─ %7708  = Base.pointerref(%7706, 1, 1)::Float64
└─────          goto #2629
2629 ─          goto #2630
2630 ─          $(Expr(:gc_preserve_end, :(%7638)))
└─────          goto #2631
2631 ─          goto #2632
2632 ─ %7714  = StrideArraysCore.getfield(u, :data)::Vector{Float64}
│      %7715  = $(Expr(:gc_preserve_begin, :(%7714)))
│      %7716  = StrideArraysCore.getfield(u, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #2637 if not true
2633 ─ %7718  = Core.tuple(5, %7404, %6994, %6991, %6984)::NTuple{5, Int64}
│      %7719  = StrideArraysCore.getfield(%7716, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %7720  = Core.getfield(%7719, 5)::Int64
│      %7721  = Base.bitcast(UInt64, %7720)::UInt64
│      %7722  = Base.bitcast(Int64, %7721)::Int64
│      %7723  = Base.sle_int(1, %7404)::Bool
│      %7724  = Base.sle_int(%7404, 4)::Bool
│      %7725  = Base.and_int(%7723, %7724)::Bool
│      %7726  = Base.sle_int(1, %6994)::Bool
│      %7727  = Base.sle_int(%6994, 4)::Bool
│      %7728  = Base.and_int(%7726, %7727)::Bool
│      %7729  = Base.sle_int(1, %6991)::Bool
│      %7730  = Base.sle_int(%6991, 4)::Bool
│      %7731  = Base.and_int(%7729, %7730)::Bool
│      %7732  = Base.sub_int(%6984, 1)::Int64
│      %7733  = Base.bitcast(UInt64, %7732)::UInt64
│      %7734  = Base.bitcast(UInt64, %7722)::UInt64
│      %7735  = Base.ult_int(%7733, %7734)::Bool
│      %7736  = Base.and_int(%7735, true)::Bool
│      %7737  = Base.and_int(%7731, %7736)::Bool
│      %7738  = Base.and_int(%7728, %7737)::Bool
│      %7739  = Base.and_int(%7725, %7738)::Bool
│      %7740  = Base.and_int(true, %7739)::Bool
└─────          goto #2635 if not %7740
2634 ─          goto #2636
2635 ─          invoke Base.throw_boundserror(%7716::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %7718::NTuple{5, Int64})::Union{}
└─────          unreachable
2636 ─          nothing::Nothing
2637 ┄ %7746  = StrideArraysCore.getfield(%7716, :ptr)::Ptr{Float64}
│      %7747  = Base.sub_int(%7404, 1)::Int64
│      %7748  = Base.sub_int(%6994, 1)::Int64
│      %7749  = Base.sub_int(%6991, 1)::Int64
│      %7750  = Base.sub_int(%6984, 1)::Int64
└─────          goto #2646 if not true
2638 ┄ %7752  = φ (#2637 => 2, #2645 => %7764)::Int64
│      %7753  = Base.sle_int(1, %7752)::Bool
└─────          goto #2640 if not %7753
2639 ─ %7755  = Base.sle_int(%7752, 5)::Bool
└─────          goto #2641
2640 ─          nothing::Nothing
2641 ┄ %7758  = φ (#2639 => %7755, #2640 => false)::Bool
└─────          goto #2643 if not %7758
2642 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %7752, true)::Static.True
│      %7761  = Base.add_int(%7752, 1)::Int64
└─────          goto #2644
2643 ─          goto #2644
2644 ┄ %7764  = φ (#2642 => %7761)::Int64
│      %7765  = φ (#2642 => false, #2643 => true)::Bool
│      %7766  = Base.not_int(%7765)::Bool
└─────          goto #2646 if not %7766
2645 ─          goto #2638
2646 ┄          goto #2647
2647 ─          goto #2648
2648 ─ %7771  = Base.mul_int(%7750, 4)::Int64
│      %7772  = Base.add_int(%7749, %7771)::Int64
│      %7773  = Base.mul_int(%7772, 4)::Int64
│      %7774  = Base.add_int(%7748, %7773)::Int64
│      %7775  = Base.mul_int(%7774, 4)::Int64
│      %7776  = Base.add_int(%7747, %7775)::Int64
│      %7777  = Base.mul_int(%7776, 5)::Int64
│      %7778  = Base.add_int(4, %7777)::Int64
│      %7779  = Base.mul_int(8, %7778)::Int64
│      %7780  = Core.bitcast(Core.UInt, %7746)::UInt64
│      %7781  = Base.bitcast(UInt64, %7779)::UInt64
│      %7782  = Base.add_ptr(%7780, %7781)::UInt64
│      %7783  = Core.bitcast(Ptr{Float64}, %7782)::Ptr{Float64}
└─────          goto #2649
2649 ─ %7785  = Base.pointerref(%7783, 1, 1)::Float64
└─────          goto #2650
2650 ─          goto #2651
2651 ─          $(Expr(:gc_preserve_end, :(%7715)))
└─────          goto #2652
2652 ─          goto #2653
2653 ─          goto #2654
2654 ─          goto #2655
2655 ─          goto #2657
2656 ─          nothing::Nothing
2657 ┄          goto #2659
2658 ─          nothing::Nothing
2659 ┄          goto #2660
2660 ─          goto #2662
2661 ─          nothing::Nothing
2662 ┄          goto #2663
2663 ─          goto #2665
2664 ─          nothing::Nothing
2665 ┄          goto #2667
2666 ─          nothing::Nothing
2667 ┄          goto #2668
2668 ─          goto #2670
2669 ─          nothing::Nothing
2670 ┄          goto #2671
2671 ─          goto #2673
2672 ─          nothing::Nothing
2673 ┄          goto #2675
2674 ─          nothing::Nothing
2675 ┄          goto #2676
2676 ─          goto #2678
2677 ─          nothing::Nothing
2678 ┄          goto #2679
2679 ─          goto #2681
2680 ─          nothing::Nothing
2681 ┄          goto #2683
2682 ─          nothing::Nothing
2683 ┄          goto #2684
2684 ─          goto #2686
2685 ─          nothing::Nothing
2686 ┄          goto #2687
2687 ─ %7825  = Base.div_float(%7147, %7070)::Float64
│      %7826  = Base.div_float(%7224, %7070)::Float64
│      %7827  = Base.div_float(%7301, %7070)::Float64
│      %7828  = Base.getfield(equations, :gamma)::Float64
│      %7829  = Base.sub_float(%7828, 1.0)::Float64
│      %7830  = Base.mul_float(%7147, %7825)::Float64
│      %7831  = Base.muladd_float(%7224, %7826, %7830)::Float64
│      %7832  = Base.muladd_float(%7301, %7827, %7831)::Float64
│      %7833  = Base.muladd_float(-0.5, %7832, %7378)::Float64
│      %7834  = Base.mul_float(%7829, %7833)::Float64
└─────          goto #2688
2688 ─          goto #2690
2689 ─          nothing::Nothing
2690 ┄          goto #2692
2691 ─          nothing::Nothing
2692 ┄          goto #2693
2693 ─          goto #2695
2694 ─          nothing::Nothing
2695 ┄          goto #2696
2696 ─          goto #2698
2697 ─          nothing::Nothing
2698 ┄          goto #2700
2699 ─          nothing::Nothing
2700 ┄          goto #2701
2701 ─          goto #2703
2702 ─          nothing::Nothing
2703 ┄          goto #2704
2704 ─          goto #2706
2705 ─          nothing::Nothing
2706 ┄          goto #2708
2707 ─          nothing::Nothing
2708 ┄          goto #2709
2709 ─          goto #2711
2710 ─          nothing::Nothing
2711 ┄          goto #2712
2712 ─          goto #2714
2713 ─          nothing::Nothing
2714 ┄          goto #2716
2715 ─          nothing::Nothing
2716 ┄          goto #2717
2717 ─          goto #2719
2718 ─          nothing::Nothing
2719 ┄          goto #2720
2720 ─          goto #2722
2721 ─          nothing::Nothing
2722 ┄          goto #2724
2723 ─          nothing::Nothing
2724 ┄          goto #2725
2725 ─          goto #2727
2726 ─          nothing::Nothing
2727 ┄          goto #2728
2728 ─          goto #2730
2729 ─          nothing::Nothing
2730 ┄          goto #2732
2731 ─          nothing::Nothing
2732 ┄          goto #2733
2733 ─          goto #2735
2734 ─          nothing::Nothing
2735 ┄          goto #2736
2736 ─          goto #2738
2737 ─          nothing::Nothing
2738 ┄          goto #2740
2739 ─          nothing::Nothing
2740 ┄          goto #2741
2741 ─          goto #2743
2742 ─          nothing::Nothing
2743 ┄          goto #2744
2744 ─          goto #2746
2745 ─          nothing::Nothing
2746 ┄          goto #2748
2747 ─          nothing::Nothing
2748 ┄          goto #2749
2749 ─          goto #2751
2750 ─          nothing::Nothing
2751 ┄          goto #2752
2752 ─ %7900  = Base.div_float(%7554, %7477)::Float64
│      %7901  = Base.div_float(%7631, %7477)::Float64
│      %7902  = Base.div_float(%7708, %7477)::Float64
│      %7903  = Base.getfield(equations, :gamma)::Float64
│      %7904  = Base.sub_float(%7903, 1.0)::Float64
│      %7905  = Base.mul_float(%7554, %7900)::Float64
│      %7906  = Base.muladd_float(%7631, %7901, %7905)::Float64
│      %7907  = Base.muladd_float(%7708, %7902, %7906)::Float64
│      %7908  = Base.muladd_float(-0.5, %7907, %7785)::Float64
│      %7909  = Base.mul_float(%7904, %7908)::Float64
└─────          goto #2753
2753 ─          goto #2755
2754 ─          nothing::Nothing
2755 ┄          goto #2757
2756 ─          nothing::Nothing
2757 ┄          goto #2758
2758 ─          goto #2760
2759 ─          nothing::Nothing
2760 ┄          goto #2761
2761 ─          goto #2763
2762 ─          nothing::Nothing
2763 ┄          goto #2765
2764 ─          nothing::Nothing
2765 ┄          goto #2766
2766 ─          goto #2768
2767 ─          nothing::Nothing
2768 ┄          goto #2769
2769 ─          goto #2771
2770 ─          nothing::Nothing
2771 ┄          goto #2773
2772 ─          nothing::Nothing
2773 ┄          goto #2774
2774 ─          goto #2776
2775 ─          nothing::Nothing
2776 ┄          goto #2777
2777 ─          goto #2779
2778 ─          nothing::Nothing
2779 ┄          goto #2781
2780 ─          nothing::Nothing
2781 ┄          goto #2782
2782 ─          goto #2784
2783 ─          nothing::Nothing
2784 ┄          goto #2785
2785 ─ %7943  = Base.muladd_float(-2.0, %7477, %7070)::Float64
│      %7944  = Base.mul_float(%7070, %7943)::Float64
│      %7945  = Base.muladd_float(%7477, %7477, %7944)::Float64
│      %7946  = Base.muladd_float(2.0, %7477, %7070)::Float64
│      %7947  = Base.mul_float(%7070, %7946)::Float64
│      %7948  = Base.muladd_float(%7477, %7477, %7947)::Float64
│      %7949  = Base.div_float(%7945, %7948)::Float64
│      %7950  = Base.lt_float(%7949, 0.0001)::Bool
└─────          goto #2787 if not %7950
2786 ─ %7952  = Base.add_float(%7070, %7477)::Float64
│      %7953  = Base.muladd_float(%7949, 0.2857142857142857, 0.4)::Float64
│      %7954  = Base.muladd_float(%7949, %7953, 0.6666666666666666)::Float64
│      %7955  = Base.muladd_float(%7949, %7954, 2.0)::Float64
│      %7956  = Base.div_float(%7952, %7955)::Float64
└─────          goto #2788
2787 ─ %7958  = Base.sub_float(%7477, %7070)::Float64
│      %7959  = Base.div_float(%7477, %7070)::Float64
│      %7960  = $(Expr(:foreigncall, "llvm.log.f64", Float64, svec(Float64), 0, :(:llvmcall), :(%7959), :(%7959)))::Float64
│      %7961  = Base.div_float(%7958, %7960)::Float64
└─────          goto #2788
2788 ┄ %7963  = φ (#2786 => %7956, #2787 => %7961)::Float64
│      %7964  = Base.mul_float(%7070, %7909)::Float64
│      %7965  = Base.mul_float(%7477, %7834)::Float64
│      %7966  = Base.muladd_float(-2.0, %7965, %7964)::Float64
│      %7967  = Base.mul_float(%7964, %7966)::Float64
│      %7968  = Base.muladd_float(%7965, %7965, %7967)::Float64
│      %7969  = Base.muladd_float(2.0, %7965, %7964)::Float64
│      %7970  = Base.mul_float(%7964, %7969)::Float64
│      %7971  = Base.muladd_float(%7965, %7965, %7970)::Float64
│      %7972  = Base.div_float(%7968, %7971)::Float64
│      %7973  = Base.lt_float(%7972, 0.0001)::Bool
└─────          goto #2790 if not %7973
2789 ─ %7975  = Base.muladd_float(%7972, 0.2857142857142857, 0.4)::Float64
│      %7976  = Base.muladd_float(%7972, %7975, 0.6666666666666666)::Float64
│      %7977  = Base.muladd_float(%7972, %7976, 2.0)::Float64
│      %7978  = Base.add_float(%7964, %7965)::Float64
│      %7979  = Base.div_float(%7977, %7978)::Float64
└─────          goto #2791
2790 ─ %7981  = Base.div_float(%7965, %7964)::Float64
│      %7982  = $(Expr(:foreigncall, "llvm.log.f64", Float64, svec(Float64), 0, :(:llvmcall), :(%7981), :(%7981)))::Float64
│      %7983  = Base.sub_float(%7965, %7964)::Float64
│      %7984  = Base.div_float(%7982, %7983)::Float64
└─────          goto #2791
2791 ┄ %7986  = φ (#2789 => %7979, #2790 => %7984)::Float64
│      %7987  = Base.mul_float(%7834, %7909)::Float64
│      %7988  = Base.mul_float(%7987, %7986)::Float64
│      %7989  = Base.add_float(%7825, %7900)::Float64
│      %7990  = Base.mul_float(0.5, %7989)::Float64
│      %7991  = Base.add_float(%7826, %7901)::Float64
│      %7992  = Base.mul_float(0.5, %7991)::Float64
│      %7993  = Base.add_float(%7827, %7902)::Float64
│      %7994  = Base.mul_float(0.5, %7993)::Float64
│      %7995  = Base.add_float(%7834, %7909)::Float64
│      %7996  = Base.mul_float(0.5, %7995)::Float64
│      %7997  = Base.mul_float(%7825, %7900)::Float64
│      %7998  = Base.muladd_float(%7826, %7901, %7997)::Float64
│      %7999  = Base.muladd_float(%7827, %7902, %7998)::Float64
│      %8000  = Base.mul_float(0.5, %7999)::Float64
│      %8001  = Base.mul_float(%7963, %7990)::Float64
│      %8002  = Base.muladd_float(%8001, %7990, %7996)::Float64
│      %8003  = Base.mul_float(%8001, %7992)::Float64
│      %8004  = Base.mul_float(%8001, %7994)::Float64
│      %8005  = Base.mul_float(%7834, %7900)::Float64
│      %8006  = Base.muladd_float(%7909, %7825, %8005)::Float64
│      %8007  = Base.getfield(equations, :inv_gamma_minus_one)::Float64
│      %8008  = Base.muladd_float(%7988, %8007, %8000)::Float64
│      %8009  = Base.mul_float(%8001, %8008)::Float64
│      %8010  = Base.muladd_float(0.5, %8006, %8009)::Float64
│      %8011  = Core.tuple(%8001, %8002, %8003, %8004, %8010)::NTuple{5, Float64}
└─────          goto #2792
2792 ─ %8013  = Base.arrayref(false, %6989, %6997, %7404)::Float64
│      %8014  = Base.copysign_float(0.0, %8013)::Float64
│      %8015  = Core.ifelse(true, %8013, %8014)::Float64
└─────          goto #2838 if not true
2793 ┄ %8017  = φ (#2792 => 1, #2837 => %8186)::Int64
│      %8018  = φ (#2792 => 1, #2837 => %8187)::Int64
│      %8019  = Base.getfield(%8011, %8017, true)::Float64
│      %8020  = StrideArraysCore.getfield(du, :data)::Vector{Float64}
│      %8021  = $(Expr(:gc_preserve_begin, :(%8020)))
│      %8022  = StrideArraysCore.getfield(du, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #2798 if not true
2794 ─ %8024  = Core.tuple(%8017, %6997, %6994, %6991, %6984)::NTuple{5, Int64}
│      %8025  = StrideArraysCore.getfield(%8022, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %8026  = Core.getfield(%8025, 5)::Int64
│      %8027  = Base.bitcast(UInt64, %8026)::UInt64
│      %8028  = Base.bitcast(Int64, %8027)::Int64
│      %8029  = Base.sle_int(1, %8017)::Bool
│      %8030  = Base.sle_int(%8017, 5)::Bool
│      %8031  = Base.and_int(%8029, %8030)::Bool
│      %8032  = Base.sle_int(1, %6997)::Bool
│      %8033  = Base.sle_int(%6997, 4)::Bool
│      %8034  = Base.and_int(%8032, %8033)::Bool
│      %8035  = Base.sle_int(1, %6994)::Bool
│      %8036  = Base.sle_int(%6994, 4)::Bool
│      %8037  = Base.and_int(%8035, %8036)::Bool
│      %8038  = Base.sle_int(1, %6991)::Bool
│      %8039  = Base.sle_int(%6991, 4)::Bool
│      %8040  = Base.and_int(%8038, %8039)::Bool
│      %8041  = Base.sub_int(%6984, 1)::Int64
│      %8042  = Base.bitcast(UInt64, %8041)::UInt64
│      %8043  = Base.bitcast(UInt64, %8028)::UInt64
│      %8044  = Base.ult_int(%8042, %8043)::Bool
│      %8045  = Base.and_int(%8044, true)::Bool
│      %8046  = Base.and_int(%8040, %8045)::Bool
│      %8047  = Base.and_int(%8037, %8046)::Bool
│      %8048  = Base.and_int(%8034, %8047)::Bool
│      %8049  = Base.and_int(%8031, %8048)::Bool
└─────          goto #2796 if not %8049
2795 ─          goto #2797
2796 ─          invoke Base.throw_boundserror(%8022::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %8024::NTuple{5, Int64})::Union{}
└─────          unreachable
2797 ─          nothing::Nothing
2798 ┄ %8055  = StrideArraysCore.getfield(%8022, :ptr)::Ptr{Float64}
│      %8056  = Base.sub_int(%8017, 1)::Int64
│      %8057  = Base.sub_int(%6997, 1)::Int64
│      %8058  = Base.sub_int(%6994, 1)::Int64
│      %8059  = Base.sub_int(%6991, 1)::Int64
│      %8060  = Base.sub_int(%6984, 1)::Int64
└─────          goto #2807 if not true
2799 ┄ %8062  = φ (#2798 => 2, #2806 => %8074)::Int64
│      %8063  = Base.sle_int(1, %8062)::Bool
└─────          goto #2801 if not %8063
2800 ─ %8065  = Base.sle_int(%8062, 5)::Bool
└─────          goto #2802
2801 ─          nothing::Nothing
2802 ┄ %8068  = φ (#2800 => %8065, #2801 => false)::Bool
└─────          goto #2804 if not %8068
2803 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %8062, true)::Static.True
│      %8071  = Base.add_int(%8062, 1)::Int64
└─────          goto #2805
2804 ─          goto #2805
2805 ┄ %8074  = φ (#2803 => %8071)::Int64
│      %8075  = φ (#2803 => false, #2804 => true)::Bool
│      %8076  = Base.not_int(%8075)::Bool
└─────          goto #2807 if not %8076
2806 ─          goto #2799
2807 ┄          goto #2808
2808 ─          goto #2809
2809 ─ %8081  = Base.mul_int(%8060, 4)::Int64
│      %8082  = Base.add_int(%8059, %8081)::Int64
│      %8083  = Base.mul_int(%8082, 4)::Int64
│      %8084  = Base.add_int(%8058, %8083)::Int64
│      %8085  = Base.mul_int(%8084, 4)::Int64
│      %8086  = Base.add_int(%8057, %8085)::Int64
│      %8087  = Base.mul_int(%8086, 5)::Int64
│      %8088  = Base.add_int(%8056, %8087)::Int64
│      %8089  = Base.mul_int(8, %8088)::Int64
│      %8090  = Core.bitcast(Core.UInt, %8055)::UInt64
│      %8091  = Base.bitcast(UInt64, %8089)::UInt64
│      %8092  = Base.add_ptr(%8090, %8091)::UInt64
│      %8093  = Core.bitcast(Ptr{Float64}, %8092)::Ptr{Float64}
└─────          goto #2810
2810 ─ %8095  = Base.pointerref(%8093, 1, 1)::Float64
└─────          goto #2811
2811 ─          goto #2812
2812 ─          $(Expr(:gc_preserve_end, :(%8021)))
└─────          goto #2813
2813 ─ %8100  = Base.muladd_float(%8015, %8019, %8095)::Float64
│      %8101  = StrideArraysCore.getfield(du, :data)::Vector{Float64}
│      %8102  = $(Expr(:gc_preserve_begin, :(%8101)))
│      %8103  = StrideArraysCore.getfield(du, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #2818 if not true
2814 ─ %8105  = Core.tuple(%8017, %6997, %6994, %6991, %6984)::NTuple{5, Int64}
│      %8106  = StrideArraysCore.getfield(%8103, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %8107  = Core.getfield(%8106, 5)::Int64
│      %8108  = Base.bitcast(UInt64, %8107)::UInt64
│      %8109  = Base.bitcast(Int64, %8108)::Int64
│      %8110  = Base.sle_int(1, %8017)::Bool
│      %8111  = Base.sle_int(%8017, 5)::Bool
│      %8112  = Base.and_int(%8110, %8111)::Bool
│      %8113  = Base.sle_int(1, %6997)::Bool
│      %8114  = Base.sle_int(%6997, 4)::Bool
│      %8115  = Base.and_int(%8113, %8114)::Bool
│      %8116  = Base.sle_int(1, %6994)::Bool
│      %8117  = Base.sle_int(%6994, 4)::Bool
│      %8118  = Base.and_int(%8116, %8117)::Bool
│      %8119  = Base.sle_int(1, %6991)::Bool
│      %8120  = Base.sle_int(%6991, 4)::Bool
│      %8121  = Base.and_int(%8119, %8120)::Bool
│      %8122  = Base.sub_int(%6984, 1)::Int64
│      %8123  = Base.bitcast(UInt64, %8122)::UInt64
│      %8124  = Base.bitcast(UInt64, %8109)::UInt64
│      %8125  = Base.ult_int(%8123, %8124)::Bool
│      %8126  = Base.and_int(%8125, true)::Bool
│      %8127  = Base.and_int(%8121, %8126)::Bool
│      %8128  = Base.and_int(%8118, %8127)::Bool
│      %8129  = Base.and_int(%8115, %8128)::Bool
│      %8130  = Base.and_int(%8112, %8129)::Bool
└─────          goto #2816 if not %8130
2815 ─          goto #2817
2816 ─          invoke Base.throw_boundserror(%8103::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %8105::NTuple{5, Int64})::Union{}
└─────          unreachable
2817 ─          nothing::Nothing
2818 ┄ %8136  = StrideArraysCore.getfield(%8103, :ptr)::Ptr{Float64}
│      %8137  = Base.sub_int(%8017, 1)::Int64
│      %8138  = Base.sub_int(%6997, 1)::Int64
│      %8139  = Base.sub_int(%6994, 1)::Int64
│      %8140  = Base.sub_int(%6991, 1)::Int64
│      %8141  = Base.sub_int(%6984, 1)::Int64
└─────          goto #2827 if not true
2819 ┄ %8143  = φ (#2818 => 2, #2826 => %8155)::Int64
│      %8144  = Base.sle_int(1, %8143)::Bool
└─────          goto #2821 if not %8144
2820 ─ %8146  = Base.sle_int(%8143, 5)::Bool
└─────          goto #2822
2821 ─          nothing::Nothing
2822 ┄ %8149  = φ (#2820 => %8146, #2821 => false)::Bool
└─────          goto #2824 if not %8149
2823 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %8143, true)::Static.True
│      %8152  = Base.add_int(%8143, 1)::Int64
└─────          goto #2825
2824 ─          goto #2825
2825 ┄ %8155  = φ (#2823 => %8152)::Int64
│      %8156  = φ (#2823 => false, #2824 => true)::Bool
│      %8157  = Base.not_int(%8156)::Bool
└─────          goto #2827 if not %8157
2826 ─          goto #2819
2827 ┄          goto #2828
2828 ─          goto #2829
2829 ─ %8162  = Base.mul_int(%8141, 4)::Int64
│      %8163  = Base.add_int(%8140, %8162)::Int64
│      %8164  = Base.mul_int(%8163, 4)::Int64
│      %8165  = Base.add_int(%8139, %8164)::Int64
│      %8166  = Base.mul_int(%8165, 4)::Int64
│      %8167  = Base.add_int(%8138, %8166)::Int64
│      %8168  = Base.mul_int(%8167, 5)::Int64
│      %8169  = Base.add_int(%8137, %8168)::Int64
│      %8170  = Base.mul_int(8, %8169)::Int64
│      %8171  = Core.bitcast(Core.UInt, %8136)::UInt64
│      %8172  = Base.bitcast(UInt64, %8170)::UInt64
│      %8173  = Base.add_ptr(%8171, %8172)::UInt64
│      %8174  = Core.bitcast(Ptr{Float64}, %8173)::Ptr{Float64}
└─────          goto #2830
2830 ─          Base.pointerset(%8174, %8100, 1, 1)::Ptr{Float64}
└─────          goto #2831
2831 ─          goto #2832
2832 ─          $(Expr(:gc_preserve_end, :(%8102)))
└─────          goto #2833
2833 ─ %8181  = (%8018 === 5)::Bool
└─────          goto #2835 if not %8181
2834 ─          goto #2836
2835 ─ %8184  = Base.add_int(%8018, 1)::Int64
└─────          goto #2836
2836 ┄ %8186  = φ (#2835 => %8184)::Int64
│      %8187  = φ (#2835 => %8184)::Int64
│      %8188  = φ (#2834 => true, #2835 => false)::Bool
│      %8189  = Base.not_int(%8188)::Bool
└─────          goto #2838 if not %8189
2837 ─          goto #2793
2838 ┄          goto #2839
2839 ─ %8193  = Base.arrayref(false, %6989, %7404, %6997)::Float64
│      %8194  = Base.copysign_float(0.0, %8193)::Float64
│      %8195  = Core.ifelse(true, %8193, %8194)::Float64
└─────          goto #2885 if not true
2840 ┄ %8197  = φ (#2839 => 1, #2884 => %8366)::Int64
│      %8198  = φ (#2839 => 1, #2884 => %8367)::Int64
│      %8199  = Base.getfield(%8011, %8197, true)::Float64
│      %8200  = StrideArraysCore.getfield(du, :data)::Vector{Float64}
│      %8201  = $(Expr(:gc_preserve_begin, :(%8200)))
│      %8202  = StrideArraysCore.getfield(du, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #2845 if not true
2841 ─ %8204  = Core.tuple(%8197, %7404, %6994, %6991, %6984)::NTuple{5, Int64}
│      %8205  = StrideArraysCore.getfield(%8202, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %8206  = Core.getfield(%8205, 5)::Int64
│      %8207  = Base.bitcast(UInt64, %8206)::UInt64
│      %8208  = Base.bitcast(Int64, %8207)::Int64
│      %8209  = Base.sle_int(1, %8197)::Bool
│      %8210  = Base.sle_int(%8197, 5)::Bool
│      %8211  = Base.and_int(%8209, %8210)::Bool
│      %8212  = Base.sle_int(1, %7404)::Bool
│      %8213  = Base.sle_int(%7404, 4)::Bool
│      %8214  = Base.and_int(%8212, %8213)::Bool
│      %8215  = Base.sle_int(1, %6994)::Bool
│      %8216  = Base.sle_int(%6994, 4)::Bool
│      %8217  = Base.and_int(%8215, %8216)::Bool
│      %8218  = Base.sle_int(1, %6991)::Bool
│      %8219  = Base.sle_int(%6991, 4)::Bool
│      %8220  = Base.and_int(%8218, %8219)::Bool
│      %8221  = Base.sub_int(%6984, 1)::Int64
│      %8222  = Base.bitcast(UInt64, %8221)::UInt64
│      %8223  = Base.bitcast(UInt64, %8208)::UInt64
│      %8224  = Base.ult_int(%8222, %8223)::Bool
│      %8225  = Base.and_int(%8224, true)::Bool
│      %8226  = Base.and_int(%8220, %8225)::Bool
│      %8227  = Base.and_int(%8217, %8226)::Bool
│      %8228  = Base.and_int(%8214, %8227)::Bool
│      %8229  = Base.and_int(%8211, %8228)::Bool
└─────          goto #2843 if not %8229
2842 ─          goto #2844
2843 ─          invoke Base.throw_boundserror(%8202::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %8204::NTuple{5, Int64})::Union{}
└─────          unreachable
2844 ─          nothing::Nothing
2845 ┄ %8235  = StrideArraysCore.getfield(%8202, :ptr)::Ptr{Float64}
│      %8236  = Base.sub_int(%8197, 1)::Int64
│      %8237  = Base.sub_int(%7404, 1)::Int64
│      %8238  = Base.sub_int(%6994, 1)::Int64
│      %8239  = Base.sub_int(%6991, 1)::Int64
│      %8240  = Base.sub_int(%6984, 1)::Int64
└─────          goto #2854 if not true
2846 ┄ %8242  = φ (#2845 => 2, #2853 => %8254)::Int64
│      %8243  = Base.sle_int(1, %8242)::Bool
└─────          goto #2848 if not %8243
2847 ─ %8245  = Base.sle_int(%8242, 5)::Bool
└─────          goto #2849
2848 ─          nothing::Nothing
2849 ┄ %8248  = φ (#2847 => %8245, #2848 => false)::Bool
└─────          goto #2851 if not %8248
2850 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %8242, true)::Static.True
│      %8251  = Base.add_int(%8242, 1)::Int64
└─────          goto #2852
2851 ─          goto #2852
2852 ┄ %8254  = φ (#2850 => %8251)::Int64
│      %8255  = φ (#2850 => false, #2851 => true)::Bool
│      %8256  = Base.not_int(%8255)::Bool
└─────          goto #2854 if not %8256
2853 ─          goto #2846
2854 ┄          goto #2855
2855 ─          goto #2856
2856 ─ %8261  = Base.mul_int(%8240, 4)::Int64
│      %8262  = Base.add_int(%8239, %8261)::Int64
│      %8263  = Base.mul_int(%8262, 4)::Int64
│      %8264  = Base.add_int(%8238, %8263)::Int64
│      %8265  = Base.mul_int(%8264, 4)::Int64
│      %8266  = Base.add_int(%8237, %8265)::Int64
│      %8267  = Base.mul_int(%8266, 5)::Int64
│      %8268  = Base.add_int(%8236, %8267)::Int64
│      %8269  = Base.mul_int(8, %8268)::Int64
│      %8270  = Core.bitcast(Core.UInt, %8235)::UInt64
│      %8271  = Base.bitcast(UInt64, %8269)::UInt64
│      %8272  = Base.add_ptr(%8270, %8271)::UInt64
│      %8273  = Core.bitcast(Ptr{Float64}, %8272)::Ptr{Float64}
└─────          goto #2857
2857 ─ %8275  = Base.pointerref(%8273, 1, 1)::Float64
└─────          goto #2858
2858 ─          goto #2859
2859 ─          $(Expr(:gc_preserve_end, :(%8201)))
└─────          goto #2860
2860 ─ %8280  = Base.muladd_float(%8195, %8199, %8275)::Float64
│      %8281  = StrideArraysCore.getfield(du, :data)::Vector{Float64}
│      %8282  = $(Expr(:gc_preserve_begin, :(%8281)))
│      %8283  = StrideArraysCore.getfield(du, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #2865 if not true
2861 ─ %8285  = Core.tuple(%8197, %7404, %6994, %6991, %6984)::NTuple{5, Int64}
│      %8286  = StrideArraysCore.getfield(%8283, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %8287  = Core.getfield(%8286, 5)::Int64
│      %8288  = Base.bitcast(UInt64, %8287)::UInt64
│      %8289  = Base.bitcast(Int64, %8288)::Int64
│      %8290  = Base.sle_int(1, %8197)::Bool
│      %8291  = Base.sle_int(%8197, 5)::Bool
│      %8292  = Base.and_int(%8290, %8291)::Bool
│      %8293  = Base.sle_int(1, %7404)::Bool
│      %8294  = Base.sle_int(%7404, 4)::Bool
│      %8295  = Base.and_int(%8293, %8294)::Bool
│      %8296  = Base.sle_int(1, %6994)::Bool
│      %8297  = Base.sle_int(%6994, 4)::Bool
│      %8298  = Base.and_int(%8296, %8297)::Bool
│      %8299  = Base.sle_int(1, %6991)::Bool
│      %8300  = Base.sle_int(%6991, 4)::Bool
│      %8301  = Base.and_int(%8299, %8300)::Bool
│      %8302  = Base.sub_int(%6984, 1)::Int64
│      %8303  = Base.bitcast(UInt64, %8302)::UInt64
│      %8304  = Base.bitcast(UInt64, %8289)::UInt64
│      %8305  = Base.ult_int(%8303, %8304)::Bool
│      %8306  = Base.and_int(%8305, true)::Bool
│      %8307  = Base.and_int(%8301, %8306)::Bool
│      %8308  = Base.and_int(%8298, %8307)::Bool
│      %8309  = Base.and_int(%8295, %8308)::Bool
│      %8310  = Base.and_int(%8292, %8309)::Bool
└─────          goto #2863 if not %8310
2862 ─          goto #2864
2863 ─          invoke Base.throw_boundserror(%8283::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %8285::NTuple{5, Int64})::Union{}
└─────          unreachable
2864 ─          nothing::Nothing
2865 ┄ %8316  = StrideArraysCore.getfield(%8283, :ptr)::Ptr{Float64}
│      %8317  = Base.sub_int(%8197, 1)::Int64
│      %8318  = Base.sub_int(%7404, 1)::Int64
│      %8319  = Base.sub_int(%6994, 1)::Int64
│      %8320  = Base.sub_int(%6991, 1)::Int64
│      %8321  = Base.sub_int(%6984, 1)::Int64
└─────          goto #2874 if not true
2866 ┄ %8323  = φ (#2865 => 2, #2873 => %8335)::Int64
│      %8324  = Base.sle_int(1, %8323)::Bool
└─────          goto #2868 if not %8324
2867 ─ %8326  = Base.sle_int(%8323, 5)::Bool
└─────          goto #2869
2868 ─          nothing::Nothing
2869 ┄ %8329  = φ (#2867 => %8326, #2868 => false)::Bool
└─────          goto #2871 if not %8329
2870 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %8323, true)::Static.True
│      %8332  = Base.add_int(%8323, 1)::Int64
└─────          goto #2872
2871 ─          goto #2872
2872 ┄ %8335  = φ (#2870 => %8332)::Int64
│      %8336  = φ (#2870 => false, #2871 => true)::Bool
│      %8337  = Base.not_int(%8336)::Bool
└─────          goto #2874 if not %8337
2873 ─          goto #2866
2874 ┄          goto #2875
2875 ─          goto #2876
2876 ─ %8342  = Base.mul_int(%8321, 4)::Int64
│      %8343  = Base.add_int(%8320, %8342)::Int64
│      %8344  = Base.mul_int(%8343, 4)::Int64
│      %8345  = Base.add_int(%8319, %8344)::Int64
│      %8346  = Base.mul_int(%8345, 4)::Int64
│      %8347  = Base.add_int(%8318, %8346)::Int64
│      %8348  = Base.mul_int(%8347, 5)::Int64
│      %8349  = Base.add_int(%8317, %8348)::Int64
│      %8350  = Base.mul_int(8, %8349)::Int64
│      %8351  = Core.bitcast(Core.UInt, %8316)::UInt64
│      %8352  = Base.bitcast(UInt64, %8350)::UInt64
│      %8353  = Base.add_ptr(%8351, %8352)::UInt64
│      %8354  = Core.bitcast(Ptr{Float64}, %8353)::Ptr{Float64}
└─────          goto #2877
2877 ─          Base.pointerset(%8354, %8280, 1, 1)::Ptr{Float64}
└─────          goto #2878
2878 ─          goto #2879
2879 ─          $(Expr(:gc_preserve_end, :(%8282)))
└─────          goto #2880
2880 ─ %8361  = (%8198 === 5)::Bool
└─────          goto #2882 if not %8361
2881 ─          goto #2883
2882 ─ %8364  = Base.add_int(%8198, 1)::Int64
└─────          goto #2883
2883 ┄ %8366  = φ (#2882 => %8364)::Int64
│      %8367  = φ (#2882 => %8364)::Int64
│      %8368  = φ (#2881 => true, #2882 => false)::Bool
│      %8369  = Base.not_int(%8368)::Bool
└─────          goto #2885 if not %8369
2884 ─          goto #2840
2885 ┄          goto #2886
2886 ─ %8373  = (%7405 === %7392)::Bool
└─────          goto #2888 if not %8373
2887 ─          goto #2889
2888 ─ %8376  = Base.add_int(%7405, 1)::Int64
└─────          goto #2889
2889 ┄ %8378  = φ (#2888 => %8376)::Int64
│      %8379  = φ (#2888 => %8376)::Int64
│      %8380  = φ (#2887 => true, #2888 => false)::Bool
│      %8381  = Base.not_int(%8380)::Bool
└─────          goto #2891 if not %8381
2890 ─          goto #2548
2891 ┄ %8384  = Base.add_int(%6994, 1)::Int64
│      %8385  = Base.sle_int(%8384, 4)::Bool
└─────          goto #2893 if not %8385
2892 ─          goto #2894
2893 ─ %8388  = Base.sub_int(%8384, 1)::Int64
└─────          goto #2894
2894 ┄ %8390  = φ (#2892 => 4, #2893 => %8388)::Int64
└─────          goto #2895
2895 ─          goto #2896
2896 ─ %8393  = Base.slt_int(%8390, %8384)::Bool
└─────          goto #2898 if not %8393
2897 ─          goto #2899
2898 ─          goto #2899
2899 ┄ %8397  = φ (#2897 => true, #2898 => false)::Bool
│      %8398  = φ (#2898 => %8384)::Int64
│      %8399  = φ (#2898 => %8384)::Int64
│      %8400  = Base.not_int(%8397)::Bool
└─────          goto #3243 if not %8400
2900 ┄ %8402  = φ (#2899 => %8398, #3242 => %9376)::Int64
│      %8403  = φ (#2899 => %8399, #3242 => %9377)::Int64
│      %8404  = StrideArraysCore.getfield(u, :data)::Vector{Float64}
│      %8405  = $(Expr(:gc_preserve_begin, :(%8404)))
│      %8406  = StrideArraysCore.getfield(u, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #2905 if not true
2901 ─ %8408  = Core.tuple(1, %6997, %8402, %6991, %6984)::NTuple{5, Int64}
│      %8409  = StrideArraysCore.getfield(%8406, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %8410  = Core.getfield(%8409, 5)::Int64
│      %8411  = Base.bitcast(UInt64, %8410)::UInt64
│      %8412  = Base.bitcast(Int64, %8411)::Int64
│      %8413  = Base.sle_int(1, %6997)::Bool
│      %8414  = Base.sle_int(%6997, 4)::Bool
│      %8415  = Base.and_int(%8413, %8414)::Bool
│      %8416  = Base.sle_int(1, %8402)::Bool
│      %8417  = Base.sle_int(%8402, 4)::Bool
│      %8418  = Base.and_int(%8416, %8417)::Bool
│      %8419  = Base.sle_int(1, %6991)::Bool
│      %8420  = Base.sle_int(%6991, 4)::Bool
│      %8421  = Base.and_int(%8419, %8420)::Bool
│      %8422  = Base.sub_int(%6984, 1)::Int64
│      %8423  = Base.bitcast(UInt64, %8422)::UInt64
│      %8424  = Base.bitcast(UInt64, %8412)::UInt64
│      %8425  = Base.ult_int(%8423, %8424)::Bool
│      %8426  = Base.and_int(%8425, true)::Bool
│      %8427  = Base.and_int(%8421, %8426)::Bool
│      %8428  = Base.and_int(%8418, %8427)::Bool
│      %8429  = Base.and_int(%8415, %8428)::Bool
│      %8430  = Base.and_int(true, %8429)::Bool
└─────          goto #2903 if not %8430
2902 ─          goto #2904
2903 ─          invoke Base.throw_boundserror(%8406::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %8408::NTuple{5, Int64})::Union{}
└─────          unreachable
2904 ─          nothing::Nothing
2905 ┄ %8436  = StrideArraysCore.getfield(%8406, :ptr)::Ptr{Float64}
│      %8437  = Base.sub_int(%6997, 1)::Int64
│      %8438  = Base.sub_int(%8402, 1)::Int64
│      %8439  = Base.sub_int(%6991, 1)::Int64
│      %8440  = Base.sub_int(%6984, 1)::Int64
└─────          goto #2914 if not true
2906 ┄ %8442  = φ (#2905 => 2, #2913 => %8454)::Int64
│      %8443  = Base.sle_int(1, %8442)::Bool
└─────          goto #2908 if not %8443
2907 ─ %8445  = Base.sle_int(%8442, 5)::Bool
└─────          goto #2909
2908 ─          nothing::Nothing
2909 ┄ %8448  = φ (#2907 => %8445, #2908 => false)::Bool
└─────          goto #2911 if not %8448
2910 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %8442, true)::Static.True
│      %8451  = Base.add_int(%8442, 1)::Int64
└─────          goto #2912
2911 ─          goto #2912
2912 ┄ %8454  = φ (#2910 => %8451)::Int64
│      %8455  = φ (#2910 => false, #2911 => true)::Bool
│      %8456  = Base.not_int(%8455)::Bool
└─────          goto #2914 if not %8456
2913 ─          goto #2906
2914 ┄          goto #2915
2915 ─          goto #2916
2916 ─ %8461  = Base.mul_int(%8440, 4)::Int64
│      %8462  = Base.add_int(%8439, %8461)::Int64
│      %8463  = Base.mul_int(%8462, 4)::Int64
│      %8464  = Base.add_int(%8438, %8463)::Int64
│      %8465  = Base.mul_int(%8464, 4)::Int64
│      %8466  = Base.add_int(%8437, %8465)::Int64
│      %8467  = Base.mul_int(%8466, 5)::Int64
│      %8468  = Base.add_int(0, %8467)::Int64
│      %8469  = Base.mul_int(8, %8468)::Int64
│      %8470  = Core.bitcast(Core.UInt, %8436)::UInt64
│      %8471  = Base.bitcast(UInt64, %8469)::UInt64
│      %8472  = Base.add_ptr(%8470, %8471)::UInt64
│      %8473  = Core.bitcast(Ptr{Float64}, %8472)::Ptr{Float64}
└─────          goto #2917
2917 ─ %8475  = Base.pointerref(%8473, 1, 1)::Float64
└─────          goto #2918
2918 ─          goto #2919
2919 ─          $(Expr(:gc_preserve_end, :(%8405)))
└─────          goto #2920
2920 ─          goto #2921
2921 ─ %8481  = StrideArraysCore.getfield(u, :data)::Vector{Float64}
│      %8482  = $(Expr(:gc_preserve_begin, :(%8481)))
│      %8483  = StrideArraysCore.getfield(u, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #2926 if not true
2922 ─ %8485  = Core.tuple(2, %6997, %8402, %6991, %6984)::NTuple{5, Int64}
│      %8486  = StrideArraysCore.getfield(%8483, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %8487  = Core.getfield(%8486, 5)::Int64
│      %8488  = Base.bitcast(UInt64, %8487)::UInt64
│      %8489  = Base.bitcast(Int64, %8488)::Int64
│      %8490  = Base.sle_int(1, %6997)::Bool
│      %8491  = Base.sle_int(%6997, 4)::Bool
│      %8492  = Base.and_int(%8490, %8491)::Bool
│      %8493  = Base.sle_int(1, %8402)::Bool
│      %8494  = Base.sle_int(%8402, 4)::Bool
│      %8495  = Base.and_int(%8493, %8494)::Bool
│      %8496  = Base.sle_int(1, %6991)::Bool
│      %8497  = Base.sle_int(%6991, 4)::Bool
│      %8498  = Base.and_int(%8496, %8497)::Bool
│      %8499  = Base.sub_int(%6984, 1)::Int64
│      %8500  = Base.bitcast(UInt64, %8499)::UInt64
│      %8501  = Base.bitcast(UInt64, %8489)::UInt64
│      %8502  = Base.ult_int(%8500, %8501)::Bool
│      %8503  = Base.and_int(%8502, true)::Bool
│      %8504  = Base.and_int(%8498, %8503)::Bool
│      %8505  = Base.and_int(%8495, %8504)::Bool
│      %8506  = Base.and_int(%8492, %8505)::Bool
│      %8507  = Base.and_int(true, %8506)::Bool
└─────          goto #2924 if not %8507
2923 ─          goto #2925
2924 ─          invoke Base.throw_boundserror(%8483::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %8485::NTuple{5, Int64})::Union{}
└─────          unreachable
2925 ─          nothing::Nothing
2926 ┄ %8513  = StrideArraysCore.getfield(%8483, :ptr)::Ptr{Float64}
│      %8514  = Base.sub_int(%6997, 1)::Int64
│      %8515  = Base.sub_int(%8402, 1)::Int64
│      %8516  = Base.sub_int(%6991, 1)::Int64
│      %8517  = Base.sub_int(%6984, 1)::Int64
└─────          goto #2935 if not true
2927 ┄ %8519  = φ (#2926 => 2, #2934 => %8531)::Int64
│      %8520  = Base.sle_int(1, %8519)::Bool
└─────          goto #2929 if not %8520
2928 ─ %8522  = Base.sle_int(%8519, 5)::Bool
└─────          goto #2930
2929 ─          nothing::Nothing
2930 ┄ %8525  = φ (#2928 => %8522, #2929 => false)::Bool
└─────          goto #2932 if not %8525
2931 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %8519, true)::Static.True
│      %8528  = Base.add_int(%8519, 1)::Int64
└─────          goto #2933
2932 ─          goto #2933
2933 ┄ %8531  = φ (#2931 => %8528)::Int64
│      %8532  = φ (#2931 => false, #2932 => true)::Bool
│      %8533  = Base.not_int(%8532)::Bool
└─────          goto #2935 if not %8533
2934 ─          goto #2927
2935 ┄          goto #2936
2936 ─          goto #2937
2937 ─ %8538  = Base.mul_int(%8517, 4)::Int64
│      %8539  = Base.add_int(%8516, %8538)::Int64
│      %8540  = Base.mul_int(%8539, 4)::Int64
│      %8541  = Base.add_int(%8515, %8540)::Int64
│      %8542  = Base.mul_int(%8541, 4)::Int64
│      %8543  = Base.add_int(%8514, %8542)::Int64
│      %8544  = Base.mul_int(%8543, 5)::Int64
│      %8545  = Base.add_int(1, %8544)::Int64
│      %8546  = Base.mul_int(8, %8545)::Int64
│      %8547  = Core.bitcast(Core.UInt, %8513)::UInt64
│      %8548  = Base.bitcast(UInt64, %8546)::UInt64
│      %8549  = Base.add_ptr(%8547, %8548)::UInt64
│      %8550  = Core.bitcast(Ptr{Float64}, %8549)::Ptr{Float64}
└─────          goto #2938
2938 ─ %8552  = Base.pointerref(%8550, 1, 1)::Float64
└─────          goto #2939
2939 ─          goto #2940
2940 ─          $(Expr(:gc_preserve_end, :(%8482)))
└─────          goto #2941
2941 ─          goto #2942
2942 ─ %8558  = StrideArraysCore.getfield(u, :data)::Vector{Float64}
│      %8559  = $(Expr(:gc_preserve_begin, :(%8558)))
│      %8560  = StrideArraysCore.getfield(u, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #2947 if not true
2943 ─ %8562  = Core.tuple(3, %6997, %8402, %6991, %6984)::NTuple{5, Int64}
│      %8563  = StrideArraysCore.getfield(%8560, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %8564  = Core.getfield(%8563, 5)::Int64
│      %8565  = Base.bitcast(UInt64, %8564)::UInt64
│      %8566  = Base.bitcast(Int64, %8565)::Int64
│      %8567  = Base.sle_int(1, %6997)::Bool
│      %8568  = Base.sle_int(%6997, 4)::Bool
│      %8569  = Base.and_int(%8567, %8568)::Bool
│      %8570  = Base.sle_int(1, %8402)::Bool
│      %8571  = Base.sle_int(%8402, 4)::Bool
│      %8572  = Base.and_int(%8570, %8571)::Bool
│      %8573  = Base.sle_int(1, %6991)::Bool
│      %8574  = Base.sle_int(%6991, 4)::Bool
│      %8575  = Base.and_int(%8573, %8574)::Bool
│      %8576  = Base.sub_int(%6984, 1)::Int64
│      %8577  = Base.bitcast(UInt64, %8576)::UInt64
│      %8578  = Base.bitcast(UInt64, %8566)::UInt64
│      %8579  = Base.ult_int(%8577, %8578)::Bool
│      %8580  = Base.and_int(%8579, true)::Bool
│      %8581  = Base.and_int(%8575, %8580)::Bool
│      %8582  = Base.and_int(%8572, %8581)::Bool
│      %8583  = Base.and_int(%8569, %8582)::Bool
│      %8584  = Base.and_int(true, %8583)::Bool
└─────          goto #2945 if not %8584
2944 ─          goto #2946
2945 ─          invoke Base.throw_boundserror(%8560::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %8562::NTuple{5, Int64})::Union{}
└─────          unreachable
2946 ─          nothing::Nothing
2947 ┄ %8590  = StrideArraysCore.getfield(%8560, :ptr)::Ptr{Float64}
│      %8591  = Base.sub_int(%6997, 1)::Int64
│      %8592  = Base.sub_int(%8402, 1)::Int64
│      %8593  = Base.sub_int(%6991, 1)::Int64
│      %8594  = Base.sub_int(%6984, 1)::Int64
└─────          goto #2956 if not true
2948 ┄ %8596  = φ (#2947 => 2, #2955 => %8608)::Int64
│      %8597  = Base.sle_int(1, %8596)::Bool
└─────          goto #2950 if not %8597
2949 ─ %8599  = Base.sle_int(%8596, 5)::Bool
└─────          goto #2951
2950 ─          nothing::Nothing
2951 ┄ %8602  = φ (#2949 => %8599, #2950 => false)::Bool
└─────          goto #2953 if not %8602
2952 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %8596, true)::Static.True
│      %8605  = Base.add_int(%8596, 1)::Int64
└─────          goto #2954
2953 ─          goto #2954
2954 ┄ %8608  = φ (#2952 => %8605)::Int64
│      %8609  = φ (#2952 => false, #2953 => true)::Bool
│      %8610  = Base.not_int(%8609)::Bool
└─────          goto #2956 if not %8610
2955 ─          goto #2948
2956 ┄          goto #2957
2957 ─          goto #2958
2958 ─ %8615  = Base.mul_int(%8594, 4)::Int64
│      %8616  = Base.add_int(%8593, %8615)::Int64
│      %8617  = Base.mul_int(%8616, 4)::Int64
│      %8618  = Base.add_int(%8592, %8617)::Int64
│      %8619  = Base.mul_int(%8618, 4)::Int64
│      %8620  = Base.add_int(%8591, %8619)::Int64
│      %8621  = Base.mul_int(%8620, 5)::Int64
│      %8622  = Base.add_int(2, %8621)::Int64
│      %8623  = Base.mul_int(8, %8622)::Int64
│      %8624  = Core.bitcast(Core.UInt, %8590)::UInt64
│      %8625  = Base.bitcast(UInt64, %8623)::UInt64
│      %8626  = Base.add_ptr(%8624, %8625)::UInt64
│      %8627  = Core.bitcast(Ptr{Float64}, %8626)::Ptr{Float64}
└─────          goto #2959
2959 ─ %8629  = Base.pointerref(%8627, 1, 1)::Float64
└─────          goto #2960
2960 ─          goto #2961
2961 ─          $(Expr(:gc_preserve_end, :(%8559)))
└─────          goto #2962
2962 ─          goto #2963
2963 ─ %8635  = StrideArraysCore.getfield(u, :data)::Vector{Float64}
│      %8636  = $(Expr(:gc_preserve_begin, :(%8635)))
│      %8637  = StrideArraysCore.getfield(u, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #2968 if not true
2964 ─ %8639  = Core.tuple(4, %6997, %8402, %6991, %6984)::NTuple{5, Int64}
│      %8640  = StrideArraysCore.getfield(%8637, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %8641  = Core.getfield(%8640, 5)::Int64
│      %8642  = Base.bitcast(UInt64, %8641)::UInt64
│      %8643  = Base.bitcast(Int64, %8642)::Int64
│      %8644  = Base.sle_int(1, %6997)::Bool
│      %8645  = Base.sle_int(%6997, 4)::Bool
│      %8646  = Base.and_int(%8644, %8645)::Bool
│      %8647  = Base.sle_int(1, %8402)::Bool
│      %8648  = Base.sle_int(%8402, 4)::Bool
│      %8649  = Base.and_int(%8647, %8648)::Bool
│      %8650  = Base.sle_int(1, %6991)::Bool
│      %8651  = Base.sle_int(%6991, 4)::Bool
│      %8652  = Base.and_int(%8650, %8651)::Bool
│      %8653  = Base.sub_int(%6984, 1)::Int64
│      %8654  = Base.bitcast(UInt64, %8653)::UInt64
│      %8655  = Base.bitcast(UInt64, %8643)::UInt64
│      %8656  = Base.ult_int(%8654, %8655)::Bool
│      %8657  = Base.and_int(%8656, true)::Bool
│      %8658  = Base.and_int(%8652, %8657)::Bool
│      %8659  = Base.and_int(%8649, %8658)::Bool
│      %8660  = Base.and_int(%8646, %8659)::Bool
│      %8661  = Base.and_int(true, %8660)::Bool
└─────          goto #2966 if not %8661
2965 ─          goto #2967
2966 ─          invoke Base.throw_boundserror(%8637::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %8639::NTuple{5, Int64})::Union{}
└─────          unreachable
2967 ─          nothing::Nothing
2968 ┄ %8667  = StrideArraysCore.getfield(%8637, :ptr)::Ptr{Float64}
│      %8668  = Base.sub_int(%6997, 1)::Int64
│      %8669  = Base.sub_int(%8402, 1)::Int64
│      %8670  = Base.sub_int(%6991, 1)::Int64
│      %8671  = Base.sub_int(%6984, 1)::Int64
└─────          goto #2977 if not true
2969 ┄ %8673  = φ (#2968 => 2, #2976 => %8685)::Int64
│      %8674  = Base.sle_int(1, %8673)::Bool
└─────          goto #2971 if not %8674
2970 ─ %8676  = Base.sle_int(%8673, 5)::Bool
└─────          goto #2972
2971 ─          nothing::Nothing
2972 ┄ %8679  = φ (#2970 => %8676, #2971 => false)::Bool
└─────          goto #2974 if not %8679
2973 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %8673, true)::Static.True
│      %8682  = Base.add_int(%8673, 1)::Int64
└─────          goto #2975
2974 ─          goto #2975
2975 ┄ %8685  = φ (#2973 => %8682)::Int64
│      %8686  = φ (#2973 => false, #2974 => true)::Bool
│      %8687  = Base.not_int(%8686)::Bool
└─────          goto #2977 if not %8687
2976 ─          goto #2969
2977 ┄          goto #2978
2978 ─          goto #2979
2979 ─ %8692  = Base.mul_int(%8671, 4)::Int64
│      %8693  = Base.add_int(%8670, %8692)::Int64
│      %8694  = Base.mul_int(%8693, 4)::Int64
│      %8695  = Base.add_int(%8669, %8694)::Int64
│      %8696  = Base.mul_int(%8695, 4)::Int64
│      %8697  = Base.add_int(%8668, %8696)::Int64
│      %8698  = Base.mul_int(%8697, 5)::Int64
│      %8699  = Base.add_int(3, %8698)::Int64
│      %8700  = Base.mul_int(8, %8699)::Int64
│      %8701  = Core.bitcast(Core.UInt, %8667)::UInt64
│      %8702  = Base.bitcast(UInt64, %8700)::UInt64
│      %8703  = Base.add_ptr(%8701, %8702)::UInt64
│      %8704  = Core.bitcast(Ptr{Float64}, %8703)::Ptr{Float64}
└─────          goto #2980
2980 ─ %8706  = Base.pointerref(%8704, 1, 1)::Float64
└─────          goto #2981
2981 ─          goto #2982
2982 ─          $(Expr(:gc_preserve_end, :(%8636)))
└─────          goto #2983
2983 ─          goto #2984
2984 ─ %8712  = StrideArraysCore.getfield(u, :data)::Vector{Float64}
│      %8713  = $(Expr(:gc_preserve_begin, :(%8712)))
│      %8714  = StrideArraysCore.getfield(u, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #2989 if not true
2985 ─ %8716  = Core.tuple(5, %6997, %8402, %6991, %6984)::NTuple{5, Int64}
│      %8717  = StrideArraysCore.getfield(%8714, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %8718  = Core.getfield(%8717, 5)::Int64
│      %8719  = Base.bitcast(UInt64, %8718)::UInt64
│      %8720  = Base.bitcast(Int64, %8719)::Int64
│      %8721  = Base.sle_int(1, %6997)::Bool
│      %8722  = Base.sle_int(%6997, 4)::Bool
│      %8723  = Base.and_int(%8721, %8722)::Bool
│      %8724  = Base.sle_int(1, %8402)::Bool
│      %8725  = Base.sle_int(%8402, 4)::Bool
│      %8726  = Base.and_int(%8724, %8725)::Bool
│      %8727  = Base.sle_int(1, %6991)::Bool
│      %8728  = Base.sle_int(%6991, 4)::Bool
│      %8729  = Base.and_int(%8727, %8728)::Bool
│      %8730  = Base.sub_int(%6984, 1)::Int64
│      %8731  = Base.bitcast(UInt64, %8730)::UInt64
│      %8732  = Base.bitcast(UInt64, %8720)::UInt64
│      %8733  = Base.ult_int(%8731, %8732)::Bool
│      %8734  = Base.and_int(%8733, true)::Bool
│      %8735  = Base.and_int(%8729, %8734)::Bool
│      %8736  = Base.and_int(%8726, %8735)::Bool
│      %8737  = Base.and_int(%8723, %8736)::Bool
│      %8738  = Base.and_int(true, %8737)::Bool
└─────          goto #2987 if not %8738
2986 ─          goto #2988
2987 ─          invoke Base.throw_boundserror(%8714::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %8716::NTuple{5, Int64})::Union{}
└─────          unreachable
2988 ─          nothing::Nothing
2989 ┄ %8744  = StrideArraysCore.getfield(%8714, :ptr)::Ptr{Float64}
│      %8745  = Base.sub_int(%6997, 1)::Int64
│      %8746  = Base.sub_int(%8402, 1)::Int64
│      %8747  = Base.sub_int(%6991, 1)::Int64
│      %8748  = Base.sub_int(%6984, 1)::Int64
└─────          goto #2998 if not true
2990 ┄ %8750  = φ (#2989 => 2, #2997 => %8762)::Int64
│      %8751  = Base.sle_int(1, %8750)::Bool
└─────          goto #2992 if not %8751
2991 ─ %8753  = Base.sle_int(%8750, 5)::Bool
└─────          goto #2993
2992 ─          nothing::Nothing
2993 ┄ %8756  = φ (#2991 => %8753, #2992 => false)::Bool
└─────          goto #2995 if not %8756
2994 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %8750, true)::Static.True
│      %8759  = Base.add_int(%8750, 1)::Int64
└─────          goto #2996
2995 ─          goto #2996
2996 ┄ %8762  = φ (#2994 => %8759)::Int64
│      %8763  = φ (#2994 => false, #2995 => true)::Bool
│      %8764  = Base.not_int(%8763)::Bool
└─────          goto #2998 if not %8764
2997 ─          goto #2990
2998 ┄          goto #2999
2999 ─          goto #3000
3000 ─ %8769  = Base.mul_int(%8748, 4)::Int64
│      %8770  = Base.add_int(%8747, %8769)::Int64
│      %8771  = Base.mul_int(%8770, 4)::Int64
│      %8772  = Base.add_int(%8746, %8771)::Int64
│      %8773  = Base.mul_int(%8772, 4)::Int64
│      %8774  = Base.add_int(%8745, %8773)::Int64
│      %8775  = Base.mul_int(%8774, 5)::Int64
│      %8776  = Base.add_int(4, %8775)::Int64
│      %8777  = Base.mul_int(8, %8776)::Int64
│      %8778  = Core.bitcast(Core.UInt, %8744)::UInt64
│      %8779  = Base.bitcast(UInt64, %8777)::UInt64
│      %8780  = Base.add_ptr(%8778, %8779)::UInt64
│      %8781  = Core.bitcast(Ptr{Float64}, %8780)::Ptr{Float64}
└─────          goto #3001
3001 ─ %8783  = Base.pointerref(%8781, 1, 1)::Float64
└─────          goto #3002
3002 ─          goto #3003
3003 ─          $(Expr(:gc_preserve_end, :(%8713)))
└─────          goto #3004
3004 ─          goto #3005
3005 ─          goto #3006
3006 ─          goto #3007
3007 ─          goto #3009
3008 ─          nothing::Nothing
3009 ┄          goto #3011
3010 ─          nothing::Nothing
3011 ┄          goto #3012
3012 ─          goto #3014
3013 ─          nothing::Nothing
3014 ┄          goto #3015
3015 ─          goto #3017
3016 ─          nothing::Nothing
3017 ┄          goto #3019
3018 ─          nothing::Nothing
3019 ┄          goto #3020
3020 ─          goto #3022
3021 ─          nothing::Nothing
3022 ┄          goto #3023
3023 ─          goto #3025
3024 ─          nothing::Nothing
3025 ┄          goto #3027
3026 ─          nothing::Nothing
3027 ┄          goto #3028
3028 ─          goto #3030
3029 ─          nothing::Nothing
3030 ┄          goto #3031
3031 ─          goto #3033
3032 ─          nothing::Nothing
3033 ┄          goto #3035
3034 ─          nothing::Nothing
3035 ┄          goto #3036
3036 ─          goto #3038
3037 ─          nothing::Nothing
3038 ┄          goto #3039
3039 ─ %8823  = Base.div_float(%7147, %7070)::Float64
│      %8824  = Base.div_float(%7224, %7070)::Float64
│      %8825  = Base.div_float(%7301, %7070)::Float64
│      %8826  = Base.getfield(equations, :gamma)::Float64
│      %8827  = Base.sub_float(%8826, 1.0)::Float64
│      %8828  = Base.mul_float(%7147, %8823)::Float64
│      %8829  = Base.muladd_float(%7224, %8824, %8828)::Float64
│      %8830  = Base.muladd_float(%7301, %8825, %8829)::Float64
│      %8831  = Base.muladd_float(-0.5, %8830, %7378)::Float64
│      %8832  = Base.mul_float(%8827, %8831)::Float64
└─────          goto #3040
3040 ─          goto #3042
3041 ─          nothing::Nothing
3042 ┄          goto #3044
3043 ─          nothing::Nothing
3044 ┄          goto #3045
3045 ─          goto #3047
3046 ─          nothing::Nothing
3047 ┄          goto #3048
3048 ─          goto #3050
3049 ─          nothing::Nothing
3050 ┄          goto #3052
3051 ─          nothing::Nothing
3052 ┄          goto #3053
3053 ─          goto #3055
3054 ─          nothing::Nothing
3055 ┄          goto #3056
3056 ─          goto #3058
3057 ─          nothing::Nothing
3058 ┄          goto #3060
3059 ─          nothing::Nothing
3060 ┄          goto #3061
3061 ─          goto #3063
3062 ─          nothing::Nothing
3063 ┄          goto #3064
3064 ─          goto #3066
3065 ─          nothing::Nothing
3066 ┄          goto #3068
3067 ─          nothing::Nothing
3068 ┄          goto #3069
3069 ─          goto #3071
3070 ─          nothing::Nothing
3071 ┄          goto #3072
3072 ─          goto #3074
3073 ─          nothing::Nothing
3074 ┄          goto #3076
3075 ─          nothing::Nothing
3076 ┄          goto #3077
3077 ─          goto #3079
3078 ─          nothing::Nothing
3079 ┄          goto #3080
3080 ─          goto #3082
3081 ─          nothing::Nothing
3082 ┄          goto #3084
3083 ─          nothing::Nothing
3084 ┄          goto #3085
3085 ─          goto #3087
3086 ─          nothing::Nothing
3087 ┄          goto #3088
3088 ─          goto #3090
3089 ─          nothing::Nothing
3090 ┄          goto #3092
3091 ─          nothing::Nothing
3092 ┄          goto #3093
3093 ─          goto #3095
3094 ─          nothing::Nothing
3095 ┄          goto #3096
3096 ─          goto #3098
3097 ─          nothing::Nothing
3098 ┄          goto #3100
3099 ─          nothing::Nothing
3100 ┄          goto #3101
3101 ─          goto #3103
3102 ─          nothing::Nothing
3103 ┄          goto #3104
3104 ─ %8898  = Base.div_float(%8552, %8475)::Float64
│      %8899  = Base.div_float(%8629, %8475)::Float64
│      %8900  = Base.div_float(%8706, %8475)::Float64
│      %8901  = Base.getfield(equations, :gamma)::Float64
│      %8902  = Base.sub_float(%8901, 1.0)::Float64
│      %8903  = Base.mul_float(%8552, %8898)::Float64
│      %8904  = Base.muladd_float(%8629, %8899, %8903)::Float64
│      %8905  = Base.muladd_float(%8706, %8900, %8904)::Float64
│      %8906  = Base.muladd_float(-0.5, %8905, %8783)::Float64
│      %8907  = Base.mul_float(%8902, %8906)::Float64
└─────          goto #3105
3105 ─          goto #3107
3106 ─          nothing::Nothing
3107 ┄          goto #3109
3108 ─          nothing::Nothing
3109 ┄          goto #3110
3110 ─          goto #3112
3111 ─          nothing::Nothing
3112 ┄          goto #3113
3113 ─          goto #3115
3114 ─          nothing::Nothing
3115 ┄          goto #3117
3116 ─          nothing::Nothing
3117 ┄          goto #3118
3118 ─          goto #3120
3119 ─          nothing::Nothing
3120 ┄          goto #3121
3121 ─          goto #3123
3122 ─          nothing::Nothing
3123 ┄          goto #3125
3124 ─          nothing::Nothing
3125 ┄          goto #3126
3126 ─          goto #3128
3127 ─          nothing::Nothing
3128 ┄          goto #3129
3129 ─          goto #3131
3130 ─          nothing::Nothing
3131 ┄          goto #3133
3132 ─          nothing::Nothing
3133 ┄          goto #3134
3134 ─          goto #3136
3135 ─          nothing::Nothing
3136 ┄          goto #3137
3137 ─ %8941  = Base.muladd_float(-2.0, %8475, %7070)::Float64
│      %8942  = Base.mul_float(%7070, %8941)::Float64
│      %8943  = Base.muladd_float(%8475, %8475, %8942)::Float64
│      %8944  = Base.muladd_float(2.0, %8475, %7070)::Float64
│      %8945  = Base.mul_float(%7070, %8944)::Float64
│      %8946  = Base.muladd_float(%8475, %8475, %8945)::Float64
│      %8947  = Base.div_float(%8943, %8946)::Float64
│      %8948  = Base.lt_float(%8947, 0.0001)::Bool
└─────          goto #3139 if not %8948
3138 ─ %8950  = Base.add_float(%7070, %8475)::Float64
│      %8951  = Base.muladd_float(%8947, 0.2857142857142857, 0.4)::Float64
│      %8952  = Base.muladd_float(%8947, %8951, 0.6666666666666666)::Float64
│      %8953  = Base.muladd_float(%8947, %8952, 2.0)::Float64
│      %8954  = Base.div_float(%8950, %8953)::Float64
└─────          goto #3140
3139 ─ %8956  = Base.sub_float(%8475, %7070)::Float64
│      %8957  = Base.div_float(%8475, %7070)::Float64
│      %8958  = $(Expr(:foreigncall, "llvm.log.f64", Float64, svec(Float64), 0, :(:llvmcall), :(%8957), :(%8957)))::Float64
│      %8959  = Base.div_float(%8956, %8958)::Float64
└─────          goto #3140
3140 ┄ %8961  = φ (#3138 => %8954, #3139 => %8959)::Float64
│      %8962  = Base.mul_float(%7070, %8907)::Float64
│      %8963  = Base.mul_float(%8475, %8832)::Float64
│      %8964  = Base.muladd_float(-2.0, %8963, %8962)::Float64
│      %8965  = Base.mul_float(%8962, %8964)::Float64
│      %8966  = Base.muladd_float(%8963, %8963, %8965)::Float64
│      %8967  = Base.muladd_float(2.0, %8963, %8962)::Float64
│      %8968  = Base.mul_float(%8962, %8967)::Float64
│      %8969  = Base.muladd_float(%8963, %8963, %8968)::Float64
│      %8970  = Base.div_float(%8966, %8969)::Float64
│      %8971  = Base.lt_float(%8970, 0.0001)::Bool
└─────          goto #3142 if not %8971
3141 ─ %8973  = Base.muladd_float(%8970, 0.2857142857142857, 0.4)::Float64
│      %8974  = Base.muladd_float(%8970, %8973, 0.6666666666666666)::Float64
│      %8975  = Base.muladd_float(%8970, %8974, 2.0)::Float64
│      %8976  = Base.add_float(%8962, %8963)::Float64
│      %8977  = Base.div_float(%8975, %8976)::Float64
└─────          goto #3143
3142 ─ %8979  = Base.div_float(%8963, %8962)::Float64
│      %8980  = $(Expr(:foreigncall, "llvm.log.f64", Float64, svec(Float64), 0, :(:llvmcall), :(%8979), :(%8979)))::Float64
│      %8981  = Base.sub_float(%8963, %8962)::Float64
│      %8982  = Base.div_float(%8980, %8981)::Float64
└─────          goto #3143
3143 ┄ %8984  = φ (#3141 => %8977, #3142 => %8982)::Float64
│      %8985  = Base.mul_float(%8832, %8907)::Float64
│      %8986  = Base.mul_float(%8985, %8984)::Float64
│      %8987  = Base.add_float(%8823, %8898)::Float64
│      %8988  = Base.mul_float(0.5, %8987)::Float64
│      %8989  = Base.add_float(%8824, %8899)::Float64
│      %8990  = Base.mul_float(0.5, %8989)::Float64
│      %8991  = Base.add_float(%8825, %8900)::Float64
│      %8992  = Base.mul_float(0.5, %8991)::Float64
│      %8993  = Base.add_float(%8832, %8907)::Float64
│      %8994  = Base.mul_float(0.5, %8993)::Float64
│      %8995  = Base.mul_float(%8823, %8898)::Float64
│      %8996  = Base.muladd_float(%8824, %8899, %8995)::Float64
│      %8997  = Base.muladd_float(%8825, %8900, %8996)::Float64
│      %8998  = Base.mul_float(0.5, %8997)::Float64
│      %8999  = Base.mul_float(%8961, %8990)::Float64
│      %9000  = Base.mul_float(%8999, %8988)::Float64
│      %9001  = Base.muladd_float(%8999, %8990, %8994)::Float64
│      %9002  = Base.mul_float(%8999, %8992)::Float64
│      %9003  = Base.mul_float(%8832, %8899)::Float64
│      %9004  = Base.muladd_float(%8907, %8824, %9003)::Float64
│      %9005  = Base.getfield(equations, :inv_gamma_minus_one)::Float64
│      %9006  = Base.muladd_float(%8986, %9005, %8998)::Float64
│      %9007  = Base.mul_float(%8999, %9006)::Float64
│      %9008  = Base.muladd_float(0.5, %9004, %9007)::Float64
│      %9009  = Core.tuple(%8999, %9000, %9001, %9002, %9008)::NTuple{5, Float64}
└─────          goto #3144
3144 ─ %9011  = Base.arrayref(false, %6989, %6994, %8402)::Float64
│      %9012  = Base.copysign_float(0.0, %9011)::Float64
│      %9013  = Core.ifelse(true, %9011, %9012)::Float64
└─────          goto #3190 if not true
3145 ┄ %9015  = φ (#3144 => 1, #3189 => %9184)::Int64
│      %9016  = φ (#3144 => 1, #3189 => %9185)::Int64
│      %9017  = Base.getfield(%9009, %9015, true)::Float64
│      %9018  = StrideArraysCore.getfield(du, :data)::Vector{Float64}
│      %9019  = $(Expr(:gc_preserve_begin, :(%9018)))
│      %9020  = StrideArraysCore.getfield(du, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #3150 if not true
3146 ─ %9022  = Core.tuple(%9015, %6997, %6994, %6991, %6984)::NTuple{5, Int64}
│      %9023  = StrideArraysCore.getfield(%9020, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %9024  = Core.getfield(%9023, 5)::Int64
│      %9025  = Base.bitcast(UInt64, %9024)::UInt64
│      %9026  = Base.bitcast(Int64, %9025)::Int64
│      %9027  = Base.sle_int(1, %9015)::Bool
│      %9028  = Base.sle_int(%9015, 5)::Bool
│      %9029  = Base.and_int(%9027, %9028)::Bool
│      %9030  = Base.sle_int(1, %6997)::Bool
│      %9031  = Base.sle_int(%6997, 4)::Bool
│      %9032  = Base.and_int(%9030, %9031)::Bool
│      %9033  = Base.sle_int(1, %6994)::Bool
│      %9034  = Base.sle_int(%6994, 4)::Bool
│      %9035  = Base.and_int(%9033, %9034)::Bool
│      %9036  = Base.sle_int(1, %6991)::Bool
│      %9037  = Base.sle_int(%6991, 4)::Bool
│      %9038  = Base.and_int(%9036, %9037)::Bool
│      %9039  = Base.sub_int(%6984, 1)::Int64
│      %9040  = Base.bitcast(UInt64, %9039)::UInt64
│      %9041  = Base.bitcast(UInt64, %9026)::UInt64
│      %9042  = Base.ult_int(%9040, %9041)::Bool
│      %9043  = Base.and_int(%9042, true)::Bool
│      %9044  = Base.and_int(%9038, %9043)::Bool
│      %9045  = Base.and_int(%9035, %9044)::Bool
│      %9046  = Base.and_int(%9032, %9045)::Bool
│      %9047  = Base.and_int(%9029, %9046)::Bool
└─────          goto #3148 if not %9047
3147 ─          goto #3149
3148 ─          invoke Base.throw_boundserror(%9020::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %9022::NTuple{5, Int64})::Union{}
└─────          unreachable
3149 ─          nothing::Nothing
3150 ┄ %9053  = StrideArraysCore.getfield(%9020, :ptr)::Ptr{Float64}
│      %9054  = Base.sub_int(%9015, 1)::Int64
│      %9055  = Base.sub_int(%6997, 1)::Int64
│      %9056  = Base.sub_int(%6994, 1)::Int64
│      %9057  = Base.sub_int(%6991, 1)::Int64
│      %9058  = Base.sub_int(%6984, 1)::Int64
└─────          goto #3159 if not true
3151 ┄ %9060  = φ (#3150 => 2, #3158 => %9072)::Int64
│      %9061  = Base.sle_int(1, %9060)::Bool
└─────          goto #3153 if not %9061
3152 ─ %9063  = Base.sle_int(%9060, 5)::Bool
└─────          goto #3154
3153 ─          nothing::Nothing
3154 ┄ %9066  = φ (#3152 => %9063, #3153 => false)::Bool
└─────          goto #3156 if not %9066
3155 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %9060, true)::Static.True
│      %9069  = Base.add_int(%9060, 1)::Int64
└─────          goto #3157
3156 ─          goto #3157
3157 ┄ %9072  = φ (#3155 => %9069)::Int64
│      %9073  = φ (#3155 => false, #3156 => true)::Bool
│      %9074  = Base.not_int(%9073)::Bool
└─────          goto #3159 if not %9074
3158 ─          goto #3151
3159 ┄          goto #3160
3160 ─          goto #3161
3161 ─ %9079  = Base.mul_int(%9058, 4)::Int64
│      %9080  = Base.add_int(%9057, %9079)::Int64
│      %9081  = Base.mul_int(%9080, 4)::Int64
│      %9082  = Base.add_int(%9056, %9081)::Int64
│      %9083  = Base.mul_int(%9082, 4)::Int64
│      %9084  = Base.add_int(%9055, %9083)::Int64
│      %9085  = Base.mul_int(%9084, 5)::Int64
│      %9086  = Base.add_int(%9054, %9085)::Int64
│      %9087  = Base.mul_int(8, %9086)::Int64
│      %9088  = Core.bitcast(Core.UInt, %9053)::UInt64
│      %9089  = Base.bitcast(UInt64, %9087)::UInt64
│      %9090  = Base.add_ptr(%9088, %9089)::UInt64
│      %9091  = Core.bitcast(Ptr{Float64}, %9090)::Ptr{Float64}
└─────          goto #3162
3162 ─ %9093  = Base.pointerref(%9091, 1, 1)::Float64
└─────          goto #3163
3163 ─          goto #3164
3164 ─          $(Expr(:gc_preserve_end, :(%9019)))
└─────          goto #3165
3165 ─ %9098  = Base.muladd_float(%9013, %9017, %9093)::Float64
│      %9099  = StrideArraysCore.getfield(du, :data)::Vector{Float64}
│      %9100  = $(Expr(:gc_preserve_begin, :(%9099)))
│      %9101  = StrideArraysCore.getfield(du, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #3170 if not true
3166 ─ %9103  = Core.tuple(%9015, %6997, %6994, %6991, %6984)::NTuple{5, Int64}
│      %9104  = StrideArraysCore.getfield(%9101, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %9105  = Core.getfield(%9104, 5)::Int64
│      %9106  = Base.bitcast(UInt64, %9105)::UInt64
│      %9107  = Base.bitcast(Int64, %9106)::Int64
│      %9108  = Base.sle_int(1, %9015)::Bool
│      %9109  = Base.sle_int(%9015, 5)::Bool
│      %9110  = Base.and_int(%9108, %9109)::Bool
│      %9111  = Base.sle_int(1, %6997)::Bool
│      %9112  = Base.sle_int(%6997, 4)::Bool
│      %9113  = Base.and_int(%9111, %9112)::Bool
│      %9114  = Base.sle_int(1, %6994)::Bool
│      %9115  = Base.sle_int(%6994, 4)::Bool
│      %9116  = Base.and_int(%9114, %9115)::Bool
│      %9117  = Base.sle_int(1, %6991)::Bool
│      %9118  = Base.sle_int(%6991, 4)::Bool
│      %9119  = Base.and_int(%9117, %9118)::Bool
│      %9120  = Base.sub_int(%6984, 1)::Int64
│      %9121  = Base.bitcast(UInt64, %9120)::UInt64
│      %9122  = Base.bitcast(UInt64, %9107)::UInt64
│      %9123  = Base.ult_int(%9121, %9122)::Bool
│      %9124  = Base.and_int(%9123, true)::Bool
│      %9125  = Base.and_int(%9119, %9124)::Bool
│      %9126  = Base.and_int(%9116, %9125)::Bool
│      %9127  = Base.and_int(%9113, %9126)::Bool
│      %9128  = Base.and_int(%9110, %9127)::Bool
└─────          goto #3168 if not %9128
3167 ─          goto #3169
3168 ─          invoke Base.throw_boundserror(%9101::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %9103::NTuple{5, Int64})::Union{}
└─────          unreachable
3169 ─          nothing::Nothing
3170 ┄ %9134  = StrideArraysCore.getfield(%9101, :ptr)::Ptr{Float64}
│      %9135  = Base.sub_int(%9015, 1)::Int64
│      %9136  = Base.sub_int(%6997, 1)::Int64
│      %9137  = Base.sub_int(%6994, 1)::Int64
│      %9138  = Base.sub_int(%6991, 1)::Int64
│      %9139  = Base.sub_int(%6984, 1)::Int64
└─────          goto #3179 if not true
3171 ┄ %9141  = φ (#3170 => 2, #3178 => %9153)::Int64
│      %9142  = Base.sle_int(1, %9141)::Bool
└─────          goto #3173 if not %9142
3172 ─ %9144  = Base.sle_int(%9141, 5)::Bool
└─────          goto #3174
3173 ─          nothing::Nothing
3174 ┄ %9147  = φ (#3172 => %9144, #3173 => false)::Bool
└─────          goto #3176 if not %9147
3175 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %9141, true)::Static.True
│      %9150  = Base.add_int(%9141, 1)::Int64
└─────          goto #3177
3176 ─          goto #3177
3177 ┄ %9153  = φ (#3175 => %9150)::Int64
│      %9154  = φ (#3175 => false, #3176 => true)::Bool
│      %9155  = Base.not_int(%9154)::Bool
└─────          goto #3179 if not %9155
3178 ─          goto #3171
3179 ┄          goto #3180
3180 ─          goto #3181
3181 ─ %9160  = Base.mul_int(%9139, 4)::Int64
│      %9161  = Base.add_int(%9138, %9160)::Int64
│      %9162  = Base.mul_int(%9161, 4)::Int64
│      %9163  = Base.add_int(%9137, %9162)::Int64
│      %9164  = Base.mul_int(%9163, 4)::Int64
│      %9165  = Base.add_int(%9136, %9164)::Int64
│      %9166  = Base.mul_int(%9165, 5)::Int64
│      %9167  = Base.add_int(%9135, %9166)::Int64
│      %9168  = Base.mul_int(8, %9167)::Int64
│      %9169  = Core.bitcast(Core.UInt, %9134)::UInt64
│      %9170  = Base.bitcast(UInt64, %9168)::UInt64
│      %9171  = Base.add_ptr(%9169, %9170)::UInt64
│      %9172  = Core.bitcast(Ptr{Float64}, %9171)::Ptr{Float64}
└─────          goto #3182
3182 ─          Base.pointerset(%9172, %9098, 1, 1)::Ptr{Float64}
└─────          goto #3183
3183 ─          goto #3184
3184 ─          $(Expr(:gc_preserve_end, :(%9100)))
└─────          goto #3185
3185 ─ %9179  = (%9016 === 5)::Bool
└─────          goto #3187 if not %9179
3186 ─          goto #3188
3187 ─ %9182  = Base.add_int(%9016, 1)::Int64
└─────          goto #3188
3188 ┄ %9184  = φ (#3187 => %9182)::Int64
│      %9185  = φ (#3187 => %9182)::Int64
│      %9186  = φ (#3186 => true, #3187 => false)::Bool
│      %9187  = Base.not_int(%9186)::Bool
└─────          goto #3190 if not %9187
3189 ─          goto #3145
3190 ┄          goto #3191
3191 ─ %9191  = Base.arrayref(false, %6989, %8402, %6994)::Float64
│      %9192  = Base.copysign_float(0.0, %9191)::Float64
│      %9193  = Core.ifelse(true, %9191, %9192)::Float64
└─────          goto #3237 if not true
3192 ┄ %9195  = φ (#3191 => 1, #3236 => %9364)::Int64
│      %9196  = φ (#3191 => 1, #3236 => %9365)::Int64
│      %9197  = Base.getfield(%9009, %9195, true)::Float64
│      %9198  = StrideArraysCore.getfield(du, :data)::Vector{Float64}
│      %9199  = $(Expr(:gc_preserve_begin, :(%9198)))
│      %9200  = StrideArraysCore.getfield(du, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #3197 if not true
3193 ─ %9202  = Core.tuple(%9195, %6997, %8402, %6991, %6984)::NTuple{5, Int64}
│      %9203  = StrideArraysCore.getfield(%9200, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %9204  = Core.getfield(%9203, 5)::Int64
│      %9205  = Base.bitcast(UInt64, %9204)::UInt64
│      %9206  = Base.bitcast(Int64, %9205)::Int64
│      %9207  = Base.sle_int(1, %9195)::Bool
│      %9208  = Base.sle_int(%9195, 5)::Bool
│      %9209  = Base.and_int(%9207, %9208)::Bool
│      %9210  = Base.sle_int(1, %6997)::Bool
│      %9211  = Base.sle_int(%6997, 4)::Bool
│      %9212  = Base.and_int(%9210, %9211)::Bool
│      %9213  = Base.sle_int(1, %8402)::Bool
│      %9214  = Base.sle_int(%8402, 4)::Bool
│      %9215  = Base.and_int(%9213, %9214)::Bool
│      %9216  = Base.sle_int(1, %6991)::Bool
│      %9217  = Base.sle_int(%6991, 4)::Bool
│      %9218  = Base.and_int(%9216, %9217)::Bool
│      %9219  = Base.sub_int(%6984, 1)::Int64
│      %9220  = Base.bitcast(UInt64, %9219)::UInt64
│      %9221  = Base.bitcast(UInt64, %9206)::UInt64
│      %9222  = Base.ult_int(%9220, %9221)::Bool
│      %9223  = Base.and_int(%9222, true)::Bool
│      %9224  = Base.and_int(%9218, %9223)::Bool
│      %9225  = Base.and_int(%9215, %9224)::Bool
│      %9226  = Base.and_int(%9212, %9225)::Bool
│      %9227  = Base.and_int(%9209, %9226)::Bool
└─────          goto #3195 if not %9227
3194 ─          goto #3196
3195 ─          invoke Base.throw_boundserror(%9200::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %9202::NTuple{5, Int64})::Union{}
└─────          unreachable
3196 ─          nothing::Nothing
3197 ┄ %9233  = StrideArraysCore.getfield(%9200, :ptr)::Ptr{Float64}
│      %9234  = Base.sub_int(%9195, 1)::Int64
│      %9235  = Base.sub_int(%6997, 1)::Int64
│      %9236  = Base.sub_int(%8402, 1)::Int64
│      %9237  = Base.sub_int(%6991, 1)::Int64
│      %9238  = Base.sub_int(%6984, 1)::Int64
└─────          goto #3206 if not true
3198 ┄ %9240  = φ (#3197 => 2, #3205 => %9252)::Int64
│      %9241  = Base.sle_int(1, %9240)::Bool
└─────          goto #3200 if not %9241
3199 ─ %9243  = Base.sle_int(%9240, 5)::Bool
└─────          goto #3201
3200 ─          nothing::Nothing
3201 ┄ %9246  = φ (#3199 => %9243, #3200 => false)::Bool
└─────          goto #3203 if not %9246
3202 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %9240, true)::Static.True
│      %9249  = Base.add_int(%9240, 1)::Int64
└─────          goto #3204
3203 ─          goto #3204
3204 ┄ %9252  = φ (#3202 => %9249)::Int64
│      %9253  = φ (#3202 => false, #3203 => true)::Bool
│      %9254  = Base.not_int(%9253)::Bool
└─────          goto #3206 if not %9254
3205 ─          goto #3198
3206 ┄          goto #3207
3207 ─          goto #3208
3208 ─ %9259  = Base.mul_int(%9238, 4)::Int64
│      %9260  = Base.add_int(%9237, %9259)::Int64
│      %9261  = Base.mul_int(%9260, 4)::Int64
│      %9262  = Base.add_int(%9236, %9261)::Int64
│      %9263  = Base.mul_int(%9262, 4)::Int64
│      %9264  = Base.add_int(%9235, %9263)::Int64
│      %9265  = Base.mul_int(%9264, 5)::Int64
│      %9266  = Base.add_int(%9234, %9265)::Int64
│      %9267  = Base.mul_int(8, %9266)::Int64
│      %9268  = Core.bitcast(Core.UInt, %9233)::UInt64
│      %9269  = Base.bitcast(UInt64, %9267)::UInt64
│      %9270  = Base.add_ptr(%9268, %9269)::UInt64
│      %9271  = Core.bitcast(Ptr{Float64}, %9270)::Ptr{Float64}
└─────          goto #3209
3209 ─ %9273  = Base.pointerref(%9271, 1, 1)::Float64
└─────          goto #3210
3210 ─          goto #3211
3211 ─          $(Expr(:gc_preserve_end, :(%9199)))
└─────          goto #3212
3212 ─ %9278  = Base.muladd_float(%9193, %9197, %9273)::Float64
│      %9279  = StrideArraysCore.getfield(du, :data)::Vector{Float64}
│      %9280  = $(Expr(:gc_preserve_begin, :(%9279)))
│      %9281  = StrideArraysCore.getfield(du, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #3217 if not true
3213 ─ %9283  = Core.tuple(%9195, %6997, %8402, %6991, %6984)::NTuple{5, Int64}
│      %9284  = StrideArraysCore.getfield(%9281, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %9285  = Core.getfield(%9284, 5)::Int64
│      %9286  = Base.bitcast(UInt64, %9285)::UInt64
│      %9287  = Base.bitcast(Int64, %9286)::Int64
│      %9288  = Base.sle_int(1, %9195)::Bool
│      %9289  = Base.sle_int(%9195, 5)::Bool
│      %9290  = Base.and_int(%9288, %9289)::Bool
│      %9291  = Base.sle_int(1, %6997)::Bool
│      %9292  = Base.sle_int(%6997, 4)::Bool
│      %9293  = Base.and_int(%9291, %9292)::Bool
│      %9294  = Base.sle_int(1, %8402)::Bool
│      %9295  = Base.sle_int(%8402, 4)::Bool
│      %9296  = Base.and_int(%9294, %9295)::Bool
│      %9297  = Base.sle_int(1, %6991)::Bool
│      %9298  = Base.sle_int(%6991, 4)::Bool
│      %9299  = Base.and_int(%9297, %9298)::Bool
│      %9300  = Base.sub_int(%6984, 1)::Int64
│      %9301  = Base.bitcast(UInt64, %9300)::UInt64
│      %9302  = Base.bitcast(UInt64, %9287)::UInt64
│      %9303  = Base.ult_int(%9301, %9302)::Bool
│      %9304  = Base.and_int(%9303, true)::Bool
│      %9305  = Base.and_int(%9299, %9304)::Bool
│      %9306  = Base.and_int(%9296, %9305)::Bool
│      %9307  = Base.and_int(%9293, %9306)::Bool
│      %9308  = Base.and_int(%9290, %9307)::Bool
└─────          goto #3215 if not %9308
3214 ─          goto #3216
3215 ─          invoke Base.throw_boundserror(%9281::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %9283::NTuple{5, Int64})::Union{}
└─────          unreachable
3216 ─          nothing::Nothing
3217 ┄ %9314  = StrideArraysCore.getfield(%9281, :ptr)::Ptr{Float64}
│      %9315  = Base.sub_int(%9195, 1)::Int64
│      %9316  = Base.sub_int(%6997, 1)::Int64
│      %9317  = Base.sub_int(%8402, 1)::Int64
│      %9318  = Base.sub_int(%6991, 1)::Int64
│      %9319  = Base.sub_int(%6984, 1)::Int64
└─────          goto #3226 if not true
3218 ┄ %9321  = φ (#3217 => 2, #3225 => %9333)::Int64
│      %9322  = Base.sle_int(1, %9321)::Bool
└─────          goto #3220 if not %9322
3219 ─ %9324  = Base.sle_int(%9321, 5)::Bool
└─────          goto #3221
3220 ─          nothing::Nothing
3221 ┄ %9327  = φ (#3219 => %9324, #3220 => false)::Bool
└─────          goto #3223 if not %9327
3222 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %9321, true)::Static.True
│      %9330  = Base.add_int(%9321, 1)::Int64
└─────          goto #3224
3223 ─          goto #3224
3224 ┄ %9333  = φ (#3222 => %9330)::Int64
│      %9334  = φ (#3222 => false, #3223 => true)::Bool
│      %9335  = Base.not_int(%9334)::Bool
└─────          goto #3226 if not %9335
3225 ─          goto #3218
3226 ┄          goto #3227
3227 ─          goto #3228
3228 ─ %9340  = Base.mul_int(%9319, 4)::Int64
│      %9341  = Base.add_int(%9318, %9340)::Int64
│      %9342  = Base.mul_int(%9341, 4)::Int64
│      %9343  = Base.add_int(%9317, %9342)::Int64
│      %9344  = Base.mul_int(%9343, 4)::Int64
│      %9345  = Base.add_int(%9316, %9344)::Int64
│      %9346  = Base.mul_int(%9345, 5)::Int64
│      %9347  = Base.add_int(%9315, %9346)::Int64
│      %9348  = Base.mul_int(8, %9347)::Int64
│      %9349  = Core.bitcast(Core.UInt, %9314)::UInt64
│      %9350  = Base.bitcast(UInt64, %9348)::UInt64
│      %9351  = Base.add_ptr(%9349, %9350)::UInt64
│      %9352  = Core.bitcast(Ptr{Float64}, %9351)::Ptr{Float64}
└─────          goto #3229
3229 ─          Base.pointerset(%9352, %9278, 1, 1)::Ptr{Float64}
└─────          goto #3230
3230 ─          goto #3231
3231 ─          $(Expr(:gc_preserve_end, :(%9280)))
└─────          goto #3232
3232 ─ %9359  = (%9196 === 5)::Bool
└─────          goto #3234 if not %9359
3233 ─          goto #3235
3234 ─ %9362  = Base.add_int(%9196, 1)::Int64
└─────          goto #3235
3235 ┄ %9364  = φ (#3234 => %9362)::Int64
│      %9365  = φ (#3234 => %9362)::Int64
│      %9366  = φ (#3233 => true, #3234 => false)::Bool
│      %9367  = Base.not_int(%9366)::Bool
└─────          goto #3237 if not %9367
3236 ─          goto #3192
3237 ┄          goto #3238
3238 ─ %9371  = (%8403 === %8390)::Bool
└─────          goto #3240 if not %9371
3239 ─          goto #3241
3240 ─ %9374  = Base.add_int(%8403, 1)::Int64
└─────          goto #3241
3241 ┄ %9376  = φ (#3240 => %9374)::Int64
│      %9377  = φ (#3240 => %9374)::Int64
│      %9378  = φ (#3239 => true, #3240 => false)::Bool
│      %9379  = Base.not_int(%9378)::Bool
└─────          goto #3243 if not %9379
3242 ─          goto #2900
3243 ┄ %9382  = Base.add_int(%6991, 1)::Int64
│      %9383  = Base.sle_int(%9382, 4)::Bool
└─────          goto #3245 if not %9383
3244 ─          goto #3246
3245 ─ %9386  = Base.sub_int(%9382, 1)::Int64
└─────          goto #3246
3246 ┄ %9388  = φ (#3244 => 4, #3245 => %9386)::Int64
└─────          goto #3247
3247 ─          goto #3248
3248 ─ %9391  = Base.slt_int(%9388, %9382)::Bool
└─────          goto #3250 if not %9391
3249 ─          goto #3251
3250 ─          goto #3251
3251 ┄ %9395  = φ (#3249 => true, #3250 => false)::Bool
│      %9396  = φ (#3250 => %9382)::Int64
│      %9397  = φ (#3250 => %9382)::Int64
│      %9398  = Base.not_int(%9395)::Bool
└─────          goto #3595 if not %9398
3252 ┄ %9400  = φ (#3251 => %9396, #3594 => %10374)::Int64
│      %9401  = φ (#3251 => %9397, #3594 => %10375)::Int64
│      %9402  = StrideArraysCore.getfield(u, :data)::Vector{Float64}
│      %9403  = $(Expr(:gc_preserve_begin, :(%9402)))
│      %9404  = StrideArraysCore.getfield(u, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #3257 if not true
3253 ─ %9406  = Core.tuple(1, %6997, %6994, %9400, %6984)::NTuple{5, Int64}
│      %9407  = StrideArraysCore.getfield(%9404, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %9408  = Core.getfield(%9407, 5)::Int64
│      %9409  = Base.bitcast(UInt64, %9408)::UInt64
│      %9410  = Base.bitcast(Int64, %9409)::Int64
│      %9411  = Base.sle_int(1, %6997)::Bool
│      %9412  = Base.sle_int(%6997, 4)::Bool
│      %9413  = Base.and_int(%9411, %9412)::Bool
│      %9414  = Base.sle_int(1, %6994)::Bool
│      %9415  = Base.sle_int(%6994, 4)::Bool
│      %9416  = Base.and_int(%9414, %9415)::Bool
│      %9417  = Base.sle_int(1, %9400)::Bool
│      %9418  = Base.sle_int(%9400, 4)::Bool
│      %9419  = Base.and_int(%9417, %9418)::Bool
│      %9420  = Base.sub_int(%6984, 1)::Int64
│      %9421  = Base.bitcast(UInt64, %9420)::UInt64
│      %9422  = Base.bitcast(UInt64, %9410)::UInt64
│      %9423  = Base.ult_int(%9421, %9422)::Bool
│      %9424  = Base.and_int(%9423, true)::Bool
│      %9425  = Base.and_int(%9419, %9424)::Bool
│      %9426  = Base.and_int(%9416, %9425)::Bool
│      %9427  = Base.and_int(%9413, %9426)::Bool
│      %9428  = Base.and_int(true, %9427)::Bool
└─────          goto #3255 if not %9428
3254 ─          goto #3256
3255 ─          invoke Base.throw_boundserror(%9404::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %9406::NTuple{5, Int64})::Union{}
└─────          unreachable
3256 ─          nothing::Nothing
3257 ┄ %9434  = StrideArraysCore.getfield(%9404, :ptr)::Ptr{Float64}
│      %9435  = Base.sub_int(%6997, 1)::Int64
│      %9436  = Base.sub_int(%6994, 1)::Int64
│      %9437  = Base.sub_int(%9400, 1)::Int64
│      %9438  = Base.sub_int(%6984, 1)::Int64
└─────          goto #3266 if not true
3258 ┄ %9440  = φ (#3257 => 2, #3265 => %9452)::Int64
│      %9441  = Base.sle_int(1, %9440)::Bool
└─────          goto #3260 if not %9441
3259 ─ %9443  = Base.sle_int(%9440, 5)::Bool
└─────          goto #3261
3260 ─          nothing::Nothing
3261 ┄ %9446  = φ (#3259 => %9443, #3260 => false)::Bool
└─────          goto #3263 if not %9446
3262 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %9440, true)::Static.True
│      %9449  = Base.add_int(%9440, 1)::Int64
└─────          goto #3264
3263 ─          goto #3264
3264 ┄ %9452  = φ (#3262 => %9449)::Int64
│      %9453  = φ (#3262 => false, #3263 => true)::Bool
│      %9454  = Base.not_int(%9453)::Bool
└─────          goto #3266 if not %9454
3265 ─          goto #3258
3266 ┄          goto #3267
3267 ─          goto #3268
3268 ─ %9459  = Base.mul_int(%9438, 4)::Int64
│      %9460  = Base.add_int(%9437, %9459)::Int64
│      %9461  = Base.mul_int(%9460, 4)::Int64
│      %9462  = Base.add_int(%9436, %9461)::Int64
│      %9463  = Base.mul_int(%9462, 4)::Int64
│      %9464  = Base.add_int(%9435, %9463)::Int64
│      %9465  = Base.mul_int(%9464, 5)::Int64
│      %9466  = Base.add_int(0, %9465)::Int64
│      %9467  = Base.mul_int(8, %9466)::Int64
│      %9468  = Core.bitcast(Core.UInt, %9434)::UInt64
│      %9469  = Base.bitcast(UInt64, %9467)::UInt64
│      %9470  = Base.add_ptr(%9468, %9469)::UInt64
│      %9471  = Core.bitcast(Ptr{Float64}, %9470)::Ptr{Float64}
└─────          goto #3269
3269 ─ %9473  = Base.pointerref(%9471, 1, 1)::Float64
└─────          goto #3270
3270 ─          goto #3271
3271 ─          $(Expr(:gc_preserve_end, :(%9403)))
└─────          goto #3272
3272 ─          goto #3273
3273 ─ %9479  = StrideArraysCore.getfield(u, :data)::Vector{Float64}
│      %9480  = $(Expr(:gc_preserve_begin, :(%9479)))
│      %9481  = StrideArraysCore.getfield(u, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #3278 if not true
3274 ─ %9483  = Core.tuple(2, %6997, %6994, %9400, %6984)::NTuple{5, Int64}
│      %9484  = StrideArraysCore.getfield(%9481, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %9485  = Core.getfield(%9484, 5)::Int64
│      %9486  = Base.bitcast(UInt64, %9485)::UInt64
│      %9487  = Base.bitcast(Int64, %9486)::Int64
│      %9488  = Base.sle_int(1, %6997)::Bool
│      %9489  = Base.sle_int(%6997, 4)::Bool
│      %9490  = Base.and_int(%9488, %9489)::Bool
│      %9491  = Base.sle_int(1, %6994)::Bool
│      %9492  = Base.sle_int(%6994, 4)::Bool
│      %9493  = Base.and_int(%9491, %9492)::Bool
│      %9494  = Base.sle_int(1, %9400)::Bool
│      %9495  = Base.sle_int(%9400, 4)::Bool
│      %9496  = Base.and_int(%9494, %9495)::Bool
│      %9497  = Base.sub_int(%6984, 1)::Int64
│      %9498  = Base.bitcast(UInt64, %9497)::UInt64
│      %9499  = Base.bitcast(UInt64, %9487)::UInt64
│      %9500  = Base.ult_int(%9498, %9499)::Bool
│      %9501  = Base.and_int(%9500, true)::Bool
│      %9502  = Base.and_int(%9496, %9501)::Bool
│      %9503  = Base.and_int(%9493, %9502)::Bool
│      %9504  = Base.and_int(%9490, %9503)::Bool
│      %9505  = Base.and_int(true, %9504)::Bool
└─────          goto #3276 if not %9505
3275 ─          goto #3277
3276 ─          invoke Base.throw_boundserror(%9481::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %9483::NTuple{5, Int64})::Union{}
└─────          unreachable
3277 ─          nothing::Nothing
3278 ┄ %9511  = StrideArraysCore.getfield(%9481, :ptr)::Ptr{Float64}
│      %9512  = Base.sub_int(%6997, 1)::Int64
│      %9513  = Base.sub_int(%6994, 1)::Int64
│      %9514  = Base.sub_int(%9400, 1)::Int64
│      %9515  = Base.sub_int(%6984, 1)::Int64
└─────          goto #3287 if not true
3279 ┄ %9517  = φ (#3278 => 2, #3286 => %9529)::Int64
│      %9518  = Base.sle_int(1, %9517)::Bool
└─────          goto #3281 if not %9518
3280 ─ %9520  = Base.sle_int(%9517, 5)::Bool
└─────          goto #3282
3281 ─          nothing::Nothing
3282 ┄ %9523  = φ (#3280 => %9520, #3281 => false)::Bool
└─────          goto #3284 if not %9523
3283 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %9517, true)::Static.True
│      %9526  = Base.add_int(%9517, 1)::Int64
└─────          goto #3285
3284 ─          goto #3285
3285 ┄ %9529  = φ (#3283 => %9526)::Int64
│      %9530  = φ (#3283 => false, #3284 => true)::Bool
│      %9531  = Base.not_int(%9530)::Bool
└─────          goto #3287 if not %9531
3286 ─          goto #3279
3287 ┄          goto #3288
3288 ─          goto #3289
3289 ─ %9536  = Base.mul_int(%9515, 4)::Int64
│      %9537  = Base.add_int(%9514, %9536)::Int64
│      %9538  = Base.mul_int(%9537, 4)::Int64
│      %9539  = Base.add_int(%9513, %9538)::Int64
│      %9540  = Base.mul_int(%9539, 4)::Int64
│      %9541  = Base.add_int(%9512, %9540)::Int64
│      %9542  = Base.mul_int(%9541, 5)::Int64
│      %9543  = Base.add_int(1, %9542)::Int64
│      %9544  = Base.mul_int(8, %9543)::Int64
│      %9545  = Core.bitcast(Core.UInt, %9511)::UInt64
│      %9546  = Base.bitcast(UInt64, %9544)::UInt64
│      %9547  = Base.add_ptr(%9545, %9546)::UInt64
│      %9548  = Core.bitcast(Ptr{Float64}, %9547)::Ptr{Float64}
└─────          goto #3290
3290 ─ %9550  = Base.pointerref(%9548, 1, 1)::Float64
└─────          goto #3291
3291 ─          goto #3292
3292 ─          $(Expr(:gc_preserve_end, :(%9480)))
└─────          goto #3293
3293 ─          goto #3294
3294 ─ %9556  = StrideArraysCore.getfield(u, :data)::Vector{Float64}
│      %9557  = $(Expr(:gc_preserve_begin, :(%9556)))
│      %9558  = StrideArraysCore.getfield(u, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #3299 if not true
3295 ─ %9560  = Core.tuple(3, %6997, %6994, %9400, %6984)::NTuple{5, Int64}
│      %9561  = StrideArraysCore.getfield(%9558, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %9562  = Core.getfield(%9561, 5)::Int64
│      %9563  = Base.bitcast(UInt64, %9562)::UInt64
│      %9564  = Base.bitcast(Int64, %9563)::Int64
│      %9565  = Base.sle_int(1, %6997)::Bool
│      %9566  = Base.sle_int(%6997, 4)::Bool
│      %9567  = Base.and_int(%9565, %9566)::Bool
│      %9568  = Base.sle_int(1, %6994)::Bool
│      %9569  = Base.sle_int(%6994, 4)::Bool
│      %9570  = Base.and_int(%9568, %9569)::Bool
│      %9571  = Base.sle_int(1, %9400)::Bool
│      %9572  = Base.sle_int(%9400, 4)::Bool
│      %9573  = Base.and_int(%9571, %9572)::Bool
│      %9574  = Base.sub_int(%6984, 1)::Int64
│      %9575  = Base.bitcast(UInt64, %9574)::UInt64
│      %9576  = Base.bitcast(UInt64, %9564)::UInt64
│      %9577  = Base.ult_int(%9575, %9576)::Bool
│      %9578  = Base.and_int(%9577, true)::Bool
│      %9579  = Base.and_int(%9573, %9578)::Bool
│      %9580  = Base.and_int(%9570, %9579)::Bool
│      %9581  = Base.and_int(%9567, %9580)::Bool
│      %9582  = Base.and_int(true, %9581)::Bool
└─────          goto #3297 if not %9582
3296 ─          goto #3298
3297 ─          invoke Base.throw_boundserror(%9558::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %9560::NTuple{5, Int64})::Union{}
└─────          unreachable
3298 ─          nothing::Nothing
3299 ┄ %9588  = StrideArraysCore.getfield(%9558, :ptr)::Ptr{Float64}
│      %9589  = Base.sub_int(%6997, 1)::Int64
│      %9590  = Base.sub_int(%6994, 1)::Int64
│      %9591  = Base.sub_int(%9400, 1)::Int64
│      %9592  = Base.sub_int(%6984, 1)::Int64
└─────          goto #3308 if not true
3300 ┄ %9594  = φ (#3299 => 2, #3307 => %9606)::Int64
│      %9595  = Base.sle_int(1, %9594)::Bool
└─────          goto #3302 if not %9595
3301 ─ %9597  = Base.sle_int(%9594, 5)::Bool
└─────          goto #3303
3302 ─          nothing::Nothing
3303 ┄ %9600  = φ (#3301 => %9597, #3302 => false)::Bool
└─────          goto #3305 if not %9600
3304 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %9594, true)::Static.True
│      %9603  = Base.add_int(%9594, 1)::Int64
└─────          goto #3306
3305 ─          goto #3306
3306 ┄ %9606  = φ (#3304 => %9603)::Int64
│      %9607  = φ (#3304 => false, #3305 => true)::Bool
│      %9608  = Base.not_int(%9607)::Bool
└─────          goto #3308 if not %9608
3307 ─          goto #3300
3308 ┄          goto #3309
3309 ─          goto #3310
3310 ─ %9613  = Base.mul_int(%9592, 4)::Int64
│      %9614  = Base.add_int(%9591, %9613)::Int64
│      %9615  = Base.mul_int(%9614, 4)::Int64
│      %9616  = Base.add_int(%9590, %9615)::Int64
│      %9617  = Base.mul_int(%9616, 4)::Int64
│      %9618  = Base.add_int(%9589, %9617)::Int64
│      %9619  = Base.mul_int(%9618, 5)::Int64
│      %9620  = Base.add_int(2, %9619)::Int64
│      %9621  = Base.mul_int(8, %9620)::Int64
│      %9622  = Core.bitcast(Core.UInt, %9588)::UInt64
│      %9623  = Base.bitcast(UInt64, %9621)::UInt64
│      %9624  = Base.add_ptr(%9622, %9623)::UInt64
│      %9625  = Core.bitcast(Ptr{Float64}, %9624)::Ptr{Float64}
└─────          goto #3311
3311 ─ %9627  = Base.pointerref(%9625, 1, 1)::Float64
└─────          goto #3312
3312 ─          goto #3313
3313 ─          $(Expr(:gc_preserve_end, :(%9557)))
└─────          goto #3314
3314 ─          goto #3315
3315 ─ %9633  = StrideArraysCore.getfield(u, :data)::Vector{Float64}
│      %9634  = $(Expr(:gc_preserve_begin, :(%9633)))
│      %9635  = StrideArraysCore.getfield(u, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #3320 if not true
3316 ─ %9637  = Core.tuple(4, %6997, %6994, %9400, %6984)::NTuple{5, Int64}
│      %9638  = StrideArraysCore.getfield(%9635, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %9639  = Core.getfield(%9638, 5)::Int64
│      %9640  = Base.bitcast(UInt64, %9639)::UInt64
│      %9641  = Base.bitcast(Int64, %9640)::Int64
│      %9642  = Base.sle_int(1, %6997)::Bool
│      %9643  = Base.sle_int(%6997, 4)::Bool
│      %9644  = Base.and_int(%9642, %9643)::Bool
│      %9645  = Base.sle_int(1, %6994)::Bool
│      %9646  = Base.sle_int(%6994, 4)::Bool
│      %9647  = Base.and_int(%9645, %9646)::Bool
│      %9648  = Base.sle_int(1, %9400)::Bool
│      %9649  = Base.sle_int(%9400, 4)::Bool
│      %9650  = Base.and_int(%9648, %9649)::Bool
│      %9651  = Base.sub_int(%6984, 1)::Int64
│      %9652  = Base.bitcast(UInt64, %9651)::UInt64
│      %9653  = Base.bitcast(UInt64, %9641)::UInt64
│      %9654  = Base.ult_int(%9652, %9653)::Bool
│      %9655  = Base.and_int(%9654, true)::Bool
│      %9656  = Base.and_int(%9650, %9655)::Bool
│      %9657  = Base.and_int(%9647, %9656)::Bool
│      %9658  = Base.and_int(%9644, %9657)::Bool
│      %9659  = Base.and_int(true, %9658)::Bool
└─────          goto #3318 if not %9659
3317 ─          goto #3319
3318 ─          invoke Base.throw_boundserror(%9635::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %9637::NTuple{5, Int64})::Union{}
└─────          unreachable
3319 ─          nothing::Nothing
3320 ┄ %9665  = StrideArraysCore.getfield(%9635, :ptr)::Ptr{Float64}
│      %9666  = Base.sub_int(%6997, 1)::Int64
│      %9667  = Base.sub_int(%6994, 1)::Int64
│      %9668  = Base.sub_int(%9400, 1)::Int64
│      %9669  = Base.sub_int(%6984, 1)::Int64
└─────          goto #3329 if not true
3321 ┄ %9671  = φ (#3320 => 2, #3328 => %9683)::Int64
│      %9672  = Base.sle_int(1, %9671)::Bool
└─────          goto #3323 if not %9672
3322 ─ %9674  = Base.sle_int(%9671, 5)::Bool
└─────          goto #3324
3323 ─          nothing::Nothing
3324 ┄ %9677  = φ (#3322 => %9674, #3323 => false)::Bool
└─────          goto #3326 if not %9677
3325 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %9671, true)::Static.True
│      %9680  = Base.add_int(%9671, 1)::Int64
└─────          goto #3327
3326 ─          goto #3327
3327 ┄ %9683  = φ (#3325 => %9680)::Int64
│      %9684  = φ (#3325 => false, #3326 => true)::Bool
│      %9685  = Base.not_int(%9684)::Bool
└─────          goto #3329 if not %9685
3328 ─          goto #3321
3329 ┄          goto #3330
3330 ─          goto #3331
3331 ─ %9690  = Base.mul_int(%9669, 4)::Int64
│      %9691  = Base.add_int(%9668, %9690)::Int64
│      %9692  = Base.mul_int(%9691, 4)::Int64
│      %9693  = Base.add_int(%9667, %9692)::Int64
│      %9694  = Base.mul_int(%9693, 4)::Int64
│      %9695  = Base.add_int(%9666, %9694)::Int64
│      %9696  = Base.mul_int(%9695, 5)::Int64
│      %9697  = Base.add_int(3, %9696)::Int64
│      %9698  = Base.mul_int(8, %9697)::Int64
│      %9699  = Core.bitcast(Core.UInt, %9665)::UInt64
│      %9700  = Base.bitcast(UInt64, %9698)::UInt64
│      %9701  = Base.add_ptr(%9699, %9700)::UInt64
│      %9702  = Core.bitcast(Ptr{Float64}, %9701)::Ptr{Float64}
└─────          goto #3332
3332 ─ %9704  = Base.pointerref(%9702, 1, 1)::Float64
└─────          goto #3333
3333 ─          goto #3334
3334 ─          $(Expr(:gc_preserve_end, :(%9634)))
└─────          goto #3335
3335 ─          goto #3336
3336 ─ %9710  = StrideArraysCore.getfield(u, :data)::Vector{Float64}
│      %9711  = $(Expr(:gc_preserve_begin, :(%9710)))
│      %9712  = StrideArraysCore.getfield(u, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #3341 if not true
3337 ─ %9714  = Core.tuple(5, %6997, %6994, %9400, %6984)::NTuple{5, Int64}
│      %9715  = StrideArraysCore.getfield(%9712, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %9716  = Core.getfield(%9715, 5)::Int64
│      %9717  = Base.bitcast(UInt64, %9716)::UInt64
│      %9718  = Base.bitcast(Int64, %9717)::Int64
│      %9719  = Base.sle_int(1, %6997)::Bool
│      %9720  = Base.sle_int(%6997, 4)::Bool
│      %9721  = Base.and_int(%9719, %9720)::Bool
│      %9722  = Base.sle_int(1, %6994)::Bool
│      %9723  = Base.sle_int(%6994, 4)::Bool
│      %9724  = Base.and_int(%9722, %9723)::Bool
│      %9725  = Base.sle_int(1, %9400)::Bool
│      %9726  = Base.sle_int(%9400, 4)::Bool
│      %9727  = Base.and_int(%9725, %9726)::Bool
│      %9728  = Base.sub_int(%6984, 1)::Int64
│      %9729  = Base.bitcast(UInt64, %9728)::UInt64
│      %9730  = Base.bitcast(UInt64, %9718)::UInt64
│      %9731  = Base.ult_int(%9729, %9730)::Bool
│      %9732  = Base.and_int(%9731, true)::Bool
│      %9733  = Base.and_int(%9727, %9732)::Bool
│      %9734  = Base.and_int(%9724, %9733)::Bool
│      %9735  = Base.and_int(%9721, %9734)::Bool
│      %9736  = Base.and_int(true, %9735)::Bool
└─────          goto #3339 if not %9736
3338 ─          goto #3340
3339 ─          invoke Base.throw_boundserror(%9712::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %9714::NTuple{5, Int64})::Union{}
└─────          unreachable
3340 ─          nothing::Nothing
3341 ┄ %9742  = StrideArraysCore.getfield(%9712, :ptr)::Ptr{Float64}
│      %9743  = Base.sub_int(%6997, 1)::Int64
│      %9744  = Base.sub_int(%6994, 1)::Int64
│      %9745  = Base.sub_int(%9400, 1)::Int64
│      %9746  = Base.sub_int(%6984, 1)::Int64
└─────          goto #3350 if not true
3342 ┄ %9748  = φ (#3341 => 2, #3349 => %9760)::Int64
│      %9749  = Base.sle_int(1, %9748)::Bool
└─────          goto #3344 if not %9749
3343 ─ %9751  = Base.sle_int(%9748, 5)::Bool
└─────          goto #3345
3344 ─          nothing::Nothing
3345 ┄ %9754  = φ (#3343 => %9751, #3344 => false)::Bool
└─────          goto #3347 if not %9754
3346 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %9748, true)::Static.True
│      %9757  = Base.add_int(%9748, 1)::Int64
└─────          goto #3348
3347 ─          goto #3348
3348 ┄ %9760  = φ (#3346 => %9757)::Int64
│      %9761  = φ (#3346 => false, #3347 => true)::Bool
│      %9762  = Base.not_int(%9761)::Bool
└─────          goto #3350 if not %9762
3349 ─          goto #3342
3350 ┄          goto #3351
3351 ─          goto #3352
3352 ─ %9767  = Base.mul_int(%9746, 4)::Int64
│      %9768  = Base.add_int(%9745, %9767)::Int64
│      %9769  = Base.mul_int(%9768, 4)::Int64
│      %9770  = Base.add_int(%9744, %9769)::Int64
│      %9771  = Base.mul_int(%9770, 4)::Int64
│      %9772  = Base.add_int(%9743, %9771)::Int64
│      %9773  = Base.mul_int(%9772, 5)::Int64
│      %9774  = Base.add_int(4, %9773)::Int64
│      %9775  = Base.mul_int(8, %9774)::Int64
│      %9776  = Core.bitcast(Core.UInt, %9742)::UInt64
│      %9777  = Base.bitcast(UInt64, %9775)::UInt64
│      %9778  = Base.add_ptr(%9776, %9777)::UInt64
│      %9779  = Core.bitcast(Ptr{Float64}, %9778)::Ptr{Float64}
└─────          goto #3353
3353 ─ %9781  = Base.pointerref(%9779, 1, 1)::Float64
└─────          goto #3354
3354 ─          goto #3355
3355 ─          $(Expr(:gc_preserve_end, :(%9711)))
└─────          goto #3356
3356 ─          goto #3357
3357 ─          goto #3358
3358 ─          goto #3359
3359 ─          goto #3361
3360 ─          nothing::Nothing
3361 ┄          goto #3363
3362 ─          nothing::Nothing
3363 ┄          goto #3364
3364 ─          goto #3366
3365 ─          nothing::Nothing
3366 ┄          goto #3367
3367 ─          goto #3369
3368 ─          nothing::Nothing
3369 ┄          goto #3371
3370 ─          nothing::Nothing
3371 ┄          goto #3372
3372 ─          goto #3374
3373 ─          nothing::Nothing
3374 ┄          goto #3375
3375 ─          goto #3377
3376 ─          nothing::Nothing
3377 ┄          goto #3379
3378 ─          nothing::Nothing
3379 ┄          goto #3380
3380 ─          goto #3382
3381 ─          nothing::Nothing
3382 ┄          goto #3383
3383 ─          goto #3385
3384 ─          nothing::Nothing
3385 ┄          goto #3387
3386 ─          nothing::Nothing
3387 ┄          goto #3388
3388 ─          goto #3390
3389 ─          nothing::Nothing
3390 ┄          goto #3391
3391 ─ %9821  = Base.div_float(%7147, %7070)::Float64
│      %9822  = Base.div_float(%7224, %7070)::Float64
│      %9823  = Base.div_float(%7301, %7070)::Float64
│      %9824  = Base.getfield(equations, :gamma)::Float64
│      %9825  = Base.sub_float(%9824, 1.0)::Float64
│      %9826  = Base.mul_float(%7147, %9821)::Float64
│      %9827  = Base.muladd_float(%7224, %9822, %9826)::Float64
│      %9828  = Base.muladd_float(%7301, %9823, %9827)::Float64
│      %9829  = Base.muladd_float(-0.5, %9828, %7378)::Float64
│      %9830  = Base.mul_float(%9825, %9829)::Float64
└─────          goto #3392
3392 ─          goto #3394
3393 ─          nothing::Nothing
3394 ┄          goto #3396
3395 ─          nothing::Nothing
3396 ┄          goto #3397
3397 ─          goto #3399
3398 ─          nothing::Nothing
3399 ┄          goto #3400
3400 ─          goto #3402
3401 ─          nothing::Nothing
3402 ┄          goto #3404
3403 ─          nothing::Nothing
3404 ┄          goto #3405
3405 ─          goto #3407
3406 ─          nothing::Nothing
3407 ┄          goto #3408
3408 ─          goto #3410
3409 ─          nothing::Nothing
3410 ┄          goto #3412
3411 ─          nothing::Nothing
3412 ┄          goto #3413
3413 ─          goto #3415
3414 ─          nothing::Nothing
3415 ┄          goto #3416
3416 ─          goto #3418
3417 ─          nothing::Nothing
3418 ┄          goto #3420
3419 ─          nothing::Nothing
3420 ┄          goto #3421
3421 ─          goto #3423
3422 ─          nothing::Nothing
3423 ┄          goto #3424
3424 ─          goto #3426
3425 ─          nothing::Nothing
3426 ┄          goto #3428
3427 ─          nothing::Nothing
3428 ┄          goto #3429
3429 ─          goto #3431
3430 ─          nothing::Nothing
3431 ┄          goto #3432
3432 ─          goto #3434
3433 ─          nothing::Nothing
3434 ┄          goto #3436
3435 ─          nothing::Nothing
3436 ┄          goto #3437
3437 ─          goto #3439
3438 ─          nothing::Nothing
3439 ┄          goto #3440
3440 ─          goto #3442
3441 ─          nothing::Nothing
3442 ┄          goto #3444
3443 ─          nothing::Nothing
3444 ┄          goto #3445
3445 ─          goto #3447
3446 ─          nothing::Nothing
3447 ┄          goto #3448
3448 ─          goto #3450
3449 ─          nothing::Nothing
3450 ┄          goto #3452
3451 ─          nothing::Nothing
3452 ┄          goto #3453
3453 ─          goto #3455
3454 ─          nothing::Nothing
3455 ┄          goto #3456
3456 ─ %9896  = Base.div_float(%9550, %9473)::Float64
│      %9897  = Base.div_float(%9627, %9473)::Float64
│      %9898  = Base.div_float(%9704, %9473)::Float64
│      %9899  = Base.getfield(equations, :gamma)::Float64
│      %9900  = Base.sub_float(%9899, 1.0)::Float64
│      %9901  = Base.mul_float(%9550, %9896)::Float64
│      %9902  = Base.muladd_float(%9627, %9897, %9901)::Float64
│      %9903  = Base.muladd_float(%9704, %9898, %9902)::Float64
│      %9904  = Base.muladd_float(-0.5, %9903, %9781)::Float64
│      %9905  = Base.mul_float(%9900, %9904)::Float64
└─────          goto #3457
3457 ─          goto #3459
3458 ─          nothing::Nothing
3459 ┄          goto #3461
3460 ─          nothing::Nothing
3461 ┄          goto #3462
3462 ─          goto #3464
3463 ─          nothing::Nothing
3464 ┄          goto #3465
3465 ─          goto #3467
3466 ─          nothing::Nothing
3467 ┄          goto #3469
3468 ─          nothing::Nothing
3469 ┄          goto #3470
3470 ─          goto #3472
3471 ─          nothing::Nothing
3472 ┄          goto #3473
3473 ─          goto #3475
3474 ─          nothing::Nothing
3475 ┄          goto #3477
3476 ─          nothing::Nothing
3477 ┄          goto #3478
3478 ─          goto #3480
3479 ─          nothing::Nothing
3480 ┄          goto #3481
3481 ─          goto #3483
3482 ─          nothing::Nothing
3483 ┄          goto #3485
3484 ─          nothing::Nothing
3485 ┄          goto #3486
3486 ─          goto #3488
3487 ─          nothing::Nothing
3488 ┄          goto #3489
3489 ─ %9939  = Base.muladd_float(-2.0, %9473, %7070)::Float64
│      %9940  = Base.mul_float(%7070, %9939)::Float64
│      %9941  = Base.muladd_float(%9473, %9473, %9940)::Float64
│      %9942  = Base.muladd_float(2.0, %9473, %7070)::Float64
│      %9943  = Base.mul_float(%7070, %9942)::Float64
│      %9944  = Base.muladd_float(%9473, %9473, %9943)::Float64
│      %9945  = Base.div_float(%9941, %9944)::Float64
│      %9946  = Base.lt_float(%9945, 0.0001)::Bool
└─────          goto #3491 if not %9946
3490 ─ %9948  = Base.add_float(%7070, %9473)::Float64
│      %9949  = Base.muladd_float(%9945, 0.2857142857142857, 0.4)::Float64
│      %9950  = Base.muladd_float(%9945, %9949, 0.6666666666666666)::Float64
│      %9951  = Base.muladd_float(%9945, %9950, 2.0)::Float64
│      %9952  = Base.div_float(%9948, %9951)::Float64
└─────          goto #3492
3491 ─ %9954  = Base.sub_float(%9473, %7070)::Float64
│      %9955  = Base.div_float(%9473, %7070)::Float64
│      %9956  = $(Expr(:foreigncall, "llvm.log.f64", Float64, svec(Float64), 0, :(:llvmcall), :(%9955), :(%9955)))::Float64
│      %9957  = Base.div_float(%9954, %9956)::Float64
└─────          goto #3492
3492 ┄ %9959  = φ (#3490 => %9952, #3491 => %9957)::Float64
│      %9960  = Base.mul_float(%7070, %9905)::Float64
│      %9961  = Base.mul_float(%9473, %9830)::Float64
│      %9962  = Base.muladd_float(-2.0, %9961, %9960)::Float64
│      %9963  = Base.mul_float(%9960, %9962)::Float64
│      %9964  = Base.muladd_float(%9961, %9961, %9963)::Float64
│      %9965  = Base.muladd_float(2.0, %9961, %9960)::Float64
│      %9966  = Base.mul_float(%9960, %9965)::Float64
│      %9967  = Base.muladd_float(%9961, %9961, %9966)::Float64
│      %9968  = Base.div_float(%9964, %9967)::Float64
│      %9969  = Base.lt_float(%9968, 0.0001)::Bool
└─────          goto #3494 if not %9969
3493 ─ %9971  = Base.muladd_float(%9968, 0.2857142857142857, 0.4)::Float64
│      %9972  = Base.muladd_float(%9968, %9971, 0.6666666666666666)::Float64
│      %9973  = Base.muladd_float(%9968, %9972, 2.0)::Float64
│      %9974  = Base.add_float(%9960, %9961)::Float64
│      %9975  = Base.div_float(%9973, %9974)::Float64
└─────          goto #3495
3494 ─ %9977  = Base.div_float(%9961, %9960)::Float64
│      %9978  = $(Expr(:foreigncall, "llvm.log.f64", Float64, svec(Float64), 0, :(:llvmcall), :(%9977), :(%9977)))::Float64
│      %9979  = Base.sub_float(%9961, %9960)::Float64
│      %9980  = Base.div_float(%9978, %9979)::Float64
└─────          goto #3495
3495 ┄ %9982  = φ (#3493 => %9975, #3494 => %9980)::Float64
│      %9983  = Base.mul_float(%9830, %9905)::Float64
│      %9984  = Base.mul_float(%9983, %9982)::Float64
│      %9985  = Base.add_float(%9821, %9896)::Float64
│      %9986  = Base.mul_float(0.5, %9985)::Float64
│      %9987  = Base.add_float(%9822, %9897)::Float64
│      %9988  = Base.mul_float(0.5, %9987)::Float64
│      %9989  = Base.add_float(%9823, %9898)::Float64
│      %9990  = Base.mul_float(0.5, %9989)::Float64
│      %9991  = Base.add_float(%9830, %9905)::Float64
│      %9992  = Base.mul_float(0.5, %9991)::Float64
│      %9993  = Base.mul_float(%9821, %9896)::Float64
│      %9994  = Base.muladd_float(%9822, %9897, %9993)::Float64
│      %9995  = Base.muladd_float(%9823, %9898, %9994)::Float64
│      %9996  = Base.mul_float(0.5, %9995)::Float64
│      %9997  = Base.mul_float(%9959, %9990)::Float64
│      %9998  = Base.mul_float(%9997, %9986)::Float64
│      %9999  = Base.mul_float(%9997, %9988)::Float64
│      %10000 = Base.muladd_float(%9997, %9990, %9992)::Float64
│      %10001 = Base.mul_float(%9830, %9898)::Float64
│      %10002 = Base.muladd_float(%9905, %9823, %10001)::Float64
│      %10003 = Base.getfield(equations, :inv_gamma_minus_one)::Float64
│      %10004 = Base.muladd_float(%9984, %10003, %9996)::Float64
│      %10005 = Base.mul_float(%9997, %10004)::Float64
│      %10006 = Base.muladd_float(0.5, %10002, %10005)::Float64
│      %10007 = Core.tuple(%9997, %9998, %9999, %10000, %10006)::NTuple{5, Float64}
└─────          goto #3496
3496 ─ %10009 = Base.arrayref(false, %6989, %6991, %9400)::Float64
│      %10010 = Base.copysign_float(0.0, %10009)::Float64
│      %10011 = Core.ifelse(true, %10009, %10010)::Float64
└─────          goto #3542 if not true
3497 ┄ %10013 = φ (#3496 => 1, #3541 => %10182)::Int64
│      %10014 = φ (#3496 => 1, #3541 => %10183)::Int64
│      %10015 = Base.getfield(%10007, %10013, true)::Float64
│      %10016 = StrideArraysCore.getfield(du, :data)::Vector{Float64}
│      %10017 = $(Expr(:gc_preserve_begin, :(%10016)))
│      %10018 = StrideArraysCore.getfield(du, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #3502 if not true
3498 ─ %10020 = Core.tuple(%10013, %6997, %6994, %6991, %6984)::NTuple{5, Int64}
│      %10021 = StrideArraysCore.getfield(%10018, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %10022 = Core.getfield(%10021, 5)::Int64
│      %10023 = Base.bitcast(UInt64, %10022)::UInt64
│      %10024 = Base.bitcast(Int64, %10023)::Int64
│      %10025 = Base.sle_int(1, %10013)::Bool
│      %10026 = Base.sle_int(%10013, 5)::Bool
│      %10027 = Base.and_int(%10025, %10026)::Bool
│      %10028 = Base.sle_int(1, %6997)::Bool
│      %10029 = Base.sle_int(%6997, 4)::Bool
│      %10030 = Base.and_int(%10028, %10029)::Bool
│      %10031 = Base.sle_int(1, %6994)::Bool
│      %10032 = Base.sle_int(%6994, 4)::Bool
│      %10033 = Base.and_int(%10031, %10032)::Bool
│      %10034 = Base.sle_int(1, %6991)::Bool
│      %10035 = Base.sle_int(%6991, 4)::Bool
│      %10036 = Base.and_int(%10034, %10035)::Bool
│      %10037 = Base.sub_int(%6984, 1)::Int64
│      %10038 = Base.bitcast(UInt64, %10037)::UInt64
│      %10039 = Base.bitcast(UInt64, %10024)::UInt64
│      %10040 = Base.ult_int(%10038, %10039)::Bool
│      %10041 = Base.and_int(%10040, true)::Bool
│      %10042 = Base.and_int(%10036, %10041)::Bool
│      %10043 = Base.and_int(%10033, %10042)::Bool
│      %10044 = Base.and_int(%10030, %10043)::Bool
│      %10045 = Base.and_int(%10027, %10044)::Bool
└─────          goto #3500 if not %10045
3499 ─          goto #3501
3500 ─          invoke Base.throw_boundserror(%10018::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %10020::NTuple{5, Int64})::Union{}
└─────          unreachable
3501 ─          nothing::Nothing
3502 ┄ %10051 = StrideArraysCore.getfield(%10018, :ptr)::Ptr{Float64}
│      %10052 = Base.sub_int(%10013, 1)::Int64
│      %10053 = Base.sub_int(%6997, 1)::Int64
│      %10054 = Base.sub_int(%6994, 1)::Int64
│      %10055 = Base.sub_int(%6991, 1)::Int64
│      %10056 = Base.sub_int(%6984, 1)::Int64
└─────          goto #3511 if not true
3503 ┄ %10058 = φ (#3502 => 2, #3510 => %10070)::Int64
│      %10059 = Base.sle_int(1, %10058)::Bool
└─────          goto #3505 if not %10059
3504 ─ %10061 = Base.sle_int(%10058, 5)::Bool
└─────          goto #3506
3505 ─          nothing::Nothing
3506 ┄ %10064 = φ (#3504 => %10061, #3505 => false)::Bool
└─────          goto #3508 if not %10064
3507 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %10058, true)::Static.True
│      %10067 = Base.add_int(%10058, 1)::Int64
└─────          goto #3509
3508 ─          goto #3509
3509 ┄ %10070 = φ (#3507 => %10067)::Int64
│      %10071 = φ (#3507 => false, #3508 => true)::Bool
│      %10072 = Base.not_int(%10071)::Bool
└─────          goto #3511 if not %10072
3510 ─          goto #3503
3511 ┄          goto #3512
3512 ─          goto #3513
3513 ─ %10077 = Base.mul_int(%10056, 4)::Int64
│      %10078 = Base.add_int(%10055, %10077)::Int64
│      %10079 = Base.mul_int(%10078, 4)::Int64
│      %10080 = Base.add_int(%10054, %10079)::Int64
│      %10081 = Base.mul_int(%10080, 4)::Int64
│      %10082 = Base.add_int(%10053, %10081)::Int64
│      %10083 = Base.mul_int(%10082, 5)::Int64
│      %10084 = Base.add_int(%10052, %10083)::Int64
│      %10085 = Base.mul_int(8, %10084)::Int64
│      %10086 = Core.bitcast(Core.UInt, %10051)::UInt64
│      %10087 = Base.bitcast(UInt64, %10085)::UInt64
│      %10088 = Base.add_ptr(%10086, %10087)::UInt64
│      %10089 = Core.bitcast(Ptr{Float64}, %10088)::Ptr{Float64}
└─────          goto #3514
3514 ─ %10091 = Base.pointerref(%10089, 1, 1)::Float64
└─────          goto #3515
3515 ─          goto #3516
3516 ─          $(Expr(:gc_preserve_end, :(%10017)))
└─────          goto #3517
3517 ─ %10096 = Base.muladd_float(%10011, %10015, %10091)::Float64
│      %10097 = StrideArraysCore.getfield(du, :data)::Vector{Float64}
│      %10098 = $(Expr(:gc_preserve_begin, :(%10097)))
│      %10099 = StrideArraysCore.getfield(du, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #3522 if not true
3518 ─ %10101 = Core.tuple(%10013, %6997, %6994, %6991, %6984)::NTuple{5, Int64}
│      %10102 = StrideArraysCore.getfield(%10099, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %10103 = Core.getfield(%10102, 5)::Int64
│      %10104 = Base.bitcast(UInt64, %10103)::UInt64
│      %10105 = Base.bitcast(Int64, %10104)::Int64
│      %10106 = Base.sle_int(1, %10013)::Bool
│      %10107 = Base.sle_int(%10013, 5)::Bool
│      %10108 = Base.and_int(%10106, %10107)::Bool
│      %10109 = Base.sle_int(1, %6997)::Bool
│      %10110 = Base.sle_int(%6997, 4)::Bool
│      %10111 = Base.and_int(%10109, %10110)::Bool
│      %10112 = Base.sle_int(1, %6994)::Bool
│      %10113 = Base.sle_int(%6994, 4)::Bool
│      %10114 = Base.and_int(%10112, %10113)::Bool
│      %10115 = Base.sle_int(1, %6991)::Bool
│      %10116 = Base.sle_int(%6991, 4)::Bool
│      %10117 = Base.and_int(%10115, %10116)::Bool
│      %10118 = Base.sub_int(%6984, 1)::Int64
│      %10119 = Base.bitcast(UInt64, %10118)::UInt64
│      %10120 = Base.bitcast(UInt64, %10105)::UInt64
│      %10121 = Base.ult_int(%10119, %10120)::Bool
│      %10122 = Base.and_int(%10121, true)::Bool
│      %10123 = Base.and_int(%10117, %10122)::Bool
│      %10124 = Base.and_int(%10114, %10123)::Bool
│      %10125 = Base.and_int(%10111, %10124)::Bool
│      %10126 = Base.and_int(%10108, %10125)::Bool
└─────          goto #3520 if not %10126
3519 ─          goto #3521
3520 ─          invoke Base.throw_boundserror(%10099::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %10101::NTuple{5, Int64})::Union{}
└─────          unreachable
3521 ─          nothing::Nothing
3522 ┄ %10132 = StrideArraysCore.getfield(%10099, :ptr)::Ptr{Float64}
│      %10133 = Base.sub_int(%10013, 1)::Int64
│      %10134 = Base.sub_int(%6997, 1)::Int64
│      %10135 = Base.sub_int(%6994, 1)::Int64
│      %10136 = Base.sub_int(%6991, 1)::Int64
│      %10137 = Base.sub_int(%6984, 1)::Int64
└─────          goto #3531 if not true
3523 ┄ %10139 = φ (#3522 => 2, #3530 => %10151)::Int64
│      %10140 = Base.sle_int(1, %10139)::Bool
└─────          goto #3525 if not %10140
3524 ─ %10142 = Base.sle_int(%10139, 5)::Bool
└─────          goto #3526
3525 ─          nothing::Nothing
3526 ┄ %10145 = φ (#3524 => %10142, #3525 => false)::Bool
└─────          goto #3528 if not %10145
3527 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %10139, true)::Static.True
│      %10148 = Base.add_int(%10139, 1)::Int64
└─────          goto #3529
3528 ─          goto #3529
3529 ┄ %10151 = φ (#3527 => %10148)::Int64
│      %10152 = φ (#3527 => false, #3528 => true)::Bool
│      %10153 = Base.not_int(%10152)::Bool
└─────          goto #3531 if not %10153
3530 ─          goto #3523
3531 ┄          goto #3532
3532 ─          goto #3533
3533 ─ %10158 = Base.mul_int(%10137, 4)::Int64
│      %10159 = Base.add_int(%10136, %10158)::Int64
│      %10160 = Base.mul_int(%10159, 4)::Int64
│      %10161 = Base.add_int(%10135, %10160)::Int64
│      %10162 = Base.mul_int(%10161, 4)::Int64
│      %10163 = Base.add_int(%10134, %10162)::Int64
│      %10164 = Base.mul_int(%10163, 5)::Int64
│      %10165 = Base.add_int(%10133, %10164)::Int64
│      %10166 = Base.mul_int(8, %10165)::Int64
│      %10167 = Core.bitcast(Core.UInt, %10132)::UInt64
│      %10168 = Base.bitcast(UInt64, %10166)::UInt64
│      %10169 = Base.add_ptr(%10167, %10168)::UInt64
│      %10170 = Core.bitcast(Ptr{Float64}, %10169)::Ptr{Float64}
└─────          goto #3534
3534 ─          Base.pointerset(%10170, %10096, 1, 1)::Ptr{Float64}
└─────          goto #3535
3535 ─          goto #3536
3536 ─          $(Expr(:gc_preserve_end, :(%10098)))
└─────          goto #3537
3537 ─ %10177 = (%10014 === 5)::Bool
└─────          goto #3539 if not %10177
3538 ─          goto #3540
3539 ─ %10180 = Base.add_int(%10014, 1)::Int64
└─────          goto #3540
3540 ┄ %10182 = φ (#3539 => %10180)::Int64
│      %10183 = φ (#3539 => %10180)::Int64
│      %10184 = φ (#3538 => true, #3539 => false)::Bool
│      %10185 = Base.not_int(%10184)::Bool
└─────          goto #3542 if not %10185
3541 ─          goto #3497
3542 ┄          goto #3543
3543 ─ %10189 = Base.arrayref(false, %6989, %9400, %6991)::Float64
│      %10190 = Base.copysign_float(0.0, %10189)::Float64
│      %10191 = Core.ifelse(true, %10189, %10190)::Float64
└─────          goto #3589 if not true
3544 ┄ %10193 = φ (#3543 => 1, #3588 => %10362)::Int64
│      %10194 = φ (#3543 => 1, #3588 => %10363)::Int64
│      %10195 = Base.getfield(%10007, %10193, true)::Float64
│      %10196 = StrideArraysCore.getfield(du, :data)::Vector{Float64}
│      %10197 = $(Expr(:gc_preserve_begin, :(%10196)))
│      %10198 = StrideArraysCore.getfield(du, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #3549 if not true
3545 ─ %10200 = Core.tuple(%10193, %6997, %6994, %9400, %6984)::NTuple{5, Int64}
│      %10201 = StrideArraysCore.getfield(%10198, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %10202 = Core.getfield(%10201, 5)::Int64
│      %10203 = Base.bitcast(UInt64, %10202)::UInt64
│      %10204 = Base.bitcast(Int64, %10203)::Int64
│      %10205 = Base.sle_int(1, %10193)::Bool
│      %10206 = Base.sle_int(%10193, 5)::Bool
│      %10207 = Base.and_int(%10205, %10206)::Bool
│      %10208 = Base.sle_int(1, %6997)::Bool
│      %10209 = Base.sle_int(%6997, 4)::Bool
│      %10210 = Base.and_int(%10208, %10209)::Bool
│      %10211 = Base.sle_int(1, %6994)::Bool
│      %10212 = Base.sle_int(%6994, 4)::Bool
│      %10213 = Base.and_int(%10211, %10212)::Bool
│      %10214 = Base.sle_int(1, %9400)::Bool
│      %10215 = Base.sle_int(%9400, 4)::Bool
│      %10216 = Base.and_int(%10214, %10215)::Bool
│      %10217 = Base.sub_int(%6984, 1)::Int64
│      %10218 = Base.bitcast(UInt64, %10217)::UInt64
│      %10219 = Base.bitcast(UInt64, %10204)::UInt64
│      %10220 = Base.ult_int(%10218, %10219)::Bool
│      %10221 = Base.and_int(%10220, true)::Bool
│      %10222 = Base.and_int(%10216, %10221)::Bool
│      %10223 = Base.and_int(%10213, %10222)::Bool
│      %10224 = Base.and_int(%10210, %10223)::Bool
│      %10225 = Base.and_int(%10207, %10224)::Bool
└─────          goto #3547 if not %10225
3546 ─          goto #3548
3547 ─          invoke Base.throw_boundserror(%10198::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %10200::NTuple{5, Int64})::Union{}
└─────          unreachable
3548 ─          nothing::Nothing
3549 ┄ %10231 = StrideArraysCore.getfield(%10198, :ptr)::Ptr{Float64}
│      %10232 = Base.sub_int(%10193, 1)::Int64
│      %10233 = Base.sub_int(%6997, 1)::Int64
│      %10234 = Base.sub_int(%6994, 1)::Int64
│      %10235 = Base.sub_int(%9400, 1)::Int64
│      %10236 = Base.sub_int(%6984, 1)::Int64
└─────          goto #3558 if not true
3550 ┄ %10238 = φ (#3549 => 2, #3557 => %10250)::Int64
│      %10239 = Base.sle_int(1, %10238)::Bool
└─────          goto #3552 if not %10239
3551 ─ %10241 = Base.sle_int(%10238, 5)::Bool
└─────          goto #3553
3552 ─          nothing::Nothing
3553 ┄ %10244 = φ (#3551 => %10241, #3552 => false)::Bool
└─────          goto #3555 if not %10244
3554 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %10238, true)::Static.True
│      %10247 = Base.add_int(%10238, 1)::Int64
└─────          goto #3556
3555 ─          goto #3556
3556 ┄ %10250 = φ (#3554 => %10247)::Int64
│      %10251 = φ (#3554 => false, #3555 => true)::Bool
│      %10252 = Base.not_int(%10251)::Bool
└─────          goto #3558 if not %10252
3557 ─          goto #3550
3558 ┄          goto #3559
3559 ─          goto #3560
3560 ─ %10257 = Base.mul_int(%10236, 4)::Int64
│      %10258 = Base.add_int(%10235, %10257)::Int64
│      %10259 = Base.mul_int(%10258, 4)::Int64
│      %10260 = Base.add_int(%10234, %10259)::Int64
│      %10261 = Base.mul_int(%10260, 4)::Int64
│      %10262 = Base.add_int(%10233, %10261)::Int64
│      %10263 = Base.mul_int(%10262, 5)::Int64
│      %10264 = Base.add_int(%10232, %10263)::Int64
│      %10265 = Base.mul_int(8, %10264)::Int64
│      %10266 = Core.bitcast(Core.UInt, %10231)::UInt64
│      %10267 = Base.bitcast(UInt64, %10265)::UInt64
│      %10268 = Base.add_ptr(%10266, %10267)::UInt64
│      %10269 = Core.bitcast(Ptr{Float64}, %10268)::Ptr{Float64}
└─────          goto #3561
3561 ─ %10271 = Base.pointerref(%10269, 1, 1)::Float64
└─────          goto #3562
3562 ─          goto #3563
3563 ─          $(Expr(:gc_preserve_end, :(%10197)))
└─────          goto #3564
3564 ─ %10276 = Base.muladd_float(%10191, %10195, %10271)::Float64
│      %10277 = StrideArraysCore.getfield(du, :data)::Vector{Float64}
│      %10278 = $(Expr(:gc_preserve_begin, :(%10277)))
│      %10279 = StrideArraysCore.getfield(du, :ptr)::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}
└─────          goto #3569 if not true
3565 ─ %10281 = Core.tuple(%10193, %6997, %6994, %9400, %6984)::NTuple{5, Int64}
│      %10282 = StrideArraysCore.getfield(%10279, :sizes)::Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}
│      %10283 = Core.getfield(%10282, 5)::Int64
│      %10284 = Base.bitcast(UInt64, %10283)::UInt64
│      %10285 = Base.bitcast(Int64, %10284)::Int64
│      %10286 = Base.sle_int(1, %10193)::Bool
│      %10287 = Base.sle_int(%10193, 5)::Bool
│      %10288 = Base.and_int(%10286, %10287)::Bool
│      %10289 = Base.sle_int(1, %6997)::Bool
│      %10290 = Base.sle_int(%6997, 4)::Bool
│      %10291 = Base.and_int(%10289, %10290)::Bool
│      %10292 = Base.sle_int(1, %6994)::Bool
│      %10293 = Base.sle_int(%6994, 4)::Bool
│      %10294 = Base.and_int(%10292, %10293)::Bool
│      %10295 = Base.sle_int(1, %9400)::Bool
│      %10296 = Base.sle_int(%9400, 4)::Bool
│      %10297 = Base.and_int(%10295, %10296)::Bool
│      %10298 = Base.sub_int(%6984, 1)::Int64
│      %10299 = Base.bitcast(UInt64, %10298)::UInt64
│      %10300 = Base.bitcast(UInt64, %10285)::UInt64
│      %10301 = Base.ult_int(%10299, %10300)::Bool
│      %10302 = Base.and_int(%10301, true)::Bool
│      %10303 = Base.and_int(%10297, %10302)::Bool
│      %10304 = Base.and_int(%10294, %10303)::Bool
│      %10305 = Base.and_int(%10291, %10304)::Bool
│      %10306 = Base.and_int(%10288, %10305)::Bool
└─────          goto #3567 if not %10306
3566 ─          goto #3568
3567 ─          invoke Base.throw_boundserror(%10279::StrideArraysCore.PtrArray{Float64, 5, (1, 2, 3, 4, 5), Tuple{Static.StaticInt{5}, Static.StaticInt{4}, Static.StaticInt{4}, Static.StaticInt{4}, Int64}, NTuple{5, Nothing}, NTuple{5, Static.StaticInt{1}}}, %10281::NTuple{5, Int64})::Union{}
└─────          unreachable
3568 ─          nothing::Nothing
3569 ┄ %10312 = StrideArraysCore.getfield(%10279, :ptr)::Ptr{Float64}
│      %10313 = Base.sub_int(%10193, 1)::Int64
│      %10314 = Base.sub_int(%6997, 1)::Int64
│      %10315 = Base.sub_int(%6994, 1)::Int64
│      %10316 = Base.sub_int(%9400, 1)::Int64
│      %10317 = Base.sub_int(%6984, 1)::Int64
└─────          goto #3578 if not true
3570 ┄ %10319 = φ (#3569 => 2, #3577 => %10331)::Int64
│      %10320 = Base.sle_int(1, %10319)::Bool
└─────          goto #3572 if not %10320
3571 ─ %10322 = Base.sle_int(%10319, 5)::Bool
└─────          goto #3573
3572 ─          nothing::Nothing
3573 ┄ %10325 = φ (#3571 => %10322, #3572 => false)::Bool
└─────          goto #3575 if not %10325
3574 ─          Base.getfield((static(true), static(true), static(true), static(true), static(true)), %10319, true)::Static.True
│      %10328 = Base.add_int(%10319, 1)::Int64
└─────          goto #3576
3575 ─          goto #3576
3576 ┄ %10331 = φ (#3574 => %10328)::Int64
│      %10332 = φ (#3574 => false, #3575 => true)::Bool
│      %10333 = Base.not_int(%10332)::Bool
└─────          goto #3578 if not %10333
3577 ─          goto #3570
3578 ┄          goto #3579
3579 ─          goto #3580
3580 ─ %10338 = Base.mul_int(%10317, 4)::Int64
│      %10339 = Base.add_int(%10316, %10338)::Int64
│      %10340 = Base.mul_int(%10339, 4)::Int64
│      %10341 = Base.add_int(%10315, %10340)::Int64
│      %10342 = Base.mul_int(%10341, 4)::Int64
│      %10343 = Base.add_int(%10314, %10342)::Int64
│      %10344 = Base.mul_int(%10343, 5)::Int64
│      %10345 = Base.add_int(%10313, %10344)::Int64
│      %10346 = Base.mul_int(8, %10345)::Int64
│      %10347 = Core.bitcast(Core.UInt, %10312)::UInt64
│      %10348 = Base.bitcast(UInt64, %10346)::UInt64
│      %10349 = Base.add_ptr(%10347, %10348)::UInt64
│      %10350 = Core.bitcast(Ptr{Float64}, %10349)::Ptr{Float64}
└─────          goto #3581
3581 ─          Base.pointerset(%10350, %10276, 1, 1)::Ptr{Float64}
└─────          goto #3582
3582 ─          goto #3583
3583 ─          $(Expr(:gc_preserve_end, :(%10278)))
└─────          goto #3584
3584 ─ %10357 = (%10194 === 5)::Bool
└─────          goto #3586 if not %10357
3585 ─          goto #3587
3586 ─ %10360 = Base.add_int(%10194, 1)::Int64
└─────          goto #3587
3587 ┄ %10362 = φ (#3586 => %10360)::Int64
│      %10363 = φ (#3586 => %10360)::Int64
│      %10364 = φ (#3585 => true, #3586 => false)::Bool
│      %10365 = Base.not_int(%10364)::Bool
└─────          goto #3589 if not %10365
3588 ─          goto #3544
3589 ┄          goto #3590
3590 ─ %10369 = (%9401 === %9388)::Bool
└─────          goto #3592 if not %10369
3591 ─          goto #3593
3592 ─ %10372 = Base.add_int(%9401, 1)::Int64
└─────          goto #3593
3593 ┄ %10374 = φ (#3592 => %10372)::Int64
│      %10375 = φ (#3592 => %10372)::Int64
│      %10376 = φ (#3591 => true, #3592 => false)::Bool
│      %10377 = Base.not_int(%10376)::Bool
└─────          goto #3595 if not %10377
3594 ─          goto #3252
3595 ┄ %10380 = (%6998 === 4)::Bool
└─────          goto #3597 if not %10380
3596 ─          goto #3598
3597 ─ %10383 = Base.add_int(%6998, 1)::Int64
└─────          goto #3598
3598 ┄ %10385 = φ (#3597 => %10383)::Int64
│      %10386 = φ (#3597 => %10383)::Int64
│      %10387 = φ (#3596 => true, #3597 => false)::Bool
│      %10388 = Base.not_int(%10387)::Bool
└─────          goto #3600 if not %10388
3599 ─          goto #2432
3600 ┄ %10391 = (%6995 === 4)::Bool
└─────          goto #3602 if not %10391
3601 ─          goto #3603
3602 ─ %10394 = Base.add_int(%6995, 1)::Int64
└─────          goto #3603
3603 ┄ %10396 = φ (#3602 => %10394)::Int64
│      %10397 = φ (#3602 => %10394)::Int64
│      %10398 = φ (#3601 => true, #3602 => false)::Bool
│      %10399 = Base.not_int(%10398)::Bool
└─────          goto #3605 if not %10399
3604 ─          goto #2431
3605 ┄ %10402 = (%6992 === 4)::Bool
└─────          goto #3607 if not %10402
3606 ─          goto #3608
3607 ─ %10405 = Base.add_int(%6992, 1)::Int64
└─────          goto #3608
3608 ┄ %10407 = φ (#3607 => %10405)::Int64
│      %10408 = φ (#3607 => %10405)::Int64
│      %10409 = φ (#3606 => true, #3607 => false)::Bool
│      %10410 = Base.not_int(%10409)::Bool
└─────          goto #3610 if not %10410
3609 ─          goto #2430
3610 ┄          goto #3611
3611 ─          goto #3612
3612 ─          goto #3613
3613 ─          nothing::Nothing
3614 ┄ %10417 = Base.add_int(%6984, 1)::Int64
└─────          goto #2427
3615 ─          goto #3616
3616 ─          goto #3618
3617 ─          goto #3618
3618 ┄          nothing::Nothing
3619 ┄          return Main.nothing
) => Nothing
```
