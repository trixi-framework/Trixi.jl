module CommonMod

using ..Jul1dge

export die


die() = exit(1)
die(msg::String) = (println(stderr, msg); die())


end
