if length(ARGS) < 1
    @error "must specify one year to download"
    exit(-1)
end

import Pkg
Pkg.activate(".")

include("../CFPModel.jl")
import .CFPModel

using DataFrames, FeatherFiles

for year in ARGS
    for week in 1:14
        df = convert(Array{CFPModel.Play}, CFPModel.getplays(parse(Int, year), week)) |> DataFrame
        df[!,:week] .= week
        df[!,:year] .= year
        df |> save("plays_$(year)_$(week).feather")
    end
end
