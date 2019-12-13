if length(ARGS) < 1
    @error "must specify one year to download"
    exit(-1)
end

import Pkg
Pkg.activate(".")

include("../CFBModel.jl")
import .CFBModel

using DataFrames, FeatherFiles

for year in ARGS
    for week in 1:14
        df = convert(Array{CFBModel.Play}, CFBModel.getplays(parse(Int, year), week)) |> DataFrame
        df[!,:week] .= week
        df[!,:year] .= year
        df |> save("plays_$(year)_$(week).feather")
        df |> CFBModel.generatedrives |> save("gendrives_$(year)_$(week).feather")
    end
end
