if length(ARGS) < 1
    @error "must specify one year to download"
    exit(-1)
end

import Pkg
Pkg.activate(".")

include("../CFBModel.jl")
import .CFBModel

using DataFrames, FeatherFiles, Dates

for year in ARGS
    for week in 1:14
        df = convert(Array{CFBModel.Game}, CFBModel.getgames(parse(Int, year), week)) |> DataFrame
        # df[!,:week] .= week
        df[!,:year] .= year
        2018-09-01T23:30:00.000Z
        df[!,:start_date] .= DateTime.(df[!,:start_date], (dateformat"y-m-dTH:M:S.sZ",))
        df[!,:home_post_win_prob] .= parse.(Float64, df[!,:home_post_win_prob])
        df[!,:away_post_win_prob] .= parse.(Float64, df[!,:away_post_win_prob])
        df |> save("games_$(year)_$(week).feather")
    end
end
