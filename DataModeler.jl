import Pkg
Pkg.activate(".")

using Flux
using DataFrames, DataFramesMeta, CSVFiles
using TextParse

function loadplays(; dir::AbstractString = pwd(), years::Vector{T} = collect(2014:2019)) where T<:Integer
    colparsers = Dict{String, Any}(
        "game_id" => String,
        "drive_id" => String,
        "play_id" => String,
        "period" => Int8,
        "half" => String,  # Bool
        "offense" => String,
        "defense" => String,
        "offense_conference" => String,
        "defense_conference" => String,
        "in_conference" => String,  # Bool
        "offense_home" => String,  # Bool
        "defense_home" => String,  # Bool
        "position" => Float32,
        "down" => Int8,
        "distance" => Float32,
        "offense_score" => Float32,
        "defense_score" => Float32,
        "play_type" => String,
        "venue" => String,
        "game_clock" => Float32,
        "time_delta" => Float32,
        "clock" => Float32,
        "home_score" => Float32,
        "away_score" => Float32,
        "last_play" => String,  # Bool
        "score_change" => Float32,
        "turnover" => String,  # Bool
        "delta_position" => String,  # Could be Inf
        "delta_distance" => String,  # Could be Inf
    )
    local df = DataFrame()
    for path in readdir(dir)
        m = match(r"plays_(\d+)_\d+\.csv", path)
        if !isnothing(m) && parse(Int16, m.captures[1]) ∈ years
            append!(df, load(path, colparsers=colparsers))
        end
    end
    df[!, :delta_position] = parse.(Float32, df[!, :delta_position])
    df[!, :delta_distance] = parse.(Float32, df[!, :delta_distance])
    df[!, :turnover] = df[!, :turnover] .== "true"
    df[!, :last_play] = df[!, :last_play] .== "true"
    df[!, :in_conference] = df[!, :in_conference] .== "true"
    df[!, :half] = df[!, :half] .== "true"
    df[!, :offense_home] = df[!, :offense_home] .== "true"
    df[!, :defense_home] = df[!, :defense_home] .== "true"
    df
end

df = loadplays(years=[2014])
longestGame = maximum(by(df, :game_id, g -> nrow(g)).x1)
m1 = by(df, :game_id) do g
    x = zeros(Float32, longestGame)
    y = zeros(Float32, longestGame)
    n = nrow(g)
    x[1:n] = g[:position]
    y[1:n] = g[:delta_position]
    (x=x, y=y)
end
    
