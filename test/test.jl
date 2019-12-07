using Pkg
pkg"activate ."

include("../CFPModel.jl")
import .CFPModel
using DataFrames, DataFramesMeta
using FeatherFiles
using Flux

traindata = DataFrame()
for year in 2014:2017
    for week in 1:14
        append!(traindata, load("plays_$(year)_$(week).feather") |> DataFrame)
    end
end

testdata = DataFrame()
for week in 1:14
    append!(testdata, load("plays_2019_$(week).feather") |> DataFrame)
end

# drop kickoffs and end-of-periods
traindata = @where(traindata, :down .∈ (1:4,))
testdata = @where(testdata, :down .∈ (1:4,))

# drop overtime
traindata = @where(traindata, :period .< 5)
testdata = @where(testdata, :period .< 5)

# drop timeout calls
traindata = @where(traindata, :play_type .!= "Timeout")
testdata = @where(testdata, :play_type .!= "Timeout")

# drop very rare plays
rarelimit = 0.01
playfreqs = Flux.frequencies(traindata.play_type)
playtypes = [play for (play, freq) in playfreqs if freq / nrow(traindata) > rarelimit]
traindata = @where(traindata, :play_type .∈ (playtypes,))
testdata = @where(testdata, :play_type .∈ (playtypes,))

# convert training data to categorical
categorical!(traindata, [:offense, :defense, :home, :away, :down, :period, :play_type])

# copy team categories over to test data
teams = union(levels(traindata.offense), levels(traindata.defense), ["unknown"])
for col in [:offense, :defense, :home, :away]
    testdata[testdata[!, col] .∉ (teams,), col] .= Ref("unknown")
    categorical!(testdata, col)
    levels!(testdata[!, col], teams)
end
categorical!(testdata, [:down, :period, :play_type])
