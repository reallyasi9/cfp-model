using Pkg
pkg"activate ."

include("../CFBModel.jl")
import .CFBModel
using SparseArrays
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

# figure out the longest game in number of plays for padding
nplays = maximum(nrow, groupby(traindata, :game_id))

# convert dataframe into an nvar x nplays x ngames (sparse) matrix
function expandfeatures(game)
    offense = Flux.onehotbatch(game.offense, teams, "unknown")
    defense = Flux.onehotbatch(game.defense, teams, "unknown")
    down = Flux.onehotbatch(game.down, collect(1:4))
    period = Flux.onehotbatch(game.period, collect(1:4))

    nfeatures = size(offense, 1) + size(defense, 1) + size(down, 1) + size(period, 1) + 3
    nextra = nplays - size(offense, 2)

    extra = spzeros(Float32, nfeatures, nextra)

    sparse(hcat(vcat(offense, defense, down, period, game.fraction_to_gain', game.clock', game.game_clock'), extra))
end

# make a model
nteams = length(teams)
teamembedding = Flux.Dense(nteams, 10, Flux.relu(1.))
model = Chain(
    x -> vcat(teamembedding(x[1:nteams]), teamembedding(x[nteams+1:2nteams]), x[2nteams+1:end]),
    Flux.LSTM(31, 32),
    Flux.Dropout(.2),
    Flux.LSTM(32, 32),
    Flux.Dropout(.2),
    Dense(32, 32, Flux.relu(1.)),
    Dense(32, 32, Flux.relu(1.)),
    Dense(32, 1, Flux.tanh)
)
ps = Flux.params(model)
opt = Flux.ADAM
loss(x, y) = Flux.mse(model(x), y)

# callback to keep track of progress
callback() = begin
    total_loss = mapreduce(x -> loss(expandfeatures(x), x.position_change), +, groupby(testdata, :game_id))
    println("Average test loss: $(total_loss / nrow(testdata))")
end

for game in groupby(traindata, :game_id)
    Flux.train!(loss, ps, [(expandfeatures(game), game.position_change)], opt, Flux.throttle(callback, 1))
end

X = map(expandfeatures, groupby(traindata, :game_id))
