using Pkg
pkg"activate ."

include("../CFBModel.jl")
import .CFBModel
using Dates
# using SparseArrays
using DataFrames, DataFramesMeta
using FeatherFiles
using Flux

traindata = DataFrame()
for year in 2014:2018
    for week in 1:14
        append!(traindata, load("gendrives_$(year)_$(week).feather") |> DataFrame)
    end
end

testdata = DataFrame()
for week in 1:14
    append!(testdata, load("gendrives_2019_$(week).feather") |> DataFrame)
end

# ignore overtime
filter!(row -> row[:period_start] ∈ 1:4, traindata)
filter!(row -> row[:period_start] ∈ 1:4, testdata)

# drop extremely rare outcomes
resultfreqs = Flux.frequencies(traindata.last_play)
resulttypes = [result for (result, freq) in resultfreqs if freq / nrow(traindata) > 0.005]
filter!(row -> row[:last_play] ∈ resulttypes, traindata)
filter!(row -> row[:last_play] ∈ resulttypes, testdata)

# convert training data to categorical
categorical!(traindata, [:offense, :defense, :home, :away, :period_start, :period_end, :last_play])

# copy team categories over to test data
teams = union(levels(traindata.offense), levels(traindata.defense), ["unknown"])
for col in [:offense, :defense, :home, :away]
    testdata[testdata[!, col] .∉ (teams,), col] .= Ref("unknown")
    categorical!(testdata, col)
    levels!(testdata[!, col], teams)
end
categorical!(testdata, [:period_start, :period_end, :last_play])

# predict offensive touches down
traindata[!,:touchdown] .= traindata.score_change .== 7
testdata[!,:touchdown] .= testdata.score_change .== 7

# upweight the relatively rare touchdown... later?
traindata[!,:weight] .= 1.
traindata[traindata.touchdown,:weight] .= (nrow(traindata) / sum(traindata.touchdown))

# build a very simple model
function expandfeatures(game)
    offense = Flux.onehotbatch(game.offense, teams, "unknown")
    defense = Flux.onehotbatch(game.defense, teams, "unknown")
    period = Flux.onehotbatch(game.period_start, collect(1:4))

    nfeatures = size(offense, 1) + size(defense, 1) + size(period, 1) + 3
    # nextra = batchsize - size(offense, 2)

    # extra = spzeros(Float32, nfeatures, nextra)

    vcat(offense, defense, period, game.clock_start', game.game_clock_start', game.position_start')
end

nteams = length(teams)
offenseembedding = Flux.Dense(nteams, 10, Flux.relu)
defenseembedding = Flux.Dense(nteams, 10, Flux.relu)
model = Chain(
    x -> vcat(offenseembedding(x[1:nteams,:]), defenseembedding(x[nteams+1:2nteams,:]), x[2nteams+1:end,:]),
    GRU(27, 16),
    Dropout(.1),
    BatchNorm(16),
    Dense(16, 16, Flux.relu),
    BatchNorm(16),
    Dense(16, 1, Flux.σ)
)
ps = Flux.params(model)
opt = Flux.ADAM()
loss(x, y, w) = sum(Flux.binarycrossentropy.(model(x), y').*w')
loss(x, y) = sum(Flux.binarycrossentropy.(model(x), y'))


# callback to keep track of progress
currentItr = 0
callback() = begin
    global currentItr
    test_loss = mapreduce(x -> loss(expandfeatures(x), x.touchdown), +, groupby(testdata, :game_id))
    train_loss = mapreduce(x -> loss(expandfeatures(x), x.touchdown), +, groupby(traindata, :game_id))
    println("$currentItr $(Dates.now()) - train loss: $(train_loss / nrow(traindata)) test loss: $(test_loss / nrow(testdata))")
end
throttled = Flux.throttle(callback, 10)

for game in groupby(traindata, :game_id)
    global currentItr += 1
    Flux.train!(loss, ps, [(expandfeatures(game), game.touchdown, game.weight)], opt, cb=throttled)
end
