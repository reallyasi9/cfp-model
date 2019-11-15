
import Pkg
Pkg.activate(".")

import HTTP
import JSON
using URIParser: URI
using BufferedStreams: BufferedInputStream
using DataFrames
using DataFramesMeta
using CSVFiles
using FeatherFiles

# Get plays
function getplays(year::Int, week::Int)

    playsURI = "https://api.collegefootballdata.com/plays?seasonType=regular&year=$(year)&week=$(week)"

    resp = HTTP.request("GET", playsURI)
    playsJSON = JSON.parse(String(resp.body))

    n = length(playsJSON)

    years = repeat([year], n)
    weeks = repeat([week], n)

    gameIDs = Vector{String}(undef, n)
    driveIDs = Vector{String}(undef, n)
    playIDs = Vector{String}(undef, n)

    gameClocks = Vector{Float64}(undef, n)
    periods = Vector{Int}(undef, n)
    clocks = Vector{Float64}(undef, n)
    oteams = Vector{String}(undef, n)
    dteams = Vector{String}(undef, n)
    oconferences = Vector{String}(undef, n)
    dconferences = Vector{String}(undef, n)
    inConferences = Vector{Bool}(undef, n)
    ohome = Vector{Bool}(undef, n)
    dhome = Vector{Bool}(undef, n)
    positions = Vector{Float64}(undef, n)
    downs = Vector{Int}(undef, n)
    distances = Vector{Float64}(undef, n)
    oscores = Vector{Int}(undef, n)
    dscores = Vector{Int}(undef, n)

    playTypes = Vector{String}(undef, n)

    # skip over period end, which is not a play!
    n = 0
    for play in playsJSON
        if play["play_type"] == "End Period" continue end
        n += 1

        gameIDs[n] = string(hash((year, week, play["home"], play["away"])), base=16)
        driveIDs[n] = play["drive_id"]
        playIDs[n] = play["id"]

        clock = (play["clock"]["minutes"] * 60 + play["clock"]["seconds"]) / 900
        gameClock = ((4 - play["period"]) + clock) / 4.

        gameClocks[n] = gameClock
        periods[n] = play["period"]
        clocks[n] = clock

        oteams[n] = play["offense"]
        dteams[n] = play["defense"]
        if play["offense_conference"] == nothing
            oconferences[n] = "none"
        else
            oconferences[n] = play["offense_conference"]
        end
        if play["defense_conference"] == nothing
            dconferences[n] = "none"
        else
            dconferences[n] = play["defense_conference"]
        end
        inConferences = play["offense_conference"] == play["defense_conference"]

        ohome[n] = play["offense"] == play["home"]
        dhome[n] = play["defense"] == play["home"]

        positions[n] = play["yard_line"]/100.
        downs[n] = play["down"]
        distances[n] = play["distance"]/10.
        oscores[n] = play["offense_score"]
        dscores[n] = play["defense_score"]

        playTypes[n] = play["play_type"]
        # Position gains are strange on kicks--the play starts at the original line of scrimmage, and the offense is the kicking team.
        # This means, e.g., a kickoff return for a touchdown looks like the defense ran 100/35 lengths of the field.  Not ideal.
    #     distanceGains[i] = play["yards_gained"]/play["distance"]
    #     positionGains[i] = play["yards_gained"]/play["yard_line"]
    end

    return first(DataFrame(
        game_id = gameIDs,
        drive_id = driveIDs,
        play_id = playIDs,
        game_clock = gameClocks,
        period = periods,
        half = periods .> 2,
        clock = clocks,
        offense = oteams,
        defense = dteams,
        offense_conference = oconferences,
        defense_conference = dconferences,
        in_conference = inConferences,
        offense_home = ohome,
        defense_home = dhome,
        position = positions,
        down = downs,
        distance = distances,
        offense_score = oscores,
        defense_score = dscores,
        play_type = playTypes,
    ), n)
end

# Get games
function getgames(year::Int, week::Int)
    gamesURI = "https://api.collegefootballdata.com/games?seasonType=regular&year=$(year)&week=$(week)"
    resp = HTTP.request("GET", gamesURI)
    gamesJSON = JSON.parse(String(resp.body))

    return DataFrame([(game_id = string(hash((year, week, game["home_team"], game["away_team"])), base=16), venue = game["venue"]) for game in gamesJSON])
end

# The times here are when the play ends.  We want when the play begins.
function gameclock(playsDf::DataFrame)
    gameClocks = by(playsDf, :game_id) do df
        n = size(df, 1)
        rotClock = vcat([1.], df[1:n-1, :game_clock])
        timeDelta = rotClock - df.game_clock
        (play_id = df.play_id, game_clock = rotClock, time_delta = timeDelta)
    end
    clocks = by(playsDf, [:game_id, :period]) do df
        n = size(df, 1)
        (play_id = df.play_id, clock = vcat([1.], df[1:n-1, :clock]),)
    end
    # Join these columns back on by play_id
    return join(gameClocks, clocks, on=[:game_id, :play_id], kind=:left)
end


# Determine scores and turns-over
# Scoring is hard because:
#  1. Kicks are strange.  The offense and defense switch after the first play of a kickoff-initiated drive.
#  2. Interceptions and fumble recoveries on the first play of a drive are likewise strange.
# So map the scores to home/away, determine when the home or away team scores, then map that back to offense/defense.
function drivescores(playsDf::DataFrame)
    driveScores = by(playsDf, [:game_id, :drive_id]) do df
        n = size(df, 1)
        homeScore = df.offense_score .* df.offense_home + df.defense_score .* df.defense_home
        awayScore = df.offense_score .* df.defense_home + df.defense_score .* df.offense_home
        lastPlay = zeros(Bool, n)
        lastPlay[n] = true
        (play_id = df.play_id, home_score = homeScore, away_score = awayScore, last_play = lastPlay)
    end
    return driveScores
end

function scoredeltas(playsDf::DataFrame)
    scoreDeltas = by(playsDf, :game_id) do df
        homeScoreDelta = diff(vcat([0], df.home_score))
        awayScoreDelta = diff(vcat([0], df.away_score))
        offenseScoreDelta = df.offense_home .* homeScoreDelta + df.defense_home .* awayScoreDelta
        defenseScoreDelta = df.offense_home .* awayScoreDelta + df.defense_home .* homeScoreDelta
        scoreChange = offenseScoreDelta - defenseScoreDelta
        (play_id = df.play_id, score_change = scoreChange, turnover = df.last_play .& (scoreChange .== 0))
    end
    return scoreDeltas
end


# Touchesdown mean the play ended at either the 0 yard line (away team scored) or the 100 yard line (home team scored)
# This matters for returns off of kicks and turnsover, because the position on field will be the original line of scrimmage,
# but the yardage gained will be off of where the turnover happend.  On touchesdown especially, the next play starts at the
# 35 or 65 yard line, meaning simple differences in field position between plays may not look right.
function positiondeltas(playsDf::DataFrame)
    deltaPositions = by(playsDf, :drive_id) do df
        n = size(df, 1)
        endPos = df.position[n]
        if abs(df.score_change[n]) > 3
            endPos = Float64(df.offense_home[n])  # home drives to 1, defense drives to 0
            if n == 1 && df.play_type[n] == "Kickoff"
                endPos = 1 - endPos  # home (offense) kicks, away (defense) drives to 0; or away (offsense) kicks, home (defense) drives to 1.
            end
        end
        deltaPosition = vcat(df.position[2:n], endPos) - df.position
        deltaDistance = deltaPosition ./ df.distance .* 10  # how much of the distance to go did you get?  Position is 1/field, distance is 1/chains.
        (play_id = df.play_id, delta_position = deltaPosition, delta_distance = deltaDistance)
    end
    return deltaPositions
end

for year in 2014:2018
    for week in 1:14
        playsDf = getplays(year, week)
        gamesDf = getgames(year, week)

        playsDf = join(playsDf, gamesDf, on=:game_id, kind=:left)

        # Sort games in clock order.
        sort!(playsDf, (order(:game_id), order(:game_clock, rev=true)))

        clockDf = gameclock(playsDf)
        select!(playsDf, Not([:game_clock, :clock]))
        playsDf = join(playsDf, clockDf, on=[:play_id, :game_id, :period], kind=:left)

        driveScores = drivescores(playsDf)
        playsDf = join(playsDf, driveScores, on=[:play_id, :game_id, :drive_id], kind=:left)

        scoreDeltas = scoredeltas(playsDf)
        playsDf = join(playsDf, scoreDeltas, on=[:play_id, :game_id], kind=:left)

        deltaPositions = positiondeltas(playsDf)
        playsDf = join(playsDf, deltaPositions, on=[:play_id, :drive_id], kind=:left)

        outname = "plays_$(year)_$(week).csv"
        playsDf |> save(outname)

    end
end
