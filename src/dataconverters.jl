import LazyJSON
using DataFrames

struct Clock
    minutes::Int
    seconds::Int
end

"""
Get the fractional time in the period from the clock time.
"""
periodtime(c::Clock)::Float32 = (c.minutes * 60 + c.seconds) / 900

struct Play
    id::String
    drive_id::String
    play_type::String
    home::String
    away::String
    clock::Clock
    period::Int8
    offense::String
    defense::String
    offense_conference::Union{String,Nothing}
    defense_conference::Union{String,Nothing}
    yard_line::Int8
    down::Int8
    distance::Int8
    offense_score::Int8
    defense_score::Int8
end

# """
# Convert a LazyJSON.Array of Plays into an Array of Plays.
# """
# plays(json::LazyJSON.Array) = convert(Array{Play}, json)

"""
Rotate the game clock so that the clock refers to when the play begins rather than when the play ends.
"""
function shiftclock(x)
    n = size(x,1)
    s = circshift(x, 1)
    s[1] = 1
    s
end

"""
Mark the last element of an array.
"""
function marklast(x)
    l = zeros(Bool,size(x))
    l[end] = true
    l
end

"""
Convert offense/defense scores to home/away scores.
"""
homescore(x) = x.offense_score .* x.offense_home .+ x.defense_score .* x.defense_home
awayscore(x) = x.offense_score .* x.defense_home .+ x.defense_score .* x.offense_home

"""
Home and away scores are stable, so use them to determine scoring plays.  Positive score changes favor offense.
"""
function scoringplays(x)
    Δh = diff(vcat([0], x.home_score))
    Δa = diff(vcat([0], x.away_score))
    Δo = x.offense_home .* Δh .+ x.defense_home .* Δa
    Δd = x.offense_home .* Δa .+ x.defense_home .* Δh
    Int8.(Δo .- Δd)
end

"""
Calculate position gained per play.
"""
function positiongains(x)
    endPos = x.position[end]
    if abs(x.score_change[end]) > 3
        endPos = Float32(x.offense_home[end])  # home drives to 1, defense drives to 0
    end
    e = circshift(x.position, -1)
    e[1] = endPos
    e - x.position
end

"""
Convert an array of Plays to a DataFrame.
"""
function DataFrames.DataFrame(plays::Array{Play})

    df = DataFrames.DataFrame()
    for play in plays
        gameID = play.id[1:9]
        clock = periodtime(play.clock)
        gameClock = Float32(((4 - play.period) + clock) / 4.)
        offenseConference = play.offense_conference
        if isnothing(offenseConference)
            offenseConference = "none"
        end
        defenseConference = play.defense_conference
        if isnothing(defenseConference)
            defenseConference = "none"
        end
        sameConference = offenseConference != "none" && offenseConference == defenseConference
        offenseHome = play.offense == play.home
        defenseHome = play.defense == play.home
        positionOnField = Float32(play.yard_line / 100)
        fractionToGain = Float32(play.distance / 10)

        push!(df, (
            id = play.id,
            game_id = gameID,
            drive_id = play.drive_id,
            play_type = play.play_type,
            home = play.home,
            away = play.away,
            clock = clock,
            game_clock = gameClock,
            minutes = play.clock.minutes,
            seconds = play.clock.seconds,
            period = play.period,
            offense = play.offense,
            defense = play.defense,
            offense_conference = offenseConference,
            defense_conference = defenseConference,
            in_conference = sameConference,
            offense_home = offenseHome,
            defense_home = defenseHome,
            yard_line = play.yard_line,
            position = positionOnField,
            down = play.down,
            distance = play.distance,
            fraction_to_gain = fractionToGain,
            offense_score = play.offense_score,
            defense_score = play.defense_score,
        ))
    end

    sort!(df, (:game_id, order(:game_clock, rev=true)))

    newdat = by(df, :game_id,
        clock = :clock => shiftclock,
        game_clock = :game_clock => shiftclock,
        home_score = (:offense_home, :defense_home, :offense_score, :defense_score) => homescore,
        away_score = (:offense_home, :defense_home, :offense_score, :defense_score) => awayscore,
        )

    df = hcat(select(df, Not([:clock, :game_clock])), select(newdat, Not(:game_id)), copycols=false)

    scoreChanges = by(df, :game_id, score_change = (:home_score, :away_score, :offense_home, :defense_home) => scoringplays)
    df = hcat(df, select(scoreChanges, Not(:game_id)), copycols=false)

    positionGains = by(df, :game_id, position_change = (:position, :score_change, :offense_home) => positiongains)
    df = hcat(df, select(positionGains, Not(:game_id)), copycols=false)

    driveEnds = by(df, :drive_id, last_play = :drive_id => marklast)
    df = hcat(df, select(driveEnds, Not(:drive_id)), copycols=false)

    df[!,:turnover] = df[!,:last_play] .& df[!,:score_change] .== 0

    df
end
