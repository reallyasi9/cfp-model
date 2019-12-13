import LazyJSON
using DataFrames

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
        gameClock = gametime(play.clock, play.period)
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

"""
Convert a DataFrame of plays to a DataFrame of drives.
"""
function generatedrives(plays::AbstractDataFrame)

    commonCols = [:year, :week, :game_id, :drive_id, :home, :away, :in_conference]
    sortedPlays = sort(plays, (:game_id, order(:game_clock, rev=true)))
    # Skip kickoffs, as they are strange.
    notKick = .!startswith.(sortedPlays.play_type, "Kick")
    drives = by(sortedPlays[notKick,:], commonCols,
        offense = (:offense) => last,  # who has the ball last?
        defense = (:defense) => last,
        plays = (:id) => length,
        position_start = (:position) => first,
        position_end = (:position) => last,
        game_clock_start = (:game_clock) => first,
        game_clock_end = (:game_clock) => last,
        clock_start = (:clock) => first,
        clock_end = (:clock) => last,
        period_start = (:period) => first,
        period_end = (:period) => last,
        score_change = (:score_change) => last,
        last_play = (:play_type) => last,
        turnover = (:turnover) => last,
    )

    # Home drives toward 1, away drives toward 0.
    # Make everyone drive toward 1.
    away = drives.offense .== drives.away
    drives[away,:position_start] = 1 .- drives[away,:position_start]
    drives[away,:position_end] = 1 .- drives[away,:position_end]

    # If a drive spans a period, the field flips, meaning the positions are messed up.
    # spanPeriod = drives.period_start .!= drives.period_end
    # drives[spanPeriod,:position_start] = 1 .- drives[spanPeriod,:position_start]
    # If a drive ends in the endzone, make sure that is noted appropriately
    offensiveTD = drives.score_change .== 7
    defensiveTD = drives.score_change .== -7
    drives[offensiveTD,:position_end] .= 1
    drives[defensiveTD,:position_end] .= 0
    drives[!,:position_advance] = (drives.position_end .- drives.position_start)
    drives[!,:plays] .= Int8.(drives.plays)

    drives
end

"""
Convert an array of Drives to a DataFrame.
"""
function DataFrames.DataFrame(drives::Array{Drive})

    df = DataFrames.DataFrame()
    for drive in drives
        gameID = string(drive.game_id)
        startClock = periodtime(drive.start_time)
        endClock = periodtime(drive.end_time)
        startGameClock = gametime(drive.start_time, drive.start_period)
        endGameClock = gametime(drive.end_time, drive.end_period)
        offenseConference = drive.offense_conference
        if isnothing(offenseConference)
            offenseConference = "none"
        end
        defenseConference = drive.defense_conference
        if isnothing(defenseConference)
            defenseConference = "none"
        end
        sameConference = offenseConference != "none" && offenseConference == defenseConference
        startYards = clamp(drive.start_yardline, 0, 100)
        endYards = clamp(drive.end_yardline, 0, 100)
        yardsGained = clamp(drive.yards, -100, 100)
        # believe start if it is sensible
        if startYards != drive.start_yardline
            @warn "Nonsensical drive start yards clamped to gridiron" drive.id Int64(drive.start_yardline)
        end
        # believe end if it is sensible
        if endYards != drive.end_yardline
            @warn "Nonsensical drive end yards clamped to gridiron" drive.id Int64(drive.end_yardline)
        end
        # believe gained if it is sensible
        if yardsGained != drive.yards
            @warn "Nonsensical drive yards gained clamped to gridiron" drive.id Int64(drive.yards)
        end
        # calculate my own delta

        startPositionOnField = Float32(startYards / 100)
        endPositionOnField = Float32(endYards / 100)

        push!(df, (
            id = drive.id,
            game_id = gameID,
            offense = drive.offense,
            offense_conference = offenseConference,
            defense = drive.defense,
            defense_conference = defenseConference,
            same_conference = sameConference,
            scoring = drive.scoring,
            start_period = drive.start_period,
            start_yardline = UInt8(startYards),
            start_position = startPositionOnField,
            start_clock = startClock,
            start_game_clock = startGameClock,
            end_period = drive.end_period,
            end_yardline = UInt8(endYards),
            end_position = endPositionOnField,
            end_clock = endClock,
            end_game_clock = endGameClock,
            elapsed = seconds(drive.elapsed),
            plays = drive.plays,
            yards = Int8(yardsGained),
            drive_result = drive.drive_result,
        ))
    end

    sort!(df, (:game_id, order(:start_game_clock, rev=true)))
end
