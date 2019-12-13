struct Clock
    minutes::Union{Int8,Nothing}
    seconds::Union{Int8,Nothing}
end

"""
Get the seconds left on the clock.
"""
function seconds(c::Clock)::Int16
    t::Int16 = 0
    if !isnothing(c.minutes)
        t += c.minutes * 60
    end
    if !isnothing(c.seconds)
        t += c.seconds
    end
    t
end

"""
Get the fractional time left in the period from the clock time.
"""
periodtime(c::Clock)::Float32 = seconds(c)/900

"""
Get the fraction time left in the game from the clock time and the period number.
"""
gametime(c::Clock, p::Int8)::Float32 = (4-p+periodtime(c))/4

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

struct Drive
    offense::String
    offense_conference::Union{String,Nothing}
    defense::String
    defense_conference::Union{String,Nothing}
    game_id::Int64  # Convert to string
    id::String
    scoring::Bool
    start_period::Int8
    start_yardline::UInt8
    start_time::Clock
    end_period::Int8
    end_yardline::UInt8
    end_time::Clock
    elapsed::Clock
    plays::Int8
    yards::Int  # Needs some serious cleaning in certain games
    drive_result::String
end
