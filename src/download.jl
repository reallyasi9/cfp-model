
import HTTP
import LazyJSON

"""
Download all CFB plays for a given year/week combination and return them as
    a LazyJSON array of objects.
"""
function getplays(year::Int, week::Int)

    playsURI = "https://api.collegefootballdata.com/plays?seasonType=regular&year=$(year)&week=$(week)"

    resp = HTTP.request("GET", playsURI)
    LazyJSON.parse(String(resp.body))
end

"""
Download all games for a given year-week combination and return them
    as a LazyJSON array of objects.
"""
function getgames(year::Int, week::Int)
    gamesURI = "https://api.collegefootballdata.com/games?seasonType=regular&year=$(year)&week=$(week)"
    resp = HTTP.request("GET", gamesURI)
    gamesJSON = JSON.parse(String(resp.body))

    return DataFrame([(game_id = string(hash((year, week, game["home_team"], game["away_team"])), base=16), venue = game["venue"]) for game in gamesJSON])
end
