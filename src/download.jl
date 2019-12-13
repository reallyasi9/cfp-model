
import HTTP
import LazyJSON

const _baseURLScheme = "https"
const _baseURLHost = "api.collegefootballdata.com"
const _headers = ["accept" => "application/json"]

"""
Request GET from a given URI and return the JSON result as a LazyJSON object.
"""
lazyjson(resp) = LazyJSON.parse(String(resp.body))

"""
Call a collegefootballdata API call with given parameters
"""
cfbget(path::String, query) = HTTP.get(HTTP.URIs.URI(scheme=_baseURLScheme, host=_baseURLHost, path=path, query=query), _headers)

"""
Download all CFB plays for a given year/week combination and return them as
    a LazyJSON array of objects.
"""
getplays(year::Int, week::Int) = lazyjson(cfbget("/plays", ["year" => year, "week" => week]))

"""
Download all games for a given year-week combination and return them
    as a LazyJSON array of objects.
"""
getgames(year::Int, week::Int) = lazyjson(cfbget("/games", ["year" => year, "week" => week]))

"""
Download all CFB drives for a given year/week combination and return them as
    a LazyJSON array of objects.
"""
getdrives(year::Int, week::Int) = lazyjson(cfbget("/drives", ["year" => year, "week" => week]))
