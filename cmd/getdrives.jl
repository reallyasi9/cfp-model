if length(ARGS) < 1
    @error "must specify one year to download"
    exit(-1)
end

import Pkg
Pkg.activate(".")

include("../CFBModel.jl")
import .CFBModel

using DataFrames, FeatherFiles

for year in ARGS
    for week in 1:14
        # drives = CFBModel.getdrives(parse(Int, year), week)
        # for (i,drive) in enumerate(drives)
        #     @show i drive
        #     convert(CFBModel.Drive, drive)
        # end

        df = convert(Array{CFBModel.Drive}, CFBModel.getdrives(parse(Int, year), week)) |> DataFrame
        df[!,:week] .= week
        df[!,:year] .= year
        df |> save("drives_$(year)_$(week).feather")
    end
end
