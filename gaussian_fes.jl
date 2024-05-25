using DelimitedFiles
using Distributions
using KernelDensity

function gaussian_fes()
    domain_cv = [(-1.5, 4.0), (-1.5, 4.5)]
    kT = 1.0
    xlist = []
    ylist = []
    weights = []
    nbins = 100
    for count in range
        data = readdlm("data/colvar_$(count)_$(r)_$(rc)_$(nbasis)_$(nsamples).txt", ' ', Float64)
        for i in axes(data, 1)
            # println("$(data[i, 2]) $(data[i, 3]) $(data[i, 4])")
            push!(xlist, data[i, 2])
            push!(ylist, data[i, 3])
            push!(weights, exp(data[i, 4] / kT))
        end
    end

    rangex = LinRange(domain_cv[1][1], domain_cv[1][2], nbins)
    rangey = LinRange(domain_cv[2][1], domain_cv[2][2], nbins)
    bw = w .* (domain_cv[1][2] - domain_cv[1][1], domain_cv[2][2] - domain_cv[2][1])
    # kde_result = kde(hcat(xlist, ylist), npoints = (nbins, nbins), bandwidth = bw)
    # ik = InterpKDE(kde_result)
    # open("data/kde.out", "w") do file
    #     for x in rangex
    #         for y in rangey
    #             write(file, "$(pdf(ik, x, y)) ")
    #         end
    #         write(file, "\n")
    #     end
    # end
    kde_result = kde(hcat(xlist, ylist), npoints = (nbins, nbins), weights = weights / sum(weights), bandwidth = bw)
    ik = InterpKDE(kde_result)
    open("data/kderw.out", "w") do file
        for x in rangex
            for y in rangey
                write(file, "$(pdf(ik, x, y)) ")
            end
            write(file, "\n")
        end
    end
end

r = parse(Int64, ARGS[1])
rc = parse(Int64, ARGS[2])
nbasis = parse(Int64, ARGS[3])
nsamples = parse(Int64, ARGS[4])
w = parse(Float64, ARGS[5])
range = parse(Int64, ARGS[6]):parse(Int64, ARGS[7])
gaussian_fes()
