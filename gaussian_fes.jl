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
        data = readdlm("colvar_$(count)_$(r)_$(rc)_$(nbasis)_$(nsamples).txt", ' ', Float64)
        for step in data
            push!(xlist, step[2])
            push!(ylist, step[3])
            push!(weights, exp(step[4] / kT))
        end
    end

    rangex = LinRange(domain_cv[1][1], domain_cv[1][2], nbins)
    rangey = LinRange(domain_cv[2][1], domain_cv[2][2], nbins)
    bw = w * (domain[1][2] - domain[1][1], domain[2][2] - domain[2][1])
    kde_result = kde(hcat(xlist, ylist), npoints = (nbins, nbins), weights = weights / sum(weights), bandwidth = bw)
    ik = InterpKDE(kde_result)
    open("data/kderw_$(count)_$(r)_$(rc)_$(nbasis)_$(nsamples).txt", "w") do file
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
