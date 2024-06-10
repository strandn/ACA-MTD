using DelimitedFiles
using Distributions
using KernelDensity
using StatsBase

function gaussian_erranalysis()
    domain = [(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
    kT = 1.0
    samples = []
    weights = []
    nbins = 1000
    for count in range
        data = readdlm("data/colvar_$(count)_$(r)_$(rc)_$(nbasis)_$(nsamples).out", ' ', Float64)
        for i in axes(data, 1)
            push!(samples, [data[i, 2], data[i, 3], data[i, 4], data[i, 5]])
            push!(weights, exp(data[i, 6] / kT))
        end
    end
    nbins = 1000
    ranges = [LinRange(d[1], d[2], nbins) for d in domain]
    for i in 1:4
        correct = readdlm("fes_correct_$i.txt", Float64)
        result = []
        idx = []
        bw = w1 .* (domain[i][2] - domain[i][1])
        kde_result = kde([step[i] for step in samples], npoints = nbins, weights = weights / sum(weights), bandwidth = bw)
        ik = InterpKDE(kde_result)
        for pos in 1:nbins
            dx = (domain[i][2] - domain[i][1]) / (nbins - 1)
            x = (pos - 1) * dx + domain[i][1]
            dens = pdf(ik, x)
            if dens > 1.0e-10
                push!(result, -log(dens) / kT)
                push!(idx, pos)
            end
        end
        offset = meanad(result, correct[idx])
        println(rmsd(result .- offset, correct[idx]))
    end
    nbins = 100
    ranges = [LinRange(d[1], d[2], nbins) for d in domain]
    for i in 1:4
        for j in i+1:4
            correct = reshape(readdlm("fes_correct_$i$j.txt", Float64), nbins ^ 2)
            result = []
            bw = w2 .* (domain[i][2] - domain[i][1], domain[j][2] - domain[j][1])
            kde_result = kde(hcat([step[i] for step in samples], [step[j] for step in samples]), npoints = (nbins, nbins), weights = weights / sum(weights), bandwidth = bw)
            ik = InterpKDE(kde_result)
            for ipos in 1:nbins
                dx = (domain[i][2] - domain[i][1]) / (nbins - 1)
                x = (ipos - 1) * dx + domain[i][1]
                for jpos in 1:nbins
                    dy = (domain[j][2] - domain[j][1]) / (nbins - 1)
                    y = (jpos - 1) * dy + domain[j][1]
                    pos = (ipos - 1) * nbins + jpos
                    dens = pdf(ik, x, y)
                    if dens > 1.0e-10
                        push!(result, -log(dens) / kT)
                        push!(idx, pos)
                    end
                end
            end
            offset = meanad(result, correct[idx])
            println(rmsd(result .- offset, correct[idx]))
        end
    end
end

r = parse(Int64, ARGS[1])
rc = parse(Int64, ARGS[2])
nbasis = parse(Int64, ARGS[3])
nsamples = parse(Int64, ARGS[4])
w1 = parse(Float64, ARGS[5])
w2 = parse(Float64, ARGS[6])
range = parse(Int64, ARGS[7]):parse(Int64, ARGS[8])
gaussian_erranalysis()
