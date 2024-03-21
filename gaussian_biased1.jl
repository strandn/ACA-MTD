using KernelDensity
using DelimitedFiles
using Plots
using MPI
using Random
using Distributions
using ForwardDiff
using LinearAlgebra

include("tt_aca.jl")

MPI.Init()
mpi_comm = MPI.COMM_WORLD

domain = ((-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0))
domain_cv = ((-1.5, 4.0), (-1.5, 4.5))
nbins = 256

data = readdlm("colvar.txt", ' ', Float64)
len = length(data[:, 1])
kde_result = kde(data[:,2:3], bandwidth = (0.9, 1.2), npoints = (nbins, nbins))
ik = InterpKDE(kde_result)
rhohat(x, y) = pdf(ik, x, y)
domain_cv_small = ((first(kde_result.x), last(kde_result.x)), (first(kde_result.y), last(kde_result.y)))
F = ResFunc(rhohat, domain_cv_small)
fp = open("pivots.txt")
initIJ(F, eval(Meta.parse(readline(fp))))
close(fp)

data = vcat(data, readdlm("colvar_bias1.txt", ' ', Float64))
weights = ones(length(data[:, 1]))
for i in len+1:length(data[:, 1])
	weights[i] = exp(Vbias(F, data[i, 2], data[i, 3]))
end
weights /= sum(weights)
println("$(minimum(data[:,2])) $(maximum(data[:,2])) $(minimum(data[:,3])) $(maximum(data[:,3]))")
# kde_result = kde(data[:,2:3], weights = weights)
kde_result = kde(data[:,2:3], weights = weights, bandwidth = (0.7, 0.6), npoints = (nbins, nbins))
println("$(kde_result.x) $(kde_result.y)")

ik = InterpKDE(kde_result)
rhohat(x, y) = pdf(ik, x, y)
open("kde.txt", "w") do file
	for x in kde_result.x
		for y in kde_result.y
			write(file, "$(rhohat(x, y)) ")
		end
		write(file, "\n")
	end
end
println()

n_chains = 100
n_samples = 1000
jump_width = 0.01
domain_cv_small = ((first(kde_result.x), last(kde_result.x)), (first(kde_result.y), last(kde_result.y)))
F = ResFunc(rhohat, domain_cv_small)
fp = open("pivots.txt")
initIJ(F, eval(Meta.parse(readline(fp))))
close(fp)
for r in 2:2
    println("Target rank $r")
    IJ = continuous_aca(F, [r], n_chains, n_samples, jump_width, mpi_comm)
	open("pivots1.txt", "w") do file
		write(file, IJ)
	end
    println(IJ)
	open("res$r.txt", "w") do file
		for x in kde_result.x
			for y in kde_result.y
				write(file, "$(abs(F(x, y))) ")
			end
			write(file, "\n")
		end
	end
	open("vbias$r.txt", "w") do file
		for x in kde_result.x
			for y in kde_result.y
				write(file, "$(Vbias(F, x, y)) ")
			end
			write(file, "\n")
		end
	end
end
