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
weights = ones(len)
kde_result = kde(data[:,2:3], bandwidth = (0.1, 0.1), npoints = (nbins, nbins))
ik = InterpKDE(kde_result)
rhohat(x, y) = pdf(ik, x, y)
domain_cv_small = ((first(kde_result.x), last(kde_result.x)), (first(kde_result.y), last(kde_result.y)))
F = ResFunc(rhohat, domain_cv_small)
fp = open("pivots.txt")
initIJ(F, eval(Meta.parse(readline(fp))))
close(fp)

data1 = readdlm("colvar_bias1.txt", ' ', Float64)
len1 = length(data1[:, 1])
weights1 = [exp(Vbias(F, data1[i, 2], data1[i, 3])) for i in 1:len1]
kde_result = kde(data1[:,2:3], bandwidth = (0.1, 0.1), npoints = (nbins, nbins))
ik = InterpKDE(kde_result)
rhohat(x, y) = pdf(ik, x, y)
domain_cv_small = ((first(kde_result.x), last(kde_result.x)), (first(kde_result.y), last(kde_result.y)))
F1 = ResFunc(rhohat, domain_cv_small)
fp = open("pivots1.txt")
initIJ(F1, eval(Meta.parse(readline(fp))))
close(fp)

data2 = readdlm("colvar_bias2.txt", ' ', Float64)
len2 = length(data2[:, 1])
weights2 = [exp(Vbias(F1, data2[i, 2], data2[i, 3])) for i in 1:len1]

data = vcat(data, data1, data2)
weights = [weights; weights1; weights2]
println("$(minimum(data[:,2])) $(maximum(data[:,2])) $(minimum(data[:,3])) $(maximum(data[:,3]))")
# kde_result = kde(data[:,2:3], weights = weights)
kde_result = kde(data[:,2:3], weights = weights, bandwidth = (0.1, 0.1), npoints = (nbins, nbins))
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
fp = open("pivots1.txt")
initIJ(F, eval(Meta.parse(readline(fp))))
close(fp)
for r in 3:3
    println("Target rank $r")
    IJ = continuous_aca(F, [r], n_chains, n_samples, jump_width, mpi_comm)
	open("pivots2.txt", "w") do file
		println(file, IJ)
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
for r in 1:3
	row, col = F.I[F.pos + 1][r], F.J[F.pos + 1][r]
	println("$(row) $(col) $(F.f(row..., col...)...)")
end