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

data = readdlm("colvar_bias1.out", ' ', Float64)
println("$(minimum(data[:,2])) $(maximum(data[:,2])) $(minimum(data[:,3])) $(maximum(data[:,3]))")
kde_result = kde(data[:,2:3])
# kde_result = kde(data[:,2:3], bandwidth = (0.7, 0.9), npoints = (256, 256))
println("$(kde_result.x) $(kde_result.y)")
p = contour(kde_result.x, kde_result.y, kde_result.density)
savefig(p, "plot.png")

ik = InterpKDE(kde_result)
rhohat(x, y) = pdf(ik, x, y)
open("kde.out", "w") do file
	for x in kde_result.x
		for y in kde_result.y
			write(file, "$(rhohat(x, y)) ")
		end
		write(file, "\n")
	end
end
println()

# n_chains = 100
# n_samples = 1000
# jump_width = 0.01
# domain_cv_small = ((first(kde_result.x), last(kde_result.x)), (first(kde_result.y), last(kde_result.y)))
# F = ResFunc(rhohat, domain_cv_small)
# F.I, F.J = ([[Float64[]], [[1.64347034774359]]], [[Float64[]], [[2.6683299694004967]]])
# for r in 2:2
#     println("Target rank $r")
#     IJ = continuous_aca(F, [r], n_chains, n_samples, jump_width, mpi_comm)
#     println(IJ)
# 	open("res$r.out", "w") do file
# 		for x in kde_result.x
# 			for y in kde_result.y
# 				write(file, "$(abs(F(x, y))) ")
# 			end
# 			write(file, "\n")
# 		end
# 	end
# end
