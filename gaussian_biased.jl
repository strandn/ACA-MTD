using Random
using Distributions
using ForwardDiff
using LinearAlgebra
using KernelDensity
using DelimitedFiles
using Plots
using MPI

include("tt_aca.jl")

MPI.Init()
mpi_comm = MPI.COMM_WORLD

large1 = [1.0, 0.0, 0.0, -1.0]
large2 = [-1.0, -1.0, 1.0, -1.0]
large3 = [-1.0, -1.0, -1.0, 1.0]
max1 = [1/3, -2/3, 2/3, -1.0]
max2 = [-1/3, -2/3, -2/3, 1/3]
max3 = [-1.0, -1.0, 1/3, -1/3]
max4 = [-1/3, -2/3, 0, -1/3]
v12 = large2 - large1
v12 = v12 / norm(v12)
v13 = large3 - large1
v13 = v13 - dot(v13, v12) / norm(v12)^2 * v12
v13 = v13 / norm(v13)

function V(r)
	x1, x2, x3, x4 = r
	return 10 * exp(-norm(r - max1) ^ 2) + 10 * exp(-norm(r - max2) ^ 2) + 10 * exp(-norm(r - max3) ^ 2) +
		15 * exp(-norm(r - max4) ^ 2) -
		15 * exp(-norm(r - large1) ^ 2) - 20 * exp(-norm(r - large2) ^ 2) - 25 * exp(-norm(r - large3) ^ 2) +
		(x1 + 1/3) ^ 4 / 5 + (x2 + 2/3) ^ 4 / 5 + x3 ^ 4 / 5 + (x4 + 1/3) ^ 4 / 5
end

grad_V(x1, x2, x3, x4) = ForwardDiff.gradient(V, [x1, x2, x3, x4])

domain = ((-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0))
domain_cv = ((-1.5, 4.0), (-1.5, 4.5))

data = readdlm("colvar.out", ' ', Float64)
println("$(minimum(data[:,2])) $(maximum(data[:,2])) $(minimum(data[:,3])) $(maximum(data[:,3]))")
kde_result = kde(data[:,2:3])
println("$(kde_result.x) $(kde_result.y)")
# open("kde.out", "w") do file
# 	for i in 1:256
# 		for j in 1:256
# 			write(file, "$(kde_result.density[i, j]) ")
# 		end
# 		write(file, "\n")
# 	end
# end
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

n_chains = 100
n_samples = 1000
jump_width = 0.01
domain_cv_small = ((first(kde_result.x), last(kde_result.x)), (first(kde_result.y), last(kde_result.y)))
F = ResFunc(rhohat, domain_cv_small)
# r = 5
# println("Target rank $r")
# IJ = continuous_aca(F, [r], n_chains, n_samples, jump_width, mpi_comm)
# println(IJ)
for r in 1:5
    println("Target rank $r")
    IJ = continuous_aca(F, [r], n_chains, n_samples, jump_width, mpi_comm)
    println(IJ)
	open("res$r.out", "w") do file
		for x in kde_result.x
			for y in kde_result.y
				write(file, "$(abs(F(x, y))) ")
			end
			write(file, "\n")
		end
	end
end
