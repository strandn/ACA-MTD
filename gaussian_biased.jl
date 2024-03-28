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
println("$(minimum(data[:,2])) $(maximum(data[:,2])) $(minimum(data[:,3])) $(maximum(data[:,3]))")
# kde_result = kde(data[:,2:3], npoints = (nbins, nbins))
# kde_result = kde(data[:,2:3], bandwidth = (0.1, 0.1), npoints = (nbins, nbins))
kde_result = kde(data[:,2:3], bandwidth = (0.5, 0.5), npoints = (nbins, nbins))
# kde_result = kde(data[:,2:3], bandwidth = (0.9, 1.2), npoints = (nbins, nbins))
# kde_result = kde(data[:,2:3], bandwidth = (0.87, 1.11), npoints = (nbins, nbins))
println("$(kde_result.x) $(kde_result.y)")
p = contour(kde_result.x, kde_result.y, kde_result.density)
savefig(p, "plot.png")

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
x_full = domain_cv[1][1]:(domain_cv[1][2]-domain_cv[1][1])/(nbins-1):domain_cv[1][2]
y_full = domain_cv[2][1]:(domain_cv[2][2]-domain_cv[2][1])/(nbins-1):domain_cv[2][2]
domain_cv_small = ((first(kde_result.x), last(kde_result.x)), (first(kde_result.y), last(kde_result.y)))
F = ResFunc(rhohat, domain_cv_small)
for r in 1:1
    println("Target rank $r")
    IJ = continuous_aca(F, [r], n_chains, n_samples, jump_width, mpi_comm)
	open("pivots.txt", "w") do file
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
end

vmax = 0.0
open("vbias1.txt", "w") do file
	for x in kde_result.x
		for y in kde_result.y
			write(file, "$(Vbias(F, vmax, x, y)) ")
		end
		write(file, "\n")
	end
end

large1 = [1.0, 0.0, 0.0, -1.0]
large2 = [-1.0, -1.0, 1.0, -1.0]
large3 = [-1.0, -1.0, -1.0, 1.0]
max1 = [0.0, -0.5, 0.5, -1.0]
max2 = [0.0, -0.5, -0.5, 0.0]
max3 = [-1.0, -1.0, 0.0, -0.0]
max4 = [-1/3, -2/3, 0.0, -1/3]
v12 = large2 - large1
v12 = v12 / norm(v12)
v13 = large3 - large1
v13 = v13 - dot(v13, v12) / norm(v12)^2 * v12
v13 = v13 / norm(v13)

x(z) = 0.82 - 0.82 * z[1] - 0.41 * z[2] + 0.41 * z[3]
y(z) = 0.98 - 0.25 * z[1] - 0.12 * z[2] - 0.62 * z[3] + 0.74 * z[4]

function grad_Vbias(r)
	h = (step(kde_result.x), step(kde_result.y))
	dx = ForwardDiff.gradient(x, r)
	dy = ForwardDiff.gradient(y, r)
	dVdx = (Vbias(F, vmax, x(r) + h[1], y(r)) - Vbias(F, vmax, x(r) - h[1], y(r))) / (2 * h[1])
	dVdy = (Vbias(F, vmax, x(r), y(r) + h[2]) - Vbias(F, vmax, x(r), y(r) - h[2])) / (2 * h[2])
	dVdx1 = dVdx * dx[1] + dVdy * dy[1]
	dVdx2 = dVdx * dx[2] + dVdy * dy[2]
	dVdx3 = dVdx * dx[3] + dVdy * dy[3]
	dVdx4 = dVdx * dx[4] + dVdy * dy[4]
	return [dVdx1, dVdx2, dVdx3, dVdx4]
end

function V(r)
	x1, x2, x3, x4 = r
	return 30 * exp(-5 * norm(r - max1) ^ 2) + 35 * exp(-5 * norm(r - max2) ^ 2) + 40 * exp(-5 * norm(r - max3) ^ 2) +
		45 * exp(-5 * norm(r - max4) ^ 2) -
		15 * exp(-norm(r - large1) ^ 2) - 20 * exp(-norm(r - large2) ^ 2) - 25 * exp(-norm(r - large3) ^ 2) +
		(x1 + 1/3) ^ 4 / 5 + (x2 + 2/3) ^ 4 / 5 + x3 ^ 4 / 5 + (x4 + 1/3) ^ 4 / 5
end

grad_V(x1, x2, x3, x4) = ForwardDiff.gradient(V, [x1, x2, x3, x4]) + grad_Vbias([x1, x2, x3, x4])

T = 1.0
gamma = 1.0
dt = 1.0e-4
steps = 1e7

# x1 = rand(Normal(-1.0, 0.1))
# x2 = rand(Normal(-1.0, 0.1))
# x3 = rand(Normal(-1.0, 0.1))
# x4 = rand(Normal(1.0, 0.1))
x1, x2, x3, x4 = data[len, 4:7]
v1 = v2 = v3 = v4 = 0.0

kb = 1.0
sigma = sqrt(2 * kb * T / (gamma * dt))
normal_dist = Normal(0.0, sigma)

stride = 100
t = 0.0

traj = []
for i in 1:steps
	grad = grad_V(x1, x2, x3, x4)

	global v1 = -(grad[1] / gamma) + rand(normal_dist)
	global v2 = -(grad[2] / gamma) + rand(normal_dist)
	global v3 = -(grad[3] / gamma) + rand(normal_dist)
	global v4 = -(grad[4] / gamma) + rand(normal_dist)

	old = [x1, x2, x3, x4]
	x0 = x([x1, x2, x3, x4])
	y0 = y([x1, x2, x3, x4])

	global x1 += v1 * dt
	global x2 += v2 * dt
	global x3 += v3 * dt
	global x4 += v4 * dt

	if isnan(x1) || isnan(x2) || isnan(x3) || isnan(x4)
		println("$old $([x0, y0]) $grad")
		exit(1)
	end
	
	x1 = clamp(x1, domain[1][1], domain[1][2])
	x2 = clamp(x2, domain[2][1], domain[2][2])
	x3 = clamp(x3, domain[3][1], domain[3][2])
	x4 = clamp(x4, domain[4][1], domain[4][2])

	if any(abs.(grad) .> 100)
		println("$t $old $([x0, y0]) $([x1, x2, x3, x4]) $([x([x1, x2, x3, x4]), y([x1, x2, x3, x4])]) $grad")
	end

	global t += dt

	if i % stride == 0
		push!(traj, (t, x([x1, x2, x3, x4]), y([x1, x2, x3, x4]), x1, x2, x3, x4))
	end
end
open("colvar_bias1.txt", "w") do file
	for step in traj
		write(file, "$(step[1]) $(step[2]) $(step[3]) $(step[4]) $(step[5]) $(step[6]) $(step[7])\n")
	end
end
