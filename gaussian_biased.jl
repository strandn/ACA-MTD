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

x(z) = 0.82 - 0.82 * z[1] - 0.41 * z[2] + 0.41 * z[3]
y(z) = 0.98 - 0.25 * z[1] - 0.12 * z[2] - 0.62 * z[3] + 0.74 * z[4]

function V(r)
	x1, x2, x3, x4 = r
	large1 = [1.0, 0.0, 0.0, -1.0]
	large2 = [-1.0, -1.0, 1.0, -1.0]
	large3 = [-1.0, -1.0, -1.0, 1.0]
	max1 = [0.0, -0.5, 0.5, -1.0]
	max2 = [0.0, -0.5, -0.5, 0.0]
	max3 = [-1.0, -1.0, 0.0, -0.0]
	max4 = [-1/3, -2/3, 0.0, -1/3]
	return 30 * exp(-5 * norm(r - max1) ^ 2) + 35 * exp(-5 * norm(r - max2) ^ 2) + 40 * exp(-5 * norm(r - max3) ^ 2) +
		45 * exp(-5 * norm(r - max4) ^ 2) -
		15 * exp(-norm(r - large1) ^ 2) - 20 * exp(-norm(r - large2) ^ 2) - 25 * exp(-norm(r - large3) ^ 2) +
		(x1 + 1/3) ^ 4 / 5 + (x2 + 2/3) ^ 4 / 5 + x3 ^ 4 / 5 + (x4 + 1/3) ^ 4 / 5
end

function dens(F, s)
	order = F.ndims
	npivots = [length(F.I[i]) for i in 2:order]
	result = zeros(1, npivots[1])
	for j in 1:npivots[1]
		result[j] = F.f((s[1], F.J[2][j]...)...)
	end
	AIJ = zeros(npivots[1], npivots[1])
	for j in 1:npivots[1]
		for k in 1:npivots[1]
			AIJ[j, k] = F.f((F.I[2][j]..., F.J[2][k]...)...)
		end
	end
	result *= inv(AIJ)
	for i in 2:order-1
		resulti = zeros(npivots[i - 1], npivots[i])
		for j in 1:npivots[i - 1]
			for k in 1:npivots[i]
				resulti[j, k] = F.f((F.I[i][j]..., s[i], F.J[i + 1][k]...)...)
			end
		end
		AIJ = zeros(npivots[i], npivots[i])
		for j in 1:npivots[i]
			for k in 1:npivots[i]
				AIJ[j, k] = F.f((F.I[i + 1][j]..., F.J[i + 1][k]...)...)
			end
		end
		result *= resulti * inv(AIJ)
	end
	R = zeros(npivots[order - 1])
	for j in 1:npivots[order - 1]
		R[j] = F.f((F.I[order][j]..., s[order])...)
	end
	result *= R
	return result[]
end

dens_adj(F, s) = 1.0e2 * max(dens(F, s), 1.0e-2)

function Vbias(s, rholist, kT)
	result = 0.0
	for F in rholist
		result += kT * log(dens_adj(F, s))
	end
	return result
end

Vbias_shifted(s, rholist, kT, Vshift) = max(Vbias(s, rholist, kT) - Vshift, 0.0)

function Vtop(rholist, kT, samples)
	max = 0.0
	for s in samples
		# result = 0.0
		# for F in rholist
		# 	result += kT * log(dens_adj(F, s))
		# end
		result = Vbias(s, rholist, kT)
		if result > max
			max = result
		end
	end
	return max
end

function dVbias(s, rholist, kT, Vshift, h)
	grad = fill(0.0, length(s))
	if Vbias(s, rholist, kT) <= Vshift
		return grad
	end
	for F in rholist
		rho = dens(F, s)
		if rho > 1.0e-2
			order = F.ndims
			npivots = [length(F.I[i]) for i in 2:order]
			outer = []
			inner = []
			mat = zeros(1, npivots[1])
			for j in 1:npivots[1]
				mat[j] = F.f((s[1], F.J[2][j]...)...)
			end
			push!(outer, mat)
			AIJ = zeros(npivots[1], npivots[1])
			for j in 1:npivots[1]
				for k in 1:npivots[1]
					AIJ[j, k] = F.f((F.I[2][j]..., F.J[2][k]...)...)
				end
			end
			push!(inner, inv(AIJ))
			for i in 2:order-1
				mat = zeros(npivots[i - 1], npivots[i])
				for j in 1:npivots[i - 1]
					for k in 1:npivots[i]
						mat[j, k] = F.f((F.I[i][j]..., s[i], F.J[i + 1][k]...)...)
					end
				end
				push!(outer, mat)
				AIJ = zeros(npivots[i], npivots[i])
				for j in 1:npivots[i]
					for k in 1:npivots[i]
						AIJ[j, k] = F.f((F.I[i + 1][j]..., F.J[i + 1][k]...)...)
					end
				end
				push!(inner, inv(AIJ))
			end
			mat = zeros(npivots[order - 1], 1)
			for j in 1:npivots[order - 1]
				mat[j] = F.f((F.I[order][j]..., s[order])...)
			end
			push!(outer, mat)
			douter = []
			mat = zeros(1, npivots[1])
			for j in 1:npivots[1]
				mat[j] = (F.f((s[1] + h[1], F.J[2][j]...)...) - F.f((s[1] - h[1], F.J[2][j]...)...)) / (2 * h[1])
				# mat[j] *= kT / F.f((s[1], F.J[2][j]...)...)
			end
			push!(douter, mat)
			for i in 2:order-1
				mat = zeros(npivots[i - 1], npivots[i])
				for j in 1:npivots[i - 1]
					for k in 1:npivots[i]
						mat[j, k] = (F.f((F.I[i][j]..., s[i] + h[i], F.J[i + 1][k]...)...) - F.f((F.I[i][j]..., s[i] - h[i], F.J[i + 1][k]...)...)) / (2 * h[i])
						# mat[j, k] *= kT / F.f((F.I[i][j]..., s[i], F.J[i + 1][k]...)...)
					end
				end
				push!(douter, mat)
			end
			mat = zeros(npivots[order - 1], 1)
			for j in 1:npivots[order - 1]
				mat[j] = (F.f((F.I[order][j]..., s[order] + h[order])...) - F.f((F.I[order][j]..., s[order] - h[order])...))/ (2 * h[order])
				# mat[j] += kT / F.f((F.I[order][j]..., s[order])...)
			end
			push!(douter, mat)
			# dVdx = douter[1] * inner[1] * outer[2]
			# dVdy = outer[1] * inner[1] * douter[2]
			# inc = [douter[1]; [outer[i] for i in 2:order]]
			inc = fill(outer[1], order - 1)
			pushfirst!(inc, douter[1])
			for i in 1:order
				for j in 2:order
					inc[i] *= inner[j - 1] * (i == j ? douter[j] : outer[j])
				end
			end
			grad += [inci[1, 1] for inci in inc] * kT / rho
		end
	end
	return grad
end

function grad_V(r, rholist, kT, Vshift, h)
	grad = ForwardDiff.gradient(V, r)
	dx = ForwardDiff.gradient(x, r)
	dy = ForwardDiff.gradient(y, r)
	dVdx, dVdy = dVbias([x(r), y(r)], rholist, kT, Vshift, h)
	dVdx1 = dVdx * dx[1] + dVdy * dy[1]
	dVdx2 = dVdx * dx[2] + dVdy * dy[2]
	dVdx3 = dVdx * dx[3] + dVdy * dy[3]
	dVdx4 = dVdx * dx[4] + dVdy * dy[4]
	grad += [dVdx1, dVdx2, dVdx3, dVdx4]
	return grad
end

function aca_mtd()
	domain = ((-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0))
	# domain_cv = ((-1.5, 4.0), (-1.5, 4.5))
	nbins = 100

	T = 1.0
	gamma = 1.0
	dt = 1.0e-4
	steps = 1e7
	stride = 100
	# steps = 10000
	# stride = 10
	nbiasupdates = 10

	x1 = rand(Normal(-1.0, 0.1))
	x2 = rand(Normal(-1.0, 0.1))
	x3 = rand(Normal(-1.0, 0.1))
	x4 = rand(Normal(1.0, 0.1))
	v1 = v2 = v3 = v4 = 0.0

	kb = 1.0
	sigma = sqrt(2 * kb * T / (gamma * dt))
	normal_dist = Normal(0.0, sigma)

	rholist = []
	Vmax = 20 * kb * T
	samples = []
	Vshift = 0.0
	# h = [domain_cv[1][2] - domain_cv[1][1], domain_cv[2][2] - domain_cv[2][1]] / nbins
	h = (0.0, 0.0)

	for count in 1:nbiasupdates
		println("Vbias update $count...")
		flush(stdout)
		t = 0.0

		traj = []
		for i in 1:steps
			grad = grad_V([x1, x2, x3, x4], rholist, kb * T, Vshift, h)

			v1 = -(grad[1] / gamma) + rand(normal_dist)
			v2 = -(grad[2] / gamma) + rand(normal_dist)
			v3 = -(grad[3] / gamma) + rand(normal_dist)
			v4 = -(grad[4] / gamma) + rand(normal_dist)

			old = [x1, x2, x3, x4]
			x0 = x([x1, x2, x3, x4])
			y0 = y([x1, x2, x3, x4])

			x1 += v1 * dt
			x2 += v2 * dt
			x3 += v3 * dt
			x4 += v4 * dt

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
				flush(stdout)
			end

			t += dt

			if i % stride == 0
				# push!(traj, (t, x([x1, x2, x3, x4]), y([x1, x2, x3, x4]), x1, x2, x3, x4))
				push!(traj, [t, x([x1, x2, x3, x4]), y([x1, x2, x3, x4]), Vbias_shifted([x1, x2, x3, x4], rholist, kb * T, Vshift)])
				push!(samples, [x([x1, x2, x3, x4]), y([x1, x2, x3, x4])])
			end
		end
		open("colvar_$count.txt", "w") do file
			for step in traj
				write(file, "$(step[1]) $(step[2]) $(step[3]) $(step[4])\n")
			end
		end

		# data = readdlm("colvar.txt", ' ', Float64)
		# len = length(data[:, 1])
		xlist = [step[2] for step in traj]
		ylist = [step[3] for step in traj]
		println("$(minimum(xlist)) $(maximum(xlist)) $(minimum(ylist)) $(maximum(ylist))")
		flush(stdout)
		# kde_result = kde(hcat(xlist, ylist), npoints = (nbins, nbins))
		kde_result = kde(hcat(xlist, ylist), npoints = (nbins, nbins), bandwidth = (0.1, 0.1))
		println("$(kde_result.x) $(kde_result.y)")
		flush(stdout)
		
		ik = InterpKDE(kde_result)
		rhohat(x, y) = pdf(ik, x, y)
		open("kde_$count.txt", "w") do file
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
		n_chains = 10
		n_samples = 100
		jump_width = 0.01
		# x_full = domain_cv[1][1]:(domain_cv[1][2]-domain_cv[1][1])/(nbins-1):domain_cv[1][2]
		# y_full = domain_cv[2][1]:(domain_cv[2][2]-domain_cv[2][1])/(nbins-1):domain_cv[2][2]
		domain_cv_small = ((first(kde_result.x), last(kde_result.x)), (first(kde_result.y), last(kde_result.y)))
		F = ResFunc(rhohat, domain_cv_small)
		rank = 50
		println("Target rank $rank")
		IJ = continuous_aca(F, [rank], n_chains, n_samples, jump_width, mpi_comm)
		println(IJ)
		println()

		open("dF_$(count).txt", "w") do file
			for x in kde_result.x
				for y in kde_result.y
					write(file, "$(-kb * T * log(dens_adj(F, [x, y]))) ")
				end
				write(file, "\n")
			end
		end

		push!(rholist, F)
		Vpeak = Vtop(rholist, kb * T, samples)
		Vshift = max(Vpeak - Vmax, 0.0)
		println("Vtop = $Vpeak Vshift = $Vshift")
		println()
		# open("res$count.txt", "w") do file
		# 	for x in kde_result.x
		# 		for y in kde_result.y
		# 			write(file, "$(abs(F(x, y))) ")
		# 		end
		# 		write(file, "\n")
		# 	end
		# end
		open("F_$count.txt", "w") do file
			for x in kde_result.x
				for y in kde_result.y
					write(file, "$(-Vbias_shifted([x, y], rholist, kb * T, Vshift)) ")
				end
				write(file, "\n")
			end
		end
		h = (step(kde_result.x), step(kde_result.y))
		# for x in kde_result.x
		# 	for y in kde_result.y
		# 		grad = dVbias([x, y], rholist, kb * T, Vshift, h)
		# 		print("$grad ")
		# 	end
		# 	println()
		# end
		open("dVbiasdx_$count.txt", "w") do filex
			open("dVbiasdy_$count.txt", "w") do filey
				for x in kde_result.x
					for y in kde_result.y
						grad = dVbias([x, y], rholist, kb * T, Vshift, h)
						write(filex, "$(grad[1]) ")
						write(filey, "$(grad[2]) ")
					end
					write(filex, "\n")
					write(filey, "\n")
				end
			end
		end
	end
end

# println(ARGS)
# flush(stdout)
# biasfactor = parse(Int64, ARGS[1])
aca_mtd()
