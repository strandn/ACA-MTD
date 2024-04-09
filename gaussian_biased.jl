using KernelDensity
using MPI
using Random
using Distributions
using ForwardDiff
using LinearAlgebra
using Interpolations

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

# function fes(F, s)
# 	return compute_func(F, s)
# end

# function Fmin(F, samples)
# 	min = Inf
# 	for s in samples
# 		result = fes(F, s)
# 		if result < min
# 			min = result
# 		end
# 	end
# 	return min
# end

# dens_adj(F, s) = 1.0e2 * max(dens(F, s), 1.0e-2)

# function Vbias(s, rholist, kT)
# 	result = 0.0
# 	for F in rholist
# 		result += kT * log(dens_adj(F, s))
# 	end
# 	return result
# end

function Vbias(s, rholist)
	result = 0.0
	for F in rholist
		result -= compute_func(F, s)
	end
	return result
end

Vbias_shifted(s, rholist, Vshift) = max(Vbias(s, rholist) - Vshift, 0.0)

function Vtop(rholist, samples)
	max = 0.0
	for s in samples
		result = Vbias(s, rholist)
		if result > max
			max = result
		end
	end
	return max
end

# function dVbias(s, rholist, Vshift, h)
# 	grad = fill(0.0, length(s))
# 	if Vbias(s, rholist) <= Vshift
# 		return grad
# 	end
# 	for F in rholist
# 		order = F.ndims
# 		npivots = [length(F.I[i]) for i in 2:order]
# 		outer = []
# 		inner = []
# 		mat = zeros(1, npivots[1])
# 		for j in 1:npivots[1]
# 			mat[j] = F.f((s[1], F.J[2][j]...)...)
# 		end
# 		push!(outer, mat)
# 		AIJ = zeros(npivots[1], npivots[1])
# 		for j in 1:npivots[1]
# 			for k in 1:npivots[1]
# 				AIJ[j, k] = F.f((F.I[2][j]..., F.J[2][k]...)...)
# 			end
# 		end
# 		push!(inner, inv(AIJ))
# 		for i in 2:order-1
# 			mat = zeros(npivots[i - 1], npivots[i])
# 			for j in 1:npivots[i - 1]
# 				for k in 1:npivots[i]
# 					mat[j, k] = F.f((F.I[i][j]..., s[i], F.J[i + 1][k]...)...)
# 				end
# 			end
# 			push!(outer, mat)
# 			AIJ = zeros(npivots[i], npivots[i])
# 			for j in 1:npivots[i]
# 				for k in 1:npivots[i]
# 					AIJ[j, k] = F.f((F.I[i + 1][j]..., F.J[i + 1][k]...)...)
# 				end
# 			end
# 			push!(inner, inv(AIJ))
# 		end
# 		mat = zeros(npivots[order - 1], 1)
# 		for j in 1:npivots[order - 1]
# 			mat[j] = F.f((F.I[order][j]..., s[order])...)
# 		end
# 		push!(outer, mat)
# 		douter = []
# 		mat = zeros(1, npivots[1])
# 		for j in 1:npivots[1]
# 			mat[j] = (F.f((s[1] + h[1], F.J[2][j]...)...) - F.f((s[1] - h[1], F.J[2][j]...)...)) / (2 * h[1])
# 		end
# 		push!(douter, mat)
# 		for i in 2:order-1
# 			mat = zeros(npivots[i - 1], npivots[i])
# 			for j in 1:npivots[i - 1]
# 				for k in 1:npivots[i]
# 					mat[j, k] = (F.f((F.I[i][j]..., s[i] + h[i], F.J[i + 1][k]...)...) - F.f((F.I[i][j]..., s[i] - h[i], F.J[i + 1][k]...)...)) / (2 * h[i])
# 				end
# 			end
# 			push!(douter, mat)
# 		end
# 		mat = zeros(npivots[order - 1], 1)
# 		for j in 1:npivots[order - 1]
# 			mat[j] = (F.f((F.I[order][j]..., s[order] + h[order])...) - F.f((F.I[order][j]..., s[order] - h[order])...))/ (2 * h[order])
# 		end
# 		push!(douter, mat)
# 		inc = fill(outer[1], order - 1)
# 		pushfirst!(inc, douter[1])
# 		for i in 1:order
# 			for j in 2:order
# 				inc[i] *= inner[j - 1] * (i == j ? douter[j] : outer[j])
# 			end
# 		end
# 		grad -= [inci[1, 1] for inci in inc]
# 	end
# 	return grad
# end

function dVbias_mats(F_Vbias, rholist, Vshift, domain, nbins)
	# TODO replace finite difference derivatives with analytical derivatives?
	order = F_Vbias.ndims
	npivots = [length(F_Vbias.I[i]) for i in 2:order]
	h = [(d[2] - d[1]) / (nbins - 1) for d in domain]
	outer = []
	# mat = zeros(1, npivots[1])
	# for j in 1:npivots[1]
		# mat[j] = F.f((s[1], F.J[2][j]...)...)
	# end
	range = domain[1][1]:(domain[1][2]-domain[1][1])/(nbins-1):domain[1][2]
	mat = [
		linear_interpolation(range, [F_Vbias.f((s, F_Vbias.J[2][j]...)...) for s in range])
		for j in 1:npivots[1]
	]
	push!(outer, mat)
	for i in 2:order-1
		# mat = zeros(npivots[i - 1], npivots[i])
		# for j in 1:npivots[i - 1]
		# 	for k in 1:npivots[i]
				# mat[j, k] = F.f((F.I[i][j]..., s[i], F.J[i + 1][k]...)...)
		# 	end
		# end
		range = domain[i][1]:(domain[i][2]-domain[i][1])/(nbins-1):domain[i][2]
		mat = [
			[
				linear_interpolation(range, [F_Vbias.f((F_Vbias.I[i][j]..., s, F_Vbias.J[i + 1][k]...)...) for s in range])
				for k in 1:npivots[i]
			]
			for j in 1:npivots[i - 1]
		]
		push!(outer, mat)
	end
	# mat = zeros(npivots[order - 1], 1)
	# for j in 1:npivots[order - 1]
		# mat[j] = F.f((F.I[order][j]..., s[order])...)
	# end
	range = domain[order][1]:(domain[order][2]-domain[order][1])/(nbins-1):domain[order][2]
	mat = [
		linear_interpolation(range, [F_Vbias.f((F_Vbias.I[order][j]..., s)...) for s in range])
		for j in 1:npivots[order - 1]
	]
	push!(outer, mat)
	douter = []
	# mat = zeros(1, npivots[1])
	# for j in 1:npivots[1]
		# mat[j] = (F.f((s[1] + h[1], F.J[2][j]...)...) - F.f((s[1] - h[1], F.J[2][j]...)...)) / (2 * h[1])
	# end
	# range = domain[1][1]:(domain[1][2]-domain[1][1])/(nbins - 1):domain[1][2]
	gridpoints = [[0.0 for _ in 1:nbins] for _ in 1:npivots[1]]
	for j in 1:npivots[1]
		for pos in 1:nbins
			s = domain[1][1] + (pos - 1) * (domain[1][2] - domain[1][1]) / (nbins - 1)
			if Vbias([s, F_Vbias.J[2][j]...], rholist) > Vshift
				for F in rholist
					gridpoints[j][pos] += (F.f((s + h[1], F.J[2][j]...)...) - F.f((s - h[1], F.J[2][j]...)...)) / (2 * h[1])
				end
			end
		end
	end
	range = domain[1][1]:(domain[1][2]-domain[1][1])/(nbins-1):domain[1][2]
	mat = [
		linear_interpolation(range, [gridpoints[j][pos] for pos in 1:nbins])
		for j in 1:npivots[1]
	]
	push!(douter, mat)
	for i in 2:order-1
		# mat = zeros(npivots[i - 1], npivots[i])
		# for j in 1:npivots[i - 1]
		# 	for k in 1:npivots[i]
				# mat[j, k] = (F.f((F.I[i][j]..., s[i] + h[i], F.J[i + 1][k]...)...) - F.f((F.I[i][j]..., s[i] - h[i], F.J[i + 1][k]...)...)) / (2 * h[i])
		# 	end
		# end
		gridpoints = [[[0.0 for _ in 1:nbins] for _ in 1:npivots[i]] for _ in 1:npivots[i - 1]]
		for j in 1:npivots[i - 1]
			for k in 1:npivots[i]
				for pos in 1:nbins
					s = domain[i][1] + (pos - 1) * (domain[i][2] - domain[i][1]) / (nbins - 1)
					if Vbias([F_Vbias.I[i][j]..., s, F_Vbias.J[i + 1][k]...], rholist) > Vshift
						for F in rholist
							gridpoints[j][k][pos] += (F.f((F.I[i][j]..., s + h[i], F.J[i + 1][k]...)...) - F.f((F.I[i][j]..., s - h[i], F.J[i + 1][k]...)...)) / (2 * h[i])
						end
					end
				end
			end
		end
		range = domain[i][1]:(domain[i][2]-domain[i][1])/(nbins-1):domain[i][2]
		mat = [
			[
				linear_interpolation(range, [gridpoints[j][k][pos] for pos in 1:nbins])
				for k in 1:npivots[i]
			]
			for j in 1:npivots[i - 1]
		]
		push!(douter, mat)
	end
	# mat = zeros(npivots[order - 1], 1)
	# for j in 1:npivots[order - 1]
		# mat[j] = (F.f((F.I[order][j]..., s[order] + h[order])...) - F.f((F.I[order][j]..., s[order] - h[order])...))/ (2 * h[order])
	# end
	gridpoints = [[0.0 for _ in 1:nbins] for _ in 1:npivots[order - 1]]
	for j in 1:npivots[order - 1]
		for pos in 1:nbins
			s = domain[order][1] + (pos - 1) * (domain[order][2] - domain[order][1]) / (nbins - 1)
			if Vbias([F_Vbias.I[order][j]..., s], rholist) > Vshift
				for F in rholist
					gridpoints[j][pos] += (F.f((F.I[order][j]..., s + h[order])...) - F.f((F.I[order][j]..., s - h[order])...))/ (2 * h[order])
				end
			end
		end
	end
	range = domain[order][1]:(domain[order][2]-domain[order][1])/(nbins-1):domain[order][2]
	mat = [
		linear_interpolation(range, [gridpoints[j][pos] for pos in 1:nbins])
		for j in 1:npivots[order - 1]
	]
	push!(douter, mat)
	return outer, douter
end

function dVbias(s, F_Vbias, outer, douter)
	order = F_Vbias.ndims
	npivots = [length(F_Vbias.I[i]) for i in 2:order]
	outermat = []
	inner = []
	mat = zeros(1, npivots[1])
	for j in 1:npivots[1]
		mat[j] = outer[1][j](s[1])
	end
	push!(outermat, mat)
	AIJ = zeros(npivots[1], npivots[1])
	for j in 1:npivots[1]
		for k in 1:npivots[1]
			AIJ[j, k] = F_Vbias.f((F_Vbias.I[2][j]..., F_Vbias.J[2][k]...)...)
		end
	end
	push!(inner, inv(AIJ))
	for i in 2:order-1
		mat = zeros(npivots[i - 1], npivots[i])
		for j in 1:npivots[i - 1]
			for k in 1:npivots[i]
				mat[j, k] = outer[i][j][k](s[i])
			end
		end
		push!(outermat, mat)
		AIJ = zeros(npivots[i], npivots[i])
		for j in 1:npivots[i]
			for k in 1:npivots[i]
				AIJ[j, k] = F_Vbias.f((F_Vbias.I[i + 1][j]..., F_Vbias.J[i + 1][k]...)...)
			end
		end
		push!(inner, inv(AIJ))
	end
	mat = zeros(npivots[order - 1], 1)
	for j in 1:npivots[order - 1]
		mat[j] = outer[order][j](s[order])
	end
	push!(outermat, mat)
	doutermat = []
	mat = zeros(1, npivots[1])
	for j in 1:npivots[1]
		mat[j] = douter[1][j](s[1])
	end
	push!(doutermat, mat)
	for i in 2:order-1
		mat = zeros(npivots[i - 1], npivots[i])
		for j in 1:npivots[i - 1]
			for k in 1:npivots[i]
				mat[j, k] = douter[i][j][k](s[i])
			end
		end
		push!(doutermat, mat)
	end
	mat = zeros(npivots[order - 1], 1)
	for j in 1:npivots[order - 1]
		mat[j] = douter[order][j](s[order])
	end
	push!(doutermat, mat)
	inc = fill(outermat[1], order - 1)
	pushfirst!(inc, doutermat[1])
	for i in 1:order
		for j in 2:order
			inc[i] *= inner[j - 1] * (i == j ? doutermat[j] : outermat[j])
		end
	end
	return [inci[1, 1] for inci in inc]
end

function grad_V(r, F_Vbias, outer, douter)
	grad = ForwardDiff.gradient(V, r)
	if isempty(outer)
		return grad
	end
	dx = ForwardDiff.gradient(x, r)
	dy = ForwardDiff.gradient(y, r)
	dVdx, dVdy = dVbias([x(r), y(r)], F_Vbias, outer, douter)
	dVdx1 = dVdx * dx[1] + dVdy * dy[1]
	dVdx2 = dVdx * dx[2] + dVdy * dy[2]
	dVdx3 = dVdx * dx[3] + dVdy * dy[3]
	dVdx4 = dVdx * dx[4] + dVdy * dy[4]
	grad += [dVdx1, dVdx2, dVdx3, dVdx4]
	return grad
end

function aca_mtd()
	domain = ((-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0))
	domain_cv = ((-1.5, 4.0), (-1.5, 4.5))
	nbins = 100

	T = 1.0
	gamma = 1.0
	dt = 1.0e-4
	# steps = 1e7
	# stride = 100
	steps = 10000
	stride = 10
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
	# Vmax = 20 * kb * T
	Vmax = Inf
	Vinc = 4.6 * kb * T
	samples = []
	Vshift = 0.0
	# h = (0.0, 0.0)
	F_Vbias = ResFunc(0, domain_cv, 0.0)
	outer = []
	douter = []

	for count in 1:nbiasupdates
		println("Vbias update $count...")
		flush(stdout)
		t = 0.0

		traj = []
		for i in 1:steps
			grad = grad_V([x1, x2, x3, x4], F_Vbias, outer, douter)

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
				# push!(traj, [t, x([x1, x2, x3, x4]), y([x1, x2, x3, x4]), Vbias_shifted([x1, x2, x3, x4], rholist, Vshift)])
				s = [x([x1, x2, x3, x4]), y([x1, x2, x3, x4])]
				push!(traj, [t, s[1], s[2], compute_func(F_Vbias, s)])
				push!(samples, s)
			end
		end
		open("data/colvar_$count.txt", "w") do file
			for step in traj
				write(file, "$(step[1]) $(step[2]) $(step[3]) $(step[4])\n")
			end
		end

		xlist = [step[2] for step in traj]
		ylist = [step[3] for step in traj]
		println("$(minimum(xlist)) $(maximum(xlist)) $(minimum(ylist)) $(maximum(ylist))")
		flush(stdout)
		kde_result = kde(hcat(xlist, ylist), npoints = (nbins, nbins))
		# kde_result = kde(hcat(xlist, ylist), npoints = (nbins, nbins), bandwidth = (0.05, 0.05))
		println("$(kde_result.x) $(kde_result.y)")
		println()
		flush(stdout)
		
		ik = InterpKDE(kde_result)
		rhohat(x, y) = pdf(ik, x, y)
		# fhat(x, y) = -kb * T * log(1.0e2 * max(rhohat(x, y), 1.0e-2))
		fhat(x, y) = -kb * T * log(abs(rhohat(x, y)))
		fmin = minimum([fhat(step[2], step[3]) for step in traj])
		# fhat_adj(x, y) = max(-(fhat(x, y) - fmin) + Vinc, 0)
		fhat_adj(x, y) = min((fhat(x, y) - fmin) - Vinc, 0)
		open("data/kde_$count.txt", "w") do file
			write(file, "$(first(kde_result.x)) $(last(kde_result.x)) $(step(kde_result.x))\n")
			write(file, "$(first(kde_result.y)) $(last(kde_result.y)) $(step(kde_result.y))\n")
			for x in kde_result.x
				for y in kde_result.y
					write(file, "$(rhohat(x, y)) ")
				end
				write(file, "\n")
			end
		end
		
		# n_chains = 100
		# n_samples = 1000
		n_chains = 10
		n_samples = 100
		jump_width = 0.01
		# rank = 50
		rank = 2
		domain_cv_small = ((first(kde_result.x), last(kde_result.x)), (first(kde_result.y), last(kde_result.y)))
		F = ResFunc(fhat_adj, domain_cv_small, 0.1)
		println("Target rank $rank")
		flush(stdout)
		IJ = continuous_aca(F, [rank], n_chains, n_samples, jump_width, mpi_comm)
		println(IJ)
		println()
		flush(stdout)

		open("data/dF_$(count).txt", "w") do file
			for x in kde_result.x
				for y in kde_result.y
					# write(file, "$(-kb * T * log(dens_adj(F, [x, y]))) ")
					write(file, "$(compute_func(F, [x, y])) ")
				end
				write(file, "\n")
			end
		end

		push!(rholist, F)
		Vpeak = Vtop(rholist, samples)
		Vshift = max(Vpeak - Vmax, 0.0)
		println("Vtop = $Vpeak Vshift = $Vshift")
		println()
		flush(stdout)

		Vbias_shifted_f(x, y) = Vbias_shifted([x, y], rholist, Vshift)
		F_Vbias = ResFunc(Vbias_shifted_f, domain_cv, 10^-3)
		println("Target rank $rank (full Vbias)")
		flush(stdout)
		IJ = continuous_aca(F_Vbias, [rank], n_chains, n_samples, jump_width, mpi_comm)
		println(IJ)
		println()
		flush(stdout)

		rangex = domain_cv[1][1]:(domain_cv[1][2]-domain_cv[1][1])/(nbins-1):domain_cv[1][2]
		rangey = domain_cv[2][1]:(domain_cv[2][2]-domain_cv[2][1])/(nbins-1):domain_cv[2][2]
		open("data/F_$count.txt", "w") do file
			for x in rangex
				for y in rangey
					# write(file, "$(-Vbias_shifted([x, y], rholist, Vshift)) ")
					write(file, "$(compute_func(F_Vbias, [x, y])) ")
				end
				write(file, "\n")
			end
		end
		# h = (step(kde_result.x), step(kde_result.y))
		outer, douter = dVbias_mats(F_Vbias, rholist, Vshift, domain_cv, nbins)

		# for x in rangex
		# 	for y in rangey
		# 		grad = dVbias([x, y], F_Vbias, outer, douter)
		# 		print("$(grad[1]) ")
		# 		print("$(grad[2]) ")
		# 	end
		# 	println()
		# 	println()
		# end

		open("data/dVbiasdx_$count.txt", "w") do filex
			open("data/dVbiasdy_$count.txt", "w") do filey
				for x in rangex
					for y in rangey
						grad = dVbias([x, y], F_Vbias, outer, douter)
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
