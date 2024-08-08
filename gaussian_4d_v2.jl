using Random
using Distributions
using KernelDensity
using ForwardDiff
using Interpolations

include("tt_sketch_v2.jl")

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

function Vbias(r, rho, basis)
	if rho == []
		return 0.0
	end
	return dens_eval(rho, basis, r)
end

Vbias_shifted(r, rho, basis, Vshift) = max(Vbias(r, rho, basis) - Vshift, 0.0)

function Vtop(rho, basis, samples)
	max = 0.0
	for r in samples
		result = Vbias(r, rho, basis)
		if result > max
			max = result
		end
	end
	return max
end

function convolution(basis, domain, nbins, nbasis)
	order = length(basis)
	ranges = [LinRange(d[1], d[2], nbins) for d in domain]
	newbasis = []
	newbasisd = []
	for i in 1:order
		gridpoints = [[0.0 for _ in 1:nbins] for _ in 1:nbasis]
		gridpoints_d = [[0.0 for _ in 1:nbins] for _ in 1:nbasis]
		for j in 1:nbasis
			for k in 1:nbins
				w = 0.02
				sigma = w * (domain[i][2] - domain[i][1])
				s = domain[i][1] + (k - 1) * (domain[i][2] - domain[i][1]) / (nbins - 1)
				f(x) = basis[i](x, j) * (1 / (sqrt(2 * pi) * sigma)) * exp(-(s - x) ^ 2 / (2 * sigma ^ 2))
				df(x) = basis[i](x, j) * ((x - s) / (sqrt(2 * pi) * sigma ^ 3)) * exp(-(s - x) ^ 2 / (2 * sigma ^ 2))
				gridpoints[j][k] = quadgk(f, domain[i]..., atol = 1.0e-10, rtol = 1.0e-6)[1]
				gridpoints_d[j][k] = quadgk(df, domain[i]..., atol = 1.0e-10, rtol = 1.0e-6)[1]
			end
		end
		vec = [
			linear_interpolation(ranges[i], [gridpoints[j][k] for k in 1:nbins])
			for j in 1:nbasis
		]
		conv(x, pos) = vec[pos](x)
		push!(newbasis, conv)
		vec_d = [
			linear_interpolation(ranges[i], [gridpoints_d[j][k] for k in 1:nbins])
			for j in 1:nbasis
		]
		conv_d(x, pos) = vec_d[pos](x)
		push!(newbasisd, conv_d)
	end
	return newbasis, newbasisd
end

function dVbias(r, rho, basis, basisd, Vshift)
	grad = zeros(length(r))
	if Vbias(r, rho, basis) <= Vshift
		return grad
	end
	return dens_grad(rho, basis, basisd, r)
end

function grad_V(r, rho, basis, basisd, Vshift)
	grad = ForwardDiff.gradient(V, r)
	if rho == []
		return grad
	end
	grad += dVbias(r, rho, basis, basisd, Vshift)
	return grad
end

function gradtop(rho, basis, basisd, samples, Vshift)
	dim = length(samples[1])
	max = zeros(dim)
	for r in samples
		result = dVbias(r, rho, basis, basisd, Vshift)
		for i in 1:dim
			if result[i] > abs(max[i])
				max[i] = abs(result[i])
			end
		end
	end
	return max
end

function sketch_mtd()
	domain = [(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
	nbins = 1000

	T = 1.0
	gamma = 1.0
	dt = 1.0e-4
	steps = nsamples * 1000000
	stride = 100
	nbiasupdates = 15

	x1 = rand(Normal(-1.0, 0.1))
	x2 = rand(Normal(-1.0, 0.1))
	x3 = rand(Normal(-1.0, 0.1))
	x4 = rand(Normal(1.0, 0.1))
	v1 = v2 = v3 = v4 = 0.0

	kb = 1.0
	sigma = sqrt(2 * kb * T / (gamma * dt))
	normal_dist = Normal(0.0, sigma)

	rho = []
	convbasis = []
	convbasisd = []
	Vinc = 4.6 * kb * T
	Vmax = 20 * kb * T
	samples = []
	weights = []
	Vshift = 0.0

	for count in 1:nbiasupdates
		println("Vbias update $count...")
		flush(stdout)
		t = 0.0

		traj = []
		for i in 1:steps
			grad = grad_V([x1, x2, x3, x4], rho, convbasis, convbasisd, Vshift)

			v1 = -(grad[1] / gamma) + rand(normal_dist)
			v2 = -(grad[2] / gamma) + rand(normal_dist)
			v3 = -(grad[3] / gamma) + rand(normal_dist)
			v4 = -(grad[4] / gamma) + rand(normal_dist)

			x1 += v1 * dt
			x2 += v2 * dt
			x3 += v3 * dt
			x4 += v4 * dt
			
			x1 = clamp(x1, domain[1][1], domain[1][2])
			x2 = clamp(x2, domain[2][1], domain[2][2])
			x3 = clamp(x3, domain[3][1], domain[3][2])
			x4 = clamp(x4, domain[4][1], domain[4][2])

			t += dt

			if i % stride == 0
				Vbiass = Vbias_shifted([x1, x2, x3, x4], rho, convbasis, Vshift)
				push!(traj, [t, x1, x2, x3, x4, Vbiass])
				push!(samples, [x1, x2, x3, x4])
				push!(weights, exp(Vbiass / (kb * T)))
			end
		end
		open("data/colvar_$(count)_$(rc)_$(nbasis)_$(nsamples).out", "w") do file
			for step in traj
				write(file, "$(step[1]) $(step[2]) $(step[3]) $(step[4]) $(step[5]) $(step[6])\n")
			end
		end

		xlist = [[step[i] for step in traj] for i in 2:5]
		domain_small = [(minimum(xlist), maximum(xlist)) for xlist in xlist]
		println(domain_small)
        println("Forming TT...")
        flush(stdout)
		G, basis, _ = para_sketch(hcat(xlist[1], xlist[2], xlist[3], xlist[4]), domain, "fourier", rc, 0.05, nbasis)

		if count == 1
			convbasis, convbasisd = convolution(basis, domain, nbins, nbasis)
		end
		rhomax = maximum([dens_eval(G, convbasis, [xlist[1][i], xlist[2][i], xlist[3][i], xlist[4][i]]) for i in 1:div(steps, stride)])
		G *= Vinc / rhomax
		rho = count == 1 ? G : update_sketch(rho, G)

		Vpeak = Vtop(rho, convbasis, samples)
		Vshift = max(Vpeak - Vmax, 0.0)
		println()
		println("Vtop = $Vpeak Vshift = $Vshift")
		println()
		flush(stdout)

		gradpeak = gradtop(rho, convbasis, convbasisd, samples, Vshift)
		println("maxgrad = $gradpeak")
		println()
		flush(stdout)

		gridbins = 1000
		ranges = [LinRange(d[1], d[2], gridbins) for d in domain]
		for i in 1:4
			bw = 0.01 .* (domain[i][2] - domain[i][1])
			kde_result = kde([step[i] for step in samples], npoints = gridbins, weights = weights / sum(weights), bandwidth = bw)
			ik = InterpKDE(kde_result)
			open("data/kde_$(i)_$(count)_$(rc)_$(nbasis)_$(nsamples).txt", "w") do file
				for x in ranges[i]
					write(file, "$(pdf(ik, x)) ")
				end
				write(file, "\n")
			end
		end

		gridbins = 100
		ranges = [LinRange(d[1], d[2], gridbins) for d in domain]
		for i in 1:4
			for j in i+1:4
				bw = 0.02 .* (domain[i][2] - domain[i][1], domain[j][2] - domain[j][1])
				kde_result = kde(hcat([step[i] for step in samples], [step[j] for step in samples]), npoints = (gridbins, gridbins), weights = weights / sum(weights), bandwidth = bw)
				ik = InterpKDE(kde_result)
				open("data/kde_$(i)$(j)_$(count)_$(rc)_$(nbasis)_$(nsamples).txt", "w") do file
					for x in ranges[i]
						for y in ranges[j]
							write(file, "$(pdf(ik, x, y)) ")
						end
						write(file, "\n")
					end
				end
			end
		end
	end
end

println(ARGS)
flush(stdout)
rc = parse(Int64, ARGS[1])
nbasis = parse(Int64, ARGS[2])
nsamples = parse(Int64, ARGS[3])
sketch_mtd()
