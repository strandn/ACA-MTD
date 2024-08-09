using Random
using Distributions
using KernelDensity
using ForwardDiff
using Interpolations

include("tt_sketch.jl")

x(z) = z[1]
y(z) = z[3]

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

function Vbias(s, rho, basis, domain)
	if rho == []
		return 0.0
	end
	return dens_eval(rho, basis, s, domain)
end

Vbias_shifted(s, rho, basis, domain, Vshift) = max(Vbias(s, rho, basis, domain) - Vshift, 0.0)

function Vtop(rho, basis, domain, samples)
	max = 0.0
	for s in samples
		result = Vbias(s, rho, basis, domain)
		if result > max
			max = result
		end
	end
	return max
end

function get_conv(domain, basis_type, nbasis, nbins)
	basis_original, _ = get_basis(domain, basis_type, nbasis)
	order = length(basis_original)
	ranges = [LinRange(d[1], d[2], nbins) for d in domain]
	basis = []
	basisd = []
	for i in 1:order
		gridpoints = [[0.0 for _ in 1:nbins] for _ in 1:nbasis]
		gridpoints_d = [[0.0 for _ in 1:nbins] for _ in 1:nbasis]
		for j in 1:nbasis
			for k in 1:nbins
				w = 0.02
				sigma = w * (domain[i][2] - domain[i][1])
				s = domain[i][1] + (k - 1) * (domain[i][2] - domain[i][1]) / (nbins - 1)
				f(x) = basis_original[i](x, j) * (1 / (sqrt(2 * pi) * sigma)) * exp(-(s - x) ^ 2 / (2 * sigma ^ 2))
				df(x) = basis_original[i](x, j) * ((x - s) / (sqrt(2 * pi) * sigma ^ 3)) * exp(-(s - x) ^ 2 / (2 * sigma ^ 2))
				L = (domain[i][2] - domain[i][1]) / 2
				gridpoints[j][k] = quadgk(f, domain[i][1] - L, domain[i][2] + L, atol = 1.0e-10, rtol = 1.0e-6)[1]
				gridpoints_d[j][k] = quadgk(df, domain[i][1] - L, domain[i][2] + L, atol = 1.0e-10, rtol = 1.0e-6)[1]
			end
		end
		vec = [
			linear_interpolation(ranges[i], [gridpoints[j][k] for k in 1:nbins])
			for j in 1:nbasis
		]
		conv(x, pos) = vec[pos](x)
		push!(basis, conv)
		vec_d = [
			linear_interpolation(ranges[i], [gridpoints_d[j][k] for k in 1:nbins])
			for j in 1:nbasis
		]
		conv_d(x, pos) = vec_d[pos](x)
		push!(basisd, conv_d)
	end
	return basis, basisd
end

function dVbias(s, rho, basis, basisd, domain, Vshift)
	grad = zeros(length(s))
	if Vbias(s, rho, basis, domain) <= Vshift
		return grad
	end
	return dens_grad(rho, basis, basisd, s, domain)
end

function grad_V(r, rho, basis, basisd, domain, Vshift)
	grad = ForwardDiff.gradient(V, r)
	if rho == []
		return grad
	end
	dx = ForwardDiff.gradient(x, r)
	dy = ForwardDiff.gradient(y, r)
	dVdx, dVdy = dVbias([x(r), y(r)], rho, basis, basisd, domain, Vshift)
	dVdx1 = dVdx * dx[1] + dVdy * dy[1]
	dVdx2 = dVdx * dx[2] + dVdy * dy[2]
	dVdx3 = dVdx * dx[3] + dVdy * dy[3]
	dVdx4 = dVdx * dx[4] + dVdy * dy[4]
	grad += [dVdx1, dVdx2, dVdx3, dVdx4]
	return grad
end

function gradtop(rho, basis, basisd, domain, samples, Vshift)
	dim = length(samples[1])
	max = zeros(dim)
	for r in samples
		result = dVbias(r, rho, basis, basisd, domain, Vshift)
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
	domain_cv = [(-2.0, 2.0), (-2.0, 2.0)]
	nbins = 100
	basis_type = "fourier"
	convbins = 1000
	# basis, basisd = get_conv(domain_cv, basis_type, nbasis, convbins)
	basis, basisd = get_basis(domain_cv, basis_type, nbasis)

	T = 1.0
	gamma = 1.0
	dt = 1.0e-4
	steps = nsamples * 1000000
	stride = 100
	nbiasupdates = 20

	x1 = rand(Normal(-1.0, 0.1))
	x2 = rand(Normal(-1.0, 0.1))
	x3 = rand(Normal(-1.0, 0.1))
	x4 = rand(Normal(1.0, 0.1))
	v1 = v2 = v3 = v4 = 0.0

	kb = 1.0
	sigma = sqrt(2 * kb * T / (gamma * dt))
	normal_dist = Normal(0.0, sigma)

	rho = []
	Vinc = 5 * kb * T
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
			grad = grad_V([x1, x2, x3, x4], rho, basis, basisd, domain_cv, Vshift)

			v1 = -(grad[1] / gamma) + rand(normal_dist)
			v2 = -(grad[2] / gamma) + rand(normal_dist)
			v3 = -(grad[3] / gamma) + rand(normal_dist)
			v4 = -(grad[4] / gamma) + rand(normal_dist)

			x1 += v1 * dt
			x2 += v2 * dt
			x3 += v3 * dt
			x4 += v4 * dt
			
			# x1 = clamp(x1, domain[1][1], domain[1][2])
			# x2 = clamp(x2, domain[2][1], domain[2][2])
			# x3 = clamp(x3, domain[3][1], domain[3][2])
			# x4 = clamp(x4, domain[4][1], domain[4][2])

			t += dt

			if i % stride == 0
				s = [x([x1, x2, x3, x4]), y([x1, x2, x3, x4])]
				Vbiass = Vbias_shifted(s, rho, basis, domain_cv, Vshift)
				push!(traj, [t, s[1], s[2], Vbiass])
				push!(samples, s)
				push!(weights, exp(Vbiass / (kb * T)))
			end
		end
		open("data/colvar_$(count)_$(rc)_$(nbasis)_$(nsamples).out", "w") do file
			for step in traj
				write(file, "$(step[1]) $(step[2]) $(step[3]) $(step[4])\n")
			end
		end

		xlist = [step[2] for step in traj]
		ylist = [step[3] for step in traj]
		println("$(minimum(xlist)) $(maximum(xlist)) $(minimum(ylist)) $(maximum(ylist))")
        println("Forming TT...")
        flush(stdout)
		G = para_sketch(hcat(xlist, ylist), domain_cv, basis_type, rc, 0.05, nbasis)

		rhomax = maximum([dens_eval(G, basis, [xlist[i], ylist[i]], domain_cv) for i in 1:div(steps, stride)])
		G *= Vinc / rhomax
		rho = count == 1 ? G : update_sketch(rho, G)

		rangex = LinRange(domain_cv[1][1], domain_cv[1][2], nbins)
		rangey = LinRange(domain_cv[2][1], domain_cv[2][2], nbins)
        open("data/ttde_$(count)_$(rc)_$(nbasis)_$(nsamples).txt", "w") do file
            for x in rangex
                for y in rangey
                    write(file, "$(dens_eval(G, basis, [x, y], domain_cv)) ")
                end
                write(file, "\n")
            end
        end

		Vpeak = Vtop(rho, basis, domain_cv, samples)
		Vshift = max(Vpeak - Vmax, 0.0)
		println()
		println("Vtop = $Vpeak Vshift = $Vshift")
		println()
		flush(stdout)

		gradpeak = gradtop(rho, basis, basisd, domain_cv, samples, Vshift)
		println("maxgrad = $gradpeak")
		println()
		flush(stdout)

		open("data/F_$(count)_$(rc)_$(nbasis)_$(nsamples).txt", "w") do file
			for x in rangex
				for y in rangey
					write(file, "$(-Vbias_shifted([x, y], rho, basis, domain_cv, Vshift)) ")
				end
				write(file, "\n")
			end
		end

		open("data/dVbiasdx_$(count)_$(rc)_$(nbasis)_$(nsamples).txt", "w") do filex
			open("data/dVbiasdy_$(count)_$(rc)_$(nbasis)_$(nsamples).txt", "w") do filey
				for x in rangex
					for y in rangey
						grad = dVbias([x, y], rho, basis, basisd, domain_cv, Vshift)
						write(filex, "$(grad[1]) ")
						write(filey, "$(grad[2]) ")
					end
					write(filex, "\n")
					write(filey, "\n")
				end
			end
		end

		gridbins = 1000
		ranges = [LinRange(d[1], d[2], gridbins) for d in domain]

		bw = 0.01 .* (domain[1][2] - domain[1][1])
		kde_result = kde([step[1] for step in samples], npoints = gridbins, weights = weights / sum(weights), bandwidth = bw)
		ik = InterpKDE(kde_result)
		open("data/kde_1_$(count)_$(rc)_$(nbasis)_$(nsamples).txt", "w") do file	
			for x in ranges[1]
				write(file, "$(pdf(ik, x)) ")
			end
			write(file, "\n")
		end

		bw = 0.01 .* (domain[3][2] - domain[3][1])
		kde_result = kde([step[2] for step in samples], npoints = gridbins, weights = weights / sum(weights), bandwidth = bw)
		ik = InterpKDE(kde_result)
		open("data/kde_3_$(count)_$(rc)_$(nbasis)_$(nsamples).txt", "w") do file	
			for x in ranges[3]
				write(file, "$(pdf(ik, x)) ")
			end
			write(file, "\n")
		end

		gridbins = 100
		ranges = [LinRange(d[1], d[2], gridbins) for d in domain]

		bw = 0.02 .* (domain[1][2] - domain[1][1], domain[3][2] - domain[3][1])
		kde_result = kde(hcat([step[1] for step in samples], [step[2] for step in samples]), npoints = (gridbins, gridbins), weights = weights / sum(weights), bandwidth = bw)
		ik = InterpKDE(kde_result)
		open("data/kde_13_$(count)_$(rc)_$(nbasis)_$(nsamples).txt", "w") do file
			for x in ranges[1]
				for y in ranges[3]
					write(file, "$(pdf(ik, x, y)) ")
				end
				write(file, "\n")
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
