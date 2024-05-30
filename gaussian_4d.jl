using Random
using Distributions
using KernelDensity
using ForwardDiff
using Interpolations

include("tt_sketch.jl")

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

function fes(rho, rhomax, kT)
	rho_adj = max(rho * 100 / rhomax, 1)
	return -kT * log(rho_adj)
end

function Vbias(r, rholist, rhomaxlist, basislist, kT)
	result = 0.0
	for i in eachindex(rholist)
        rho = dens_eval(rholist[i], basislist[i], r)
        result -= fes(rho, rhomaxlist[i], kT)
	end
	return result
end

Vbias_shifted(r, rholist, rhomaxlist, basislist, kT, Vshift) = max(Vbias(r, rholist, rhomaxlist, basislist, kT) - Vshift, 0.0)

function Vtop(rholist, rhomaxlist, basislist, kT, samples)
	max = 0.0
	for r in samples
		result = Vbias(r, rholist, rhomaxlist, basislist, kT)
		if result > max
			max = result
		end
	end
	return max
end

function update_conv(basis, basislist, basisdlist, domain, nbins, nbasis)
	order = length(basis)
	ranges = [LinRange(d[1], d[2], nbins) for d in domain]
	push!(basislist, [])
	push!(basisdlist, [])
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
				gridpoints[j][k] = quadgk(f, domain[i]...)[1]
				gridpoints_d[j][k] = quadgk(df, domain[i]...)[1]
			end
		end
		vec = [
			linear_interpolation(ranges[i], [gridpoints[j][k] for k in 1:nbins])
			for j in 1:nbasis
		]
		conv(x, pos) = vec[pos](x)
		push!(last(basislist), conv)
		vec_d = [
			linear_interpolation(ranges[i], [gridpoints_d[j][k] for k in 1:nbins])
			for j in 1:nbasis
		]
		conv_d(x, pos) = vec_d[pos](x)
		push!(last(basisdlist), conv_d)
	end
end

function dVbias(r, rholist, rhomaxlist, basislist, basisdlist, kT, Vshift)
	grad = zeros(length(r))
	if Vbias(r, rholist, rhomaxlist, basislist, kT) <= Vshift
		return grad
	end
	for i in eachindex(rholist)
        rho = dens_eval(rholist[i], basislist[i], r)
        if rho * 100 / rhomaxlist[i] > 1
            grad += dens_grad(rholist[i], basislist[i], basisdlist[i], r) * kT / rho
        end
	end
	return grad
end

function grad_V(r, rholist, rhomaxlist, basislist, basisdlist, kT, Vshift)
	grad = ForwardDiff.gradient(V, r)
	if isempty(rholist)
		return grad
	end
	grad += dVbias(r, rholist, rhomaxlist, basislist, basisdlist, kT, Vshift)
	return grad
end

function sketch_mtd()
	domain = [(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
	nbins = 100

	T = 1.0
	gamma = 1.0
	dt = 1.0e-4
	steps = nsamples * 1e6
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

	rholist = []
    rhomaxlist = []
	basislist = []
    basisdlist = []
	Vmax = 30 * kb * T
	samples = []
	Vshift = 0.0

	for count in 1:nbiasupdates
		println("Vbias update $count...")
		flush(stdout)
		t = 0.0

		traj = []
		weights = []
		for i in 1:steps
			grad = grad_V([x1, x2, x3, x4], rholist, rhomaxlist, basislist, basisdlist, kb * T, Vshift)

			v1 = -(grad[1] / gamma) + rand(normal_dist)
			v2 = -(grad[2] / gamma) + rand(normal_dist)
			v3 = -(grad[3] / gamma) + rand(normal_dist)
			v4 = -(grad[4] / gamma) + rand(normal_dist)

			old = [x1, x2, x3, x4]

			x1 += v1 * dt
			x2 += v2 * dt
			x3 += v3 * dt
			x4 += v4 * dt

			if isnan(x1) || isnan(x2) || isnan(x3) || isnan(x4)
				println("$old $grad")
				exit(1)
			end
			
			x1 = clamp(x1, domain[1][1], domain[1][2])
			x2 = clamp(x2, domain[2][1], domain[2][2])
			x3 = clamp(x3, domain[3][1], domain[3][2])
			x4 = clamp(x4, domain[4][1], domain[4][2])

			if any(abs.(grad) .> 100)
				println("$t $old $([x1, x2, x3, x4]) $grad")
				flush(stdout)
			end

			t += dt

			if i % stride == 0
				Vbiass = Vbias_shifted([x1, x2, x3, x4], rholist, rhomaxlist, basislist, kb * T, Vshift)
				push!(traj, [t, x1, x2, x3, x4, Vbiass])
				push!(weights, exp(Vbiass / (kb * T)))
				push!(samples, [x1, x2, x3, x4])
			end
		end
		open("data/colvar_$(count)_$(r)_$(rc)_$(nbasis)_$(nsamples).txt", "w") do file
			for step in traj
				write(file, "$(step[1]) $(step[2]) $(step[3]) $(step[4]) $(step[5]) $(step[6])\n")
			end
		end

		# x1list = [step[2] for step in traj]
		# x2list = [step[3] for step in traj]
		# x3list = [step[4] for step in traj]
		# x4list = [step[5] for step in traj]
		xlist = [[step[i] for step in traj] for i in 2:5]
		domain_small = [(minimum(xlist), maximum(xlist)) for xlist in xlist]
		println(domain_small)
        println("Forming TT...")
        flush(stdout)
		G, basis, _ = para_sketch(hcat(xlist[1], xlist[2], xlist[3], xlist[4]), domain_small, "fourier", r, rc, 0.2, nbasis, ones(Int64(div(steps, stride))))

		push!(rholist, G)
		update_conv(basis, basislist, basisdlist, domain, nbins, nbasis)
		push!(rhomaxlist, maximum([dens_eval(G, last(basislist), [xlist[1][i], xlist[2][i], xlist[3][i], xlist[4][i]]) for i in 1:Int64(div(steps, stride))]))

		Vpeak = Vtop(rholist, rhomaxlist, basislist, kb * T, samples)
		Vshift = max(Vpeak - Vmax, 0.0)
		println("Vtop = $Vpeak Vshift = $Vshift")
		println()
		flush(stdout)

		nbins = 1000
		ranges = [LinRange(d[1], d[2], nbins) for d in domain]
		for i in 1:4
			bw = 0.01 .* (domain[i][2] - domain[i][1])
			kde_result = kde(xlist[i], npoints = nbins, weights = weights / sum(weights), bandwidth = bw)
			ik = InterpKDE(kde_result)
			open("data/fes_$(i)_$(count).txt", "w") do file
				for x in ranges[i]
					write(file, "$(pdf(ik, x)) ")
				end
				write(file, "\n")
			end
		end

		nbins = 100
		ranges = [LinRange(d[1], d[2], nbins) for d in domain]
		for i in 1:4
			for j in i:4
				bw = 0.02 .* (domain[i][2] - domain[i][1], domain[j][2] - domain[j][1])
				kde_result = kde(hcat(xlist[i], xlist[j]), npoints = (nbins, nbins), weights = weights / sum(weights), bandwidth = bw)
				ik = InterpKDE(kde_result)
				open("data/fes_$(i)$(j)_$(count).txt", "w") do file
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
r = parse(Int64, ARGS[1])
rc = parse(Int64, ARGS[2])
nbasis = parse(Int64, ARGS[3])
nsamples = parse(Int64, ARGS[4])
sketch_mtd()
