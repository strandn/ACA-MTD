using Random
using Distributions
using KernelDensity
using ForwardDiff

include("tt_sketch.jl")

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

function Vbias(s, rho, rhomax, basis, Vmax)
	if rho == undef
		return 0.0
	end
	dens = dens_eval(rho, basis, s)
	return dens * Vmax / rhomax
end

function dVbias(s, rho, rhomax, basis, basisd, Vmax)
	if rho == undef
		return zeros(length(s))
	end
	grad = dens_grad(rho, basis, basisd, s)
	return grad * Vmax / rhomax
end

function grad_V(r, rho, rhomax, basis, basisd, Vmax)
	grad = ForwardDiff.gradient(V, r)
	dx = ForwardDiff.gradient(x, r)
	dy = ForwardDiff.gradient(y, r)
	dVdx, dVdy = dVbias([x(r), y(r)], rho, rhomax, basis, basisd, Vmax)
	dVdx1 = dVdx * dx[1] + dVdy * dy[1]
	dVdx2 = dVdx * dx[2] + dVdy * dy[2]
	dVdx3 = dVdx * dx[3] + dVdy * dy[3]
	dVdx4 = dVdx * dx[4] + dVdy * dy[4]
	grad += [dVdx1, dVdx2, dVdx3, dVdx4]
	return grad
end

function sketch_mtd()
	domain = [(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
	domain_cv = [(-1.5, 4.0), (-1.5, 4.5)]
	nbins = 100

	T = 1.0
	gamma = 1.0
	dt = 1.0e-4
	steps = 1e6
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

	rho = undef
    rhomax = 0.0
    basis = undef
    basisd = undef
	Vmax = 35 * kb * T
	samples = []

	for count in 1:nbiasupdates
		println("Vbias update $count...")
		flush(stdout)
		t = 0.0

		traj = []
		for i in 1:steps
			grad = grad_V([x1, x2, x3, x4], rho, rhomax, basis, basisd, Vmax)

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

			if any(abs.(grad) .> 200)
				println("$t $old $([x0, y0]) $([x1, x2, x3, x4]) $([x([x1, x2, x3, x4]), y([x1, x2, x3, x4])]) $grad")
				flush(stdout)
			end

			t += dt

			if i % stride == 0
				s = [x([x1, x2, x3, x4]), y([x1, x2, x3, x4])]
				push!(traj, [t, s[1], s[2], Vbias(s, rho, rhomax, basis, Vmax)])
				push!(samples, s)
			end
		end
		open("data/colvar_$(count)_$(r)_$(rc)_$(nbasis).txt", "w") do file
			for step in traj
				write(file, "$(step[1]) $(step[2]) $(step[3]) $(step[4])\n")
			end
		end

		domain_cv_small = [(minimum([s[1] for s in samples]), maximum([s[1] for s in samples])), (minimum([s[2] for s in samples]), maximum([s[2] for s in samples]))]
		println("$(domain_cv_small[1][1]) $(domain_cv_small[1][2]) $(domain_cv_small[2][1]) $(domain_cv_small[2][2])")
        println("Forming TT...")
        flush(stdout)
		rho, basis, basisd = para_sketch([s[i] for s in samples, i in 1:length(samples[1])], domain_cv_small, "gaussian", r, rc, 0.05, nbasis)
        
		rhomax = maximum([dens_eval(rho, basis, s) for s in samples])
		
		rangex_small = LinRange(domain_cv_small[1][1], domain_cv_small[1][2], nbins)
		rangey_small = LinRange(domain_cv_small[2][1], domain_cv_small[2][2], nbins)
        open("data/ttde_$(count)_$(r)_$(rc)_$(nbasis).txt", "w") do file
            write(file, "$(first(rangex_small)) $(last(rangex_small)) $(step(rangex_small))\n")
            write(file, "$(first(rangey_small)) $(last(rangey_small)) $(step(rangey_small))\n")
            for x in rangex_small
                for y in rangey_small
                    write(file, "$(dens_eval(rho, basis, [x, y])) ")
                end
                write(file, "\n")
            end
        end

		kde_result = kde([s[i] for s in samples, i in 1:length(samples[1])], npoints = (nbins, nbins))
		ik = InterpKDE(kde_result)
		open("data/kde_$(count)_$(r)_$(rc)_$(nbasis).txt", "w") do file
            for x in rangex_small
                for y in rangey_small
                    write(file, "$(pdf(ik, x, y)) ")
                end
                write(file, "\n")
            end
        end

		rangex = LinRange(domain_cv[1][1], domain_cv[1][2], nbins)
		rangey = LinRange(domain_cv[2][1], domain_cv[2][2], nbins)
		open("data/F_$(count)_$(r)_$(rc)_$(nbasis).txt", "w") do file
			for x in rangex
				for y in rangey
					write(file, "$(-Vbias([x, y], rho, rhomax, basis, Vmax)) ")
				end
				write(file, "\n")
			end
		end

		open("data/dVbiasdx_$(count)_$(r)_$(rc)_$(nbasis).txt", "w") do filex
			open("data/dVbiasdy_$(count)_$(r)_$(rc)_$(nbasis).txt", "w") do filey
				for x in rangex
					for y in rangey
						grad = dVbias([x, y], rho, rhomax, basis, basisd, Vmax)
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

println(ARGS)
flush(stdout)
r = parse(Int64, ARGS[1])
rc = parse(Int64, ARGS[2])
nbasis = parse(Int64, ARGS[3])
sketch_mtd()
