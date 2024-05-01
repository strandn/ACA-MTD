using Random
using Distributions

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

function fes(rho, rhomax, kT)
	rho_adj = max(rho * 100 / rhomax, 1)
	return -kT * log(rho_adj)
end

function Vbias(s, rholist, rhomaxlist, basislist, kT)
	result = 0.0
	for i in eachindex(rholist)
		# result += kT * log(dens_eval(rholist[i], basislist[i], s))
        rho = dens_eval(rholist[i], basislist[i], s)
        result -= fes(rho, rhomaxlist[i], kT)
	end
	return result
end

Vbias_shifted(s, rholist, basislist, rhomaxlist, kT, Vshift) = max(Vbias(s, rholist, rhomaxlist, basislist, kT) - Vshift, 0.0)

function Vtop(rholist, rhomaxlist, basislist, kT, samples)
	max = 0.0
	for s in samples
		result = Vbias(s, rholist, rhomaxlist, basislist, kT)
		if result > max
			max = result
		end
	end
	return max
end

function dVbias(s, rholist, rhomaxlist, basislist, basisdlist, kT, Vshift)
	grad = fill(0.0, length(s))
	if Vbias(s, rholist, rhomaxlist, basislist, kT) <= Vshift
		return grad
	end
	for i in eachindex(rholist)
        rho = dens_eval(rholist[i], basislist[i], s)
        if rho * 100 / rhomaxlist[i] > 1
            grad += dens_grad(rholist[i], basislist[i], basisdlist[i], s) * kT / rho
        end
	end
	return grad
end

function grad_V(r, rholist, rhomaxlist, basislist, basisdlist, kT, Vshift)
	grad = ForwardDiff.gradient(V, r)
	if isempty(rholist)
		return grad
	end
	dx = ForwardDiff.gradient(x, r)
	dy = ForwardDiff.gradient(y, r)
	dVdx, dVdy = dVbias([x(r), y(r)], rholist, rhomaxlist, basislist, basisdlist, kT, Vshift)
	dVdx1 = dVdx * dx[1] + dVdy * dy[1]
	dVdx2 = dVdx * dx[2] + dVdy * dy[2]
	dVdx3 = dVdx * dx[3] + dVdy * dy[3]
	dVdx4 = dVdx * dx[4] + dVdy * dy[4]
	grad += [dVdx1, dVdx2, dVdx3, dVdx4]
	return grad
end

function sketch_mtd()
	domain = ((-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0))
	domain_cv = ((-1.5, 4.0), (-1.5, 4.5))
	domain_cv_full = ((-2.46, 4.1), (-2.48, 4.5))
	nbins = 100

	T = 1.0
	gamma = 1.0
	dt = 1.0e-4
	steps = 1e6
	stride = 100
	# steps = 10000
	# stride = 10
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
	# Vmax = 20 * kb * T
	Vmax = Inf
	# Vinc = 4.6 * kb * T
	samples = []
	Vshift = 0.0

	for count in 1:nbiasupdates
		println("Vbias update $count...")
		flush(stdout)
		t = 0.0

		traj = []
		for i in 1:steps
			grad = grad_V([x1, x2, x3, x4], rholist, rhomaxlist, basislist, basisdlist, kb * T, Vshift)

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
				s = [x([x1, x2, x3, x4]), y([x1, x2, x3, x4])]
				push!(traj, [t, s[1], s[2], Vbias_shifted(s, rholist, rhomaxlist, basislist, kT, Vshift)])
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
        println("Forming TT...")
        println()
        flush(stdout)
		G, basis, basis_d = para_sketch(hcat(xlist, ylist), domain_cv, "poly", 2, 30, 0.05)
        
        push!(rhomaxlist, maximum([dens_eval(G, basis, [xlist[i], ylist[i]]) for i in 1:Int64(div(steps, stride))]))
		
		rangex_small = LinRange(minimum(xlist), maximum(xlist), nbins)
		rangey_small = LinRange(minimum(ylist), maximum(ylist), nbins)
        open("data/ttde_$count.txt", "w") do file
            write(file, "$(first(rangex_small)) $(last(rangex_small)) $(step(rangex_small))\n")
            write(file, "$(first(rangey_small)) $(last(rangey_small)) $(step(rangey_small))\n")
            for x in rangex_small
                for y in rangey_small
                    write(file, "$(dens_eval(G, basis, [x, y])) ")
                end
                write(file, "\n")
            end
        end

		open("data/dF_$count.txt", "w") do file
            for x in rangex_small
                for y in rangey_small
                    write(file, "$(fes(dens_eval(G, basis, [x, y]), last(rhomaxlist), kT)) ")
                end
                write(file, "\n")
            end
        end

		push!(rholist, G)
        push!(basislist, basis)
        push!(basisdlist, basis_d)
		Vpeak = Vtop(rholist, rhomaxlist, basislist, kb * T, samples)
		Vshift = max(Vpeak - Vmax, 0.0)
		println("Vtop = $Vpeak Vshift = $Vshift")
		println()
		flush(stdout)

		rangex = LinRange(domain_cv[1][1], domain_cv[1][2], nbins)
		rangey = LinRange(domain_cv[2][1], domain_cv[2][2], nbins)
		open("data/F_$count.txt", "w") do file
			for x in rangex
				for y in rangey
					write(file, "$(-Vbias_shifted([x, y], rholist, basislist, rhomaxlist, kb * T, Vshift)) ")
				end
				write(file, "\n")
			end
		end
		update_dVbias(F, outer, inner, douter, domain_cv_full, nbins)

		open("data/dVbiasdx_$count.txt", "w") do filex
			open("data/dVbiasdy_$count.txt", "w") do filey
				for x in rangex
					for y in rangey
						grad = dVbias([x, y], rholist, rhomaxlist, basislist, basisdlist, kb * T, Vshift)
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

sketch_mtd()
