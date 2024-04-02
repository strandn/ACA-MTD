using Random
using Distributions
using ForwardDiff
using LinearAlgebra

include("tt_aca.jl")

x(r) = 0.82 - 0.82 * r[1] - 0.41 * r[2] + 0.41 * r[3]
y(r) = 0.98 - 0.25 * r[1] - 0.12 * r[2] - 0.62 * r[3] + 0.74 * r[4]

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

function Vbias(r, slist, w, sigma)
	result = 0.0
	for i in eachindex(slist)
		result += w[i] * exp(-(x(r) - slist[i][1]) ^ 2 / (2 * sigma[1] ^ 2) - (y(r) - slist[i][2]) ^ 2 / (2 * sigma[2] ^ 2))
	end
	return result
end

function grad_V(r, slist, w, sigma)
	grad = ForwardDiff.gradient(V, r)
	for i in eachindex(slist)
		Vinc(r) = w[i] * exp(-(x(r) - slist[i][1]) ^ 2 / (2 * sigma[1] ^ 2) - (y(r) - slist[i][2]) ^ 2 / (2 * sigma[2] ^ 2))
		grad += ForwardDiff.gradient(Vinc, r)
	end
	return grad
end

function well_tempered_mtd(biasfactor::Float64)
	domain = ((-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0))
	
	T = 1.0
	gamma = 1.0
	dt = 1.0e-4
	steps = 1e8
	
	x1 = rand(Normal(-1.0, 0.1))
	x2 = rand(Normal(-1.0, 0.1))
	x3 = rand(Normal(-1.0, 0.1))
	x4 = rand(Normal(1.0, 0.1))
	v1 = v2 = v3 = v4 = 0.0
	
	kb = 1.0
	normal_dist = Normal(0.0, sqrt(2 * kb * T / (gamma * dt)))
	
	stride = 100
	t = 0.0

	pace = 500
	height = 1.2
	# biasfactor = 6
	sigma = [0.15, 0.15]
	
	traj = []
	slist = []
	w = []
	if isfile("hills.txt")
		rm("hills.txt")
	end
	for i in 1:steps
		grad = grad_V([x1, x2, x3, x4], slist, w, sigma)
	
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
		end
	
		if i % stride == 0
			push!(traj, (t, x([x1, x2, x3, x4]), y([x1, x2, x3, x4]), Vbias([x1, x2, x3, x4], slist, w, sigma)))
		end

		if i % pace == 0
			s_new = [x([x1, x2, x3, x4]), y([x1, x2, x3, x4])]
			w_new = height * exp(-Vbias([x1, x2, x3, x4], slist, w, sigma) / (kb * T * (biasfactor - 1)))
			push!(slist, s_new)
			push!(w, w_new)
			open("hills.txt", "a") do file
				write(file, "$t $s_new $w_new\n")
			end
		end

		t += dt
	end
	open("colvar.txt", "w") do file
		for step in traj
			write(file, "$(step[1]) $(step[2]) $(step[3]) $(step[4])\n")
		end
	end
	# open("hills.txt", "w") do file
	# 	for i in eachindex(slist)
	# 		write(file, "$(pace * dt) $(slist[i]) $(w[i])\n")
	# 	end
	# end
end

println(ARGS)
flush(stdout)
biasfactor = parse(Int64, ARGS[1])
well_tempered_mtd(biasfactor)
