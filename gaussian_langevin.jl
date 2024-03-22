using Random
using Distributions
using ForwardDiff
using LinearAlgebra

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

function V(r)
	x1, x2, x3, x4 = r
	return 30 * exp(-5 * norm(r - max1) ^ 2) + 35 * exp(-5 * norm(r - max2) ^ 2) + 40 * exp(-5 * norm(r - max3) ^ 2) +
		45 * exp(-5 * norm(r - max4) ^ 2) -
		15 * exp(-norm(r - large1) ^ 2) - 20 * exp(-norm(r - large2) ^ 2) - 25 * exp(-norm(r - large3) ^ 2) +
		(x1 + 1/3) ^ 4 / 5 + (x2 + 2/3) ^ 4 / 5 + x3 ^ 4 / 5 + (x4 + 1/3) ^ 4 / 5
end

grad_V(x1, x2, x3, x4) = ForwardDiff.gradient(V, [x1, x2, x3, x4])

domain = ((-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0))
domain_cv = ((-1.5, 4.0), (-1.5, 4.5))
T = 1.0
gamma = 1.0
# mass = 1.0
dt = 1.0e-4
steps = 1e7

x1 = rand(Normal(-1.0, 0.1))
x2 = rand(Normal(-1.0, 0.1))
x3 = rand(Normal(-1.0, 0.1))
x4 = rand(Normal(1.0, 0.1))
v1 = v2 = v3 = v4 = 0.0

kb = 1.0
# sigma = sqrt(2 * kb * T * gamma / mass)
sigma = sqrt(2 * kb * T / (gamma * dt))
normal_dist = Normal(0.0, sigma)

stride = 100
t = 0.0

traj = []
Vmax, Vmin = -Inf, Inf
for i in 1:steps
	grad = grad_V(x1, x2, x3, x4)
	if V([x1, x2, x3, x4]) > Vmax
		global Vmax = V([x1, x2, x3, x4])
	end
	if V([x1, x2, x3, x4]) < Vmin
		global Vmin = V([x1, x2, x3, x4])
	end
	
	# global v1 -= (gamma * v1 + grad[1]) / mass * dt + rand(normal_dist) * sqrt(dt)
	# global v2 -= (gamma * v2 + grad[2]) / mass * dt + rand(normal_dist) * sqrt(dt)
	# global v3 -= (gamma * v3 + grad[3]) / mass * dt + rand(normal_dist) * sqrt(dt)
	# global v4 -= (gamma * v4 + grad[4]) / mass * dt + rand(normal_dist) * sqrt(dt)

	global v1 = -(grad[1] / gamma) + rand(normal_dist)
	global v2 = -(grad[2] / gamma) + rand(normal_dist)
	global v3 = -(grad[3] / gamma) + rand(normal_dist)
	global v4 = -(grad[4] / gamma) + rand(normal_dist)

	global x1 += v1 * dt
	global x2 += v2 * dt
	global x3 += v3 * dt
	global x4 += v4 * dt
	
	x1 = clamp(x1, domain[1][1], domain[1][2])
	x2 = clamp(x2, domain[2][1], domain[2][2])
	x3 = clamp(x3, domain[3][1], domain[3][2])
	x4 = clamp(x4, domain[4][1], domain[4][2])

	x = 0.82 - 0.82 * x1 - 0.41 * x2 + 0.41 * x3
	y = 0.98 - 0.25 * x1 - 0.12 * x2 - 0.62 * x3 + 0.74 * x4
	global t += dt

	if i % stride == 0
		push!(traj, (t, x, y, x1, x2, x3, x4))
	end
end
open("colvar.txt", "w") do file
	for step in traj
		write(file, "$(step[1]) $(step[2]) $(step[3]) $(step[4]) $(step[5]) $(step[6]) $(step[7])\n")
	end
end
println("Vmax = $Vmax")
println("Vmin = $Vmin")
