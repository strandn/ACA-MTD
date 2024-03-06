using HCubature
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
function V(x1, x2, x3, x4)
	r = [x1, x2, x3, x4]
	return 30 * exp(-5 * norm(r - max1) ^ 2) + 35 * exp(-5 * norm(r - max2) ^ 2) + 40 * exp(-5 * norm(r - max3) ^ 2) +
		45 * exp(-5 * norm(r - max4) ^ 2) -
		20 * exp(-norm(r - large1) ^ 2) - 25 * exp(-norm(r - large2) ^ 2) - 30 * exp(-norm(r - large3) ^ 2) +
		(x1 + 1/3) ^ 4 / 5 + (x2 + 2/3) ^ 4 / 5 + x3 ^ 4 / 5 + (x4 + 1/3) ^ 4 / 5
end

beta = 1
P(x1, x2, x3, x4) = exp(-beta * V(x1, x2, x3, x4))
domain = ((-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0))
domain_cv = ((-1.5, 4.0), (-1.5, 4.5))
nbins = 50
dx = (domain_cv[1][2] - domain_cv[1][1]) / nbins
dy = (domain_cv[2][2] - domain_cv[2][1]) / nbins
G = zeros(nbins - 1, nbins - 1)
Z = hcubature(x->P(x[1], x[2], x[3], x[4]), [domain[k][1] for k in 1:4], [domain[k][2] for k in 1:4]; rtol = 10^-4)
println(Z)
flush(stdout)
rho_total = 0.0
for i in 1:nbins-1
	Threads.@threads for j in 1:nbins-1
		(x, y) = ((i - 1) * dx + domain_cv[1][1], (j - 1) * dy + domain_cv[2][1])
		f(x1, x2, x3, x4) = if 0.82 - 0.82 * x1 - 0.41 * x2 + 0.41 * x3 >= x && 0.82 - 0.82 * x1 - 0.41 * x2 + 0.41 * x3 < x + dx && 0.98 - 0.25 * x1 - 0.12 * x2 - 0.62 * x3 + 0.74 * x4 >= y && 0.98 - 0.25 * x1 - 0.12 * x2 - 0.62 * x3 + 0.74 * x4 < y + dy
			P(x1, x2, x3, x4)
		else
			0.0
		end
		rho, error = hcubature(x->f(x[1], x[2], x[3], x[4]), [domain[k][1] for k in 1:4], [domain[k][2] for k in 1:4]; maxevals = 10^7, initdiv = 20)
		# rho = max(rho, 10^-10)
		global rho_total += rho
		G[i, j] = -log(rho) / beta
		# print("$(-log(rho) / beta) ")
	end
	# println()
	println("i = $i done!")
	flush(stdout)
end
println(G)
println(rho_total)
