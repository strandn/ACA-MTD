using HCubature
using LinearAlgebra

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

function gaussian_marginals()
	beta = 1.0
	P(x1, x2, x3, x4) = exp(-beta * V([x1, x2, x3, x4]))
	domain = [(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
	Z = hcubature(x->P(x[1], x[2], x[3], x[4]), [d[1] for d in domain], [d[2] for d in domain])
	println(Z)
	flush(stdout)

	nbins = 1000
	ranges = [LinRange(d[1], d[2], nbins) for d in domain]

	rho_total = 0.0
	println("Computing F(x1)...")
	flush(stdout)
	open("fes_correct_1.txt", "w") do file
		Threads.@threads for x in ranges[1]
			f(x2, x3, x4) = P(x, x2, x3, x4)
			rho, _ = hcubature(x->f(x[1], x[2], x[3]), [domain[k][1] for k in [2, 3, 4]], [domain[k][2] for k in [2, 3, 4]])
			rho_total += rho
			write(file, "$(-log(rho) / beta) ")
		end
		write(file, "\n")
	end
	println(rho_total)
	println()

	rho_total = 0.0
	println("Computing F(x2)...")
	flush(stdout)
	open("fes_correct_2.txt", "w") do file
		Threads.@threads for x in ranges[2]
			f(x1, x3, x4) = P(x1, x, x3, x4)
			rho, _ = hcubature(x->f(x[1], x[2], x[3]), [domain[k][1] for k in [1, 3, 4]], [domain[k][2] for k in [1, 3, 4]])
			global rho_total += rho
			write(file, "$(-log(rho) / beta) ")
		end
		write(file, "\n")
	end
	println(rho_total)
	println()

	rho_total = 0.0
	println("Computing F(x3)...")
	flush(stdout)
	open("fes_correct_3.txt", "w") do file
		Threads.@threads for x in ranges[3]
			f(x1, x2, x4) = P(x1, x2, x, x4)
			rho, _ = hcubature(x->f(x[1], x[2], x[3]), [domain[k][1] for k in [1, 2, 4]], [domain[k][2] for k in [1, 2, 4]])
			global rho_total += rho
			write(file, "$(-log(rho) / beta) ")
		end
		write(file, "\n")
	end
	println(rho_total)
	println()

	rho_total = 0.0
	println("Computing F(x4)...")
	flush(stdout)
	open("fes_correct_4.txt", "w") do file
		Threads.@threads for x in ranges[4]
			f(x1, x2, x3) = P(x1, x2, x3, x)
			rho, _ = hcubature(x->f(x[1], x[2], x[3]), [domain[k][1] for k in [1, 2, 3]], [domain[k][2] for k in [1, 2, 3]])
			global rho_total += rho
			write(file, "$(-log(rho) / beta) ")
		end
		write(file, "\n")
	end
	println(rho_total)
	println()

	nbins = 100
	ranges = [LinRange(d[1], d[2], nbins) for d in domain]

	rho_total = 0.0
	println("Computing F(x1, x2)...")
	flush(stdout)
	open("fes_correct_12.txt", "w") do file
		Threads.@threads for x in ranges[1]
			for y in ranges[2]
				f(x3, x4) = P(x, y, x3, x4)
				rho, _ = hcubature(x->f(x[1], x[2]), [domain[k][1] for k in [3, 4]], [domain[k][2] for k in [3, 4]])
				rho_total += rho
				write(file, "$(-log(rho) / beta) ")
			end
			write(file, "\n")
		end
	end
	println(rho_total)
	println()

	rho_total = 0.0
	println("Computing F(x1, x3)...")
	flush(stdout)
	open("fes_correct_13.txt", "w") do file
		Threads.@threads for x in ranges[1]
			for y in ranges[3]
				f(x2, x4) = P(x, x2, y, x4)
				rho, _ = hcubature(x->f(x[1], x[2]), [domain[k][1] for k in [2, 4]], [domain[k][2] for k in [2, 4]])
				rho_total += rho
				write(file, "$(-log(rho) / beta) ")
			end
			write(file, "\n")
		end
	end
	println(rho_total)
	println()

	rho_total = 0.0
	println("Computing F(x1, x4)...")
	flush(stdout)
	open("fes_correct_14.txt", "w") do file
		Threads.@threads for x in ranges[1]
			for y in ranges[4]
				f(x2, x3) = P(x, x2, x3, y)
				rho, _ = hcubature(x->f(x[1], x[2]), [domain[k][1] for k in [2, 3]], [domain[k][2] for k in [2, 3]])
				rho_total += rho
				write(file, "$(-log(rho) / beta) ")
			end
			write(file, "\n")
		end
	end
	println(rho_total)
	println()

	rho_total = 0.0
	println("Computing F(x2, x3)...")
	flush(stdout)
	open("fes_correct_23.txt", "w") do file
		Threads.@threads for x in ranges[2]
			for y in ranges[3]
				f(x1, x4) = P(x1, x, y, x4)
				rho, _ = hcubature(x->f(x[1], x[2]), [domain[k][1] for k in [1, 4]], [domain[k][2] for k in [1, 4]])
				rho_total += rho
				write(file, "$(-log(rho) / beta) ")
			end
			write(file, "\n")
		end
	end
	println(rho_total)
	println()

	rho_total = 0.0
	println("Computing F(x2, x4)...")
	flush(stdout)
	open("fes_correct_24.txt", "w") do file
		Threads.@threads for x in ranges[2]
			for y in ranges[4]
				f(x1, x3) = P(x1, x, x3, y)
				rho, _ = hcubature(x->f(x[1], x[2]), [domain[k][1] for k in [1, 3]], [domain[k][2] for k in [1, 3]])
				rho_total += rho
				write(file, "$(-log(rho) / beta) ")
			end
			write(file, "\n")
		end
	end
	println(rho_total)
	println()

	rho_total = 0.0
	println("Computing F(x3, x4)...")
	flush(stdout)
	open("fes_correct_34.txt", "w") do file
		Threads.@threads for x in ranges[3]
			for y in ranges[4]
				f(x1, x2) = P(x1, x2, x, y)
				rho, _ = hcubature(x->f(x[1], x[2]), [domain[k][1] for k in [1, 2]], [domain[k][2] for k in [1, 2]])
				rho_total += rho
				write(file, "$(-log(rho) / beta) ")
			end
			write(file, "\n")
		end
	end
	println(rho_total)
	println()
end

gaussian_marginals()
