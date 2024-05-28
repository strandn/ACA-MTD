domain_cv = [(-1.5, 4.0), (-1.5, 4.5)]
nbins = 100

biasfactor = parse(Int64, ARGS[1])
width = parse(Int64, ARGS[2])
sigma = width * 0.01
height = 1.0
# sigma = [0.15, 0.15]
stride = 1000

strlist = readlines("hills_$(biasfactor)_$(width).txt")
data = [parse.(Float64, split(replace(str, r"[^\d\.-]+" => " "))) for str in strlist]

slist = [line[2:3] for line in data]
w = [line[4] for line in data]

open("fes_$(biasfactor)_$(width).txt", "w") do file
	result = zeros(nbins, nbins)
	for i in eachindex(slist)
		for j in 1:nbins
			for k in 1:nbins
				x = domain_cv[1][1] + (j - 1) * (domain_cv[1][2] - domain_cv[1][1]) / (nbins - 1)
				y = domain_cv[2][1] + (k - 1) * (domain_cv[2][2] - domain_cv[2][1]) / (nbins - 1)
				result[j, k] += w[i] * exp(-(x - slist[i][1]) ^ 2 / (2 * (sigma * (domain_cv[1][2] - domain_cv[1][1])) ^ 2) - (y - slist[i][2]) ^ 2 / (2 * (sigma * (domain_cv[2][2] - domain_cv[2][1])) ^ 2))
				if i % stride == 0
					write(file, "$(-result[j, k]) ")
				end
			end
			if i % stride == 0
				write(file, "\n")
			end
		end
	end
end
