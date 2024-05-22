using Random
using Distributions
using ForwardDiff
using KernelDensity

include("tt_sketch.jl")

function V(r)
	# x1, x2 = r
	# return x1 ^ 2 + x2 ^ 2 + 1.0
    return 0.0
end

function grad_V(r)
	return ForwardDiff.gradient(V, r)
end

function sketch_mtd()
	domain = [(-6.0, 6.0), (-8.0, 8.0)]
	nbins = 100

	T = 1.0
	gamma = 1.0
	dt = 1.0e-4
	steps = 1e8
	stride = 100

	x1 = rand(Normal(0.0, 0.1))
	x2 = rand(Normal(0.0, 0.1))
	v1 = v2 = 0.0

	kb = 1.0
	sigma = sqrt(2 * kb * T / (gamma * dt))
	normal_dist = Normal(0.0, sigma)

    traj = []
    weights = []
    for i in 1:steps
        grad = grad_V([x1, x2])

        v1 = -(grad[1] / gamma) + rand(normal_dist)
        v2 = -(grad[2] / gamma) + rand(normal_dist)

        x1 += v1 * dt
        x2 += v2 * dt
        
        x1 = clamp(x1, domain[1][1], domain[1][2])
        x2 = clamp(x2, domain[2][1], domain[2][2])

        if i % stride == 0
            push!(traj, [x1, x2])
            # push!(weights, x1 ^ 4 + x2 ^ 4 + 1.0)
            push!(weights, x1 + x2 > 0 ? exp(15.0) : 1.0)
        end
    end

    xlist = [step[1] for step in traj]
    ylist = [step[2] for step in traj]
    G, basis, _ = para_sketch(hcat(xlist, ylist), domain, "poly", 6, 30, 0.2, 30, ones(Int64(div(steps, stride))))

    rangex = LinRange(domain[1][1], domain[1][2], nbins)
    rangey = LinRange(domain[2][1], domain[2][2], nbins)
    open("ttde.out", "w") do file
        for x in rangex
            for y in rangey
                write(file, "$(dens_eval(G, basis, [x, y])) ")
            end
            write(file, "\n")
        end
    end

    kde_result = kde(hcat(xlist, ylist), npoints = (nbins, nbins))
    ik = InterpKDE(kde_result)
    open("kde.out", "w") do file
        for x in rangex
            for y in rangey
                write(file, "$(pdf(ik, x, y)) ")
            end
            write(file, "\n")
        end
    end

    G, basis, _ = para_sketch(hcat(xlist, ylist), domain, "poly", 6, 30, 0.2, 30, weights / sum(weights))

    rangex = LinRange(domain[1][1], domain[1][2], nbins)
    rangey = LinRange(domain[2][1], domain[2][2], nbins)
    open("ttderw.out", "w") do file
        for x in rangex
            for y in rangey
                write(file, "$(dens_eval(G, basis, [x, y])) ")
            end
            write(file, "\n")
        end
    end
end

sketch_mtd()
