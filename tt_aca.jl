using Random
using Distributions

mutable struct ResFunc{T, N}
    f
    ndims::Int64
    pos::Int64
    domain::NTuple{N, Tuple{T, T}}
    I::Vector{Vector{Vector{T}}}
    J::Vector{Vector{Vector{T}}}
    resfirst::Vector{T}
    minp::Vector{T}

    function ResFunc(f, domain::NTuple{N, Tuple{T, T}}) where {T, N}
        new{T, N}(f, N, 0, domain, [[[T[]]]; [Vector{T}[] for _ in 2:N]], [[[T[]]]; [Vector{T}[] for _ in 2:N]], Vector{T}[], fill(Inf, N - 1))
    end
end

function (F::ResFunc{T, N})(elements::T...) where {T, N}
    (x, y) = ([elements[i] for i in 1:F.pos], [elements[i] for i in F.pos+1:F.ndims])
    k = length(F.I[F.pos + 1])
    old = new = zeros(1, 1)
    for iter in 0:k
        new = zeros(k - iter + 1, k - iter + 1)
        for idx in CartesianIndices(new)
            if iter == 0
                row = idx[1] == k + 1 ? x : F.I[F.pos + 1][idx[1]]
                col = idx[2] == k + 1 ? y : F.J[F.pos + 1][idx[2]]
                new[idx] = F.f((row..., col...)...)
            else
                new[idx] = old[idx[1] + 1, idx[2] + 1] - old[idx[1] + 1, 1] * old[1, idx[2] + 1] / old[1, 1]
            end
        end
        old = deepcopy(new)
    end
    return new[]
end


function Vbias(F::ResFunc{T, N}, elements::T...) where {T, N}
    # if length(F.I[F.pos + 1]) == 1
    #     return -10 * log(abs(F(elements...)) + 1.0e-6)
    # elseif length(F.I[F.pos + 1]) == 2
    #     return -30 * log(abs(F(elements...)) + 1.0e-5)
    # end
    # (x, y) = ([elements[i] for i in 1:F.pos], [elements[i] for i in F.pos+1:F.ndims])
    # k = length(F.I[F.pos + 1])
    # old = new = zeros(1, 1)
    # for iter in 0:k
    #     new = zeros(k - iter + 1, k - iter + 1)
    #     for idx in CartesianIndices(new)
    #         if iter == 0
    #             row = idx[1] == k + 1 ? x : F.I[F.pos + 1][idx[1]]
    #             col = idx[2] == k + 1 ? y : F.J[F.pos + 1][idx[2]]
    #             # new[idx] = log(F.f((row..., col...)...)) - log(0.1 * F.minp[F.pos])
    #             eps = 1.0e-12
    #             new[idx] = log(max(abs(F.f((row..., col...)...)), eps)) - log(eps)
    #         else
    #             new[idx] = old[idx[1] + 1, idx[2] + 1] - old[idx[1] + 1, 1] * old[1, idx[2] + 1] / old[1, 1]
    #         end
    #     end
    #     old = deepcopy(new)
    # end
    # return -abs(new[])
    
    # eps, alpha = if length(F.I[F.pos + 1]) == 1
    #     [1.0e-2], [1.0]
    # elseif length(F.I[F.pos + 1]) == 2
    #     [1.0e-2, 1.0e-2], [1.0, 0.16]
    # elseif length(F.I[F.pos + 1]) == 3
    #     [1.0e-6, 1.0e-6, 1.0e-6], [1.0, 0.08, 0.03]
    # elseif length(F.I[F.pos + 1]) == 4
    #     [1.0e-6, 1.0e-6, 1.0e-6, 1.0e-6], [1.0, 0.08, 0.03, 0.1]
    # end
    # rank = length(F.I[F.pos + 1])
    # (x, y) = ([elements[i] for i in 1:F.pos], [elements[i] for i in F.pos+1:F.ndims])
    # (xlist, ylist) = (F.I[F.pos + 1], F.J[F.pos + 1])
    # kde1 = [F.f((x..., yk...)...) for yk in ylist]
    # kde2 = [F.f((xk..., y...)...) for xk in xlist]
    # kde12 = [F.f((xlist[i]..., ylist[i]...)...) for i in 1:rank]
    # f1 = [-log(max(kde1[i], eps[i])) + log(eps[i]) for i in 1:rank]
    # f2 = [-log(max(kde2[i], eps[i])) + log(eps[i]) for i in 1:rank]
    # f12 = [-log(max(kde12[i], eps[i])) + log(eps[i]) for i in 1:rank]
    # return -sum(alpha .* f1 .* f2 ./ f12)

    Vinc = 5.0
    alpha = [1.0, 1.0, 1.0, 1.0]
    rank = length(F.I[F.pos + 1])
    (x, y) = ([elements[i] for i in 1:F.pos], [elements[i] for i in F.pos+1:F.ndims])
    (xlist, ylist) = (F.I[F.pos + 1], F.J[F.pos + 1])
    kde1 = [F.f((x..., yk...)...) for yk in ylist]
    kde2 = [F.f((xk..., y...)...) for xk in xlist]
    kde12 = [F.f((xlist[i]..., ylist[i]...)...) for i in 1:rank]
    f1 = -log.(abs.(kde1))
    f2 = -log.(abs.(kde2))
    f12 = -log.(abs.(kde12))
    f1 = [max(-(f1[i] - f12[i]) + Vinc, 0) for i in 1:rank]
    f2 = [max(-(f2[i] - f12[i]) + Vinc, 0) for i in 1:rank]
    f12 = fill(Vinc, rank)
    return sum(alpha .* f1 .* f2 ./ f12)
    # return Vtop > Vmax ? max(sum(f1 .* f2 ./ f12) - (Vtop - Vmax), 0.0) : sum(f1 .* f2 ./ f12)
    # result = 0.0
    # for i in 1:rank
    #     result += f1[i] * f2[i] / f12[i]
    #     result = max(result - offsets[i], 0.0)
    # end
    # return result
end

function initIJ(F::ResFunc{T, N}, IJ::Tuple{Vector{Vector{Vector{T}}}, Vector{Vector{Vector{T}}}}) where {T, N}
    order = F.ndims
    (F.I, F.J) = IJ
    F.pos += 1
    for i in 1:order-1
        push!(F.resfirst, F.f((F.I[i + 1][1]..., F.J[i + 1][1]...)...))
    end
end

function updateIJ(F::ResFunc{T, N}, ij::NTuple{N, T}) where {T, N}
    push!(F.I[F.pos + 1], [ij[j] for j in 1:F.pos])
    push!(F.J[F.pos + 1], [ij[j] for j in F.pos+1:F.ndims])
end

function continuous_aca(F::ResFunc{T, N}, rank::Vector{Int64}, n_chains::Int64, n_samples::Int64, jump_width::Float64, mpi_comm::MPI.Comm) where {T, N}
    order = F.ndims
    if order == 1 && mpi_rank == 0
        error(
            "`continuous_aca` currently does not support system sizes of 1.",
        )
    end

    mpi_rank = MPI.Comm_rank(mpi_comm)
    mpi_size = MPI.Comm_size(mpi_comm)

    F.pos = 0
    for i in 1:order-1
        if mpi_rank == 0
            println("pos = $i")
            flush(stdout)
        end
        F.pos += 1
        
        n_pivots = length(F.I[i])
        n_chains_reduced = max(ceil(Int64, n_chains / n_pivots), ceil(Int64, mpi_size / n_pivots))
        n_chains_total = n_pivots * n_chains_reduced
        xylist = fill(Tuple(fill(0.0, order)), n_chains_total)
        reslist = fill(0.0, n_chains_total)
        resminlist = fill(0.0, n_chains_total)
        res_new = 0.0
        for r in length(F.I[i + 1])+1:rank[i]
            elements_per_task = floor(Int64, n_chains_total / mpi_size)
            remainder = n_chains_total % mpi_size
            local_xy = fill(Tuple(fill(0.0, order)), elements_per_task)
            local_res = fill(0.0, elements_per_task)
            local_resmin = fill(0.0, elements_per_task)
            for k in 1:elements_per_task
                global_idx = mpi_rank * elements_per_task + k
                pidx = floor(Int64, (global_idx - 1) / n_chains_reduced) + 1
                local_xy[k], local_res[k], local_resmin[k] = max_metropolis(F, F.I[i][pidx], n_samples, jump_width)
            end
            xydata = MPI.Gather(local_xy, 0, mpi_comm)
            resdata = MPI.Gather(local_res, 0, mpi_comm)
            resmindata = MPI.Gather(local_resmin, 0, mpi_comm)
            if mpi_rank == 0
                xylist[1:mpi_size*elements_per_task] .= xydata
                reslist[1:mpi_size*elements_per_task] .= resdata
                resminlist[1:mpi_size*elements_per_task] .= resmindata
            end
            if mpi_rank == 0 && remainder > 0
                for k in mpi_size*elements_per_task+1:n_chains_total
                    pidx = floor(Int64, (k - 1) / n_chains_reduced) + 1
                    xylist[k], reslist[k], resminlist[k] = max_metropolis(F, F.I[i][pidx], n_samples, jump_width)
                end
            end
            xylist = reshape(xylist, (n_pivots, n_chains_reduced))
            reslist = reshape(reslist, (n_pivots, n_chains_reduced))
            resminlist = reshape(resminlist, (n_pivots, n_chains_reduced))
            idx = argmax(reslist)
            xy = [xylist[idx]]
            MPI.Bcast!(xy, 0, mpi_comm)
            res_new = [reslist[idx]]
            MPI.Bcast!(res_new, 0, mpi_comm)
            if isempty(F.I[i + 1])
                push!(F.resfirst, res_new[])
            elseif res_new[] > F.resfirst[i]
                F.resfirst[i] = res_new[]
            elseif res_new[] / F.resfirst[i] < 1e-6
                break
            end
            updateIJ(F, xy[])
            if mpi_rank == 0
                println("rank = $r res = $(res_new[]) xy = $(xy[])")
                flush(stdout)
                if minimum(resmindata) < F.minp[i]
                    F.minp[i] = minimum(resmindata)
                end
            end
        end
    end

    return F.I, F.J
end

function max_metropolis(F::ResFunc{T, N}, pivot::Vector{T}, n_samples::Int64, jump_width::Float64) where {T, N}
    order = F.ndims - F.pos + 1
    
    lb = [F.domain[i][1] for i in F.pos:F.ndims]
    ub = [F.domain[i][2] for i in F.pos:F.ndims]

    chain = zeros(n_samples, order)

    max_res = 0.0
    min_res = Inf
    max_xy = zeros(F.ndims)

    for k in 1:order
        chain[1, k] = rand() * (ub[k] - lb[k]) + lb[k]
    end
    while abs(F([pivot; [chain[1, k] for k in 1:order]]...)) == 0.0
        for k in 1:order
            chain[1, k] = rand() * (ub[k] - lb[k]) + lb[k]
        end
    end

    for i in 2:n_samples
        # println(abs(F([pivot; [chain[i - 1, k] for k in 1:order]]...)))
        p_new = zeros(order)
        for k in 1:order
            p_new[k] = rand(Normal(chain[i - 1, k], jump_width * (ub[k] - lb[k])))
            if p_new[k] < lb[k]
                p_new[k] = lb[k] + abs(p_new[k] - lb[k])
            elseif p_new[k] > ub[k]
                p_new[k] = ub[k] - abs(p_new[k] - ub[k])
            end
        end

        arg_old = [pivot; [chain[i - 1, k] for k in 1:order]]
        arg_new = [pivot; [p_new[k] for k in 1:order]]
        f_old = abs(F(arg_old...))
        f_new = abs(F(arg_new...))
        acceptance_prob = min(1, f_new / f_old)
        
        if isnan(acceptance_prob) || rand() < acceptance_prob
            chain[i, :] = p_new
            if f_new > max_res
                max_res = f_new
                max_xy = arg_new
            end
            if f_new < min_res
                min_res = f_new
            end
        else
            chain[i, :] = chain[i - 1, :]
        end
    end

    return Tuple(max_xy), max_res, min_res
end

function compute_norm(F::ResFunc{T, N}) where {T, N}
    order = F.ndims
    npivots = [length(F.I[i]) for i in 2:order]
    norm = zeros(1, npivots[1])
    for j in 1:npivots[1]
        f(x) = F.f((x, F.J[2][j]...)...)
        norm[j] = quadgk(f, F.domain[1]...)[1]
    end
    AIJ = zeros(npivots[1], npivots[1])
    for j in 1:npivots[1]
        for k in 1:npivots[1]
            AIJ[j, k] = F.f((F.I[2][j]..., F.J[2][k]...)...)
        end
    end
    norm *= inv(AIJ)
    for i in 2:order-1
        normi = zeros((npivots[i - 1], npivots[i]))
        for j in 1:npivots[i - 1]
            for k in 1:npivots[i]
                f(x) = F.f((F.I[i][j]..., x, F.J[i + 1][k]...)...)
                normi[j, k] = quadgk(f, F.domain[i]...)[1]
            end
        end
        AIJ = zeros(npivots[i], npivots[i])
        for j in 1:npivots[i]
            for k in 1:npivots[i]
                AIJ[j, k] = F.f((F.I[i + 1][j]..., F.J[i + 1][k]...)...)
            end
        end
        norm *= normi * inv(AIJ)
    end
    R = zeros(npivots[order - 1])
    for j in 1:npivots[order - 1]
        f(x) = F.f((F.I[order][j]..., x)...)
        R[j] = quadgk(f, domain[order]...)[1]
    end
    norm *= R
    return norm[]
end

function compute_mu(F::ResFunc{T, N}, norm::T) where {T, N}
    order = F.ndims
    npivots = [length(F.I[i]) for i in 2:order]
    mu = [zeros(1, npivots[1]) for _ in 1:order]
    for j in 1:npivots[1]
        for pos in 1:order
            f(x) = if pos == 1
                x * F.f((x, F.J[2][j]...)...)
            else
                F.f((x, F.J[2][j]...)...)
            end
            mu[pos][j] = quadgk(f, domain[1]...)[1]
        end
    end
    AIJ = zeros(npivots[1], npivots[1])
    for j in 1:npivots[1]
        for k in 1:npivots[1]
            AIJ[j, k] = F.f((F.I[2][j]..., F.J[2][k]...)...)
        end
    end
    for pos in 1:order
        mu[pos] *= inv(AIJ)
    end
    for i in 2:order-1
        normi = [zeros((npivots[i - 1], npivots[i])) for _ in 1:order]
        prev = deepcopy(mu)
        mu = [zeros(1, npivots[i]) for _ in 1:order]
        for j in 1:npivots[i - 1]
            for k in 1:npivots[i]
                for pos in 1:order
                    f(x) = if pos == i
                        x * F.f((F.I[i][j]..., x, F.J[i + 1][k]...)...)
                    else
                        F.f((F.I[i][j]..., x, F.J[i + 1][k]...)...)
                    end
                    normi[pos][j, k] = quadgk(f, domain[i]...)[1]
                end
            end
        end
        AIJ = zeros(npivots[i], npivots[i])
        for j in 1:npivots[i]
            for k in 1:npivots[i]
                AIJ[j, k] = F.f((F.I[i + 1][j]..., F.J[i + 1][k]...)...)
            end
        end
        for pos in 1:order
            mu[pos] = prev[pos] * normi[pos] * inv(AIJ)
        end
    end
    R = [zeros(npivots[order - 1]) for _ in 1:order]
    prev = deepcopy(mu)
    mu = zeros(order)
    for j in 1:npivots[order - 1]
        for pos in 1:order
            f(x) = if pos == order
                x * F.f((F.I[order][j]..., x)...)
            else
                F.f((F.I[order][j]..., x)...)
            end
            R[pos][j] = quadgk(f, domain[order]...)[1]
        end
    end
    for pos in 1:order
        mu[pos] = (prev[pos] * R[pos])[]
    end
    return [mu[pos] / norm[] for pos in 1:order]
end

function compute_var(F::ResFunc{T, N}, norm::T, mu::Vector{T}) where {T, N}
    order = F.ndims
    npivots = [length(F.I[i]) for i in 2:order]
    var = [zeros(1, npivots[1]) for _ in 1:order]
    for j in 1:npivots[1]
        for pos in 1:order
            f(x) = if pos == 1
                (x - mu[1]) ^ 2 * F.f((x, F.J[2][j]...)...)
            else
                F.f((x, F.J[2][j]...)...)
            end
            var[pos][j] = quadgk(f, domain[1]...)[1]
        end
    end
    AIJ = zeros(npivots[1], npivots[1])
    for j in 1:npivots[1]
        for k in 1:npivots[1]
            AIJ[j, k] = F.f((F.I[2][j]..., F.J[2][k]...)...)
        end
    end
    for pos in 1:order
        var[pos] *= inv(AIJ)
    end
    for i in 2:order-1
        normi = [zeros((npivots[i - 1], npivots[i])) for _ in 1:order]
        prev = deepcopy(var)
        var = [zeros(1, npivots[i]) for _ in 1:order]
        for j in 1:npivots[i - 1]
            for k in 1:npivots[i]
                for pos in 1:order
                    f(x) = if pos == i
                        (x - mu[i]) ^ 2 * F.f((F.I[i][j]..., x, F.J[i + 1][k]...)...)
                    else
                        F.f((F.I[i][j]..., x, F.J[i + 1][k]...)...)
                    end
                    normi[pos][j, k] = quadgk(f, domain[i]...)[1]
                end
            end
        end
        AIJ = zeros(npivots[i], npivots[i])
        for j in 1:npivots[i]
            for k in 1:npivots[i]
                AIJ[j, k] = F.f((F.I[i + 1][j]..., F.J[i + 1][k]...)...)
            end
        end
        for pos in 1:order
            var[pos] = prev[pos] * normi[pos] * inv(AIJ)
        end
    end
    R = [zeros(npivots[order - 1]) for _ in 1:order]
    prev = deepcopy(var)
    var = zeros(order)
    for j in 1:npivots[order - 1]
        for pos in 1:order
            f(x) = if pos == order
                (x - mu[order]) ^ 2 * F.f((F.I[order][j]..., x)...)
            else
                F.f((F.I[order][j]..., x)...)
            end
            R[pos][j] = quadgk(f, domain[order]...)[1]
        end
    end
    for pos in 1:order
        var[pos] = (prev[pos] * R[pos])[]
    end
    return [var[pos] / norm[] for pos in 1:order]
end
