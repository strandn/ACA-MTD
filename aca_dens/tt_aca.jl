include("tt_sketch.jl")

mutable struct ResFunc{T, N}
    f
    ndims::Int64
    pos::Int64
    domain::NTuple{N, Tuple{T, T}}
    I::Vector{Vector{Vector{T}}}
    J::Vector{Vector{Vector{T}}}
    u::Vector{Vector{T}}
    v::Vector{Vector{T}}
    resfirst::Vector{T}
    cutoff::T

    function ResFunc(f, domain::NTuple{N, Tuple{T, T}}, cutoff::T) where {T, N}
        new{T, N}(f, N, 0, domain, [[[T[]]]; [Vector{T}[] for _ in 2:N]], [[[T[]]]; [Vector{T}[] for _ in 2:N]], Vector{T}[], Vector{T}[], T[], cutoff)
    end
end

# function (F::ResFunc{T, N})(elements::T...) where {T, N}
#     (x, y) = ([elements[i] for i in 1:F.pos], [elements[i] for i in F.pos+1:F.ndims])
#     k = length(F.I[F.pos + 1])
#     old = new = zeros(1, 1)
#     for iter in 0:k
#         new = zeros(k - iter + 1, k - iter + 1)
#         for idx in CartesianIndices(new)
#             if iter == 0
#                 row = idx[1] == k + 1 ? x : F.I[F.pos + 1][idx[1]]
#                 col = idx[2] == k + 1 ? y : F.J[F.pos + 1][idx[2]]
#                 new[idx] = F.f([row; col]...)
#             else
#                 new[idx] = old[idx[1] + 1, idx[2] + 1] - old[idx[1] + 1, 1] * old[1, idx[2] + 1] / old[1, 1]
#             end
#         end
#         old = deepcopy(new)
#     end
#     return new[]
# end

function aca_partial(F::ResFunc{T, N}, samples, is::Int64, ilist::Vector{Int64}) where {T, N}
    r = length(F.I[F.pos + 1]) + 1
    if r == 1
        evals = zeros(length(samples))
        Threads.@threads for k in eachindex(samples)
            evals[k] = F.f(samples[k]...)
        end
        is = argmax(abs.(evals))
    end
    Rj = zeros(length(samples))
    x = [samples[is][i] for i in 1:F.pos]
    Threads.@threads for k in eachindex(samples)
        yk = [samples[k][i] for i in F.pos+1:F.ndims]
        Rj[k] = F.f([x; yk]...)
    end
    for l in 1:r-1
        Rj -= F.u[l][is] * F.v[l]
    end
    js = argmax(abs.(Rj))
    dk = Rj[js]

    Ri = zeros(length(samples))
    y = [samples[js][i] for i in F.pos+1:F.ndims]
    Threads.@threads for k in eachindex(samples)
        xk = [samples[k][i] for i in 1:F.pos]
        Ri[k] = F.f([xk; y]...)
    end
    for l in 1:r-1
        Ri -= F.u[l] * F.v[l][js]
    end
    push!(F.u, Ri)
    push!(F.v, Rj / dk)

    # if isempty(F.I[F.pos + 1])
    #     push!(F.resfirst, abs(dk))
    # end

    # push!(F.I[F.pos + 1], x)
    # push!(F.J[F.pos + 1], y)
    push!(ilist, is)
    ulast = deepcopy(F.u[r])
    is = argmax(abs.(ulast))
    while is in ilist
        ulast[is] = 0
        is = argmax(abs.(ulast))
    end
    return is, abs(dk), x, y
end

# function updateIJ(F::ResFunc{T, N}, ij::NTuple{N, T}) where {T, N}
#     push!(F.I[F.pos + 1], [ij[j] for j in 1:F.pos])
#     push!(F.J[F.pos + 1], [ij[j] for j in F.pos+1:F.ndims])
# end

function continuous_aca(F::ResFunc{T, N}, rank::Vector{Int64}, samples) where {T, N}
    order = F.ndims

    F.pos = 0
    for i in 1:order-1
        println("pos = $i")
        flush(stdout)
        F.pos += 1
        
        res_new = 0.0
        ilist = Int64[]
        is = 1
        # is = length(samples)
        empty!(F.u)
        empty!(F.v)
        for r in 1:rank[i]
            # results = zeros(n_samples)
            # Threads.@threads for k in 1:n_samples
            #     pivot = F.I[i][(k - 1) % n_pivots + 1]
            #     arg = [pivot; samples[k][F.pos:F.ndims]]
            #     results[k] = abs(F(arg...))
            # end
            
            # top = argmax(results)
            # pivot_top = F.I[i][(top - 1) % n_pivots + 1]
            # arg_top = [pivot_top; samples[top][F.pos:F.ndims]]
            # res_new = results[top]
            # xy = Tuple(arg_top)

            is, res_new, x, y = aca_partial(F, samples, is, ilist)
            if isempty(F.I[i + 1])
                push!(F.resfirst, res_new)
            elseif res_new > F.resfirst[i]
                F.resfirst[i] = res_new
            elseif res_new / F.resfirst[i] < F.cutoff
                break
            end
        
            push!(F.I[F.pos + 1], x)
            push!(F.J[F.pos + 1], y)

            println("rank = $r res = $res_new xy = $([x; y])")
            flush(stdout)
        end
    end

    return F.I, F.J
end

function update_vb(vb::MPS, G::MPS, basis, convbasis, n::Int64, domain::Vector{Tuple{Float64, Float64}}, samples, kT, vshift::Float64)
    d = length(basis)
    P(x...) = if length(vb) == 0
        kT * log(max(dens_eval(G, convbasis, [elt for elt in x]), 0.1))
    else
        max(dens_eval(vb, convbasis, [elt for elt in x]) + kT * log(max(dens_eval(G, convbasis, [elt for elt in x]), 1)) - vshift, -2 * kT)
    end
    F = ResFunc(P, Tuple(domain), 1.0e-6)

    println()
    println("Starting TT-cross ACA...")
    continuous_aca(F, fill(5, d - 1), samples)

    sites = siteinds(n, d)
    l = Vector{Index}(undef, d - 1)
    psi = Vector{ITensor}(undef, d)
    ranks = [length(F.I[i]) for i in 2:d]
    print("Determinants ")
    flush(stdout)
    for ii in eachindex(sites)
        s = sites[ii]
        if ii != d
            l[ii] = Index(ranks[ii], "Link,l=$ii")
        end

        if ii == 1
            psi[1] = ITensor(s, l[1]')
            Threads.@threads for ss in eachval(s)
                for lr in eachval(l[1])
                    f(x) = P([x; F.J[2][lr]]...) * basis[1](x, ss)
                    # psi[1][s => ss, l[1]' => lr] = quadgk(f, domain[1]..., atol = 1.0e-12)[1]
                    psi[1][s => ss, l[1]' => lr] = quadgk(f, domain[1]..., atol = 1.0e-8, rtol = 1.0e-6)[1]
                    # psi[1][s => ss, l[1]' => lr] = quadgk(f, domain[1]..., atol = 1.0e-6, rtol = 1.0e-4)[1]
                end
            end
        elseif ii == d
            psi[d] = ITensor(s, l[d - 1])
            Threads.@threads for ss in eachval(s)
                for ll in eachval(l[d - 1])
                    f(x) = P([F.I[d][ll]; x]...) * basis[d](x, ss)
                    # psi[d][s => ss, l[d - 1] => ll] = quadgk(f, domain[d]..., atol = 1.0e-12)[1]
                    psi[d][s => ss, l[d - 1] => ll] = quadgk(f, domain[d]..., atol = 1.0e-8, rtol = 1.0e-6)[1]
                    # psi[d][s => ss, l[d - 1] => ll] = quadgk(f, domain[d]..., atol = 1.0e-6, rtol = 1.0e-4)[1]
                end
            end
        else
            psi[ii] = ITensor(s, l[ii - 1], l[ii]')
            Threads.@threads for ss in eachval(s)
                for ll in eachval(l[ii - 1])
                    for lr in eachval(l[ii])
                        f(x) = P([F.I[ii][ll]; x; F.J[ii + 1][lr]]...) * basis[ii](x, ss)
                        # psi[ii][s => ss, l[ii - 1] => ll, l[ii]' => lr] = quadgk(f, domain[ii]..., atol = 1.0e-12)[1]
                        psi[ii][s => ss, l[ii - 1] => ll, l[ii]' => lr] = quadgk(f, domain[ii]..., atol = 1.0e-8, rtol = 1.0e-6)[1]
                        # psi[ii][s => ss, l[ii - 1] => ll, l[ii]' => lr] = quadgk(f, domain[ii]..., atol = 1.0e-6, rtol = 1.0e-4)[1]
                    end
                end
            end
        end

        if ii != d
            Ahat = zeros(ranks[ii], ranks[ii])
            for jj in 1:ranks[ii]
                for kk in 1:ranks[ii]
                    Ahat[jj, kk] = P([F.I[ii + 1][jj]; F.J[ii + 1][kk]]...)
                end
            end
            print("$(det(Ahat)) ")
            flush(stdout)
            psi[ii] *= ITensor(inv(Ahat), l[ii]', l[ii])
        end
    end
    println()
    flush(stdout)

    return MPS(psi)
end
