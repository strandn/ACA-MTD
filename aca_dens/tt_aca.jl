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

function updateIJ(F::ResFunc{T, N}, ij::NTuple{N, T}) where {T, N}
    push!(F.I[F.pos + 1], [ij[j] for j in 1:F.pos])
    push!(F.J[F.pos + 1], [ij[j] for j in F.pos+1:F.ndims])
end

function aca_diag(F::ResFunc{T, N}, samples, Rk::Vector{Float64}) where {T, N}
    k = length(F.I[F.pos + 1]) + 1
    ik = argmax(abs.(Rk))
    dk = Rk[ik]
    Ri = zeros(length(samples))
    Rj = zeros(length(samples))
    Threads.@threads for i in eachindex(samples)
        Ri[i] = F.f([[samples[i][j] for j in 1:F.pos]; [samples[ik][j] for j in F.pos+1:F.ndims]]...)
        Rj[i] = F.f([[samples[ik][j] for j in 1:F.pos]; [samples[i][j] for j in F.pos+1:F.ndims]]...)
        for l in 1:k-1
            Ri[i] -= F.u[l][i] * F.v[l][ik]
            Rj[i] -= F.u[l][ik] * F.v[l][i]
        end
    end
    push!(F.u, Ri)
    push!(F.v, Rj / dk)
    return abs(dk), samples[ik]
end

function continuous_aca(F::ResFunc{T, N}, rank::Vector{Int64}, samples) where {T, N}
    order = F.ndims

    F.pos = 0
    for i in 1:order-1
        println("pos = $i")
        flush(stdout)
        F.pos += 1
        
        res_new = 0.0
        Rk = [F.f(samples[i]...) for i in eachindex(samples)]
        empty!(F.u)
        empty!(F.v)
        for r in 1:rank[i]
            res_new, arg_top = aca_diag(F, samples, Rk)
            xy = Tuple(arg_top)
            if isempty(F.I[i + 1])
                push!(F.resfirst, res_new)
            elseif res_new > F.resfirst[i]
                F.resfirst[i] = res_new
            elseif res_new / F.resfirst[i] < F.cutoff
                break
            end
    
            updateIJ(F, xy)
            Rk -= F.u[r] .* F.v[r]
            println("rank = $r res = $res_new xy = $xy")
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
        # max(dens_eval(vb, basis, [elt for elt in x]) + kT * log(max(dens_eval(G, convbasis, [elt for elt in x]), 1)) - vshift, -2 * kT)
        max(dens_eval(vb, convbasis, [elt for elt in x]) + kT * log(max(dens_eval(G, convbasis, [elt for elt in x]), 1)) - vshift, -2 * kT)
    end
    F = ResFunc(P, Tuple(domain), 0.05)

    println()
    println("Starting TT-cross ACA...")
    continuous_aca(F, fill(30, d - 1), samples)

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
                end
            end
        elseif ii == d
            psi[d] = ITensor(s, l[d - 1])
            Threads.@threads for ss in eachval(s)
                for ll in eachval(l[d - 1])
                    f(x) = P([F.I[d][ll]; x]...) * basis[d](x, ss)
                    # psi[d][s => ss, l[d - 1] => ll] = quadgk(f, domain[d]..., atol = 1.0e-12)[1]
                    psi[d][s => ss, l[d - 1] => ll] = quadgk(f, domain[d]..., atol = 1.0e-8, rtol = 1.0e-6)[1]
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
