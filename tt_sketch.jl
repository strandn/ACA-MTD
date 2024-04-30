using LegendrePolynomials
using ITensors
using LinearAlgebra

function fourier_basis(x::Vector{Float64}, n::Int64, dom::Tuple{Float64, Float64})
    L = (dom[2] - dom[1]) / 2
    shift = (dom[2] + dom[1]) / 2
    y = zeros(length(x), 2 * n + 1)
    y0 = ones(length(x)) * 1 / sqrt(2 * L)
    y1 = zeros(length(x), n)
    y2 = zeros(length(x), n)
    for i in 1:n
        y1[:, i] = sqrt(1 / L) * cos.(pi * (x .- shift) * i / L)
        y2[:, i] = sqrt(1 / L) * sin.(pi * (x .- shift) * i / L)
    end

    y[:, 1] = y0
    y[:, 2:2:end-1] = y1
    y[:, 3:2:end] = y2
    return y
end
    
function legendre_basis(x::Vector{Float64}, n::Int64, dom::Tuple{Float64, Float64})
    L = (dom[2] - dom[1]) / 2
    shift = (dom[2] + dom[1]) / 2
    y = zeros(length(x), n)
    for i in 1:n
        y[:, i] = sqrt(i - 1 / 2) * Pl.((x .- shift) / L, i - 1)
    end
    return y
end

function create_TT_coeff(n::Int64, d::Int64, r::Int64, a::Float64)
    # sites = [Index(n, "Site,n=" * string(i)) for i in 1:d]
    sites = siteinds(n, d)
    coeff = randomMPS(sites; linkdims = r)
    for i in 1:d
        # if i == 1
        #     coeff[1][:, :] = 0.5 * ones(n, r)
        # elseif i == d
        #     coeff[d][:, :] = 0.5 * ones(r, n)
        # else
        #     coeff[i][:, :, :] = 0.5 * ones(r, n, r)
        # end

        A = diagITensor(a, sites[i], sites[i]')
        A[1, 1] = 1
        coeff[i] *= A
        noprime!(coeff[i])
    end
    return coeff
end

function int_basis_sample(basis, samples::Array{Float64, 2}, is::IndexSet)
    d = size(samples, 2)
    # is_new = [Index(size(samples, 1), "Site,n=" * string(i)) for i in 1:d]
    is_new = siteinds(size(samples, 1), d)
    # M = [ITensor(is[i], is_new[i]) for i in 1:d]
    M = Vector{ITensor}(undef, d)
    for i in 1:d
        M[i] = ITensor(basis[i](samples[:, i]), is_new[i], is[i])
    end
    return M, is_new
end

function form_tensor_moment(M::Vector{ITensor}, coeff::MPS, is::IndexSet)
    d = length(M);
    N = size(M[1], 1);
    # nb = dim(siteind(coeff, 1))
    rc = dim(linkind(coeff, 1))
    L = deepcopy(coeff)

    for i in 1:d
        L[i] *= M[i]
    end

    envi_L = Vector{Matrix}(undef, d)
    envi_L[2] = matrix(L[1], is[1], linkind(coeff, 1))
    for i in 3:d
        L_arr = array(L[i - 1], linkind(coeff, i - 2), is[i - 1], linkind(coeff, i - 1))
        envi_L[i] = zeros(N, rc)
        for j in 1:N
            envi_L[i][j, :] = envi_L[i - 1][j, :]' * L_arr[:, j, :]
        end
    end

    envi_R = Vector{Matrix}(undef, d)
    envi_R[d - 1] = matrix(L[d], is[d], linkind(coeff, d - 1))
    for i in d-2:-1:1
        L_arr = array(L[i + 1], linkind(coeff, i), is[i + 1], linkind(coeff, i + 1))
        envi_R[i] = zeros(N, rc)
        for j in 1:N
            envi_R[i][j, :] = envi_R[i + 1][j, :]' * L_arr[:, j, :]
        end
    end

    B = Vector{ITensor}(undef, d)
    for core_id in 1:d
        if core_id == 1
            B[1] = ITensor(envi_R[1], is[1], linkind(coeff, 1)) * M[1] / N
        elseif core_id == d
            B[d] = ITensor(envi_L[d], is[d], linkind(coeff, d - 1)) * M[d] / N
        else
            # B[core_id] = M[core_id] * envi_L[core_id] * envi_R[core_id]
            B[core_id] = ITensor(linkind(coeff, core_id - 1), is[core_id], linkind(coeff, core_id))
            for i in 1:rc
                for j in 1:rc
                    B[core_id][i, :, j] = envi_L[core_id][:, i] .* envi_R[core_id][:, j]
                end
            end
            B[core_id] *= M[core_id] / N
        end
    end

    return MPS(B), envi_L, envi_R
end

function para_sketch(samples::Array{Float64, 2}, domain::Vector{Tuple{Float64, Float64}}, basis_type::String, r::Int64, rc::Int64, alpha::Float64)
    d = size(samples, 2)
    basis, nb = if basis_type == "fourier"
        nb0 = 10;
        ([b(x) = fourier_basis(x, nb0, domain[i]) for i in 1:d], 2 * nb0 + 1)
    elseif basis_type == "poly"
        nb0 = 18;
        ([b(x) = legendre_basis(x, nb0, domain[i]) for i in 1:d], nb0)
    end

    coeff = create_TT_coeff(nb, d, rc, alpha)
    M, is = int_basis_sample(basis, samples, siteinds(coeff))
    G = Vector{ITensor}(undef, d)

    Bemp, envi_L, envi_R = form_tensor_moment(M, coeff, is)
    V = Vector{ITensor}(undef, d)
    for core_id in 1:d
        if core_id == 1
            G[1] = Bemp[1]
        else
            l = linkind(coeff, core_id - 1)
            # A = envi_L[core_id] * delta(l', l) * envi_R[core_id - 1]
            # G[core_id] = ITensor(pinv(matrix(A, l', l)), l', l) * Bemp[core_id]
            # noprime!(G[core_id])
            # _, _, V[core_id] = svd(A, l', maxdim = r, righttags = tags(l))
            A = envi_L[core_id]' * envi_R[core_id - 1]
            G[core_id] = ITensor(pinv(A), l', l) * Bemp[core_id]
            noprime!(G[core_id])
            _, _, V[core_id] = svd(ITensor(A, l', l), l', maxdim = r, righttags = tags(l))
        end
    end

    for core_id in 1:d
        if core_id == 1
            G[1] *= V[2]
        elseif core_id == d
            G[d] *= V[d]
        else
            G[core_id] *= V[core_id]
            G[core_id] *= V[core_id + 1]
        end
    end

    return MPS(G), basis
end

function eval(G::MPS, basis, elements::Vector{Float64})
    d = length(elements)
    result = G[1] * ITensor(basis[1]([elements[1]]), siteind(G, 1))
    # phi = Vector{ITensor}(undef, d)
    for i in 2:d
        result *= G[i] * ITensor(basis[i]([elements[i]]), siteind(G, i))
    end
    return result[]
end

G, basis = para_sketch([-0.9 -0.8 -0.6; -0.3 0.1 0.6; -0.4 0.3 -0.7], [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)], "poly", 2, 4, 0.05)
result = eval(G, basis, [-0.9, -0.8, -0.6])
println(result)
result = eval(G, basis, [-0.95, -0.85, -0.65])
println(result)
