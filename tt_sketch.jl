using LegendrePolynomials
using ITensors
using LinearAlgebra
using ForwardDiff
using QuadGK

# function fourier_basis(x::Vector{Float64}, n::Int64, dom::Tuple{Float64, Float64})
#     L = (dom[2] - dom[1]) / 2
#     shift = (dom[2] + dom[1]) / 2
#     y = zeros(length(x), 2 * n + 1)
#     y0(x) = x > dom[1] && x < dom[2] ? 1 / sqrt(2 * L) : 0.0
#     y[:, 1] = y0.(x)
#     for i in 1:n
#         y1(x) = x > dom[1] && x < dom[2] ? sqrt(1 / L) * cos(pi * (x - shift) * i / L) : 0.0
#         y2(x) = x > dom[1] && x < dom[2] ? sqrt(1 / L) * sin(pi * (x - shift) * i / L) : 0.0
#         y[:, 2 * i] = y1.(x)
#         y[:, 2 * i + 1] = y2.(x)
#     end
#     return y
# end
    
# function legendre_basis(x::Vector{Float64}, n::Int64, dom::Tuple{Float64, Float64})
#     L = (dom[2] - dom[1]) / 2
#     shift = (dom[2] + dom[1]) / 2
#     y = zeros(length(x), n)
#     for i in 1:n
#         f(x) = x > dom[1] && x < dom[2] ? sqrt(i - 1 / 2) * Pl((x - shift) / L, i - 1) : 0.0
#         y[:, i] = f.(x)
#     end
#     return y
# end

# function fourier_d(x::Vector{Float64}, n::Int64, dom::Tuple{Float64, Float64})
#     L = (dom[2] - dom[1]) / 2
#     shift = (dom[2] + dom[1]) / 2
#     dy = zeros(length(x), 2 * n + 1)
#     for i in 1:n
#         y1(x) = sqrt(1 / L) * cos(pi * (x - shift) * i / L)
#         dy1(x) = x > dom[1] && x < dom[2] ? ForwardDiff.derivative(y1, x) : 0.0
#         y2(x) = sqrt(1 / L) * sin(pi * (x - shift) * i / L)
#         dy2(x) = x > dom[1] && x < dom[2] ? ForwardDiff.derivative(y2, x) : 0.0
#         dy[:, 2 * i] = dy1.(x)
#         dy[:, 2 * i + 1] = dy2.(x)
#     end
#     return dy
# end

# function legendre_d(x::Vector{Float64}, n::Int64, dom::Tuple{Float64, Float64})
#     L = (dom[2] - dom[1]) / 2
#     shift = (dom[2] + dom[1]) / 2
#     dy = zeros(length(x), n)
#     for i in 1:n
#         f(x) = sqrt(i - 1 / 2) * Pl((x - shift) / L, i - 1)
#         df(x) = x > dom[1] && x < dom[2] ? ForwardDiff.derivative(f, x) : 0.0
#         dy[:, i] = df.(x)
#     end
#     return dy
# end

function fourier_basis(x::Float64, pos::Int64, dom::Tuple{Float64, Float64})
    if x < dom[1] || x > dom[2]
        return 0.0
    end
    L = (dom[2] - dom[1]) / 2
    shift = (dom[2] + dom[1]) / 2
    if pos == 1
        return 1 / sqrt(2 * L)
    elseif pos % 2 == 0
        return sqrt(1 / L) * cos(pi * (x - shift) * div(pos, 2) / L)
    else
        return sqrt(1 / L) * sin(pi * (x - shift) * div(pos, 2) / L)
    end
end
    
function legendre_basis(x::Float64, pos::Int64, dom::Tuple{Float64, Float64})
    if x < dom[1] || x > dom[2]
        return 0.0
    end
    L = (dom[2] - dom[1]) / 2
    shift = (dom[2] + dom[1]) / 2
    xadj = (x - shift) / L
    if xadj < -1.0
        xadj = -1.0
    end
    if xadj > 1.0
        xadj = 1.0
    end
    return sqrt(pos / L - 1 / (2 * L)) * Pl(xadj, pos - 1)
end

function gaussian_basis(x::Float64, pos::Int64, dom::Tuple{Float64, Float64}, n::Int64)
    if x < dom[1] || x > dom[2]
        return 0.0
    end
    dx = (dom[2] - dom[1]) / (n - 2)
    # centers = dom[1]:dx:dom[2]
    centers = LinRange(dom[1], dom[2], n - 1)
    if pos == 1
        return 1.0
    else
        return exp(-(x - centers[pos - 1]) ^ 2 / (2 * (0.5*dx) ^ 2))
    end
end

function fourier_d(x::Float64, pos::Int64, dom::Tuple{Float64, Float64})
    if x < dom[1] || x > dom[2]
        return 0.0
    end
    L = (dom[2] - dom[1]) / 2
    shift = (dom[2] + dom[1]) / 2
    f(x) = if pos == 1
        1 / sqrt(2 * L)
    elseif pos % 2 == 0
        sqrt(1 / L) * cos(pi * (x - shift) * pos / L)
    else
        sqrt(1 / L) * sin(pi * (x - shift) * pos / L)
    end
    df(x) = ForwardDiff.derivative(f, x)
    return df(x)
end
    
function legendre_d(x::Float64, pos::Int64, dom::Tuple{Float64, Float64})
    if x < dom[1] || x > dom[2]
        return 0.0
    end
    L = (dom[2] - dom[1]) / 2
    shift = (dom[2] + dom[1]) / 2
    function f(x)
        xadj = (x - shift) / L
        if xadj < -1.0
            xadj = -1.0
        end
        if xadj > 1.0
            xadj = 1.0
        end
        return sqrt(pos / L - 1 / (2 * L)) * Pl(xadj, pos - 1)
    end
    df(x) = ForwardDiff.derivative(f, x)
    return df(x)
end

function gaussian_d(x::Float64, pos::Int64, dom::Tuple{Float64, Float64}, n::Int64)
    if x < dom[1] || x > dom[2]
        return 0.0
    end
    dx = (dom[2] - dom[1]) / (n - 2)
    # centers = dom[1]:dx:dom[2]
    centers = LinRange(dom[1], dom[2], n - 1)
    f(x) = if pos == 1
        1.0
    else
        exp(-(x - centers[pos - 1]) ^ 2 / (2 * dx ^ 2))
    end
    df(x) = ForwardDiff.derivative(f, x)
    return df(x)
end

function create_TT_coeff(n::Int64, d::Int64, r::Int64, a::Float64)
    sites = siteinds(n, d)
    coeff = randomMPS(sites; linkdims = r)
    # println(coeff)
    for i in 1:d
        # if i == 1
        #     coeff[1][:, :] = randn(n, r)
        #     # coeff[1][:, :] = 0.5 * ones(n, r)
        # elseif i == d
        #     coeff[d][:, :] = randn(r, n)
        #     # coeff[d][:, :] = 0.5 * ones(r, n)
        # else
        #     coeff[i][:, :, :] = randn(r, n, r)
        #     # coeff[i][:, :, :] = 0.5 * ones(r, n, r)
        # end
        A = diagITensor(a, sites[i], sites[i]')
        A[1, 1] = 1
        coeff[i] *= A
        noprime!(coeff[i])
    end
    return coeff
end

function int_basis_sample(basis, samples::Array{Float64, 2}, is::IndexSet, sample_weight::Vector{Float64}, nb::Int64)
    N, d = size(samples)
    is_new = siteinds(size(samples, 1), d)
    M = Vector{ITensor}(undef, d)
    for i in 1:d
        # M[i] = ITensor((sample_weight / sum(sample_weight)) .^ d .* basis[i](samples[:, i]), is_new[i], is[i])
        # M[i] = ITensor((sample_weight / sum(sample_weight)) .^ d .* basis[i].(samples[:, i], 1:nb), is_new[i], is[i])
        M[i] = ITensor((sample_weight * N / sum(sample_weight)) .^ d .* [basis[i](x, pos) for x in samples[:, i], pos in 1:nb], is_new[i], is[i])
        # M[i] = ITensor((sample_weight) .^ d .* [basis[i](x, pos) for x in samples[:, i], pos in 1:nb], is_new[i], is[i])
    end
    return M, is_new
end

function form_tensor_moment(M::Vector{ITensor}, coeff::MPS, is::IndexSet)
    d = length(M);
    N = size(M[1], 1);
    rc = linkdim(coeff, 1)
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

function para_sketch(samples::Array{Float64, 2}, domain::Vector{Tuple{Float64, Float64}}, basis_type::String, r::Int64, rc::Int64, alpha::Float64, nb::Int64, sample_weight::Vector{Float64})
    d = size(samples, 2)
    # basis, basis_d, nb = if basis_type == "fourier"
    #     ([b(x) = fourier_basis(x, nb0, domain[i]) for i in 1:d], [db(x) = fourier_d(x, nb0, domain[i]) for i in 1:d], 2 * nb0 + 1)
    # elseif basis_type == "poly"
    #     ([b(x) = legendre_basis(x, nb0, domain[i]) for i in 1:d], [b(x) = legendre_d(x, nb0, domain[i]) for i in 1:d], nb0)
    # end
    basis, basis_d = if basis_type == "fourier"
        ([b(x, pos) = fourier_basis(x, pos, domain[i]) for i in 1:d], [db(x, pos) = fourier_d(x, pos, domain[i]) for i in 1:d])
    elseif basis_type == "poly"
        ([b(x, pos) = legendre_basis(x, pos, domain[i]) for i in 1:d], [db(x, pos) = legendre_d(x, pos, domain[i]) for i in 1:d])
    elseif basis_type == "gaussian"
        ([b(x, pos) = gaussian_basis(x, pos, domain[i], nb) for i in 1:d], [db(x, pos) = gaussian_d(x, pos, domain[i], nb) for i in 1:d])
    end

    coeff = create_TT_coeff(nb, d, rc, alpha)
    M, is = int_basis_sample(basis, samples, siteinds(coeff), sample_weight, nb)
    # for i in 1:d
    #     M[i] .*= sample_weight .^ (1 / d) / sum(sample_weight)
    # end
    G = Vector{ITensor}(undef, d)

    Bemp, envi_L, envi_R = form_tensor_moment(M, coeff, is)
    V = Vector{ITensor}(undef, d)
    for core_id in 1:d
        if core_id == 1
            G[1] = Bemp[1]
        else
            l = linkind(coeff, core_id - 1)
            A = envi_L[core_id]' * envi_R[core_id - 1]
            G[core_id] = ITensor(pinv(A), l', l) * Bemp[core_id]
            noprime!(G[core_id])
            _, _, V[core_id] = svd(ITensor(A, l', l), l', maxdim = r, righttags = tags(l))
        end
    end

    for core_id in 1:d
        basis_int = zeros(nb, nb)
        for s in 1:nb
            for t in s:nb
                f(x) = basis[core_id](x, s) * basis[core_id](x, t)
                basis_int[s, t] = basis_int[t, s] = quadgk(f, domain[core_id]...)[1]
            end
        end
        # display(basis_int)
        G[core_id] *= ITensor(inv(basis_int), siteind(coeff, core_id), siteind(coeff, core_id)')
        noprime!(G[core_id])
        if core_id == 1
            G[1] *= V[2]
        elseif core_id == d
            G[d] *= V[d]
        else
            G[core_id] *= V[core_id]
            G[core_id] *= V[core_id + 1]
        end
    end

    return MPS(G), basis, basis_d
end

function dens_eval(G::MPS, basis, elements::Vector{Float64})
    d = length(elements)
    s = siteinds(G)
    result = G[1] * ITensor(basis[1].(elements[1], 1:ITensors.dim(s[1])), s[1])
    for i in 2:d
        result *= G[i] * ITensor(basis[i].(elements[i], 1:ITensors.dim(s[i])), s[i])
    end
    return result[]
end

function dens_grad(G::MPS, basis, basis_d, elements::Vector{Float64})
    d = length(elements)
    s = siteinds(G)
    grad = zeros(d)
    # display(basis)
    # display(basis_d)
    for k in 1:d
        result = G[1] * ITensor((k == 1 ? basis_d : basis)[1].(elements[1], 1:ITensors.dim(s[1])), s[1])
        for i in 2:d
            result *= G[i] * ITensor((k == i ? basis_d : basis)[i].(elements[i], 1:ITensors.dim(s[i])), s[i])
        end
        grad[k] = result[]
    end
    return grad
end

# G, basis, basis_d = para_sketch([-0.9 -0.8 -0.6; -0.3 0.1 0.6; -0.4 0.3 -0.7], [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)], "gaussian", 2, 4, 0.05, 21, [1.0, 1.0, 1.0])
# result = dens_eval(G, basis, [-0.9, -0.8, -0.6])
# println(result)
# result = dens_eval(G, basis, [-0.95, -0.85, -0.65])
# println(result)
# result = dens_grad(G, basis, basis_d, [-0.9, -0.8, -0.6])
# println(result)
# result = dens_grad(G, basis, basis_d, [-0.95, -0.85, -0.65])
# println(result)
