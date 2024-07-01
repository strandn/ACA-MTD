#include <cmath>
#include <gsl/gsl_integration.h>
#include <Eigen/Dense>
#include <Eigen/QR>
#include "itensor/util/print_macro.h"
#include "tt_sketch.h"

namespace itensor {

struct GSLParams { BasisFunc* instance; int j; int k; };

BasisFunc::
BasisFunc() 
    : 
    dom_(std::make_pair(-1.0, 1.0)),
    nbasis_(20),
    conv_(false),
    nbins_(100),
    L_(1.0),
    shift_(0.0)
    { }

BasisFunc::
BasisFunc(std::pair<Real, Real> dom, int nbasis) 
    : 
    dom_(dom),
    nbasis_(nbasis),
    conv_(false),
    nbins_(100),
    L_((dom.second - dom.first) / 2),
    shift_((dom.second + dom.first) / 2),
    grid_(nbasis, std::vector<Real>(100, 0.0)),
    gridd_(nbasis, std::vector<Real>(100, 0.0)),
    xdata_(100, 0.0)
    {
    gsl_integration_workspace* workspace = gsl_integration_workspace_alloc(1000);
    Real result, error;
    for(auto j : range(nbasis_))
        {
        for(auto k : range(nbins_))
            {
            GSLParams gsl_params = { this, j + 1, k + 1 };
            gsl_function F;
            F.function = &f;
            F.params = &gsl_params;
            gsl_integration_qag(&F, dom.first, dom.second, 1.0e-10, 1.0e-6, 1000, 2, workspace, &result, &error);
            grid_[j][k] = result;
            gsl_function DF;
            DF.function = &df;
            DF.params = &gsl_params;
            gsl_integration_qag(&DF, dom.first, dom.second, 1.0e-10, 1.0e-6, 1000, 2, workspace, &result, &error);
            gridd_[j][k] = result;
            }
        }
    gsl_integration_workspace_free(workspace);

    for(auto i : range(nbins_)) xdata_[i] = dom_.first + i * (dom_.second - dom_.first) / (nbins_ - 1);
    }

Real BasisFunc::
fourier(Real x, int pos) const
    {
    if(x < dom_.first || x > dom_.second)
        {
        return 0.0;
        }
    if(pos == 1)
        {
        return 1 / std::sqrt(2 * L_);
        }
    else if(pos % 2 == 0)
        {
        return std::sqrt(1 / L_) * cos(M_PI * (x - shift_) * (pos / 2) / L_);
        }
    else
        {
        return std::sqrt(1 / L_) * sin(M_PI * (x - shift_) * (pos / 2) / L_);
        }
    }

Real BasisFunc::
operator()(Real x, int pos) const
    {
    if(conv_)
        {
        return interpolate(x, pos, false);
        }
    else
        {
        return fourier(x, pos);
        }
    }

Real BasisFunc::
grad(Real x, int pos) const
    {
    if(conv_)
        {
        return interpolate(x, pos, true);
        }
    else
        {
        if(x < dom_.first || x > dom_.second)
            {
            return 0.0;
            }
        if(pos == 1)
            {
            return 0.0;
            }
        else if(pos % 2 == 0)
            {
            return -pow(1 / L_, 3 / 2) * M_PI * (pos / 2) * sin(M_PI * (x - shift_) * (pos / 2) / L_);
            }
        else
            {
            return pow(1 / L_, 3 / 2) * M_PI * (pos / 2) * cos(M_PI * (x - shift_) * (pos / 2) / L_);
            }
        }
    }

Real BasisFunc::
interpolate(Real x, int pos, bool grad) const
    {
    int i = 0;
    if(x >= xdata_[nbins_ - 2])
        {
        i = nbins_ - 2;
        }
    else
        {
        while(x > xdata_[i + 1]) ++i;
        }
    Real xL = xdata_[i];
    Real yL = grad ? gridd_[pos - 1][i] : grid_[pos - 1][i];
    Real xR = xdata_[i + 1];
    Real yR = grad ? gridd_[pos - 1][i + 1] : grid_[pos - 1][i + 1];
    assert(x >= xL && x <= xR);
    Real dydx = (yR - yL) / (xR - xL);
    return yL + dydx * (x - xL);
    }

Real
f(Real x, void* params)
    {
    GSLParams* gsl_params = (GSLParams*)params;
    Real w = 0.02;
    Real sigma = w * (gsl_params->instance->dom().second - gsl_params->instance->dom().first);
    Real s = gsl_params->instance->dom().first + (gsl_params->k - 1) * (gsl_params->instance->dom().second - gsl_params->instance->dom().first) / (gsl_params->instance->nbins() - 1);
    return gsl_params->instance->fourier(x, gsl_params->j) * (1 / (std::sqrt(2 * M_PI) * sigma)) * exp(-pow(s - x, 2) / (2 * pow(sigma, 2)));
    }

Real
df(Real x, void* params)
    {
    GSLParams* gsl_params = (GSLParams*)params;
    Real w = 0.02;
    Real sigma = w * (gsl_params->instance->dom().second - gsl_params->instance->dom().first);
    Real s = gsl_params->instance->dom().first + (gsl_params->k - 1) * (gsl_params->instance->dom().second - gsl_params->instance->dom().first) / (gsl_params->instance->nbins() - 1);
    return gsl_params->instance->fourier(x, gsl_params->j) * ((x - s) / (std::sqrt(2 * M_PI) * pow(sigma, 3))) * exp(-pow(s - x, 2) / (2 * pow(sigma, 2)));
    }

MPS
paraSketch(std::vector<std::vector<Real>> const& samples, std::vector<BasisFunc> const& basis, int rc)
    {
    assert(samples.size() > 0);
    int N = samples.size();
    int d = samples[0].size();
    
    assert(basis.size() > 0);
    int nb = basis[0].nbasis();
    auto coeff = createTTCoeff(nb, d, rc);
    auto result1 = intBasisSample(basis, samples, coeff.sites());
    auto M = result1.first;
    auto is = result1.second;
    MPS G(d);

    auto result2 = formTensorMoment(M, coeff, is);
    auto Bemp = std::get<0>(result2);
    auto envi_L = std::get<1>(result2);
    auto envi_R = std::get<2>(result2);
    // auto links = linkInds(coeff);
    std::vector<ITensor> V(d);
    for(auto core_id : range1(d))
        {
        if(core_id == 1)
            {
            G.Aref(1) = Bemp.A(1);
            }
        else
            {
            Eigen::MatrixXd LMat(N, rc), RMat(N, rc);
            auto l = linkInd(coeff, core_id - 1);
            for(auto i : range1(N))
                {
                for(auto j : range1(rc))
                    {
                    LMat(i - 1, j - 1) = envi_L[core_id - 1].real(IndexVal(is(core_id), i), IndexVal(l, j));
                    RMat(i - 1, j - 1) = envi_R[core_id - 2].real(IndexVal(is(core_id - 1), i), IndexVal(l, j));
                    }
                }
            Eigen::MatrixXd AMat = LMat.transpose() * RMat;
            Eigen::MatrixXd PMat = AMat.completeOrthogonalDecomposition().pseudoInverse();
            ITensor A(prime(l), l), Pinv(prime(l), l);
            for(auto i : range1(rc))
                {
                for(auto j : range1(rc))
                    {
                    A.set(IndexVal(prime(l), i), IndexVal(l, j), AMat(i - 1, j - 1));
                    Pinv.set(IndexVal(prime(l), i), IndexVal(l, j), PMat(i - 1, j - 1));
                    }
                }
            G.Aref(core_id) = Pinv * Bemp.A(core_id);
            // G.Aref(core_id) *= delta(prime(l), l);
            G.Aref(core_id).noprime();
            auto original_link_name = l.name();
            ITensor U, S;
            V[core_id - 1] = ITensor(l);
            svd(A, U, S, V[core_id - 1], {"Cutoff=", 1.0e-6, "RightIndexName=", original_link_name});
            }
        }
    for(auto i : range1(d - 1)) print(linkInd(G, i), " ");
    println();

    for(auto core_id : range1(d))
        {
        if(core_id == 1)
            {
            G.Aref(1) *= V[1];
            }
        else if(core_id == d)
            {
            G.Aref(d) *= V[d - 1];
            }
        else
            {
            G.Aref(core_id) *= V[core_id - 1];
            G.Aref(core_id) *= V[core_id];
            }
        }
    for(auto i : range1(d - 1)) print(linkInd(G, i), " ");
    println();

    return G;
    }

MPS
createTTCoeff(int n, int d, int r)
    {
    SiteSet sites(d, n);
    // auto coeff = randomMPS(sites, r);
    MPS coeff(d);
    std::vector<Index> a(d - 1);
    for(auto i : range1(d - 1)) a[i] = Index(nameint("a", i));
    for(auto i : range1(d))
        {
        if(i == 1)
            {
            coeff.Aref(1) = ITensor(sites(1), a[1]);
            }
        else if (i == d)
            {
            coeff.Aref(i) = ITensor(dag(a[i - 1]), sites(i), a[i]);
            }
        else
            {
            coeff.Aref(d) = ITensor(dag(a[d - 1]), sites(d));
            }
        randomize(coeff.Aref(i));
        }
    Real alpha = 0.05;
    for(auto i : range1(d))
        {
        auto s = sites(i);
        auto sp = prime(s);
        std::vector<Real> Avec(n, alpha);
        Avec[0] = 1.0;
        auto A = diagTensor(Avec, s, sp);
        coeff.Aref(i) *= A;
        // coeff.Aref(i) *= delta(s, sp);
        coeff.Aref(i).noprime();
        }
    return coeff;
    }

std::pair<std::vector<ITensor>, SiteSet>
intBasisSample(std::vector<BasisFunc> const& basis, std::vector<std::vector<Real>> const& samples, SiteSet const& is)
    {
    int N = samples.size();
    int d = samples[0].size();
    int nb = basis[0].nbasis();
    SiteSet sites_new(d, N);
    std::vector<ITensor> M;
    std::vector<Index> is_new;
    for(auto i : range1(d))
        {
        M.push_back(ITensor(sites_new(i), is(i)));
        is_new.push_back(sites_new(i));
        for(auto j : range1(N))
            {
            for(auto k : range1(nb))
                {
                Real basisval = pow(1.0 / N, 1.0 / d) * basis[i - 1](samples[j - 1][i - 1], k);
                M.back().set(IndexVal(sites_new(i), j), IndexVal(is(i), k), basisval);
                }
            }
        }
    return std::make_pair(M, is_new);
    }

std::tuple<MPS, std::vector<ITensor>, std::vector<ITensor>>
formTensorMoment(std::vector<ITensor> const& M, MPS const& coeff, SiteSet const& is)
    {
    int d = M.size();
    int N = dim(is(1));
    // auto links = linkInds(coeff);
    int r = dim(linkInd(coeff, 1));
    auto L = coeff;

    for(auto i : range1(d)) L.Aref(i) *= M[i - 1];

    std::vector<ITensor> envi_L(d);
    envi_L[1] = L.A(1) * delta(is(1), is(2));
    for(int i = 2; i < d; ++i)
        {
        envi_L[i] = ITensor(is(i + 1), linkInd(coeff, i));
        for(auto j : range1(N))
            {
            for(auto k : range1(r))
                {
                ITensor LHS(linkInd(coeff, i - 1)), RHS(linkInd(coeff, i - 1));
                for(auto ii : range1(r))
                    {
                    LHS.set(IndexVal(linkInd(coeff, i - 1), ii), envi_L[i - 1].real(IndexVal(is(i), j), IndexVal(linkInd(coeff, i - 1), ii)));
                    RHS.set(IndexVal(linkInd(coeff, i - 1), ii), L.A(i).real(IndexVal(linkInd(coeff, i - 1), ii), IndexVal(is(i), j), IndexVal(linkInd(coeff, i), k)));
                    }
                auto next = LHS * RHS;
                envi_L[i].set(IndexVal(is(i + 1), j), IndexVal(linkInd(coeff, i), k), next.real());
                }
            }
        }

    std::vector<ITensor> envi_R(d);
    envi_R[d - 2] = L.A(d) * delta(is(d), is(d - 1));
    for(int i = d - 3; i >= 0; --i)
        {
        envi_R[i] = ITensor(is(i + 1), linkInd(coeff, i + 1));
        for(auto j : range1(N))
            {
            for(auto k : range1(r))
                {
                ITensor LHS(linkInd(coeff, i + 2)), RHS(linkInd(coeff, i + 2));
                for(auto ii : range1(r))
                    {
                    LHS.set(IndexVal(linkInd(coeff, i + 2), ii), envi_R[i + 1].real(IndexVal(is(i + 2), j), IndexVal(linkInd(coeff, i + 2), ii)));
                    RHS.set(IndexVal(linkInd(coeff, i + 2), ii), L.A(i + 2).real(IndexVal(linkInd(coeff, i + 2), ii), IndexVal(is(i + 2), j), IndexVal(linkInd(coeff, i + 1), k)));
                    }
                auto next = LHS * RHS;
                envi_R[i].set(IndexVal(is(i + 1), j), IndexVal(linkInd(coeff, i + 1), k), next.real());
                }
            }
        }

    MPS B(d);
    for(auto core_id : range1(d))
        {
        if(core_id == 1)
            {
            B.Aref(1) = envi_R[0] * M[0];
            }
        else if(core_id == d)
            {
            B.Aref(d) = envi_L[d - 1] * M[d - 1];
            }
        else
            {
            B.Aref(core_id) = ITensor(linkInd(coeff, core_id - 1), is(core_id), linkInd(coeff, core_id));
            for(auto i : range1(r))
                {
                for(auto j : range1(r))
                    {
                    for(auto k : range1(N))
                        {
                        Real Lelt = envi_L[core_id - 1].real(IndexVal(is(core_id), k), IndexVal(linkInd(coeff, core_id - 1), i));
                        Real Relt = envi_R[core_id - 1].real(IndexVal(is(core_id), k), IndexVal(linkInd(coeff, core_id), j));
                        B.Aref(core_id).set(IndexVal(linkInd(coeff, core_id - 1), i), IndexVal(is(core_id), k), IndexVal(linkInd(coeff, core_id), j), Lelt * Relt);
                        }
                    }
                }
            B.Aref(core_id) *= M[core_id - 1];
            }
        }
    
    return std::make_tuple(B, envi_L, envi_R);
    }

Real
densEval(MPS const& G, std::vector<BasisFunc> const& basis, std::vector<Real> const& elements)
    {
    int d = elements.size();
    // auto s = siteInds(G);
    std::vector<ITensor> basis_evals(d);
    for(auto i : range1(d))
        {
        auto s = G.sites()(i);
        basis_evals[i - 1] = ITensor(s);
        for(auto j : range1(dim(s))) basis_evals[i - 1].set(IndexVal(s, j), basis[i - 1](elements[i - 1], j));
        }
    auto result = G.A(1) * basis_evals[0];
    for(int i = 2; i <= d; ++i) result *= G.A(i) * basis_evals[i - 1];
    return result.real();
    }

std::vector<Real>
densGrad(MPS const& G, std::vector<BasisFunc> const& basis, std::vector<Real> const& elements)
    {
    int d = elements.size();
    // auto s = siteInds(G);
    std::vector<Real> grad(d, 0.0);
    std::vector<ITensor> basis_evals(d), basisd_evals(d);
    for(auto i : range1(d))
        {
        auto s = G.sites()(i);
        basis_evals[i - 1] = basisd_evals[i - 1] = ITensor(s);
        for(auto j : range1(dim(s))) basis_evals[i - 1].set(IndexVal(s, j), basis[i - 1](elements[i - 1], j));
        for(auto j : range1(dim(s))) basisd_evals[i - 1].set(IndexVal(s, j), basis[i - 1].grad(elements[i - 1], j));
        }
    for(auto k : range1(d))
        {
        auto result = G.A(1) * (k == 1 ? basisd_evals[0] : basis_evals[0]);
        for(int i = 2; i <= d; ++i) result *= G.A(i) * (k == i ? basisd_evals[i - 1] : basis_evals[i - 1]);
        grad[k - 1] = result.real();
        }
    return grad;
    }

} // namespace itensor
