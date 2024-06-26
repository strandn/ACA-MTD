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
    nbins_(0),
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
            // println(j);
            // println(k);
            GSLParams gsl_params = { this, j + 1, k + 1 };
            gsl_function F;
            F.function = &f;
            F.params = &gsl_params;
            gsl_integration_qag(&F, dom.first, dom.second, 1.0e-10, 1.0e-6, 1000, 2, workspace, &result, &error);
            grid_[j][k] = result;
            // println("grad");
            // println();
            gsl_function DF;
            DF.function = &df;
            DF.params = &gsl_params;
            gsl_integration_qag(&DF, dom.first, dom.second, 1.0e-10, 1.0e-6, 1000, 2, workspace, &result, &error);
            gridd_[j][k] = result;
            }
        }
    gsl_integration_workspace_free(workspace);

    for(auto i : range(nbins_))
        {
        xdata_[i] = dom_.first + i * (dom_.second - dom_.first) / (nbins_ - 1);
        }
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
    // println(x);
    // println(gsl_params->j);
    // println(gsl_params->k);
    // println();
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
paraSketch(std::vector<std::vector<Real>> const& samples, std::vector<std::pair<Real, Real>> const& domain, std::vector<BasisFunc> const& basis, int rc)
    {
    assert(samples.size() > 0);
    int N = samples.size();
    int d = samples[0].size();
    
    assert(basis.size() > 0);
    int nb = basis[0].nbasis();
    auto coeff = createTTCoeff(nb, d, rc);
    auto result1 = intBasisSample(basis, samples, siteInds(coeff));
    auto M = result1.first;
    auto is = result1.second;
    MPS G(d);

    auto result2 = formTensorMoment(M, coeff, is);
    auto Bemp = std::get<0>(result2);
    auto envi_L = std::get<1>(result2);
    auto envi_R = std::get<2>(result2);
    auto links = linkInds(coeff);
    std::vector<ITensor> V(d);
    for(auto core_id : range1(d))
        {
        if(core_id == 1)
            {
            G.ref(1) = Bemp(1);
            }
        else
            {
            Eigen::MatrixXd LMat(N, rc), RMat(N, rc);
            for(auto i : range1(N))
                {
                for(auto j : range1(rc))
                    {
                    LMat(i - 1, j - 1) = envi_L[core_id - 1].elt(is(core_id) = i, links(core_id - 1) = j);
                    RMat(i - 1, j - 1) = envi_R[core_id - 2].elt(is(core_id - 1) = i, links(core_id - 1) = j);
                    }
                }
            Eigen::MatrixXd AMat = LMat.transpose() * RMat;
            Eigen::MatrixXd PMat = AMat.completeOrthogonalDecomposition().pseudoInverse();
            ITensor A(prime(links(core_id - 1)), links(core_id - 1)), Pinv(prime(links(core_id - 1)), links(core_id - 1));
            for(auto i : range1(rc))
                {
                for(auto j : range1(rc))
                    {
                    A.set(prime(links(core_id - 1)) = i, links(core_id - 1) = j, AMat(i - 1, j - 1));
                    Pinv.set(prime(links(core_id - 1)) = i, links(core_id - 1) = j, PMat(i - 1, j - 1));
                    }
                }
            G.ref(core_id) = noPrime(Pinv * Bemp(core_id));
            auto original_link_tags = tags(links(core_id - 1));
            ITensor U, S;
            V[core_id - 1] = ITensor(links(core_id - 1));
            svd(A, U, S, V[core_id - 1], {"Cutoff=", 1.0e-6, "RightTags=", original_link_tags});
            }
        println(core_id);
        PrintData(V[core_id - 1]);
        }
    PrintData(linkInds(G));
    PrintData(G);

    for(auto core_id : range1(d))
        {
        if(core_id == 1)
            {
            G.ref(1) *= V[1];
            }
        else if(core_id == d)
            {
            G.ref(d) *= V[d - 1];
            }
        else
            {
            G.ref(core_id) *= V[core_id - 1];
            G.ref(core_id) *= V[core_id];
            }
        }
    PrintData(linkInds(G));
    PrintData(G);

    return G;
    }

MPS
createTTCoeff(int n, int d, int r)
    {
    auto sites = SiteSet(d, n);
    auto coeff = randomMPS(sites, r);
    // PrintData(coeff);
    Real alpha = 0.05;
    for(auto i : range1(d))
        {
        coeff.ref(i).fill(0.5);
        auto s = sites(i);
        auto sp = prime(s);
        std::vector<Real> Avec(n, alpha);
        Avec[0] = 1.0;
        auto A = diagITensor(Avec, s, sp);
        coeff.ref(i) *= A;
        coeff.ref(i) = noPrime(coeff(i));
        }
    // PrintData(coeff);
    return coeff;
    }

std::pair<std::vector<ITensor>, IndexSet>
intBasisSample(std::vector<BasisFunc> const& basis, std::vector<std::vector<Real>> const& samples, IndexSet const& is)
    {
    int N = samples.size();
    int d = samples[0].size();
    int nb = basis[0].nbasis();
    auto sites_new = SiteSet(d, N);
    std::vector<ITensor> M;
    std::vector<Index> is_new;
    for(auto i : range1(d))
        {
        M.push_back(ITensor(sites_new(i), is(i)));
        is_new.push_back(sites_new(i));
        for(auto j : range1(N))
            {
            for(auto k : range1(nb)) M.back().set(sites_new(i) = j, is(i) = k, pow(1.0 / N, 1.0 / d) * basis[i - 1](samples[j - 1][i - 1], k));
            }
        // println(i);
        // PrintData(M.back());
        // PrintData(is_new.back());
        }
    return make_pair(M, IndexSet(is_new));
    }

std::tuple<MPS, std::vector<ITensor>, std::vector<ITensor>>
formTensorMoment(std::vector<ITensor> const& M, MPS const& coeff, IndexSet const& is)
    {
    int d = M.size();
    int N = dim(is(1));
    auto links = linkInds(coeff);
    int r = dim(links(1));
    auto L = coeff;

    for(auto i : range1(d))
        {
        L.ref(i) *= M[i - 1];
        }

    std::vector<ITensor> envi_L(d);
    envi_L[1] = L(1) * delta(is(1), is(2));
    for(int i = 2; i < d; ++i)
        {
        envi_L[i] = ITensor(is(i + 1), links(i));
        for(auto j : range1(N))
            {
            for(auto k : range1(r))
                {
                ITensor LHS(links(i - 1)), RHS(links(i - 1));
                for(auto ii : range1(r))
                    {
                    LHS.set(links(i - 1) = ii, envi_L[i - 1].elt(is(i) = j, links(i - 1) = ii));
                    RHS.set(links(i - 1) = ii, L(i).elt(links(i - 1) = ii, is(i) = j, links(i) = k));
                    }
                envi_L[i].set(is(i + 1) = j, links(i) = k, elt(LHS * RHS));
                }
            }
        }

    std::vector<ITensor> envi_R(d);
    envi_R[d - 2] = L(d) * delta(is(d), is(d - 1));
    for(int i = d - 3; i >= 0; --i)
        {
        envi_R[i] = ITensor(is(i + 1), links(i + 1));
        for(auto j : range1(N))
            {
            for(auto k : range1(r))
                {
                ITensor LHS(links(i + 2)), RHS(links(i + 2));
                for(auto ii : range1(r))
                    {
                    LHS.set(links(i + 2) = ii, envi_R[i + 1].elt(is(i + 2) = j, links(i + 2) = ii));
                    RHS.set(links(i + 2) = ii, L(i + 2).elt(links(i + 2) = ii, is(i + 2) = j, links(i + 1) = k));
                    }
                envi_R[i].set(is(i + 1) = j, links(i + 1) = k, elt(LHS * RHS));
                }
            }
        }

    MPS B(d);
    for(auto core_id : range1(d))
        {
        if(core_id == 1)
            {
            // PrintData(envi_R[0]);
            // PrintData(M[0]);
            B.ref(1) = envi_R[0] * M[0];
            }
        else if(core_id == d)
            {
            // PrintData(envi_L[d - 1]);
            // PrintData(M[d - 1]);
            B.ref(d) = envi_L[d - 1] * M[d - 1];
            }
        else
            {
            B.ref(core_id) = ITensor(links(core_id - 1), is(core_id), links(core_id));
            for(auto i : range1(r))
                {
                for(auto j : range1(r))
                    {
                    for(auto k : range1(N))
                        {
                        Real Lelt = envi_L[core_id - 1].elt(is(core_id) = k, links(core_id - 1) = i);
                        Real Relt = envi_R[core_id - 1].elt(is(core_id) = k, links(core_id) = j);
                        B.ref(core_id).set(links(core_id - 1) = i, is(core_id) = k, links(core_id) = j, Lelt * Relt);
                        }
                    }
                }
            B.ref(core_id) *= M[core_id - 1];
            }
        // println(core_id);
        // PrintData(envi_L[core_id - 1]);
        // PrintData(envi_R[core_id - 1]);
        }
    
    // PrintData(B);
    return std::make_tuple(B, envi_L, envi_R);
    }

} // namespace itensor
