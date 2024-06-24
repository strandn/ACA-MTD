#include <cmath>
#include <gsl/gsl_integration.h>
#include "tt_sketch.h"

namespace itensor {

using std::pair;
using std::make_pair;
using std::tuple

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
    L_((dom[1] - dom[0]) / 2),
    shift_((dom[1] + dom[0]) / 2)
    {
    grid_.resize(nbins_, std::vector<Real>(nbasis_, 0.0));
    gridd_.resize(nbins_, std::vector<Real>(nbasis_, 0.0));
    gsl_integration_workspace* workspace = gsl_integration_workspace_alloc(1000);
    Real result, error;
    for(auto j : range(nbasis_))
        {
        for(auto k : range(nbins_))
            {
            int jk[2] = {j, k}
            gsl_function F;
            F.function = &f;
            F.params = &jk;
            gsl_integration_qag(&F, dom[0], dom[1], 0, 1e-7, 1000, 6, workspace, &result, &error);
            grid_[j][k] = result;
            gsl_function DF;
            F.function = &df;
            F.params = &jk;
            gsl_integration_qag(&DF, dom[0], dom[1], 0, 1e-7, 1000, 6, workspace, &result, &error);
            gridd_[j][k] = result;
            }
        }
    }

Real BasisFunc::
fourier(Real x, int pos) const
    {
    if(x < dom_[0] || x > dom_[1])
        {
        return 0.0;
        }
    if(pos == 0)
        {
        return 1 / sqrt(2 * L_);
        }
    else if(pos % 2 == 0)
        {
        return sqrt(1 / L_) * sin(M_PI * (x - shift_) * ((pos + 1) / 2) / L_);
        }
    else
        {
        return sqrt(1 / L_) * cos(M_PI * (x - shift_) * ((pos + 1) / 2) / L_);
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
        return fourier(x, pos)
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
        if(x < dom_[0] || x > dom_[1])
            {
            return 0.0;
            }
        if(pos == 0)
            {
            return 1 / sqrt(2 * L_);
            }
        else if(pos % 2 == 0)
            {
            return pow(1 / L_, 3 / 2) * M_PI * ((pos + 1) / 2) * cos(M_PI * (x - shift_) * ((pos + 1) / 2) / L_);
            }
        else
            {
            return -pow(1 / L_, 3 / 2) * M_PI * ((pos + 1) / 2) * sin(M_PI * (x - shift_) * ((pos + 1) / 2) / L_);
            }
        }
    }

Real BasisFunc::
f(Real x, void* params) const
    {
    int* jk = (int*)params;
    int j = jk[0];
    int k = jk[1];
    w = 0.02;
    sigma = w * (dom_[1] - dom_[0]);
    s = dom_[0] + k * (dom_[1] - dom_[0]) / (nbins_ - 1);
    return fourier(x, j) * (1 / (sqrt(2 * M_PI) * sigma)) * exp(-pow(s - x, 2) / (2 * pow(sigma, 2)));
    }

Real BasisFunc::
df(Real x, void* params) const
    {
    int* jk = (int*)params;
    int j = jk[0];
    int k = jk[1];
    w = 0.02;
    sigma = w * (dom_[1] - dom_[0]);
    s = dom_[0] + k * (dom_[1] - dom_[0]) / (nbins_ - 1);
    return fourier(x, j) * ((x - s) / (sqrt(2 * M_PI) * pow(sigma, 3))) * exp(-pow(s - x, 2) / (2 * pow(sigma, 2)));
    }

Real BasisFunc::
interpolate(Real x, int pos, bool grad) const
    {
    std::vector<Real> xdata(nbins_, 0.0);
    for(auto i : range(nbins_))
        {
        xdata[i] = dom_[0] + i * (dom_[1] - dom_[0]) / (nbins_ - 1);
        }
    int i = 0;
    if(x >= xdata[size - 2])
        {
        i = size - 2;
        }
    else
        {
        while(x > xdata[i + 1]) ++i;
        }
    Real xL = xdata[i];
    Real yL = grad ? gridd_[pos][i] : grid_[pos][i];
    Real xR = xdata[i + 1];
    Real yR = grad ? gridd_[pos][i + 1] : grid_[pos][i + 1];
    assert(x >= xL && x <= xR);
    Real dydx = (yR - yL) / (xR - xL);
    return yL + dydx * (x - xL);
    }

MPS
paraSketch(std::vector<std::vector<Real>> const& samples, std::vector<std::pair<Real, Real>> const& domain, std::vector<BasisFunc> const& basis, int rc, int nb)
    {
    assert(samples.size() > 0);
    int d = samples[0].size();
    
    auto coeff = createTTCoeff(nb, d, rc);
    auto result = intBasisSample(basis, samples, siteinds(coeff), nb);
    M = get<0>(result);
    is = get<1>(result);
    auto G = MPS(d);
    }

MPS
createTTCoeff(int n, int d, int r)
    {
    auto sites = SiteSet(n, d);
    auto coeff = randomMPS(sites);
    Real alpha = 0.05;
    for(auto i : range1(d))
        {
        auto s = sites(i);
        auto sp = prime(s);
        auto A = diagITensor(std::vector<Real>(n, alpha), s, sp);
        A.set(s = 1, sp = 1, 1.0);
        coeff.ref(i) *= A;
        noprime(coeff.ref(i));
        }
    return coeff;
    }

std::pair<std::vector<ITensor>, IndexSet>
intBasisSample(std::vector<BasisFunc> const& basis, std::vector<std::vector<Real>> const& samples, IndexSet const& is, int nb)
    {
    int N = samples.size();
    int d = samples[0].size();
    auto sites_new = SiteSet(N, d);
    std::vector<ITensor> M;
    std::vector<Index> is_new;
    for(auto i : range1(d))
        {
        M.push_back(ITensor(sites_new(i), is(i)));
        is_new.push_back(sites_new(i));
        for(auto j : range1(N))
            {
            for(auto k : range1(nb)) M.back().set(sites_new(i) = j, is(i) = k, pow(1.0 / N, 1 / d) * basis[i - 1](samples[j][i], k));
            }
        }
    return make_pair(M, IndexSet(is_new));
    }

std::tuple<MPS, Matrix, Matrix>
formTensorMoment(std::vector<ITensor> const& M, MPS const& coeff, IndexSet const& is)
    {
    int d = M.size();
    int N = dim(is(1));
    auto links = linkInds(coeff);
    int r = dim(links(1));
    auto L = coeff;

    for(auto i : range1(d))
        {
        L.ref(i) *= M(i);
        }
    }

    // envi_L = Vector{Matrix}(undef, d)
    // envi_L[2] = matrix(L[1], is[1], linkind(coeff, 1))
    // for i in 3:d
    //     L_arr = array(L[i - 1], linkind(coeff, i - 2), is[i - 1], linkind(coeff, i - 1))
    //     envi_L[i] = zeros(N, rs[i - 1])
    //     for j in 1:N
    //         envi_L[i][j, :] = envi_L[i - 1][j, :]' * L_arr[:, j, :]
    //     end
    // end

    // envi_R = Vector{Matrix}(undef, d)
    // envi_R[d - 1] = matrix(L[d], is[d], linkind(coeff, d - 1))
    // for i in d-2:-1:1
    //     L_arr = array(L[i + 1], linkind(coeff, i + 1), is[i + 1], linkind(coeff, i))
    //     envi_R[i] = zeros(N, rs[i])
    //     for j in 1:N
    //         envi_R[i][j, :] = envi_R[i + 1][j, :]' * L_arr[:, j, :]
    //     end
    // end

    // std::vector<Eigen::MatrixXd> envi_L(d);
    // envi_L[1] = Eigen::MatrixXd(dim(is(1)), dim(coeff.linkInd(1)));
    // for(auto j : range1(dim(is(1))))
    //     {
    //     for(auto k : range1(dim(coeff.linkInd(1))))
    //         {
    //         envi_L[1](j, k) = L(1).elt(is(1) = j, coeff.linkInd(1) = k);
    //         }
    //     }
    std::vector<ITensor> envi_L(d);
    envi_L[1] = L(1);
    for(int i = 2; i < d; ++i)
        {
        envi_L[i] = ITensor(is(i), links(i));
        for(auto j : range1(N))
            {
            for(auto k : range1(r))
                {
                // auto L_arr = ITensor(links(i - 1), is(i));
                // for(auto ii : range1(r))
                //     {
                //     for(auto jj : range1(N))
                //         {
                //         L_arr.set(links(i - 1) = ii, is(i) = jj, L(i).elt(links(i - 1) = ii, is(i) = jj, links(i) = k))
                //         }
                //     }
                // envi_L[i]().set(is(i) = )
                ITensor LHS(links(i - 1)), RHS(links(i - 1));;
                for(auto ii : range1(r))
                    {
                    LHS.set(links(i - 1) = ii, envi_L[i - 1].elt(is(i - 1) = j, links(i - 1) = ii));
                    RHS.set(links(i - 1) = ii, L(i).elt(links(i - 1) = ii, is(i - 1) = j, links(i) = k));
                    }
                }
                envi_L[i]().set(is(i - 1) = j, links(i) = k, elt(LHS * RHS));
            }
        }

} // namespace itensor