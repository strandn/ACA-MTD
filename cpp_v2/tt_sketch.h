#ifndef __TT_SKETCH_
#define __TT_SKETCH_

#include "itensor/all.h"

namespace itensor {

class BasisFunc
    {
    private:
    std::pair<Real, Real> dom_;
    int nbasis_;
    bool conv_;
    int nbins_;
    Real L_;
    Real shift_;
    std::vector<std::vector<Real>> grid_,
        gridd_;
    std::vector<Real> xdata_;
    public:

    BasisFunc();

    BasisFunc(std::pair<Real, Real> dom, int nbasis);

    Real fourier(Real x, int pos) const;

    Real operator()(Real x, int pos) const;

    Real grad(Real x, int pos) const;

    void setConv(bool status) { conv_ = status; }

    Real interpolate(Real x, int pos, bool grad) const;

    int nbasis() const { return this->nbasis_; }

    std::pair<Real, Real> const& dom() const { return this->dom_; }

    int nbins() const { return this->nbins_; }

    }; // class BasisFunc

Real f(Real x, void* params);

Real df(Real x, void* params);

MPS
paraSketch(std::vector<std::vector<Real>> const& samples, std::vector<BasisFunc> const& basis, int rc);

MPS
createTTCoeff(int n, int d, int r);

std::pair<std::vector<ITensor>, SiteSet>
intBasisSample(std::vector<BasisFunc> const& basis, std::vector<std::vector<Real>> const& samples, SiteSet const& is);

std::tuple<MPS, std::vector<ITensor>, std::vector<ITensor>>
formTensorMoment(std::vector<ITensor> const& M, MPS const& coeff, SiteSet const& is);

Real
densEval(MPS const& G, std::vector<BasisFunc> const& basis, std::vector<Real> const& elements);

std::vector<Real>
densGrad(MPS const& G, std::vector<BasisFunc> const& basis, std::vector<Real> const& elements);

} // namespace itensor

#endif //__TT_SKETCH_
