#ifndef __TT_SKETCH_
#define __TT_SKETCH_

#include "itensor/all.h"

namespace itensor {

class BasisFunc
    {
    private:
    std::pair<Real, Real> dom_;
    int nbasis_;
    std::vector<vector<Real>> grid_,
        gridd_;
    bool conv_;
    int nbins_;
    Real L_;
    Real shift_;
    public:

    BasisFunc();

    BasisFunc(std::pair<Real, Real> dom, int nbasis);

    Real operator()(Real x, int pos) const;

    Real grad(Real x, int pos) const;

    void set_conv(bool status) { conv_ = status; }

    } // class BasisFunc

} // namespace itensor

#endif //__TT_SKETCH_
