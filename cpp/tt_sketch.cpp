#include <cmath>
#include "tt_sketch.h"

namespace itensor {

using std::pair;
using std::make_pair;

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
    L_((dom[2] - dom[1]) / 2),
    shift_((dom[2] + dom[1]) / 2)
    {
    
    }

Real BasisFunc::
operator()(Real x, int pos) const
    {
    if(x < dom_[1] || x > dom_[2])
        {
        return 0.0;
        }
    if(conv_)
        {

        }
    else
        {
        if(pos == 1)
            {
            return 1 / sqrt(2 * L_);
            }
        else if(pos % 2 == 0)
            {
            return sqrt(1 / L_) * cos(M_PI * (x - shift_) * (pos / 2) / L_);
            }
        else
            {
            return sqrt(1 / L_) * sin(M_PI * (x - shift_) * (pos / 2) / L_);
            }
        }
    }

Real BasisFunc::
grad(Real x, int pos) const
    {
    if(x < dom_[1] || x > dom_[2])
        {
        return 0.0;
        }
    if(conv_)
        {

        }
    else
        {
        if(pos == 1)
            {
            return 1 / sqrt(2 * L_);
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

} // namespace itensor