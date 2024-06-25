#include "tt_sketch.h"
using namespace std;
using namespace itensor;

int
main()
    {
    vector<vector<Real>> samples(3, vector<Real>(3, 0.0));
    samples[0][0] = -0.9;
    samples[0][1] = -0.8;
    samples[0][2] = -0.6;
    samples[1][0] = -0.3;
    samples[1][1] = 0.1;
    samples[1][2] = 0.6;
    samples[2][0] = -0.4;
    samples[2][1] = 0.3;
    samples[2][2] = -0.7;
    vector<pair<Real, Real>> domain;
    domain.push_back(make_pair(-1.0, 1.0));
    domain.push_back(make_pair(-1.0, 1.0));
    domain.push_back(make_pair(-1.0, 1.0));
    vector<BasisFunc> basis(3, BasisFunc(make_pair(-1.0, -1.0), 10));
    auto G = paraSketch(samples, domain, basis, 5);
    }