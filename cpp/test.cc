#include "tt_sketch.h"
using namespace std;
using namespace itensor;

int
main()
    {
    srand(time(NULL));
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
    // vector<pair<Real, Real>> domain;
    // domain.push_back(make_pair(-1.0, 1.0));
    // domain.push_back(make_pair(-1.0, 1.0));
    // domain.push_back(make_pair(-1.0, 1.0));
    vector<BasisFunc> basis(3, BasisFunc(make_pair(-1.0, 1.0), 10));
    auto G = paraSketch(samples, basis, 5);
    auto f = h5_open("data.h5", 'w');
    h5_write(f,"itensor_G0", G); 
    close(f);
    auto f = h5_open("data.h5", 'w');
    h5_write(f,"itensor_G1", G); 
    close(f);
    basis[0].setConv(false);
    basis[1].setConv(false);
    basis[2].setConv(false);
    cout << densEval(G, basis, { -0.9, -0.8, -0.6 }) << endl;
    cout << densEval(G, basis, { -0.95, -0.85, -0.65 }) << endl;
    vector<Real> dens_grad1 = densGrad(G, basis, { -0.9, -0.8, -0.6 });
    vector<Real> dens_grad2 = densGrad(G, basis, { -0.95, -0.85, -0.65 });
    cout << dens_grad1[0] << " " << dens_grad1[1] << " " << dens_grad1[2] << " " << endl;
    cout << dens_grad2[0] << " " << dens_grad2[1] << " " << dens_grad2[2] << " " << endl;
    basis[0].setConv(true);
    basis[1].setConv(true);
    basis[2].setConv(true);
    cout << densEval(G, basis, { -0.9, -0.8, -0.6 }) << endl;
    cout << densEval(G, basis, { -0.95, -0.85, -0.65 }) << endl;
    dens_grad1 = densGrad(G, basis, { -0.9, -0.8, -0.6 });
    dens_grad2 = densGrad(G, basis, { -0.95, -0.85, -0.65 });
    cout << dens_grad1[0] << " " << dens_grad1[1] << " " << dens_grad1[2] << " " << endl;
    cout << dens_grad2[0] << " " << dens_grad2[1] << " " << dens_grad2[2] << " " << endl;
    }
