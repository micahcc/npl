/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file img_lanczos_interp_test4.cpp Tests lanczos interpolation on a periodic
 * function including the wrapping boundary condition. In this, the POSITION
 * in the image, rather than the index is used, that is how it differs from
 * test3.
 *
 *****************************************************************************/

#include <iostream>

#include "mrimage.h"
#include "mrimage_utils.h"
#include "iterators.h"
#include "accessors.h"

using namespace std;
using namespace npl;


double foo(size_t len, double* x, size_t* period)
{
    if(len != 4)
        return 0;
    return cos(2*M_PI*x[0]/period[0]) + cos(2*M_PI*x[1]/period[1]) +
        sin(2*M_PI*x[2]/period[2]) + sin(2*M_PI*x[3]/period[3]);
};

int main()
{
    std::random_device rd;
    std::default_random_engine rng;

    double TOL = 0.4;

	/* Create an image with: x+y*100+z*10000*/
	std::vector<size_t> sz({9, 4, 5, 7});
	shared_ptr<MRImage> testimg = createMRImage(sz, FLOAT64);
    double pt[4];
    size_t ntest = 100;

    // set orientation
    MatrixXd dir(4,4);
    dir << -0.746358,0.650412,0.14112,0,0.664326,0.74088,0.0988343,0,
        -0.04027,0.167515,-0.985047,0,0,0,0,1;
    testimg->setDirection(dir, true);
    VectorXd origin(4);
    origin << 0.32,10,-8,32;
    VectorXd spacing(4);
    origin << 1.1,3.0,7,.3;

    // fill image with foo
    for(NDIter<double> it(testimg); !it.eof(); ++it) {
        it.index(4, pt);
        testimg->indexToPoint(4, pt, pt);
        it.set(foo(4, pt, sz.data()));
    }

    LanczosInterpNDView<double> interp(testimg);
    interp.m_ras = true;

    // check that lanczos interpolation is close to foo for some random points
    // inside the image
	for(size_t ii=0; ii<ntest; ii++) {

        // choose random point inside image
        for(size_t dd=0; dd<4; dd++) {
            std::uniform_real_distribution<double> unif(0, sz[dd]-1);
            pt[dd] = unif(rng);
        }
        testimg->indexToPoint(4, pt, pt);

        double intv = interp.get(4, pt);
        double truev = foo(4, pt, sz.data());

        if(fabs(intv - truev) > TOL) {
            cerr << "During Inside-the-FOV Test\n"
                << "Difference Between Interpolated and Real Result: " <<
                intv << " vs " << truev << endl;
            return -1;
        }
    }

	return 0;
}


