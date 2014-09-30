/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * 	http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file img_nn_interp_test4.cpp Tests Nearest Neighbor interpolation on a
 * periodic function including the wrapping boundary condition. This test 
 * doesn't really work for any rotation. 
 * I need to find a better way to test NN interpolation
 * because it can result in large errors during rotation.
 *
 *****************************************************************************/

#include <iostream>

#include "mrimage.h"
#include "mrimage_utils.h"
#include "iterators.h"
#include "accessors.h"

using namespace std;
using namespace npl;


double foo(size_t len, double* xi, size_t* period)
{
    if(len != 4)
        return 0;
    double x[4];
    x[0] = round(xi[0]);
    x[1] = round(xi[1]);
    x[2] = round(xi[2]);
    x[3] = round(xi[3]);
    return cos(2*M_PI*x[0]/period[0]) + cos(2*M_PI*x[1]/period[1]) +
        sin(2*M_PI*x[2]/period[2]) + sin(2*M_PI*x[3]/period[3]); 
};

int main()
{
    std::random_device rd;
    std::default_random_engine rng;

    double TOL = 0.00000001;

	/* Create an image with: x+y*100+z*10000*/
	std::vector<size_t> sz({9, 4, 5, 7});
	shared_ptr<MRImage> testimg = createMRImage(sz, FLOAT64);
    double pt[4];
    size_t ntest = 100;

    // set orientation
    MatrixXd dir(4,4);
    dir << -1,0,0,0,0,-1,0,0,0,0,1,0,0,0,0,1;
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

    NNInterpNDView<double> interp(testimg);
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


