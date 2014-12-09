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
 * @file shooting_test.cpp Test shooting algorithm for L1-regularized
 * regression. This is essentially the lasso estimator of regression.
 *
 *****************************************************************************/

#include <string>
#include <stdexcept>
#include <random>
#include <memory>

#include "statistics.h"
#include "basic_plot.h"

using std::string;
using std::shared_ptr;

using namespace Eigen;
using namespace npl;
using namespace std;

void plotMat(std::string filename, const MatrixXd& in)
{
    Plotter plot;
    std::vector<double> tmp(in.rows());
    for(size_t ii=0; ii<in.cols(); ii++) {
        for(size_t rr=0; rr<in.rows(); rr++)
            tmp[rr] = in(rr, ii);
        plot.addArray(tmp.size(), tmp.data());
    }

    plot.write(filename);
}

int main()
{
    std::default_random_engine rng;
    std::uniform_real_distribution<double> unifdist(-1,1);
    std::normal_distribution<> gaussdist(0,1);

    /* 
     * Create the Test Data
     */

    size_t ntimes = 100;
    size_t ndims = 3;
    MatrixXd X(ntimes, ndims);
    VectorXd y(ntimes);
    VectorXd beta(ndims);

    // create square wave
    bool high = false;
    for(size_t ii=0; ii<ntimes; ii++) {
        if(ii % 20 == 0)
            high = !high;
        X(ii,0) = high;
    }
    // create sin wave
    for(size_t ii=0; ii<ntimes; ii++) 
        X(ii,1) = sin(ii/10.);

    // create gaussian 
    for(size_t ii=0; ii<ntimes; ii++) 
        X(ii,2) = gaussdist(rng);

    plotMat("before_mix.svg", X);

    /* 
     * Mix The Data
     */

    // create random beta
    for(size_t ii=0; ii<ndims; ii++)
		beta[ii] = unifdist(rng);

    y = X*beta;
	
	// add noise
    for(size_t ii=0; ii<ntimes; ii++) 
        y[ii] += gaussdist(rng);
    
	plotMat("after_mix.svg", X);
   

	/* 
	 * Perform Regression 
	 */
	VectorXd ebeta = shootingRegr(X, y, 0.1);
	
	cerr << "True Beta: " << beta << endl;
	cerr << "Est. Beta: " << ebeta << endl;

    return 0;
}


