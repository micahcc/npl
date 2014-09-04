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
 * @file pca_test.cpp Test PCA.
 *
 *****************************************************************************/

#include <string>
#include <stdexcept>
#include <random>
#include <memory>

#include "statistics.h"
#include "basic_plot.h"
#include "basic_functions.h"

using std::string;

using namespace Eigen;
using namespace npl;

int corrCompare(const VectorXd& v, const VectorXd& u)
{
    if(v.rows() != u.rows())
        throw std::invalid_argument("Error input matrix and vector must have "
                "same number of rows");

    // compare square wave
    double corr = 0;
    double mu1 = 0;
    double mu2 = 0;
    double ss1 = 0;
    double ss2 = 0;
    for(size_t tt=0; tt<u.rows(); tt++) {
        mu1 += v[tt];
        ss1 += v[tt]*v[tt];
        mu2 += u[tt];
        ss2 += u[tt]*u[tt];
        corr += v[tt]*u[tt];
    }
    
    return sample_corr(u.rows(), mu1, mu2, ss1, ss2, corr);
}

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

    size_t ntimes = 1000;
    size_t ndims = 3;
    MatrixXd data(ntimes, ndims);
    MatrixXd mix(ndims, ndims);

    // create square wave
    bool high = false;
    for(size_t ii=0; ii<ntimes; ii++) {
        if(ii%20 == 0)
            high = !high;
        data(ii,0) = high;
    }
    // create sin wave
    for(size_t ii=0; ii<ntimes; ii++) 
        data(ii,1) = sin(ii/20.);

    // create gaussian 
    for(size_t ii=0; ii<ntimes; ii++) 
        data(ii,2) = gaussdist(rng);

    plotMat("before_mix.svg", data);

    /* 
     * Mix The Data
     */

    // create random mixing matrix
    for(size_t ii=0; ii<ndims; ii++) {
        for(size_t jj=0; jj<ndims; jj++) 
            mix(ii,jj) = unifdist(rng);
    }

    data = data*mix;
    plotMat("after_mix.svg", data);

    data = pca(data, 0.001);
    plotMat("after_pca.svg", data);
 
    // check that the output IS'NT correlated
    for(size_t ii=0; ii<ndims; ii++) {
        for(size_t jj=0; jj<ii; jj++) {
            double cor = corrCompare(data.row(ii), data.row(jj));
            std::cerr << "Cor: " << cor << std::endl;
            if(cor > 0.001) {
                std::cerr << "PCA output should be uncorrelated!" << std::endl;
                return -1;
            }
        }
    }
    return 0;
}

