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
 * @file ica_test.cpp Test ICA.
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
using std::shared_ptr;

using namespace Eigen;
using namespace npl;

int corrCompare(const VectorXd& v, const MatrixXd& data)
{
    if(v.rows() != data.rows())
        throw std::invalid_argument("Error input matrix and vector must have "
                "same number of rows");

    // compare square wave
    for(size_t dd=0; dd<data.cols(); dd++) {
        double corr = 0;
        double mu1 = 0;
        double mu2 = 0;
        double ss1 = 0;
        double ss2 = 0;
        for(size_t tt=0; tt<data.rows(); tt++) {
            mu1 += v[tt];
            ss1 += v[tt]*v[tt];
            mu2 += data(tt,dd);
            ss2 += data(tt,dd)*data(tt,dd);
            corr += v[tt]*data(tt,dd);
        }
        corr = sample_corr(data.rows(), mu1, mu2, ss1, ss2, corr);
        std::cerr << corr << std::endl;
        if(fabs(corr) > 0.99) {
            std::cerr << "Corr="<<corr<<" good enough" << std::endl;
            return 0;
        }
    }

    return -1;
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
    MatrixXd tdata(ntimes, ndims);
    MatrixXd mix(ndims, ndims);

    // create square wave
    bool high = false;
    for(size_t ii=0; ii<ntimes; ii++) {
        if(ii % 20 == 0)
            high = !high;
        tdata(ii,0) = high;
    }
    // create sin wave
    for(size_t ii=0; ii<ntimes; ii++)
        tdata(ii,1) = sin(ii/10.);

    // create gaussian
    for(size_t ii=0; ii<ntimes; ii++)
        tdata(ii,2) = gaussdist(rng);

    plotMat("before_mix.svg", tdata);

    /*
     * Mix The Data
     */

    // create random mixing matrix
    for(size_t ii=0; ii<ndims; ii++) {
        for(size_t jj=0; jj<ndims; jj++)
            mix(ii,jj) = unifdist(rng);
    }

    MatrixXd data = tdata*mix;
    plotMat("after_mix.svg", data);

    // remove mean/variance
    for(size_t cc=0; cc<data.cols(); cc++)  {
        double sum = 0;
        double sumsq = 0;
        for(size_t rr=0; rr<data.rows(); rr++)  {
            sum += data(rr,cc);
            sumsq += data(rr,cc)*data(rr,cc);
        }
        double sigma = sqrt(sample_var(data.rows(), sum, sumsq));
        double mean = sum/data.rows();

        for(size_t rr=0; rr<data.rows(); rr++)
            data(rr,cc) = (data(rr,cc)-mean)/sigma;
    }

	std::cerr << "PCA...";
    auto pdata = pca(data, .98);
	std::cerr << "Done" << std::endl;
    plotMat("after_pca.svg", pdata);

	std::cerr << "ICA...";
    auto idata = ica(pdata);
	std::cerr << "Done" << std::endl;
    plotMat("after_ica.svg", idata);

    /*
     * compare signals
     */

    // compare square wave
    if(corrCompare(tdata.col(0), idata) != 0)
        return -1;

    // compare sin wave
    if(corrCompare(tdata.col(1), idata) != 0)
        return -1;

    // compare random
    if(corrCompare(tdata.col(2), idata) != 0)
        return -1;

    return 0;
}

