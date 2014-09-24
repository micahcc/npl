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
 * @file expmax_test1.cpp Basic test of the ExpMax class.
 *
 *****************************************************************************/

#include "statistics.h"
#include "basic_plot.h"
#include "npltypes.h"
#include <iostream>
#include <Eigen/Dense>
#include <ctime>

using namespace std;
using namespace npl;

void generateMeanCov(size_t ndim, size_t ncluster, MatrixXd& mean, MatrixXd&
        cov, MatrixXd& stddev)
{
    std::random_device rd;
    std::default_random_engine rng(rd());
    std::normal_distribution<double> randGD(0, 1);
    std::uniform_real_distribution<double> randUD(0, .5);
    
    // distance will be at least 1
    mean.resize(ncluster, ndim);
    cov.resize(ndim*ncluster, ndim);
    stddev.resize(ndim*ncluster, ndim);
    MatrixXd affine(ndim, ndim);
    for(size_t ii=0; ii<ncluster; ii++) {
        // mean
        for(size_t jj=0; jj<ndim; jj++) 
            mean(ii, jj) = randGD(rng)*10;

        //affine
        affine.setZero();
        for(size_t jj=0; jj<ndim; jj++) {
            for(size_t kk=0; kk<=jj; kk++) 
                affine(jj, kk) = randUD(rng);
        }

        // turn covariance into standard deviation
        // compute the Cholesky decomposition of A
        stddev.block(ii*ndim, 0, ndim, ndim) = affine;
        cov.block(ii*ndim, 0, ndim, ndim) = affine*affine.transpose();
    }
}

void generateMVGaussians(size_t ncluster, size_t nsamples, size_t ndim, 
        const MatrixXd& mean, const MatrixXd& stddev,
        MatrixXd& samples, Eigen::VectorXi& classes)
{

    std::random_device rd;
    std::default_random_engine rng(rd());
    std::uniform_int_distribution<int> randUI(0, ncluster-1);
    std::normal_distribution<double> randGD(0, 1);

    // fill samples with normal vector, then multiplied by the stddev matrix
    samples.resize(nsamples, ndim);
    classes.resize(nsamples);
    for(size_t ii=0; ii<nsamples; ii++) {
        for(size_t jj=0; jj<ndim; jj++) 
            samples(ii, jj) = randGD(rng);

        // randomly select group
        int c = randUI(rng);
        classes[ii] = c;
        samples.row(ii) = mean.row(c)+samples.row(ii)
            *stddev.block(c*ndim, 0, ndim, ndim).transpose();
        cerr << samples.row(ii) << endl;
    }

}

int main()
{
    /****************************
     * create points randomly clustered around a few means
     ***************************/
    const size_t NCLUSTER = 4;
    const size_t NDIM = 2;
    const size_t NSAMPLES = 100;

    std::random_device rd;
    std::default_random_engine rng(rd());
    std::uniform_int_distribution<int> randUI(0, NCLUSTER-1);
    std::normal_distribution<double> randGD(0, 1);
    
    MatrixXd truemean;
    MatrixXd truecov;
    MatrixXd truestddev;
    Eigen::VectorXi trueclass;
    MatrixXd samples;
    generateMeanCov(NDIM, NCLUSTER, truemean, truecov, truestddev);
    generateMVGaussians(NCLUSTER, NSAMPLES, NDIM, truemean, 
            truestddev, samples, trueclass);

    /****************************
     * Perform Clustering
     ***************************/
    ExpMax cluster(NDIM, NCLUSTER);
    cluster.compute(samples);
    auto oclass = cluster.classify(samples);

    /****************************
     * Test output
     ***************************/
    // align truemean with estmeans
    Eigen::VectorXi cmap(NCLUSTER);
    double err = 0;
    for(size_t ii=0; ii<NCLUSTER; ii++) {
        double best = INFINITY;
        int bi = -1;
        for(size_t jj=0; jj<NCLUSTER; jj++) {
            double d = (truemean.row(ii)-cluster.getMeans().row(jj)).norm();
            if(d < best) {
                best = d;
                bi = jj;
            }
        }
        err += best;
        cmap[bi] = ii;
    }

    cerr << "Class Map: " << cmap.transpose() << endl;
    
    cerr << "Means:\n";
    for(size_t ii=0; ii<NCLUSTER; ii++) {
        cerr << "True:\n" << truemean.row(cmap[ii]) << endl;
        cerr << "Est:\n" << cluster.getMeans().row(ii) << endl;
    }
    cerr << "Error: " << err/(NDIM*NCLUSTER) << endl;
   
    cerr << "True Covariance:\n" << truecov << endl << endl;
    cerr << "calc Covariance:\n" << cluster.getCovs() << endl << endl;

    size_t misscount = 0;
    for(size_t ii=0; ii<NSAMPLES; ii++) {
        if(cmap[oclass[ii]] != trueclass[ii])
            misscount++;
    }

    cerr << misscount << "/" << NSAMPLES << " (" << 
        (double)misscount/NSAMPLES << ") Incorrect" << endl;

    if(err/(NDIM*NCLUSTER) > 0.1) {
        cerr << "Fail" << endl;
        return -1;
    }

    cerr << "OK!" << endl;
    return 0;
}


