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
 * @file array_test1.cpp
 *
 *****************************************************************************/

#include "statistics.h"
#include "npltypes.h"
#include <iostream>
#include <Eigen/Dense>
#include <ctime>

using namespace std;
using namespace npl;

int main()
{
    /****************************
     * create points randomly clustered around a few means
     ***************************/
    const size_t NCLUSTER = 4;
    const size_t NDIM = 4;
    const size_t NSAMPLES = 10000;

    std::random_device rd;
//    std::default_random_engine rng(rd());
    std::default_random_engine rng(13);
    std::uniform_int_distribution<int> randUI(0, NCLUSTER-1);
    std::normal_distribution<double> randGD(0, 1);

    // distance will be at least 1
    MatrixXd truemeans(NCLUSTER, NDIM);
    for(size_t ii=0; ii<NCLUSTER; ii++) {
        for(size_t jj=0; jj<NDIM; jj++) {
            truemeans(ii, jj) = randUI(rng);
        }
    }

    // fill samples with noise
    MatrixXd samples(NSAMPLES, NDIM);
    for(size_t ii=0; ii<NSAMPLES; ii++) {
        for(size_t jj=0; jj<NDIM; jj++) {
            samples(ii, jj) = randGD(rng);
        }
    }

    // add mean to each sample
    Eigen::VectorXi trueclass(NSAMPLES);
    for(size_t ii=0; ii<NSAMPLES; ii++) {
        // choose random group
        int c = randUI(rng);
        samples.row(ii) += truemeans.row(c);
        trueclass[ii] = c;
    }
 
    /****************************
     * Perform K-Means
     ***************************/
    KMeans kmeans(NDIM, NCLUSTER);
    kmeans.compute(samples);
    auto oclass = kmeans.classify(samples);

    /****************************
     * Test output
     ***************************/
    // align truemeans with estmeans
    Eigen::VectorXi cmap(NCLUSTER);
    double err = 0;
    for(size_t ii=0; ii<NCLUSTER; ii++) {
        double best = INFINITY;
        int bi = -1;
        for(size_t jj=0; jj<NCLUSTER; jj++) {
            double d = (truemeans.row(ii)-kmeans.getMeans().row(jj)).norm();
            if(d < best) {
                best = d;
                bi = jj;
            }
        }
        err += best;
        cmap[bi] = ii;
    }

    cerr << "Class Map: " << cmap.transpose() << endl;
    
    cerr << "True Means:\n";
    for(size_t ii=0; ii<NCLUSTER; ii++) {
        cerr << truemeans.row(cmap[ii]) << endl;
    }
    cerr << "Est. Means:\n" << kmeans.getMeans() << endl;
    cerr << "Error: " << err/(NDIM*NCLUSTER) << endl;

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

