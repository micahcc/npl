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
 * @file generalLinearModel.cpp Tests large scale GLM
 *
 *****************************************************************************/

#include <version.h>
#include <string>
#include <stdexcept>
#include <random>

#include <Eigen/Dense>
#include "statistics.h"
#include "mrimage.h"
#include "mrimage_utils.h"
#include "ndarray_utils.h"
#include "kernel_slicer.h"
#include "kdtree.h"
#include "iterators.h"
#include "accessors.h"
#include "utility.h"
#include "nplio.h"
#include "basic_plot.h"

using std::string;
using Eigen::MatrixXd;

using namespace npl;

int main()
{   
    // TODO add intercept
    
    /* 
     * Create Test Image: 4 Images
     */
    double intercept = 0;
    vector<size_t> voldim({12,17,13});
    vector<size_t> fdim({12,17,13,1024});

    ptr<MRImage> beta;

    // make array
    {
        vector<ptr<NDArray>> prebeta;
        prebeta.push_back(createMRImage(voldim.size(), voldim.data(), FLOAT32));
        prebeta.push_back(createMRImage(voldim.size(), voldim.data(), FLOAT32));
        prebeta.push_back(createMRImage(voldim.size(), voldim.data(), FLOAT32));
        prebeta.push_back(createMRImage(voldim.size(), voldim.data(), FLOAT32));

        fillLinear(prebeta[0]);
        fillCircle(prebeta[1], 3, 1);
        fillCircle(prebeta[2], 3, 10);
        fillGaussian(prebeta[3]);
        beta = dPtrCast<MRImage>(concatElevate(prebeta));
    }

    auto fmri = createMRImage(fdim.size(), fdim.data(), FLOAT32);
    fillGaussian(fmri);
 
    // create X and fill it with sin waves
    MatrixXd X(fdim[3], 4);
    for(size_t rr=0; rr<X.rows(); rr++ ) {
        for(size_t cc=0; cc<X.cols(); cc++) {
            X(rr,cc) = sin(M_PI*(rr+1)/100*cc);
        }
    }

    // add values from beta*X, cols of X correspond to the highest dim of beta
    Vector3DIter<double> bit(beta);
    bit.goBegin();
    for(size_t cc=0; cc<X.cols(); cc++, ++bit) {

        // for each voxel in fMRI,
        Vector3DIter<double> it(fmri);
        for(it.goBegin(); !it.eof(); ++it) {
            // for each row in X/time in fMRI 
            for(size_t rr=0; rr<X.rows(); rr++) 
                it.set(rr, it[rr] + bit[cc]*X(rr,cc));
        }
    }

    writeMRImage(fmri, "signal_noise.nii.gz");
    writeMRImage(beta, "betas.nii.gz");

//    /* Perform Regression */
//	int tlen = fmri->tlen();
//	double TR = fmri->spacing(3);
//
//    // create output images
//    std::list<ptr<MRImage>> tImgs;
//    std::list<ptr<MRImage>> pImgs;
//    std::list<NDAccess<double>> tAccs;
//    std::list<NDAccess<double>> pAccs;
//    for(size_t ii=0; ii<X.cols(); ii++) {
//        tImgs.push_back(dPtrCast<MRImage>(fmri->extractCast(3, fmri->dim(), FLOAT64)));
//        tAccs.push_back(NDAccess<double>(tImgs.back()));
//        pImgs.push_back(dPtrCast<MRImage>(fmri->extractCast(3, fmri->dim(), FLOAT64)));
//        pAccs.push_back(NDAccess<double>(pImgs.back()));
//    }
//
//    vector<int64_t> ind(3);
//
//    // Cache Reused Vectors
//    auto Xinv = pseudoInverse(X);
//    auto covInv = pseudoInverse(X.transpose()*X);
//
//    const double MAX_T = 100;
//    const double STEP_T = 0.1;
//    auto student_cdf = students_t_cdf(X.rows()-1, STEP_T, MAX_T);
//
//    // regress each timesereies
//    ChunkIter<double> it(fmri);
//    it.setLineChunk(3);
//    Eigen::VectorXd signal(tlen);
//    for(it.goBegin(); !it.eof(); it.nextChunk()) {
//
//        // copy to signal
//        it.goChunkBegin();
//        for(size_t tt=0; !it.eoc(); ++tt, ++it) 
//            signal[tt] = *it;
//
//        RegrResult ret = regress(signal, X, covInv, Xinv, student_cdf);
//
//        auto t_it = tAccs.begin();
//        auto p_it = pAccs.begin();
//        for(size_t ii=0; ii<X.cols(); ii++) {
//            (*t_it).set(ret.t[ii], ind);
//            (*p_it).set(ret.p[ii], ind);
//            ++t_it;
//            ++p_it;
//        }
//    }
//
//
//    /* Test Results */
//
//    auto t_it = tImgs.begin();
//    auto p_it = pImgs.begin();
//    for(size_t ii=0; ii<X.cols(); ii++) {
//        (*t_it)->write(a_odir.getValue()+"/t_"+to_string(ii)+".nii.gz");
//        (*p_it)->write(a_odir.getValue()+"/p_"+to_string(ii)+".nii.gz");
//    }
    
    return 0;
}



