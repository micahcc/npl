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
#include "macros.h"

using std::string;
using Eigen::MatrixXd;

using namespace npl;

int main()
{   
    // TODO add intercept
   
    vector<size_t> voldim({12,17,13});
    vector<size_t> fdim({12,17,13,1024});
    ptr<MRImage> realbeta;
    ptr<MRImage> fmri;

    // create X and fill it with sin waves
    MatrixXd X(fdim[3], 4);
    for(size_t rr=0; rr<X.rows(); rr++ ) {
        for(size_t cc=0; cc<X.cols(); cc++) {
            X(rr,cc) = sin(M_PI*(rr+1)/100*cc);
        }
    }
    // make the first an intercept
    for(size_t rr=0; rr<X.rows(); rr++)
        X(rr,0) = 1;

    {
        /* 
         * Create Test Image: 4 Images
         */

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
            realbeta = dPtrCast<MRImage>(concatElevate(prebeta));
        }

        fmri = createMRImage(fdim.size(), fdim.data(), FLOAT32);
        fillGaussian(fmri);

        // add values from beta*X, cols of X correspond to the highest dim of beta
        Vector3DIter<double> bit(realbeta);
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
        writeMRImage(realbeta, "betas.nii.gz");
    }


    /* Perform Regression */
	int tlen = fmri->tlen();
    if(fmri->ndim() != 4) {
        throw INVALID_ARGUMENT("Input Image should be 4D!");
    }

    // create output images
    vector<size_t> osize(4, 0);
    for(size_t ii=0; ii<3; ii++) {
        osize[ii] = fmri->dim(ii);
    }
    osize[3] = X.cols();

    auto t_est = dPtrCast<MRImage>(fmri->copyCast(4, osize.data()));
    auto p_est = dPtrCast<MRImage>(fmri->copyCast(4, osize.data()));
    auto b_est = dPtrCast<MRImage>(fmri->copyCast(4, osize.data()));
    
    vector<int64_t> ind(3);

    // Cache Reused Vectors
    auto Xinv = pseudoInverse(X);
    auto covInv = pseudoInverse(X.transpose()*X);

    const double MAX_T = 20;
    const double STEP_T = 0.01;
    auto student_pdf = students_t_pdf(X.rows()-1, STEP_T, MAX_T);
    auto student_cdf = students_t_cdf(X.rows()-1, STEP_T, MAX_T);
//    writePlot("t_pdf.svg", student_pdf);
//    writePlot("t_cdf.svg", student_cdf);
    VectorXd signal(fmri->dim(3));

    // regress each timesereies
    Vector3DIter<double> tit(t_est), pit(p_est), bit(b_est); 
    tit.goBegin(); 
    pit.goBegin();
    bit.goBegin();
    for(Vector3DConstIter<double> fit(fmri); !fit.eof(); ++fit, ++tit, ++pit, ++bit) {

        // copy to signal
        for(size_t tt=0; tt<tlen; tt++)
            signal[tt] = fit[tt];

        RegrResult ret = regress(signal, X, covInv, Xinv, student_cdf);

        for(size_t ii=0; ii<X.cols(); ii++) {
            tit.set(ret.t[ii], ii);
            pit.set(ret.p[ii], ii);
            bit.set(ret.bhat[ii], ii);
        }
    }


    /* Test Results */
    writeMRImage(b_est, "beta_est.nii.gz");
    writeMRImage(p_est, "p_est.nii.gz");
    writeMRImage(t_est, "t_est.nii.gz");
    return 0;
}



