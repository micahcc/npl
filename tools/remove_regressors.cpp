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
 * @file convertDeform.cpp
 *
 *****************************************************************************/

#include <version.h>
#include <tclap/CmdLine.h>
#include <string>
#include <stdexcept>

#include <Eigen/SVD>
#include <Eigen/Dense>

#include "mrimage.h"
#include "mrimage_utils.h"
#include "iterators.h"
#include "accessors.h"
#include "statistics.h"

using namespace npl;

/**
 * @brief Removes the effects of X from signal (y). Note that this takes both X
 * and the pseudoinverse of X because the bulk of the computation will be on
 * the pseudoinverse. 
 *
 * Beta in OLS may be computed with the pseudoinverse (P):
 * B = Py
 * where P is the pseudoinverse:
 * P = VE^{-1}U^* 
 *
 * @param signal response term (y), will be modified to remove the effects of X
 * @param X Design matrix, or independent values in colums
 * @param Xinv the pseudoinverse of X
 */
inline
void regressOutLS(VectorXd& signal, const MatrixXd& X, const MatrixXd& Xinv)
{
    signal = signal - (Xinv*signal)*X;
}

int main(int argc, char** argv)
{
	try {
	/*
	 * Command Line
	 */

	TCLAP::CmdLine cmd("Performs regression on each timeseries then removes "
			"(decorrelates) the timeseries with the input regressors. This is "
            "useful, for instance, for removing motion parameters.",
			' ', __version__ );

	TCLAP::ValueArg<string> a_fmri("i", "input", "Input 4D image.",
			true, "", "*.nii.gz", cmd);

	TCLAP::MultiArg<string> a_regressors("r", "regressor", "Regressor image, "
            "must be 4D. You may provide multiple of these and they will all "
            "be used simultaneously. Each voxel within the image is treated as "
            " a regressor. If you have data in a txt file, use nplMakeImage to "
            "create a 4D image.", true, "*.nii.gz", cmd);

	TCLAP::ValueArg<string> a_out("o", "out", "Output image.",
			true, "", "*.nii.gz", cmd);

	cmd.parse(argc, argv);

	std::shared_ptr<MRImage> fmri = readMRImage(a_fmri.getValue());
    if(fmri->ndim() != 4)
        throw std::invalid_argument("Input "+a_fmri.getValue()+" must be 4D!");
    size_t tlen = fmri->tlen();

    std::list<std::vector<double>> regressor_list;
    for(auto it=a_regressors.begin(); it != a_regressors.end(); ++it) {
        auto regimg = readMRImage(*it);

        // check image
        if(regimg->ndim() != 4|| regimg->tlen() != tlen) {
            throw std::invalid_argument("Input size in time direction (" +
                    to_string(regimg->tlen()) + ") does not match fMRI");
        }

        ChunkIter<double> pit(regimg);
        pit.setLineChunk(3);
        for(pit.goBegin(); !pit.eof(); pit.nextChunk()) {
            // add regressor to list
            regressor_list.push_back(std::vector<double>());
            regressor_list.back().resize(tlen);
         
            pit.goChunkBegin();
            for(size_t tt=0; !pit.eoc(); ++pit, ++tt) 
                regressor_list.back()[tt] = *pit;
        }
    }
    
    // create regressor matrix
    MatrixXd X(tlen, regressor_list.size());
    size_t cc = 0;
    for(auto& regr : regressor_list) {
        for(size_t tt=0; tt<tlen; tt++)
            X(tt, cc) = regr[tt];
    }

    // since most of the computation time will be solve (X^TX)X^T, precompute
    MatrixXd Xinv = pseudoInverse(X);

    // perform regressions
    ChunkIter<double> it(fmri);
    it.setLineChunk(3);
    VectorXd signal(tlen);
    // regress each timesereies
    for(it.goBegin(); !it.eof(); it.nextChunk()) {

        // copy to signal
        it.goChunkBegin();
        for(size_t tt=0; !it.eoc(); ++tt, ++it) 
            signal[tt] = *it;

        regressOutLS(signal, X, Xinv);

        // write out
        it.goChunkBegin();
        for(size_t tt=0; !it.eoc(); ++tt, ++it) 
            it.set(signal[tt]);
    }

    fmri->write(a_out.getValue());

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
}


