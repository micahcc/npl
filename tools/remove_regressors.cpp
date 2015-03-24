/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file remove_regressors.cpp Performs regression on each timesereies then
 * removes (decorrelates) the  timeseries with the input regressors. This is
 * useful for removing motion parameters.
 *
 *****************************************************************************/

#include <version.h>
#include <tclap/CmdLine.h>
#include <string>
#include <stdexcept>

#include <Eigen/SVD>
#include <Eigen/Dense>

#include "mrimage.h"
#include "nplio.h"
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
	cerr << "Version: " << __version__ << endl;
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

	TCLAP::MultiArg<string> a_regressors("r", "regressor", "Regressors. Could "
			"be an image or csv file. Spatial information will not be used "
			"if an image is given. Values should correspond to the value at "
			"each TR of image. If you have the 3-column format of FSL, that "
			"needs to be sampled at the correct image TR.",
			true, "*csv|*.nii.gz", cmd);

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


