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
 * @file fmri_ica.cpp Tool for performing ICA on a fMRI image.
 *
 *****************************************************************************/

#include <version.h>
#include <tclap/CmdLine.h>
#include <string>
#include <stdexcept>

#include <Eigen/Dense>
#include <Eigen/IterativeSolvers>

#include "mrimage.h"
#include "nplio.h"
#include "iterators.h"
#include "statistics.h"
#include "macros.h"

using std::string;
using std::shared_ptr;
using std::to_string;

using namespace Eigen;
using namespace npl;

TCLAP::MultiSwitchArg a_verbose("v", "verbose", "Write lots of output. "
		"Be wary for large datasets!", 1);


int main(int argc, char** argv)
{
	cerr << "Version: " << __version__ << endl;
	try {
	/*
	 * Command Line
	 */

	TCLAP::CmdLine cmd("Perform ICA analysis on an image, or group of images. "
			"By default this will be a temporal ICA. You can perform "
			"group-analysis by concatinating in space or time. ORDER OF INPUTS "
			"MATTERS IF YOU CONCATINATE IN BOTH. Concatination in time occurs "
			"for adjacent arguments. So to concatinate (a+b and c+d) in time "
			"then (ab+cd) in space you would do -t 2 -s 2 -i a -i b -i c -i d "
			". To concatinate a+b+c and d+e+f in time then abc+def in space. "
			"You sould do -t 3 -s 2 -i a -i b -i c -i d -i e -i f . Note that "
			"for efficiency we assume all the images are the same size, in "
			"the future there may be a way to explicity specify the final "
			"dimensions. By default a Temporal ICA will be performed, to do a "
			"spatial ICA select -S.", ' ', __version__ );

	TCLAP::SwitchArg a_spatial_ica("S", "spatial-ica", "Perform a spatial ICA"
			", reducing unmixing timepoints to produce spatially independent "
			"maps.", cmd);

	TCLAP::MultiArg<string> a_in("i", "input", "Input fMRI image.",
			true, "*.nii.gz", cmd);
	TCLAP::ValueArg<int> a_time_append("t", "time-appends", "Number of images "
			"to append in the matrix of images, in the time direction.", false,
			1, "int", cmd);
	TCLAP::ValueArg<int> a_space_append("s", "space-appends", "Number of images "
			"to append in the matrix of images, in the space direction.", false,
			1, "int", cmd);

    TCLAP::ValueArg<string> a_components("o", "out-components", "Output "
            "Independent Components as a 1x1xCxT image.",
			true, "", "*.nii.gz", cmd);
    TCLAP::ValueArg<string> a_mapdir("d", "mapdir", "Output "
            "directory for ICA significance maps. The number of maps will "
            "depend on the number of components, and will be in the same "
            "space as the input fMRI image. names will be the "
            "$mapdir/$input_$num.nii.gz where $mapdir is the mapdir, $input "
            "is the basename from -i and $num is the component number",
            true, "./", "/", cmd);

	TCLAP::ValueArg<double> a_evthresh("T", "ev-thresh", "Threshold on "
			"ratio of total variance to account for (default 0.99)", false,
			0.99, "ratio", cmd);
	TCLAP::ValueArg<int> a_simultaneous("V", "simul-vectors", "Simultaneous "
			"vectors to estimate eigenvectors for in lambda. Bump this up "
			"if you have convergence issues. This is part of the "
			"Band Lanczos Eigenvalue Decomposition of the covariance matrix. "
			"By default the covariance size is used, which is conservative. "
			"Values less than 1 will default back to that", false, -1, 
			"#vecs", cmd);
	TCLAP::ValueArg<int> a_maxrank("R", "max-rank", "Maximum rank of output "
			"in PCA. This sets are hard limit on the number of estimated "
			"eigenvectors in the Band Lanczos Solver (and thus the maximum "
			"number of singular values in the SVD)", false, -1, "#vecs", cmd);

	cmd.add(a_verbose);
	cmd.parse(argc, argv);

	/**********
	 * Input
	 *********/
	if(!a_in.isSet()) {
		cerr << "Need to provide at least 1 input image!" << endl;
		return -1;
	}

	// read first image for example
	auto example = readMRImage(a_in.getValue()[0]);

	size_t tcat = a_time_append.getValue(); // subjects to concat in space
	size_t scat = a_space_append.getValue(); // subjects to conctat in time
	size_t tsing = example->tlen(); // single timelength
	size_t ssing = example->elements()/example->tlen(); // single spacelength
	size_t tlen = tsing*tcat; // total timelength
	size_t slen = ssing*scat; // total spacelength
	example.reset();
	
	MatrixXd cov;

	/*
	 * First Perform SVD
	 */

	if(a_verbose.getValue() >= 1) 
		cerr << "Computing Covariance..." << endl;
	
	// Data is formed with 
	// X = space x time
	// Compute either M M^T or M^T M
	if(tlen < slen) {
		// Time is the dimesion in the covariance matrix, 
		// thus computing X
		cov.resize(tlen, tlen);
		cov.setZero();

		// Buffer to hold all timepoints at one time
		// essientially X(ss*ssing:ss*ssing+ssing, :)
		MatrixXd buff(ssing, tlen);
		
		for(size_t ss=0; ss<scat; ss++) { // big space 
			for(size_t tt=0; tt<tcat; tt++) { // big time
				example = readMRImage(a_in.getValue()[ss*tcat + tt]);
				if(example->tlen() != tlen || example->elements() != ssing*tlen){
					cerr << "Error Image sizes differ!" << endl;
					return -1;
				}

				Vector3DIter<double> it(example);
				for(size_t s=0; s<ssing; ++it, s++) { // small space
					for(size_t t=0; t<tsing; t++) // small time
						buff(s, t+tt*tsing) = it[t];
				}
			}
			
			if(a_verbose.getValue() >= 2) 
				cerr << "X_" << ss << " = " << endl << endl << buff << endl << endl;
			cov += buff.transpose()*buff;
		}

		if(a_verbose.getValue() >= 2) 
			cerr << "Final Covariance:\n\n" << cov << endl << endl;
	} else {

		// Space is the dimension in the covariance matrix
		cov.resize(slen, slen);
		cov.setZero();
	
		// Buffer to hold all spatial elements at one time
		// essientially X(:,tt*tsing:tt*tsing+tsing)
		MatrixXd buff(slen, tsing);
		
		for(size_t tt=0; tt<tcat; tt++) { // big time
			for(size_t ss=0; ss<scat; ss++) { // big space 
				example = readMRImage(a_in.getValue()[ss*tcat + tt]);
				if(example->tlen() != tlen || example->elements() != ssing*tlen){
					cerr << "Error Image sizes differ!" << endl;
					return -1;
				}

				Vector3DIter<double> it(example);
				for(size_t s=0; s<ssing; ++it, s++) { // small space
					for(size_t t=0; t<tsing; t++) // small time
						buff(s+ss*ssing, t) = it[t];
				}
			}
			
			if(a_verbose.getValue() >= 2) 
				cerr << "X_" << tt << " = " << endl << endl << buff << endl << endl;

			cov += buff*buff.transpose();
		}
		
		if(a_verbose.getValue() >= 2) 
			cerr << "Final Covariance:\n\n" << cov << endl << endl;
	}

	if(a_verbose.getValue() >= 1) 
		cerr << "Done with Covariance." << endl;
	if(a_verbose.getValue() >= 2)
		cerr << endl << cov << endl;

	BandLanczosSelfAdjointEigenSolver<double> esolve;
	
	int simvecs = a_simultaneous.getValue();
	if(simvecs <= 1) 
		simvecs = cov.rows();

	if(a_verbose.getValue() >= 1) {
		cerr << "EigenSolving Covariance." << endl;
		cerr << "Using " << simvecs << " simultaneous eigenvectors " << endl;
		cerr << "Using " << a_evthresh.getValue() << "% of eigenvalues" << endl;
	}
	if(a_evthresh.isSet())
		esolve.setTraceSqrStop(a_evthresh.getValue());
	esolve.compute(cov, simvecs);

	// Reduce singular values to > 0 components
	int eigrows = esolve.eigenvalues().rows();
	int rank = 0;
	VectorXd singvals(eigrows);
	for(int cc=0; cc<eigrows; cc++) {
		if(esolve.eigenvalues()[cc] > 1e-10)
			rank++;
	}

	if(a_verbose.getValue() >= 2)
		cerr << "Singular Values: " << esolve.eigenvalues().tail(rank).transpose() << endl;

	MatrixXd whitened;

	// Note that because Eigen Solvers usually sort eigenvalues in
	// increasing order but singular value decomposers do decreasing order,
	// we need to reverse the singular value and singular vectors found.
	if(tlen < slen) {
		// Computed Covariance with time as dim, X^T X, so got right singular
		// values, V
		if(a_spatial_ica.isSet()) {
			// X = USV*, U = XVS^-1
			cerr << "Computing U from V, for spatial PCA\n";
			whitened.resize(slen, rank);

			cerr << "Computing Projection Matrix" << endl;
			MatrixXd proj = esolve.eigenvectors().rightCols(rank)*
				esolve.eigenvalues().tail(rank).cwiseInverse().asDiagonal();
			cerr << "Done:\n\n" << proj << endl << endl; 

			// Buffer to hold all timepoints at one time
			MatrixXd buff(ssing, tlen);

			for(size_t ss=0; ss<scat; ss++) { // big space 
				for(size_t tt=0; tt<tcat; tt++) { // big time
					example = readMRImage(a_in.getValue()[ss*tcat + tt]);
					if(example->tlen() != tlen || example->elements() != ssing*tlen){
						cerr << "Error Image sizes differ!" << endl;
						return -1;
					}

					Vector3DIter<double> it(example);
					for(size_t s=0; s<ssing; ++it, s++) { // small space
						for(size_t t=0; t<tsing; t++) // small time
							buff(s, t+tt*tsing) = it[t];
					}
				}

				if(a_verbose.getValue() >= 2) 
					cerr << "X_" << ss << " = " << endl << endl << buff << endl;
				whitened.middleRows(ss*ssing, ssing) = buff*proj;
				if(a_verbose.getValue() >= 2) 
					cerr << "White:\n" << endl << whitened << endl << endl;
			}
		} else { 
			// Already Have V
			if(a_verbose.getValue() >= 2)
				cerr << "Already have V, for spatial PCA\n";
			whitened = esolve.eigenvectors().rightCols(rank);
		}

		if(a_verbose.getValue() >= 2)
			cerr << "Done finding whitened data for spatial PCA\n";

	} else {
		// Computed left singular values (U), since we did X X^T
		if(a_spatial_ica.isSet()) {
			// Already Have U
			if(a_verbose.getValue() >= 2)
				cerr << "Already have U, for Spatial PCA\n";
			whitened = esolve.eigenvectors().rightCols(rank);
		} else {
			// X = USV*, X^T = VSU*, V = X^T U S^-1
			cerr << "Computing V from U, for Temporal PCA\n";
			whitened.resize(tlen, rank);

			cerr << "Computing Projection Matrix" << endl;
			MatrixXd proj = esolve.eigenvectors().rightCols(rank)*
				esolve.eigenvalues().tail(rank).cwiseInverse().asDiagonal();
			cerr << "Done:\n\n" << proj << endl << endl; 
			
			// Buffer to hold all spatial elements at one time
			MatrixXd buff(slen, tsing);

			for(size_t tt=0; tt<tcat; tt++) { // big time
				for(size_t ss=0; ss<scat; ss++) { // big space 
					example = readMRImage(a_in.getValue()[ss*tcat + tt]);
					if(example->tlen() != tlen || example->elements() != ssing*tlen){
						cerr << "Error Image sizes differ!" << endl;
						return -1;
					}

					Vector3DIter<double> it(example);
					for(size_t s=0; s<ssing; ++it, s++) { // small space
						for(size_t t=0; t<tsing; t++) // small time
							buff(s+ss*ssing, t) = it[t];
					}
				}

				if(a_verbose.getValue() >= 2) 
					cerr << "X_" << tt << " = " << endl << endl << buff << endl;
				
				whitened.middleRows(tt*tsing, tsing) = buff.transpose()*proj;
				if(a_verbose.getValue() >= 2) 
					cerr << "White:\n" << endl << whitened << endl << endl;
			}
		}
	}


//	// perform ICA
//	std::cerr << "ICA...";
//	MatrixXd X_ic = ica(X_pc, 0.01);
//	std::cerr << "Done" << endl;
//
//    // 
//	MatrixXd regressors = reduce(inimg);
//    for(size_t cc = 0; cc < regressors.cols(); cc++) {
//        // perform regression
//        //RegrResult tmp = regress(inimg, regressors.row(cc));
//
//        // write out each of the images
//        //tmp.rsqr->write("rsqr_"+to_string(cc)+".nii.gz");
//        //tmp.T->write("T_"+to_string(cc)+".nii.gz");
//        //tmp.p->write("p_"+to_string(cc)+".nii.gz");
//        //tmp.beta->write("beta_"+to_string(cc)+".nii.gz");
//    }

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }

    return 0;
}


