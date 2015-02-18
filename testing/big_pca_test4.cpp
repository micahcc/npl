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
 * @file big_pca_test4.cpp This tests the Randomized Subspace Iteration method
 * of computing the SVD.
 *
 *****************************************************************************/

#include <string>

#include "ica_helpers.h"
#include "mrimage_utils.h"
#include "mrimage.h"
#include "iterators.h"
#include "utility.h"
#include "nplio.h"
#include "version.h"

#include <Eigen/SVD>
#include <Eigen/Dense>
#include <Eigen/IterativeSolvers>

#include <tclap/CmdLine.h>

using namespace npl;
using namespace std;

using Eigen::Success;
using Eigen::ComputeThinV;
using Eigen::ComputeThinU;
using Eigen::JacobiSVD;
using Eigen::BDCSVD;

size_t approxrank(const Ref<const VectorXd> singvals, double thresh)
{
	double var = 0;
	double totalvar = singvals.sum();
	size_t rank = 0;
	for(rank = 0; var < totalvar*thresh && rank < singvals.rows(); rank++)
		var += singvals[rank];

	return rank;
}

int testTallPCAJoin(const MatrixReorg& reorg, std::string prefix, double svt,
		int estrank, size_t poweriters)
{
	cerr<<"Tall PCA"<<endl;
	double thresh = 0.1;

	size_t totrows = reorg.rows();
	size_t totcols = reorg.cols();
	size_t curcol = 0;

	// Create Full Matrix to Compare Against
	MatrixXd full(totrows, totcols);
	for(size_t ii=0; ii<reorg.ntall(); ++ii) {
		MatMap mat(prefix+to_string(ii));

		full.middleCols(curcol, reorg.tallMatCols()[ii]) = mat.mat;
		curcol += reorg.tallMatCols()[ii];
	}

	// Perform Full SVD
	cerr<<"Full SVD:"<<full.rows()<<"x"<<full.cols()<<endl;
	JacobiSVD<MatrixXd> fullsvd(full, ComputeThinU | ComputeThinV);
	const auto& fullS = fullsvd.singularValues();
	const auto& fullU = fullsvd.matrixU();

	MatrixXd mergeU, mergeV;
	VectorXd mergeS = onDiskSVD(reorg, estrank, poweriters, svt,
			&mergeU, &mergeV);

	size_t rank = mergeS.rows();
	cerr << "SVD Rank: " << rank << endl;

	cerr<<"Comparing Full S with Merge S"<<endl;
	cerr<<"Full S\n"<<fullS.transpose()<<endl;
	cerr<<"Full U\n"<<fullU.transpose()<<endl;
	cerr<<"Merge S\n"<<mergeS.transpose()<<endl;
	cerr<<"Merge U\n"<<mergeU.transpose()<<endl;
	for(size_t ii=0; ii<rank; ++ii) {
		cerr << fullS[ii] << " vs " << mergeS[ii] << endl;
		if(2*fabs(mergeS[ii] - fullS[ii])/fabs(mergeS[ii]+fullS[ii]) > thresh) {
			cerr<<"Difference in Singular Value "<<ii<<": "<<mergeS[ii]<<" vs "
				<<fullS[ii]<<endl;
			return -1;
		}
	}

	cerr<<"Comparing Full U with Merge U"<<endl;
	cerr<<fullS.rows()<<endl;
	cerr<<mergeS.rows()<<endl;
	for(size_t ii=0; ii<rank;  ++ii) {
		cerr<<"Dot "<<ii<<":"<<(mergeU.col(ii).dot(fullU.col(ii)))<<endl;
		if(1-fabs(mergeU.col(ii).dot(fullU.col(ii))) > thresh) {
			cerr<<"Difference in U col "<<ii<<": "<<mergeU.col(ii)<<" vs "
				<<fullU.col(ii)<<endl;
			return -1;
		}
	}

	return 0;
}

int main(int argc, char** argv)
{
	cerr << "Version: " << __version__ << endl;
	try {
	/*
	 * Command Line
	 */

	TCLAP::CmdLine cmd("Performs an SVD. If an input matrix is given then it "
			"performs the SVD on that matrix, otherwise it creates a random "
			"SVD-able matrix and compares the true SVD with the estimated. ",
			' ', __version__ );

	TCLAP::ValueArg<string> a_in("i", "input", "Input Matrix. Format: "
			"size_t size_t double double ...., where the first two size_t's "
			"are the number rows and columns and then the matrix data "
			"immediately follow.", false, "", "mat.bin", cmd);
	TCLAP::SwitchArg a_verbose("v", "verbose", "Print matrices", cmd);

	TCLAP::ValueArg<size_t> a_rows("r", "rows", "Rows in random matrix (only "
		"applied if no input matrix given.", false, 3, "rows", cmd);
	TCLAP::ValueArg<size_t> a_cols("c", "cols", "Cols in random matrix (only "
		"applied if no input matrix given.", false, 2, "cols", cmd);
	TCLAP::ValueArg<size_t> a_timepoints("t", "times", "Number of timepoints ",
			false, 10, "rank", cmd);
	TCLAP::ValueArg<size_t> a_hidden("s", "signals", "Number of hidden "
			"variables which are the acual signals",
			false, 10, "truesigs", cmd);

	TCLAP::ValueArg<double> a_svthresh("", "sv-thresh", "During dimension "
			"reduction, A singular value will be considered nonzero if its "
			"value is strictly greater than "
			"|singular value| < threshold x |max singular value|. "
			"By default this is 0.01.",
			false, 0.01, "ratio", cmd);
	TCLAP::ValueArg<int> a_rank("", "rank", "Minimum Rank to "
			"approximate with", false, 100, "rank", cmd);
	TCLAP::ValueArg<int> a_powerit("", "poweriter", "Number of power iterations, "
			"this enhaces accuracy at the cost of more passes over the input",
			false, 0, "iters", cmd);

	TCLAP::ValueArg<string> a_out("R", "randmat", "Output generated (random) "
			"matrix.", false, "", "mat.bin", cmd);

	cmd.parse(argc, argv);

	double vthresh = 0.95;

	if(argc == 2)
		vthresh = atof(argv[1]);

	std::random_device rd;
	unsigned int seed = rd();
	size_t timepoints = a_timepoints.getValue();
	size_t numhidden = a_hidden.getValue();
	size_t nrows = a_rows.getValue();
	size_t ncols = a_cols.getValue();
//	seed = 3888816431;

	cerr<<"Seed: "<<seed<<endl;
	std::default_random_engine rng(seed);
	std::uniform_real_distribution<double> dist(-1,1);

	std::string pref = "pca4";
	MatrixXd hidden(timepoints*nrows, numhidden);
	for(size_t cc=0; cc<hidden.cols(); cc++) {
		for(size_t rr=0; rr<hidden.rows(); rr++) {
			hidden(rr, cc) = dist(rng);
		}
	}

	// create random images
	vector<ptr<MRImage>> inputs(ncols*nrows);
	vector<ptr<MRImage>> masks(ncols);
	vector<std::string> fn_inputs(ncols*nrows);
	vector<std::string> fn_masks(ncols);
	for(size_t cc = 0; cc<ncols; cc++) {
		masks[cc] = randImage(INT8, 0, 1, 5, 17, 19, 0);
		fn_masks[cc] = pref+"mask_"+to_string(cc)+".nii.gz";
		masks[cc]->write(fn_masks[cc]);

		// Add 3 Primary Signals and Make the weights
		auto weights = randImage(FLOAT64, 0, 1, 5, 17, 19, numhidden);

		for(size_t rr = 0; rr<nrows; rr++) {
			inputs[rr+cc*nrows] = randImage(FLOAT64, 0, 0.01, 5, 17, 19, timepoints);

			// Add primary signals
			for(size_t ww=0; ww<numhidden; ww++) {
				Vector3DIter<double> wit(weights);
				Vector3DIter<double> it(inputs[rr+cc*nrows]);
				for( ; !it.eof(); ++it, ++wit) {
					for(size_t tt=0; tt<timepoints; tt++)
						it.set(tt, wit[ww]*hidden(tt+rr*timepoints, ww)+it[tt]);
				}
			}

			fn_inputs[rr+cc*nrows] = pref+to_string(cc)+"_"+
						to_string(rr)+".nii.gz";
			inputs[rr+cc*nrows]->write(fn_inputs[rr+cc*nrows]);
		}
	}

	MatrixReorg reorg(pref, 500000, true);
	if(reorg.createMats(nrows, ncols, fn_masks, fn_inputs) != 0)
		return -1;

	if(testTallPCAJoin(reorg, pref+"_tall_", vthresh, a_rank.getValue(),
				a_powerit.getValue()) != 0)
		return -1;

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr<<"error: "<<e.error()<<" for arg "<<e.argId()<<std::endl;}

}



