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
 * @file fmri_gica2.cpp New version of group ICA that performs two SVD's to
 * reduce the memory footprint, prior to performing ICA.
 *
 *****************************************************************************/

#include <version.h>
#include <tclap/CmdLine.h>
#include <string>
#include <stdexcept>

#include <Eigen/Dense>
#include <Eigen/IterativeSolvers>

#include "ica_helpers.h"
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

/**
 * @brief The basic idea is to split the rows into digesteable chunks, then
 * perform the SVD on each of them.
 *
 * A = [A1 A2 A3 ... ]
 * A = [UEV1 UEV2 .... ]
 * A = [UE1 UE2 UE3 ...] diag([V1, V2, V3...])
 *
 * UE1 should have far fewer columns than rows so that where A is RxC,
 * with R < C, [UE1 ... ] should have be R x LN with LN < R
 *
 * Say we are concatinating S subjects each with T timepoints, then
 * A is STxC, assuming a rank of L then [UE1 ... ] will be ST x SL
 *
 * Even if L = T / 2 then this is a 1/4 savings in the SVD computation
 *
 * @param timeblocks Number of fMRI images to append in time direction
 * @param spaceblocks Number of fMRI images to append in space direction
 * @param prefix Prefix for output files (matrices)
 * @param masks Masks, one per spaceblock (columns of matching space)
 * @param files Files in time-major order, [s0t0 s0t1 s0t2 s1t0 s1t1 s1t2]
 * where s0 means 0th space-appended image, and t0 means the same for time
 * @param evthresh Threshold for eigenvalues (ratio of maximum)
 * @param sumvar Total variance to explain with SVD/PCA
 * @param lancvec Number of lanczos vectors to initialize SVD/BandLanczos Eigen
 * Solver with, a good starting point is 2* the number of expected PC's, if
 * convergence fails, use more
 * @param iters Maximum number of iterations in BandLanczos Eigen Solve
 * @param gbmax Maximum number of gigabytes of memory to use
 * @param spatial Perform Spatial ICA, if not a temporal ICA is done
 *
 * @return
 */
MatrixXd fmri_gica(size_t timeblocks, size_t spaceblocks, string prefix,
		const vector<string>& masks, const vector<string>& files,
		double evthresh, double sumvar, int lancvec, int iters, double gbmax,
		bool spatial);

int main(int argc, char** argv)
{
	cerr << "Version: " << __version__ << endl;
	try {
	/*
	 * Command Line
	 */

	TCLAP::CmdLine cmd("Perform ICA analysis on an image, or group of images. "
			"By default this will be a temporal ICA. Group analysis is "
			"done by concatinating in time. By default a Temporal ICA will "
			"be performed, to do a spatial ICA select -S.", ' ', __version__ );

	TCLAP::MultiArg<string> a_in("i", "input", "Input fMRI images. Ordered in "
			"time-major (adjacent inputs will be next to each other during "
			"concatination). Every s images will correspond to a mask (where "
			"s is the value provided by -s).",
			true, "*.nii.gz", cmd);
	TCLAP::MultiArg<string> a_masks("m", "mask", "Mask images.",
			true, "*.nii.gz", cmd);

	TCLAP::ValueArg<string> a_prefix("p", "prefix", "Output "
			"prefix for ICA significance maps and intermediate data files. "
			"The number of maps will "
			"depend on the number of components, and will be in the same "
			"space as the input fMRI image. names will be the "
			"$prefix_$input_$num.nii.gz where $prefix is this, $input "
			"is the basename from -i and $num is the component number",
			true, "./", "/", cmd);

	TCLAP::ValueArg<int> a_time_append("t", "time-concat", "Number of images "
			"in a rows of time-concatination. ", false, -1,
			"#vecs", cmd);
	TCLAP::ValueArg<int> a_space_append("s", "space-concat", "Number of images "
			"in a rows of spatial-concatination. ", false, -1,
			"#vecs", cmd);

	TCLAP::ValueArg<double> a_evthresh("", "ev-thresh", "Threshold on "
			"ratio of total variance to account for (default 0.99)",
			false, 0.1, "ratio", cmd);
	TCLAP::ValueArg<double> a_varthresh("", "ev-thresh", "Threshold on "
			"ratio of total variance to account for (default 0.99)",
			false, 0.99, "ratio", cmd);
	TCLAP::ValueArg<int> a_simultaneous("V", "simul-vectors", "Simultaneous "
			"vectors to estimate eigenvectors for in lambda. Bump this up "
			"if you have convergence issues. This is part of the "
			"Band Lanczos Eigenvalue Decomposition of the covariance matrix. "
			"By default the covariance size is used, which is conservative. "
			"Values less than 1 will default back to that", false, -1,
			"#vecs", cmd);
	TCLAP::ValueArg<int> a_iters("", "max-iters", "Maximum iterations "
			"in PCA. This sets are hard limit on the number of estimated "
			"eigenvectors in the Band Lanczos Solver (and thus the maximum "
			"number of singular values in the SVD. Be wary of making it "
			"too small). -1 removes limit", false, -1, "#vecs", cmd);
	TCLAP::ValueArg<double> a_gbram("M", "memory-max", "Maximum number of GB "
			"of RAM to use for chunks. This is needed to decide how to divide "
			"up data into spatial chunks. ", false, -1, "#vecs", cmd);

	TCLAP::SwitchArg a_spatial_ica("S", "spatial-ica", "Perform a spatial ICA"
			", reducing unmixing timepoints to produce spatially independent "
			"maps.", cmd);

	cmd.add(a_verbose);
	cmd.parse(argc, argv);

	/**********
	 * Input
	 *********/
	if(!a_in.isSet()) {
		cerr << "Need to provide at least 1 input image!" << endl;
		return -1;
	}

	MatrixXd ics = fmri_gica(a_time_append.getValue(),
			a_space_append.getValue(), a_prefix.getValue(), a_masks.getValue(),
			a_in.getValue(), a_evthresh.getValue(), a_varthresh.getValue(),
			a_simultaneous.getValue(), a_iters.getValue(), a_gbram.getValue(),
			a_spatial_ica.isSet());

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }

	return 0;
}

/**
 * @brief The basic idea is to split the rows into digesteable chunks, then
 * perform the SVD on each of them.
 *
 * A = [A1 A2 A3 ... ]
 * A = [UEV1 UEV2 .... ]
 * A = [UE1 UE2 UE3 ...] diag([V1, V2, V3...])
 *
 * UE1 should have far fewer columns than rows so that where A is RxC,
 * with R < C, [UE1 ... ] should have be R x LN with LN < R
 *
 * Say we are concatinating S subjects each with T timepoints, then
 * A is STxC, assuming a rank of L then [UE1 ... ] will be ST x SL
 *
 * Even if L = T / 2 then this is a 1/4 savings in the SVD computation
 *
 * @param timeblocks Number of fMRI images to append in time direction
 * @param spaceblocks Number of fMRI images to append in space direction
 * @param prefix Prefix for output files (matrices)
 * @param masks Masks, one per spaceblock (columns of matching space)
 * @param files Files in time-major order, [s0t0 s0t1 s0t2 s1t0 s1t1 s1t2]
 * where s0 means 0th space-appended image, and t0 means the same for time
 * @param evthresh Threshold for eigenvalues (ratio of maximum)
 * @param sumvar Total variance to explain with SVD/PCA
 * @param lancvec Number of lanczos vectors to initialize SVD/BandLanczos Eigen
 * Solver with, a good starting point is 2* the number of expected PC's, if
 * convergence fails, use more
 * @param iters Maximum number of iterations in BandLanczos Eigen Solve
 * @param gbmax Maximum number of gigabytes of memory to use
 * @param spatial Perform Spatial ICA, if not a temporal ICA is done
 *
 * @return
 */
MatrixXd fmri_gica(size_t timeblocks, size_t spaceblocks, string prefix,
		const vector<string>& masks, const vector<string>& files,
		double evthresh, double sumvar, int lancvec, int iters, double gbmax,
		bool spatial)
{
	// Don't use more than half of memory on each block of rows
	cerr << "Reorganizing data into matrices...";
	MatrixReorg reorg(prefix, (size_t)0.5*gbmax*(1<<27), a_verbose.isSet());
	int status = reorg.createMats(timeblocks, spaceblocks, masks, files);
	if(status != 0)
		throw RUNTIME_ERROR("Error while reorganizing data into 2D Matrices");

	cerr<<"Done"<<endl;

	double thresh = 0.1;
	size_t totrows = reorg.rows();
	size_t totcols = reorg.cols();
	size_t curcol = 0;
	size_t catcols = 0; // Number of Columns when concatinating horizontally
	size_t maxrank = 0;

	for(size_t ii=0; ii<reorg.ntall(); ii++) {
		MatMap talldata(reorg.tallMatName(ii));

		cerr<<"Chunk SVD:"<<talldata.mat.rows()<<"x"<<talldata.mat.cols()<<endl;
		TruncatedLanczosSVD<MatrixXd> svd;
		svd.setThreshold(evthresh);
		svd.setTraceStop(sumvar);
		svd.setLanczosBasis(lancvec);
		svd.compute(talldata.mat, ComputeThinU | ComputeThinV);
		if(svd.info() == NoConvergence)
			throw RUNTIME_ERROR("Error computing Tall SVD, might want to "
					"increase # of lanczos vectors");

		cerr << "SVD Rank: " << svd.rank() << endl;
		maxrank = std::max<size_t>(maxrank, svd.rank());

		// write
		string usname = prefix+"US_"+to_string(ii);
		string vname = prefix +"V_"+to_string(ii);

		MatMap tmpmat;

		// Create UE
		tmpmat.create(usname, talldata.mat.rows(), svd.rank());
		tmpmat.mat = svd.matrixU().leftCols(svd.rank())*
			svd.singularValues().head(svd.rank());

		tmpmat.create(vname, talldata.mat.cols(), svd.rank());
		tmpmat.mat = svd.matrixV().leftCols(svd.rank());

		catcols += svd.rank();
	}

	// Merge / Construct Column(EV^T, EV^T, ... )
	MatrixXd mergedUE(totrows, catcols);
	curcol = 0;
	for(size_t ii=0; ii<reorg.ntall(); ii++) {
		MatMap tmpmat(prefix+"US_"+to_string(ii));

		mergedUE.middleCols(curcol, tmpmat.cols) = tmpmat.mat;
		curcol += tmpmat.cols;
	}

	cerr<<"Merge SVD:"<<mergedUE.rows()<<"x"<<mergedUE.cols()<<endl;
	MatrixXd U, V;
	VectorXd E;
	{
		TruncatedLanczosSVD<MatrixXd> svd;
		svd.setThreshold(evthresh);
		svd.setTraceStop(sumvar);
		svd.setLanczosBasis(lancvec);
		svd.compute(mergedUE, ComputeThinU | ComputeThinV);
		if(svd.info() == NoConvergence)
			throw RUNTIME_ERROR("Error computing Merged SVD, might want to "
					"increase # of lanczos vectors");

		U = svd.matrixU();
		E = svd.singularValues();
		V = svd.matrixV();
	}

	/*
	 * Recall:
	 *
	 * Assume A1 = T1 S1 C1, etc
	 * (note C is actually C^T and V is actually V^T)
	 *
	 * A = [TS1 TS2 TS3 ...] diag([C1, C2, C3...])
	 *
	 * Given [TS1 ... ] = UEV
	 * A = UEV diag([C1, C2, C3...])
	 * U_A = U
	 * E_A = E
	 * V_A^T = V^T diag([C1^T, C2^T, C3^T...])
	 * V_A = diag([C1 C2 C3...]) V
	 *
	 *
	 * */
	if(!spatial) {
		cerr << "Performing ICA" << endl;
		return ica(U);
	} else {
		cerr << "Constructing Full V...";
		// store fullV in U, since unneeded
		U.resize(reorg.rows(), V.cols());

		size_t currow = 0;
		for(size_t ii=0; ii<reorg.ntall(); ii++) {
			string vname = prefix +"V_"+to_string(ii);
			MatMap C(vname);

			U.middleRows(currow, C.mat.rows()) = C.mat*V;
			currow += C.mat.rows();
		}
		cerr << "Done\nPerforming ICA" << endl;
		return ica(U);
	}
}


