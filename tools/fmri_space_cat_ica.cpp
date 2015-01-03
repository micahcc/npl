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
 * @file fmri_psd_ica.cpp Tool for performing ICA on multiple fMRI images by
 * first concatinating in space. This requires much less memory for the SVD
 * than concatinating in time. For more general usage you can optionally 
 * perform ICA on Power Spectral Density, which makes the specific time-courses
 * irrelevent, and insteady groups data by frequency.
 *****************************************************************************/

#include <version.h>
#include <tclap/CmdLine.h>
#include <string>
#include <stdexcept>

#include <Eigen/Dense>
#include <Eigen/IterativeSolvers>

#include "fftw3.h"

#include "mrimage.h"
#include "nplio.h"
#include "iterators.h"
#include "statistics.h"
#include "utility.h"
#include "ndarray_utils.h"
#include "macros.h"

using std::string;
using std::shared_ptr;
using std::to_string;

using namespace Eigen;
using namespace npl;

TCLAP::MultiSwitchArg a_verbose("v", "verbose", "Write lots of output. "
		"Be wary for large datasets!", 1);

int fillMat(double* rawdata, size_t nrow, size_t ncol, 
		ptr<const MRImage> img, ptr<const MRImage> mask);

int fillMatPSD(double* rawdata, size_t nrow, size_t ncol, 
		ptr<const MRImage> img, ptr<const MRImage> mask);

int spacecat_ica(bool psd, const vector<string>& imgnames, 
		const vector<string>& masknames, string workdir, double evthresh, 
		int lancbasis, int maxrank, bool spatial);

int main(int argc, char** argv)
{
	cerr << "Version: " << __version__ << endl;
	try {
	/*
	 * Command Line
	 */

	TCLAP::CmdLine cmd("Perform ICA analysis on an image, or group of "
			"images by concating in space. This only works if 1) the images "
			"have identical timecourses or 2) you select -p to transform the "
			"time-series into the power-spectral-density prior to performing "
			"ICA analysis. The SVD memory usage only depends on the squre of "
			"the longest single subjects time-course rather than the "
			"concatined time-course (which is what the normal memory "
			"complexity of PCA is.", ' ', __version__ );

	TCLAP::SwitchArg a_spatial("S", "spatial-ica", "Compute the spatial ICA. "
			"Note that LxS values must fit into memory to do this, where L is "
			"the PCA-reduced dataset and S is the totally number of masked "
			"spatial locations across all subjects", cmd);
	
	TCLAP::SwitchArg a_psd("p", "power-spectral-density", "Compute the power "
			"spectral density of timer-series. The advantage of this is that "
			"concating in space will work even if the timeseries of the "
			"individual images are not identical. The only assumption is that "
			"similar tasks have similar domonant frequencies. ", cmd);

	TCLAP::MultiArg<string> a_in("i", "input", "Input fMRI images.",
			true, "*.nii.gz", cmd);
	TCLAP::MultiArg<string> a_masks("m", "mask", "Input masks for fMRI. Note "
			"that it is not necessary to mask if all the time-series outsize "
			"the brain are 0, since those regions will be removed anyway",
			false, "*.nii.gz", cmd);

//    TCLAP::ValueArg<string> a_components("o", "out-components", "Output "
//            "Independent Components as a 1x1xCxT image.",
//			true, "", "*.nii.gz", cmd);
    TCLAP::ValueArg<string> a_workdir("d", "dir", "Output "
            "directory. Large intermediate matrices will be written here as "
			"well as the spatial maps. ", true, "./", "path", cmd);

	TCLAP::ValueArg<double> a_evthresh("T", "ev-thresh", "Threshold on "
			"ratio of total variance to account for (default 0.99)", false,
			INFINITY, "ratio", cmd);
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

	return spacecat_ica(a_psd.isSet(), a_in.getValue(), a_masks.getValue(), 
			a_workdir.getValue(), a_evthresh.getValue(),
			a_simultaneous.getValue(), a_maxrank.getValue(), 
			a_spatial.isSet());

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }

    return 0;
}

int spatial_cat_orthog(string workdir, double evthresh,
		int lancbasis, int maxrank, bool spatial, const vector<int>& ncols,
		const MatrixXd& XXT, MatrixXd& regr)
{
	// Compute EigenVectors (U)
	BandLanczosSelfAdjointEigenSolver<double> eig;
	eig.setTraceStop(evthresh);
	eig.setRank(maxrank);
	eig.compute(XXT, lancbasis);

	if(eig.info() == NoConvergence) {
		cerr << "Non Convergence of BandLanczosSelfAdjointEigenSolver" << endl;
		return -1;
	}

	double sum = 0;
	double totalev = eig.eigenvalues().sum();
	int eigrows = eig.eigenvalues().rows();
	int rank = 0;
	VectorXd S(eigrows);
	for(int cc=0; cc<eigrows; cc++) {
		double v = eig.eigenvalues()[eigrows-1-cc];
		if(v < std::numeric_limits<double>::epsilon() ||
				(evthresh >= 0 && evthresh <= 1 && (sum > totalev*evthresh)))
			S[cc] = 0;
		else {
			S[cc] = std::sqrt(eig.eigenvalues()[eigrows-1-cc]);
			rank++;
		}
		sum += v;
	}
	S.conservativeResize(rank);

	// Computed left singular values (U)
	// A = USV*, A^T = VSU*, V = A^T U S^-1
	MatrixXd U(eig.eigenvectors().rows(), rank);
	for(int cc=0; cc<rank; cc++)
		U.col(cc) = eig.eigenvectors().col(eigrows-1-cc);

	if(!spatial) {
		// U is what we need
		regr = U;
	} else {
		// Compute V = X^T U S^-1
		// rows of V correspond to cols of X
		size_t totalcols = 0;
		for(auto C : ncols) totalcols += C;
		regr.resize(totalcols, S.rows());

		int curr_row = 0;
		for(int ii=0; ii<ncols.size(); ii++) {
			string fn = workdir+"/psdmat_"+to_string(ii);
			MemMap mmap(fn, ncols[ii]*U.rows()+sizeof(double), false);
			if(mmap.size() <= 0)
				return -1;

			Eigen::Map<MatrixXd> X((double*)mmap.data(), U.rows(), ncols[ii]);
			regr.middleRows(curr_row, ncols[ii]) = X.transpose()*U*
						S.cwiseInverse().asDiagonal();

			curr_row += ncols[ii];
		}
	}
	return 0;
}

int spacecat_ica(bool dopsd, const vector<string>& imgnames, 
		const vector<string>& masknames, string workdir, double evthresh, 
		int lancbasis, int maxrank, bool spatial)
{
	/**********
	 * Input
	 *********/
	if(imgnames.empty()) {
		cerr << "Need to provide at least 1 input image!" << endl;
		return -1;
	}

	int nrows = -1;
	vector<int> ncols(imgnames.size());
	int totalcols = 0;

	// Read Each Image, Create a Mask, save the mask
	for(int ii=0; ii<imgnames.size(); ii++) {

		// Read Image
		auto img = readMRImage(imgnames[ii]);

		// Load or Compute Mask
		ptr<MRImage> mask;
		if(ii < masknames.size()) {
			mask = readMRImage(masknames[ii]);
		} else {
			mask = dPtrCast<MRImage>(varianceT(img));
		}

		// Figure out number of columns from mask
		ncols[ii] = 0;
		for(FlatIter<int> it(mask); !it.eof(); ++it) {
			if(*it != 0)
				ncols[ii]++;
		}
		totalcols += ncols[ii];

		// Write Mask
		mask->write(workdir+"/mask_"+to_string(ii)+"_"+".nii.gz");
		
		// Create Output File to buffer file, if we are going to FFT, the 
		// round to the nearest power of 2
		if(dopsd) {
			if(nrows <= 0)
				nrows = round2(img->tlen());
			else if(nrows != round2(img->tlen())) {
				cerr << "Number of time-points differ in inputs!" << endl;
				return -1;
			}
		} else {
			if(nrows <= 0)
				nrows = img->tlen();
			else if(nrows != img->tlen()) {
				cerr << "Number of time-points differ in inputs!" << endl;
				return -1;
			}
		}

		string fn = workdir+"/psdmat_"+to_string(ii);
		MemMap mmap(fn, ncols[ii]*nrows+sizeof(double), true);
		if(mmap.size() <= 0)
			return -1;

		double* ptr = (double*)mmap.data();
		if(dopsd) {
			if(fillMatPSD(ptr, nrows, ncols[ii], img, mask) != 0) {
				cerr<<"Error computing Power Spectral Density"<<endl;
				return -1;
			}
		} else {
			if(fillMat(ptr, nrows, ncols[ii], img, mask) != 0) {
				cerr<<"Error Filling Matrix"<<endl;
				return -1;
			}
		}
	}

	// Compute SVD of inputs
	MatrixXd XXT(nrows, nrows);
	XXT.setZero();

	// Sum XXT_i for i in 1...subjects
	for(int ii=0; ii<imgnames.size(); ii++) {
		string fn = workdir+"/psdmat_"+to_string(ii);
		MemMap mmap(fn, ncols[ii]*nrows+sizeof(double), false);
		if(mmap.size() <= 0)
			return -1;

		// Sum up XXT from each chunk
		Eigen::Map<MatrixXd> X((double*)mmap.data(), nrows, ncols[ii]);
		XXT += X*X.transpose();
	}

	// Each column is a regressor
	MatrixXd regressors;
	if(spatial_cat_orthog(workdir, evthresh, lancbasis, maxrank,
				spatial, ncols, XXT, regressors) != 0) {
		return -1;
	}

	// Now Perform ICA
	MatrixXd X_ics = ica(regressors);

	// Create Maps
	// TODO

	return 0;
}

int fillMat(double* rawdata, size_t nrows, size_t ncols,
		ptr<const MRImage> img, ptr<const MRImage> mask)
{
	if(!mask->matchingOrient(img, false, true)) {
		cerr << "Mask and Image have different orientation or size!" << endl;
		return -1;
	}
		
	// Create View of Matrix, and fill with PSD in each column
	Eigen::Map<MatrixXd> mat(rawdata, nrows, ncols);
	
	Vector3DConstIter<double> iit(img);
	NDConstIter<int> mit(mask);
	size_t cc=0;
	for(; !mit.eof() && !iit.eof(); ++iit, ++mit) {
		if(*mit != 0)
			continue;

		if(nrows != img->tlen()) {
			cerr<<"Input matrix size does not match time-length!"<<endl;
			return -1;
		}

		for(int rr=0; rr<nrows; rr++)
			mat(rr, cc) =  iit[rr];

		// Only Increment Columns for Non-Zero Mask Pixels
		++cc;
	}

	return 0;
}

int fillMatPSD(double* rawdata, size_t nrows, size_t ncols,
		ptr<const MRImage> img, ptr<const MRImage> mask)
{
	if(!mask->matchingOrient(img, false, true)) {
		cerr << "Mask and Image have different orientation or size!" << endl;
		return -1;
	}

	// Create View of Matrix, and fill with PSD in each column
	Eigen::Map<MatrixXd> mat(rawdata, nrows, ncols);

	double* ibuffer;
	fftw_complex* obuffer;
	fftw_plan fwd;
	ibuffer = fftw_alloc_real(mat.rows());
	obuffer = fftw_alloc_complex(mat.rows());
	fwd = fftw_plan_dft_r2c_1d(mat.rows(), ibuffer, obuffer, FFTW_MEASURE);
	for(size_t ii=0; ii<mat.rows(); ii++)
		ibuffer[ii] = 0;

	size_t tlen = img->tlen();
	Vector3DConstIter<double> iit(img);
	NDConstIter<int> mit(mask);
	size_t cc=0;
	for(; !mit.eof() && !iit.eof(); ++iit, ++mit) {
		if(*mit != 0)
			continue;

		// Perform FFT
		for(int tt=0; tt<tlen; tt++)
			ibuffer[tt] = iit[tt];
		fftw_execute(fwd);

		// Convert FFT to Power Spectral Density
		for(int rr=0; rr<mat.rows(); rr++) {
			std::complex<double> cv (obuffer[rr][0], obuffer[rr][1]);
			double rv = std::abs(cv);
			mat(rr, cc) = rv*rv;
		}

		// Only Increment Columns for Non-Zero Mask Pixels
		++cc;
	}

	return 0;
}
