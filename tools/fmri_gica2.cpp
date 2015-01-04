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

int fmri_gica();

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

	TCLAP::MultiArg<string> a_in("i", "input", "Input fMRI image.",
			true, "*.nii.gz", cmd);

	TCLAP::ValueArg<string> a_mapdir("d", "mapdir", "Output "
			"directory for ICA significance maps. The number of maps will "
			"depend on the number of components, and will be in the same "
			"space as the input fMRI image. names will be the "
			"$mapdir/$input_$num.nii.gz where $mapdir is the mapdir, $input "
			"is the basename from -i and $num is the component number",
			true, "./", "/", cmd);

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

	return fmri_gica(a_spatial_ica.isSet(), a_in.getValue(),
			a_mapdir.getValue(), a_evthresh.getValue(),
			a_simultaneous.getValue(), a_maxrank.getValue()
			a_gbram.getValue());

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }

	return 0;
}

int fmri_gica(bool spacial, const vector<string>& files,
		string maskfile, string workdir, double evthresh,
		int lancvec, int maxrank, double gbmax)
{
	ptr<MRImage> mask;

	// Number of Rows in each Row (time-concat set of images)
	vector<int> nrows(files.size(), -1);
	int totalrows = 0;
	
	// Convert Each Image into a Flat Matrix
	int ncols = 0;
	
	// Load Each Image and change to matrix
	for(int tt=0; tt<files.size(); tt++) {

		// Read Image
		auto img = readMRImage(files[tt]);
		nrows[tt] = img->tlen();
		totalrows += nrows[tt];

		// Load or Compute Mask
		if(!mask) {
			if(!maskfile.empty())
				mask = readMRImage(maskfile);
			else
				mask = dPtrCast<MRImage>(binarize(varianceT(img), 0));
			
			// Figure out number of columns from mask
			int ncols = 0;
			for(FlatIter<int> it(mask); !it.eof(); ++it) {
				if(*it != 0)
					ncols++
			}
		
			// Write Mask
			mask->write(workdir+"/mask.nii.gz");
		}

		// Create mat file by memory mapping
		string fn = workdir+"/mat_"+to_string(tt);
		MemMap mmap(fn, nrows[ii]*ncols+sizeof(double), true);
		if(mmap.size() <= 0)
			return -1;

		// Fill Matrix from Image
		double* ptr = (double*)mmap.data();
		if(fillMat(ptr, nrows[ii], ncols, img, mask) != 0) {
			cerr<<"Error Filling Matrix"<<endl;
			return -1;
		}
	}
	
	if(a_verbose.getValue() >= 1) {
		cerr << "Data Dimensions: " << endl;
		cerr << "# Concatination in Time: " << tcat << endl;
		cerr << "Total Time Size: " << totalrows << endl;
		cerr << "(Single) Image Space Size: " << ncols << endl;
	}
	
	// Convert Row Blocks to Col Blocks
	convert
	TODO	
	// Perform SVD on each block
	
	// Perform SVD on [U_1S_1 U_2S_2 ... ]
	
	// First 
	size_t nscalar = ncols*totalrows;

	MatrixXd cov;

	/*
	 * First Perform SVD, Whiten
	 */

	MatrixLoader loader(tlen, slen, tsing, ssing, a_in.getValue());
	MatrixXd white = whiten(a_spatial_ica.isSet(), a_simultaneous.getValue(),
				a_evthresh.getValue(), a_maxrank.getValue(), 1e-10, tlen, slen,
				a_time_append.getValue(), a_space_append.getValue(), loader);

	if(a_verbose.getValue() >= 4) {
		cout << "Whitened = [";
		for(size_t rr=0; rr<white.rows(); rr++) {
			if(rr) cout << "],\n";
			for(size_t cc=0; cc<white.cols(); cc++) {
				if(cc) cout << ",";
				else cout << "[";
				cout << white(rr,cc);
			}
		}
		cout << "]]\n";
	}

	// Each Column is a Dimension Now
	// perform ICA
	std::cerr << "ICA...";
	MatrixXd X_ic = ica(white);
	std::cerr << "Done" << endl;

//	//
//	MatrixXd regressors = reduce(inimg);
//	for(size_t cc = 0; cc < regressors.cols(); cc++) {
//		// perform regression
//		//RegrResult tmp = regress(inimg, regressors.row(cc));
//
//		// write out each of the images
//		//tmp.rsqr->write("rsqr_"+to_string(cc)+".nii.gz");
//		//tmp.T->write("T_"+to_string(cc)+".nii.gz");
//		//tmp.p->write("p_"+to_string(cc)+".nii.gz");
//		//tmp.beta->write("beta_"+to_string(cc)+".nii.gz");
//	}

}

MatrixXd whiten(bool whiterows, int initrank, double varthresh, int maxrank,
		double svthresh, int trows, int tcols, int rchunks, int cchunks,
		MatrixLoader& loader)
{
	MatrixXd whitened;
	MatrixXd C;

	/**************************************************************************
	 * Load Individual Images and Concatinate to Produce Parts of X
	 *************************************************************************/
	if(trows > tcols) {
		// Compute right singular vlaues (V)
		C.resize(tcols, tcols);
		C.setZero();

		// we need to load entire rows (all cols) at a time
		MatrixXd buffer(trows/rchunks, tcols);
		for(int rr=0; rr<rchunks; rr++) {
			for(int cc=0; cc<cchunks; cc++) {
				loader.load(buffer, 0, cc*tcols/cchunks, rr, cc);
			}
			C += buffer.transpose()*buffer;
		}

	} else {
		// Computed left singular values (U)
		C.resize(trows, trows);
		C.setZero();

		// we need to load entire cols (all rows) at a time
		MatrixXd buffer(trows, tcols/cchunks);
		for(int cc=0; cc<cchunks; cc++) {
			for(int rr=0; rr<rchunks; rr++) {
				loader.load(buffer, rr*trows/rchunks, 0, rr, cc);
			}
			C += buffer*buffer.transpose();
		}
	}

	/**************************************************************************
	 * Perform EigenValue Decomp on either X^T X or X X^T
	 *************************************************************************/
	// This is a bit hackish, really the user should set this
	if(initrank <= 1)
		initrank = std::max<int>(trows, tcols);

	BandLanczosSelfAdjointEigenSolver<double> eig;
	eig.setTraceStop(varthresh);
	eig.setRank(maxrank);
	eig.compute(C, initrank);

	if(eig.info() == NoConvergence) {
		throw "Non-convergence!";
	}

	int eigrows = eig.eigenvalues().rows();
	int rank = 0;
	VectorXd singvals(eigrows);
	for(int cc=0; cc<eigrows; cc++) {
		if(eig.eigenvalues()[eigrows-1-cc] < svthresh)
			singvals[cc] = 0;
		else {
			singvals[cc] = std::sqrt(eig.eigenvalues()[eigrows-1-cc]);
			rank++;
		}
	}

	singvals.conservativeResize(rank);

	/**************************************************************************
	 * If we want Independent Cols then we need U, for Independent Rows, V^T
	 *************************************************************************/
	// Note that because Eigen Solvers usually sort eigenvalues in
	// increasing order but singular value decomposers do decreasing order,
	// we need to reverse the singular value and singular vectors found.
	if(trows > tcols) {
		// Computed right singular vlaues (V)
		// A = USV*, U = AVS^-1

		// reverse and fill V
		MatrixXd V(eig.eigenvectors().rows(), rank);
		for(int cc=0; cc<rank; cc++)
			V.col(cc) = eig.eigenvectors().col(eigrows-1-cc);

		// Compute U if needed
		if(!whiterows) {
			// Need to load entire rows (all cols) of A, then set
			// corresponding rows in U, U = A*VS;
			MatrixXd VS = V*(singvals.cwiseInverse()).asDiagonal();
			MatrixXd U(trows, rank);
			MatrixXd buffer(trows/rchunks, tcols);
			for(int rr=0; rr<rchunks; rr++) {
				for(int cc=0; cc<cchunks; cc++) {
					loader.load(buffer, 0, cc*tcols/cchunks, rr, cc);
				}
				U.middleRows(rr*trows/rchunks, trows/rchunks) = buffer*VS;
			}

			return U;
		} else {
			// Whiten Rows, already have V
			return V;
		}
	} else {
		// Computed left singular values (U)
		// A = USV*, A^T = VSU*, V = A^T U S^-1

		// reverse and fill U
		MatrixXd U(eig.eigenvectors().rows(), rank);
		for(int cc=0; cc<rank; cc++)
			U.col(cc) = eig.eigenvectors().col(eigrows-1-cc);

		if(!whiterows) {
			// Whiten Cols , already have U
			return U;
		} else {

			// To Compute A^T US, we will compute R rows of V at a time, which
			// corresponds to R columns of A (R rows of A^T)
			// (all rows) of A at a time
			MatrixXd US = U*(singvals.cwiseInverse()).asDiagonal();
			MatrixXd V(tcols, rank);
			MatrixXd buffer(trows, tcols/cchunks);
			for(int cc=0; cc<rchunks; cc++) {
				for(int rr=0; rr<rchunks; rr++) {
					loader.load(buffer, rr*trows/rchunks, 0, rr, cc);
				}

				V.middleRows(cc*tcols/cchunks, tcols/cchunks) = buffer.transpose()*US;
			}

			return V;
		}
	}
}


