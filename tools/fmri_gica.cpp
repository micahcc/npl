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
 * @file fmri_gica.cpp Tool for performing ICA on multiple fMRI images, 
 * note that this assumes either X^T X or XX^T can be stored in memory
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

class MatrixLoader
{
public:
	MatrixLoader(int rows, int cols, int b_rows, int b_cols,
			const vector<string>& filenames)
	{
		m_sz[0] = rows;
		m_sz[1] = cols;
		m_bsz[0] = b_rows;
		m_bsz[1] = b_cols;
		m_chunks[0] = rows/b_rows;
		m_chunks[1] = cols/b_cols;
		assert(rows%b_rows == 0);
		assert(cols%b_cols == 0);

		m_fnames = filenames;
	};

	void load(MatrixXd& buffer, int buff_row, int buff_col, int brow, int bcol)
	{
		std::string fn = m_fnames[bcol*m_chunks[0]+brow];
		auto img = readMRImage(fn);
		if(!img || img->tlen() != m_bsz[0] || 
				img->elements()/m_bsz[0]!= m_bsz[1]) {
			throw std::string("Error reading ")+fn+
				std::string(" or Image sizes differ!");
		}
		if(a_verbose.getValue() >= 3) {
			cerr << *img << endl;
		}

		Vector3DIter<double> it(img);
		for(size_t s=0; s<m_bsz[1]; ++it, s++) { // small space
			for(size_t t=0; t<m_bsz[0]; t++) // small time
				buffer(buff_row+t, buff_col+s) = it[t];
		}
	};

	vector<string> m_fnames;
	int m_sz[2];
	int m_bsz[2];
	int m_chunks[2];
};

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

	if(a_verbose.getValue() >= 1) {
		cerr << "Data Dimensions: " << endl;
		cerr << "# Concatination in Time: " << tcat << endl;
		cerr << "# Concatination in Space : " << scat << endl;
		cerr << "Single Image Time Size: " << tsing << endl;
		cerr << "Single Image Space Size: " << ssing << endl;
		cerr << "Total Time Size: " << tlen << endl;
		cerr << "Toal Space Size: " << slen << endl;
	}
	
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

