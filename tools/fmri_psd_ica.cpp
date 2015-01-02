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
 * first computing the power spectral density of each image's timeseries
 * then concatinating in space. This requires much less memory for the SVD
 * than concatinating in time.
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
		cerr << *img << endl;

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
#ifdef VERYVERBOSE
		cerr << "Computing A^T*A" << endl;
		cerr << endl << C << endl << endl;
#endif //VERYVERBOSE
		// Compute right singular vlaues (V)
		C.resize(tcols, tcols);
		C.setZero();

		// we need to load entire rows (all cols) at a time
		MatrixXd buffer(trows/rchunks, tcols);
		for(int rr=0; rr<rchunks; rr++) {
			for(int cc=0; cc<cchunks; cc++) {
				loader.load(buffer, 0, cc*tcols/cchunks, rr, cc);
			}
#ifdef VERYVERBOSE
			cerr << "Full Buffer " << rr << "\n" << buffer << endl << endl;
#endif //VERYVERBOSE
			C += buffer.transpose()*buffer;
		}

#ifdef VERYVERBOSE
		cerr << "Computed A^T A" << endl;
		cerr << endl << C << endl << endl;
#endif //VERYVERBOSE
	} else {
#ifdef VERYVERBOSE
		cerr << "Computing A*A^T" << endl;
		cerr << endl << C << endl << endl;
#endif //VERYVERBOSE
		// Computed left singular values (U)
		C.resize(trows, trows); 
		C.setZero();

		// we need to load entire cols (all rows) at a time
		MatrixXd buffer(trows, tcols/cchunks);
		for(int cc=0; cc<cchunks; cc++) {
			for(int rr=0; rr<rchunks; rr++) {
				loader.load(buffer, rr*trows/rchunks, 0, rr, cc);
			}
#ifdef VERYVERBOSE
			cerr << "Full Buffer " << cc << "\n" << buffer << endl << endl;
#endif //VERYVERBOSE
			C += buffer*buffer.transpose();
		}
#ifdef VERYVERBOSE
		cerr << "Computed A*A^T" << endl;
		cerr << endl << C << endl << endl;
#endif //VERYVERBOSE
	}

	/**************************************************************************
	 * Perform EigenValue Decomp on either X^T X or X X^T
	 *************************************************************************/
	// This is a bit hackish, really the user should set this
	if(initrank <= 1)
		initrank = std::max<int>(trows, tcols);

	BandLanczosSelfAdjointEigenSolver<double> eig;
	eig.setTraceSqrStop(varthresh);
	eig.setRank(maxrank);
	eig.compute(C, initrank);

	if(eig.info() == NoConvergence) {
		throw "Non-convergence!";
	}

#ifdef VERYVERBOSE
	cerr << "Eigenvalues: " << eig.eigenvalues().transpose() << endl;
	cerr << "EigenVectors: " << endl << eig.eigenvectors() << endl << endl;
#endif //VERYVERBOSE
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

#ifdef VERYVERBOSE
	for(int ee=0; ee<rank; ee++) {
		double err = (eig.eigenvalues()[ee]*eig.eigenvectors().col(ee) -
				C*eig.eigenvectors().col(ee)).squaredNorm();
		cerr << "Error = " << err << endl;
	}
#endif //VERYVERBOSE

	singvals.conservativeResize(rank);
#ifdef VERYVERBOSE
	cerr << "Singular Values: " << singvals.transpose() << endl;
#endif //VERYVERBOSE

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

#ifdef VERYVERBOSE
		cerr << "V Matrix: " << endl << V << endl << endl;
#endif //VERYVERBOSE

		// Compute U if needed
		if(!whiterows) {
			cerr << "Whiten Columns, need to compute U" << endl;

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
#ifdef VERYVERBOSE
				cerr << "Full Buffer " << rr << "\n" << buffer << endl << endl;
				cerr << "Partial U" << rr << "\n" << U << endl << endl;
#endif //VERYVERBOSE
			}

			cerr << "Finished Computing U" << endl;
			return U;
		} else {
			cerr << "Whiten Rows, already have V" << endl;
			return V;
		}
	} else {
		// Computed left singular values (U)
		// A = USV*, A^T = VSU*, V = A^T U S^-1

		// reverse and fill U
		MatrixXd U(eig.eigenvectors().rows(), rank);
		for(int cc=0; cc<rank; cc++)
			U.col(cc) = eig.eigenvectors().col(eigrows-1-cc);

#ifdef VERYVERBOSE
		cerr << "U Matrix: " << endl << U << endl << endl;
#endif //VERYVERBOSE

		if(!whiterows) {
			cerr << "Whiten Columns, already have U" << endl;
			return U;
		} else {
			cerr << "Whiten Rows, need to compute V" << endl;

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
#ifdef VERYVERBOSE
				cerr << "Full Buffer " << cc << "\n" << buffer << endl << endl;
				cerr << "Partial V" << cc << "\n" << V << endl << endl;
#endif //VERYVERBOSE
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

	TCLAP::CmdLine cmd("Perform ICA analysis on an image, or group of "
			"images. Image Power Spectral Density is used so that specific "
			"time-course does not affect the outcome, just the frequencies."
			"As a result the SVD only depends on the longest single subjects "
			"time-course rather than the concatined time-course",
			' ', __version__ );

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

	/**********
	 * Input
	 *********/
	if(!a_in.isSet()) {
		cerr << "Need to provide at least 1 input image!" << endl;
		return -1;
	}

	for(int ii=0; ii<a_in.size(); ii++) {

		// Read Image
		auto img = readMRImage(a_in->getValue()[ii]);

		ptr<MRImage> mask;
		if(ii < a_masks.size()) {
			mask = readMRImage(a_masks->getValue()[ii]);
		} else {
			mask = varianceT(img);
		}

		auto psdimg = psd(img, mask);
		psdimg->write(a_odir.getValue()+"/psd_"+to_string(ii)+"_"+".nii.gz");
	}

	// Dear Micah, You should work hard. 
	// Compute SUM X_i^T X_i
	// Compute EigenVectors (U)
	// Compute V 
	// Use V as input to ICA , Cols of VT correspond to columns of X
	// j

	// 
	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }

    return 0;
}

ptr<NDArray> flattenPSD(ptr<const MRImage> img, ptr<const MRImage> mask)
{
	// Check Mask
	if(!mask || !mask->matchingOrient(img, false, true))
		throw INVALID_ARGUMENT("Invalid Mask: "+a_mask->getValue()[ii]);

	// sum up nonzero and fill blocksz
	int nrows = img->tlen();
	int pnrows = round2(nrows);
	int ncols = 0;
	for(FlatIter<int> it(mask); !it.eof(); ++it) {
		if(*it != 0) ncols++;
	}
	
	double* ibuffer = fftw_alloc_real(pnrows);
	fftw_complex* obuffer = fftw_alloc_complex(pnrows);
	fftw_plan fwd = fftw_plan_dft_r2c_1d(nrows, ibuffer, obuffer, FFTW_FORWARD,
			FFTW_MEASURE);

	for(size_t ii=0; ii<pnrows; ii++)
		ibuffer[ii] = 0;

	// Create Output NDArray
	ptr<NDArrayStore<2,double>> out(new NDArrayStore<2, double>({pnrows, ncols}));

	// Fill With Masked Timeseries
	NDiter<int> mit(mask);
	Vector3DIter<double> iit(img);
	int cc = 0;
	for(; !mit.eof() && !iit.eof(); ++iit, ++mit) {
		if(*mit != 0) {
			// fill input buffer
			for(int rr=0; rr<nrows; rr++)
				ibuffer[rr] = iit[rr];

			// Fourier Transform
			fftw_execute(fwd);

			// Save Power Spectral Density
			for(int rr=0; rr<nrows; rr++) {
				std::complex<double> tmp(obuffer[rr][0], obuffer[rr][1]);
				(*buff)[{rr, cc}] = std::abs(tmp)*std::abs(tmp);
			}
			++cc;
		}
	}

	return out;
}



