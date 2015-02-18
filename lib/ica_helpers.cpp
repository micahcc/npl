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
 * @file ica_helpers.cpp Tools for performing ICA, including rewriting images as
 * matrices. All the main functions for real world ICA.
 *
 *****************************************************************************/

#include "ica_helpers.h"

#include <Eigen/Dense>
#include <Eigen/IterativeSolvers>

#include <iostream>
#include <fstream>
#include <string>

#include "fftw3.h"

#include "utility.h"
#include "npltypes.h"
#include "ndarray_utils.h"
#include "mrimage.h"
#include "nplio.h"
#include "iterators.h"
#include "statistics.h"

using namespace std;

namespace npl {

/*****************************************************************************
 * High Level Functions for Performing Large Scale ICA Analysis
 ****************************************************************************/

/**
 * @brief Helper function for large-scale ICA analysis. This takes
 * a working directory, which should already have 'mat_#' files with data
 * (one per element of ncols) and orthogonalizes the data to produce a
 * set of variables which are orthogonal.
 *
 * This assumes that the input is a set of matrices which will be concat'd in
 * the col (spatial) direction. By default it is assumed that the columns
 * represent dimensions and the rows samples. This means that the output
 * of Xorth will normally have one row for every row of the original matrices
 * and far fewer columns. If rowdims is true then Xorth will have one row
 * for each of the columns in the merge mat_* files, and fewer columns than
 * the original matrices had rows.
 *
 * @param prefix Directory which should have mat_0, mat_1, ... up to
 * ncols.size()-1
 * @param svthresh Ratio of sum of eigenvalues to capture.
 * @param initbasis Number of starting basis vectors to initialize the
 * BandLanczos algorithm with. If this is <= 1, one dimension of of XXT will be
 * used.
 * @param maxiters Maximum number of iterations to perform in EV decomp
 * @param rowdims Perform reduction on the original inputs rows. Note that the
 * output will still be in column format, where each columns is a dimension.
 * This makes it easier to perform ICA.
 * the original data).
 * @param ncols Number of columns in each block of columns
 * @param XXT Covariance (X*X.transpose())
 * @param Xorth Output orthogonal version of X
 *
 * @return
 */
int spcat_orthog(std::string prefix, double svthresh,
		int initbasis, int maxiters, bool rowdims, const std::vector<int>& ncols,
		const MatrixXd& XXT, MatrixXd& Xorth)
{
	(void)initbasis;
	(void)maxiters;
	// Compute EigenVectors (U)
	Eigen::SelfAdjointEigenSolver<MatrixXd> eig(XXT);
//	Eigen::BandLanczosSelfAdjointEigenSolver<MatrixXd> eig;
//	eig.setDeflationTol(std::numeric_limits<double>::epsilon());
//	eig.setDesiredRank(maxiters);
//	eig.compute(XXT, initbasis);

	if(eig.info() == Eigen::NoConvergence)
		throw RUNTIME_ERROR("Non Convergence of BandLanczosSelfAdjointEigen"
				"Solver, try increasing lanczos basis or maxiters");

	double minev = sqrt(std::numeric_limits<double>::epsilon());
	double sum = 0;
	double totalev = eig.eigenvalues().sum();
	int eigrows = eig.eigenvalues().rows();
	int rank = 0;
	VectorXd S(eigrows);
	for(int cc=0; cc<eigrows; cc++) {
		double v = eig.eigenvalues()[eigrows-1-cc];
		if(v > minev && (svthresh<0 || svthresh>1 || sum>totalev*svthresh)) {
			S[cc] = std::sqrt(eig.eigenvalues()[eigrows-1-cc]);
			rank++;
		}
		sum += v;
	}
	S.conservativeResize(rank);

	// Computed left singular values (U), reverse order
	// A = USV*, A^T = VSU*, V = A^T U S^-1
	MatrixXd U(eig.eigenvectors().rows(), rank);
	for(int cc=0; cc<rank; cc++)
		U.col(cc) = eig.eigenvectors().col(eigrows-1-cc);

	if(!rowdims) {
		// U is what we need, since each COLUMN represents a dim
		Xorth = U;
	} else {
		// Need V since each ROW represents a DIM, in V each COLUMN will be a
		// DIM, which will correspond to a ROW of X
		// Compute V = X^T U S^-1
		// rows of V correspond to cols of X
		size_t totalcols = 0;
		for(auto C : ncols) totalcols += C;
		Xorth.resize(totalcols, S.rows());

		int curr_row = 0;
		for(int ii=0; ii<ncols.size(); ii++) {
			string fn = prefix+"_mat_"+to_string(ii);
			MemMap mmap(fn, ncols[ii]*U.rows()*sizeof(double), false);
			if(mmap.size() < 0)
				return -1;

			Eigen::Map<MatrixXd> X((double*)mmap.data(), U.rows(), ncols[ii]);
			Xorth.middleRows(curr_row, ncols[ii]) = X.transpose()*U*
						S.cwiseInverse().asDiagonal();

			curr_row += ncols[ii];
		}
	}
	return 0;
}

/**
 * @brief Uses randomized subspace approximation to reduce the input matrix
 * (made up of blocks stored on disk with a given prefix). This assumes that
 * the matrix is a wide matrix (which is generally a good assumption in
 * fMRI) and that it therefore is better to reduce the number of columns.
 *
 * To achieve this, we transpose the algorithm of 4.4 from
 * Halko N, Martinsson P-G, Tropp J A. Finding structure with randomness:
 * Probabilistic algorithms for constructing approximate matrix decompositions.
 * 2009;1–74. Available from: http://arxiv.org/abs/0909.4061
 *
 * @param prefix File prefix
 * @param tol Tolerance for stopping
 * @param startrank Initial rank (est rank double each time, -1 to start at
 * log(min(rows,cols)))
 * @param maxrank Maximum rank (or -1 to select the min(rows,cols))
 * @param poweriters
 * @param U Output U matrix, if null then ignored
 * @param V Output V matrix, if null then ignored
 *
 * @return Vector of singular values
 */
VectorXd onDiskSVD(const MatrixReorg& A,
		int rank, size_t poweriters, MatrixXd* U, MatrixXd* V)
{

	// Algorithm 4.4
	MatrixXd Yc;
	MatrixXd Yhc;
	MatrixXd Qtmp;
	MatrixXd Qhat;
	MatrixXd Omega;

	// From the Original Algorithm A -> At everywhere
	Yc.resize(A.cols(), rank);
	Yhc.resize(A.rows(), rank);
	Omega.resize(A.rows(), rank);

	fillGaussian<MatrixXd>(Omega);
	A.postMult(Yc, Omega, true);

	Eigen::HouseholderQR<MatrixXd> qr(Yc);
	Qtmp = qr.householderQ()*MatrixXd::Identity(A.cols(), rank);
	Eigen::HouseholderQR<MatrixXd> qrh;
	cerr << "Power Iteration: ";
	for(size_t ii=0; ii<poweriters; ii++) {
		cerr<<ii<<" ";
		A.postMult(Yhc, Qtmp);
		qrh.compute(Yhc);
		Qhat = qrh.householderQ()*MatrixXd::Identity(A.rows(), rank);
		A.postMult(Yc, Qhat, true);
		qr.compute(Yc);
		Qtmp = qr.householderQ()*MatrixXd::Identity(A.cols(), rank);
	}
	cerr << "Done" << endl;

	// Form B = Q* x A
	cerr<<"Making Low Rank Approximation ("<<Qtmp.cols()<<"x"<<A.rows()<<")"<<endl;
	MatrixXd B(Qtmp.cols(), A.rows());;
	A.preMult(B, Qtmp.transpose(), true);
	Eigen::JacobiSVD<MatrixXd> smallsvd(B, Eigen::ComputeThinU | Eigen::ComputeThinV);

	cerr<<"Done with SVD Estimation"<<endl;
	// These are swapped because we did A^T
	if(V)
		*V = Qtmp*smallsvd.matrixU();
	if(U)
		*U = smallsvd.matrixV();

	return smallsvd.singularValues();
}

/**
 * @brief Computes the the ICA of temporally concatinated images. This is mostly
 * for comparison purposes with the more optimzes image processing methods.
 *
 * @param imgnames List of images to load. Data is concatinated from left to
 * right.
 * @param maskname Common mask for the all the input images. If empty, then
 * non-zero variance areas will be used.
 * @param prefix Directory to create mat_* files and mask_* files in
 * @param varthresh Threshold for percentage of variance to account for.
 * @param spatial Whether to do spatial ICA. Warning this is much more memory
 * and CPU intensive than PSD/Time ICA.
 *
 * @return Matrix with independent components in columns
 *
 */
void tcat_ica(const vector<string>& imgnames, string maskname,
		string prefix, double varthresh, bool spatial)
{
	/**********
	 * Input
	 *********/
	if(imgnames.empty())
		throw INVALID_ARGUMENT("Need to provide at least 1 input image!");
	vector<std::string> masks;
	masks.push_back(maskname);

	if(varthresh < 0 || varthresh > 1)
		varthresh = 0.9;

	int rank = 0; // Rank of Singular Value Decomp
	VectorXd singvals;
	MatrixXd matrixU;
	MatrixXd matrixV;

	// Matrix Reorg Creates Tall Matrices and Wide Matrices, we need the tall
	size_t maxd = (1<<30); // roughly 8GB
	vector<string> masks(1, maskname);
	MatrixReorg reorg(prefix, maxd, false);
	reorg.createMats(imgnames.size(), 1, masks, imgnames, true);

	// Create AA* matrix
	MatrixXd AAt(reorg.rows, reorg.rows);
	for(size_t cc = 0; cc < reorg.ntall(); cc++) {
		MatMap m(reorg.tallMatName(cc));
		AAt += m.mat*m.mat.transpose();
	}

	/*
	 * Perform SVD
	 */
	SelfAdjointEigenSolver<MatrixXd> eig(AAt);
	const auto& evec = eig.eigenvectors();
	const auto& evals = eig.eigenvalues();

	// Compute Rank
	double totalvar = 0;
	double sum = 0;
	for(size_t ii=0; ii<AAt.rows(); ii++) {
		if(evals[AAt.rows()-ii-1] > std::numeric_limits<double>::epsilon())
			totalvar += sqrt(eig.eigenvalues()[AAt.rows()-ii-1]);
	}

	for(size_t ii=0; ii<AAt.rows(); ii++) {
		if(evals[AAt.rows()-ii-1] > std::numeric_limits<double>::epsilon()) {
			sum += sqrt(evals[AAt.rows()-ii-1]);
			if(sum < totalvar*varthresh)
				rank++;
			else
				break;
		}
	}

	matrixU.resize(reorg.rows, rank);
	matrixV.resize(reorg.cols, rank);
	singvals.resize(rank);
	for(size_t ii=0; ii<rank; ii++) {
		singvals[ii] = sqrt(evals[AAt.rows()-ii-1]);
		matruxU.col(ii) = evecs.col(AAt.rows()-ii-1);
	}

	// V = A^TUE^-1
	reorg.postMult(matrixV, matrixU, true);
	for(size_t cc=0; cc<rank; cc++)
		matrixV.col(cc) /= singvals[cc];

#ifdef DEBUG
	MatrixXd inv = V*(singvals.cwiseInverse()).asDiagonal()*U.transpose();
	MatrixXd tmp(reorg.rows, reorg.rows);
	reorg.postMult(tmp, inv, false);
	cerr << "Error in tcat_ica inverse with SVD: " << tmp.cwiseAbs().sum() << endl;
#endif //DEBUG

	/*
	 * Compute ICA and spatial maps
	 */
	if(!spatial) {
		cerr << "Temporal ICA" << endl;
		MatrixXd ics = ica(matrixU);

		cerr << "Regressing Temporal Components" << endl;
		// need to compute the CDF for students_t_cdf
		const double MAX_T = 2000;
		const double STEP_T = 0.1;
		StudentsT distrib(ics.rows()-1, STEP_T, MAX_T);
		cerr << "Computing Inverse of ICs ("<<ics.rows()<<"x"
			<<ics.cols()<<")"<<endl;
		MatrixXd Xinv = pseudoInverse(ics);
		VectorXd Cinv = pseudoInverse(ics.transpose()*ics).diagonal();

		// Iterate through mask as we iterate through the columns of the ICs
		ptr<MRImage> mask;
		ptr<MRImage> out;
		FlatIter<int> mit;
		Vector3DIter<double> oit;

		// Create 4D Image matching mask
		size_t odim[4] = {0,0,0, ics.cols()};

		int ii; // iterate through tall matrices
		int cc = 0; // iterate through columns of tall matrix
		int mm = -1; // iterate through masks
		RegrResult result;
		/*
		 * Convert statistics of each column of each tall matrix back to
		 * spatial position
		 */
		MatMap tall;
		for(ii=0; tall.open(reorg.tallMatName(ii)) == 0; ii++) {

			// Iterate through columns of tall matrix
			for(cc=0; cc<tall.cols; ) {
				// Match iterations of tall columns with mask iteration
				if(!mask || mit.eof()) {
					cerr << "Loading Mask" << endl;
					if(cc != 0) {
						throw RUNTIME_ERROR("Error Tall Matrix:\n"+m_pref+
								"_tall_"+to_string(ii)+"\nMismatches mask:"+
								m_pref+"_mask_"+to_string(mm));
					}

					// If there is currently an open output image, write
					if(out)
						out->write(m_pref+"_tmap_m"+to_string(mm)+".nii.gz");
					mm++;

					// Read Next Mask
					cerr<<"Reading mask ("<<mm<<") = "<<m_pref<<"_mask_"
							<<mm<<".nii.gz"<<endl;
					mask = readMRImage(m_pref+"_mask_"+to_string(mm)+".nii.gz");
					if(mask->ndim() != 3)
						throw RUNTIME_ERROR("Error input mask is not 3D!");
					for(size_t dd=0; dd<3; dd++)
						odim[dd] = mask->dim(dd);
					odim[3] = ics.cols();

					out = dPtrCast<MRImage>(mask->createAnother(4,
								odim, FLOAT32));
					mit.setArray(mask);
					oit.setArray(out);
					mit.goBegin();
					oit.goBegin();
				}

				if(*mit) {
					// regress each component with current column of tall
					regress(&result, tall.mat.col(cc), ics,
							Cinv, Xinv, distrib);
					for(size_t comp=0; comp<ics.cols(); comp++)
						oit.set(comp, result.t[comp]);
					cc++;
				} else {
					for(size_t comp=0; comp<ics.cols(); comp++)
						oit.set(comp, 0);
				}

				++mit;
				++oit;
			}
		}

		// If there is currently an open output image, write as multiple, last
		if(out)
			out->write(m_pref+"_tmap_m"+to_string(mm)+".nii.gz");
		cerr << "Done with Regression" << endl;
	} else {
		cerr << "Spatial ICA" << endl;
		MatrixXd ics = ica(matrixV);

		// Estimate Error with A - UEV*, for each row / Column in A compute
		// variance then divide ic/sigma^2

		// Convert each signal matrix into a t-score
		VectorXd tmp(ics.rows());
		for(size_t cc = 0; cc < ics.cols; cc++) {
			mixtureModel(tmp, ics.col(cc));
			ics.col(cc) = tmp;
		}

		// Re-associate each column with a spatial signal (map)
		// Columns of ics correspond to rows of the wide matrices/voxels
		int npt = 0; // number of points in the current image
		int nptii = 0; // iterate through total points in all images
		int mm = 0; // current map
		int comp = 0; // current Independent Component
		int rr=0; // Current Row/Sample of IC
		for(size_t fullrows=0; fullrows != ics.rows(); fullrows += npt) {
			for(mm=0; nptii<ics.rows(); nptii+=npt, ++mm) {
				auto mask = readMRImage(m_pref+"_mask_"+to_string(mm)+".nii.gz");
				auto out = mask->copyCast(FLOAT32);
				FlatIter<int> mit(mask);
				FlatIter<double> oit(out);

				// count cols
				npt = 0;
				for(mit.goBegin(); !mit.eof(); ++mit) {
					if(*mit != 0)
						npt++;
				}

				// Iterate through Components
				for(comp=0; comp<ics.cols(); comp++) {
					for(rr=0, mit.goBegin(),oit.goBegin(); !mit.eof(); ++mit, ++oit) {
						if(*mit != 0) {
							oit.set(ics().mat(rr, comp));
							rr++;
						}
					}
					out->write(m_pref+"_map_m"+to_string(mm)+"_c"+to_string(comp)+
							".nii.gz");
				}
			}
			if(nptii > ics.rows()) {
				throw RUNTIME_ERROR("Error:\n"+m_pref+"_SICA\nSize does not match "
						"input masks");
			}

		}

	}

}

/**
 * @brief Computes the the ICA of spatially concatinated images. Optionally
 * the data may be converted from time-series to a power spectral density,
 * making this function applicable to resting state data.
 *
 * @param psd Compute the power spectral density prior to PCA/ICA
 * @param imgnames List of images to load. Data is concatinated from left to
 * right.
 * @param masknames List of masks fo each of the in input image. May be empty
 * or have missing masks at the end (in which case zero-variance timeseries are
 * assumed to be outside the mask)
 * @param prefix Directory to create mat_* files and mask_* files in
 * @param svthresh Threshold for singular values (drop values this ratio of the
 * max)
 * @param deftol Threshold for eigenvalues of XXT and singular values. Scale
 * is a ratio from 0 to 1 relative to the total sum of eigenvalues/variance.
 * Infinity will the BandLanczos Algorithm to run to completion and only
 * singular values > sqrt(epsilon) to be kept
 * @param initbasis Basis size of BandLanczos Algorithm. This is used as the
 * seed for the Krylov Subspace.
 * @param maxiters Maximum number of iterations for PCA
 * @param spatial Whether to do spatial ICA. Warning this is much more memory
 * and CPU intensive than PSD/Time ICA.
 *
 * @return Matrix with independent components in columns
 *
 */
MatrixXd spcat_ica(bool psd, const vector<string>& imgnames,
		const vector<string>& masknames, string prefix, double deftol,
		double svthresh, int initbasis, int maxiters, bool spatial)
{
	(void)deftol;
	(void)initbasis;
	(void)maxiters;

	/**********
	 * Input
	 *********/
	if(imgnames.empty())
		throw INVALID_ARGUMENT("Need to provide at least 1 input image!");

	const double MINSV = sqrt(std::numeric_limits<double>::epsilon());
	MatrixXd XXt;
	int nrows = -1;
	vector<int> ncols(imgnames.size());
	int totalcols = 0;

	int rank = 0; // Rank of Singular Value Decomp
	VectorXd singvals; // E in X = UEV*
	MatrixXd matrixU; // U in X = UEV*

	// Read Each Image, Create a Mask, save the mask
	for(int ii=0; ii<imgnames.size(); ii++) {

		// Read Image
		auto img = readMRImage(imgnames[ii]);

		// Load or Compute Mask from Variance
		ptr<MRImage> mask;
		if(ii < masknames.size())
			mask = readMRImage(masknames[ii]);
		else
			mask = dPtrCast<MRImage>(varianceT(img));

		// Figure out number of columns from mask
		ncols[ii] = 0;
		for(FlatIter<int> it(mask); !it.eof(); ++it) {
			if(*it != 0)
				ncols[ii]++;
		}
		totalcols += ncols[ii];

		// Write Mask
		mask->write(prefix+"_mask_"+to_string(ii)+"_"+".nii.gz");

		// Create Output File to buffer file, if we are going to FFT, the
		// round to the nearest power of 2
		int tmprows = psd ? round2(img->tlen()) : img->tlen();

		if(nrows <= 0) {
			// Set rows and check allocation
			nrows = tmprows;
			try {
				XXt.resize(nrows, nrows);
			} catch(...) {
				throw RUNTIME_ERROR("Not enough memory for matrix of "+
						to_string(nrows)+"^2 doubles");
			}
		} else if(nrows != tmprows)
			throw INVALID_ARGUMENT("Number of time-points differ in inputs!");

		string fn = prefix+"_mat_"+to_string(ii);
		MemMap mmap(fn, ncols[ii]*nrows+sizeof(double), true);
		if(mmap.size() <= 0)
			throw RUNTIME_ERROR("Error opening "+fn+" for writing");

		double* ptr = (double*)mmap.data();
		if(psd)
			fillMatPSD(ptr, nrows, ncols[ii], img, mask);
		else
			fillMat(ptr, nrows, ncols[ii], img, mask);
	}

	/*
	 * Compute SVD of input
	 */
	XXt.setZero();

	// Sum XXt_i for i in 1...subjects
	for(int ii=0; ii<imgnames.size(); ii++) {
		string fn = prefix+"_mat_"+to_string(ii);
		MemMap mmap(fn, ncols[ii]*nrows+sizeof(double), false);
		if(mmap.size() <= 0)
			throw RUNTIME_ERROR("Error opening "+fn+" for reading");

		// Sum up XXt from each chunk
		Eigen::Map<MatrixXd> X((double*)mmap.data(), nrows, ncols[ii]);
		XXt += X*X.transpose();
	}

	// Compute EigenVectors (U)
	Eigen::SelfAdjointEigenSolver<MatrixXd> eig(XXt);
//	Eigen::BandLanczosSelfAdjointEigenSolver<MatrixXd> eig;
//	eig.setDesiredRank(maxiters);
//	eig.setDeflationTol(deftol);
//	eig.compute(XXt, initbasis);

	if(eig.info() == Eigen::NoConvergence)
		throw RUNTIME_ERROR("Non Convergence of BandLanczosSelfAdjointEigen"
				"Solver, try increasing lanczos basis or maxiters");

	double tmp = 0;
	int eigrows = eig.eigenvalues().rows();
	double maxev = eig.eigenvalues().maxCoeff();
	tmp = 0;
	singvals.resize(eigrows);
	for(int cc=0; cc<eigrows; cc++) {
		double v = eig.eigenvalues()[eigrows-1-cc];
		if(v > MINSV && (svthresh<0 || svthresh>1 || v > maxev*svthresh)) {
			singvals[cc] = std::sqrt(eig.eigenvalues()[eigrows-1-cc]);
			rank++;
		}
		tmp += v;
	}
	singvals.conservativeResize(rank);

	// Computed left singular values (U), reverse order
	// A = USV*, A^T = VSU*, V = A^T U S^-1
	matrixU.resize(nrows, rank);
	for(int cc=0; cc<rank; cc++)
		matrixU.col(cc) = eig.eigenvectors().col(eigrows-1-cc);

	if(!spatial) {
		// U is what we need, since each COLUMN represents a dim
		return ica(matrixU);
	} else {
		MatrixXd matrixV(totalcols, rank);
		// Need V since each ROW represents a DIM, in V each COLUMN will be a
		// DIM, which will correspond to a ROW of X
		// Compute V = X^T U S^-1
		// rows of V correspond to cols of X
		int curr_row = 0; // Current row in input (col in V)
		for(int ii=0; ii<ncols.size(); ii++) {
			string fn = prefix+"_mat_"+to_string(ii);
			MemMap mmap(fn, ncols[ii]*nrows*sizeof(double), false);
			if(mmap.size() < 0)
				throw RUNTIME_ERROR("Error opening "+fn+" for reading");

			Eigen::Map<MatrixXd> X((double*)mmap.data(), nrows, ncols[ii]);
			matrixV.middleRows(curr_row, ncols[ii]) = X.transpose()*matrixU*
						singvals.cwiseInverse().asDiagonal();

			curr_row += ncols[ii];
		}
		return ica(matrixV);
	}
}

MatrixReorg::MatrixReorg(std::string prefix, size_t maxdoubles, bool verbose)
{
	m_prefix = prefix;
	m_maxdoubles = maxdoubles;
	m_verbose = verbose;
};

/**
 * @brief Loads existing matrices by first reading ${prefix}_tall_0,
 * ${prefix}_wide_0, and ${prefix}_mask_*, and checking that all the dimensions
 * can be made to match (by loading the appropriate number of matrices/masks).
 *
 * @return 0 if succesful, -1 if read failure, -2 if write failure
 */
int MatrixReorg::checkMats()
{
	m_outcols.clear();
	std::string info = m_prefix + ".txt";
	std::string tallpr = m_prefix + "_tall_";
	std::string maskpr = m_prefix + "_mask_";

	ifstream ifs(info);
	ifs >> m_totalrows >> m_totalcols;
	if(m_verbose) {
		cerr << "Information File: " << info << endl;
		cerr << "Tall Matrix Prefix: " << tallpr << endl;
		cerr << "Mask Prefix:        " << maskpr << endl;
		cerr << "Total Rows/Timepoints: " << m_totalrows<< endl;
		cerr << "Total Cols/Voxels:     " << m_totalcols << endl;
	}
	if(m_totalcols == 0 || m_totalrows == 0)
		throw RUNTIME_ERROR("Error zero size input from "+info);

	/* Read First Tall and Wide to get totalrows and totalcols */
	MatMap map;
	int cols = 0;
	for(size_t bb=0; cols < m_totalcols; bb++) {
		if(map.open(tallpr+to_string(bb)))
			throw RUNTIME_ERROR("Error opening "+tallpr+to_string(bb));
		if(m_totalrows != map.rows)
			throw RUNTIME_ERROR("Error, mismatch in file size ("+
					to_string(map.rows)+"x"+to_string(map.cols)+" vs "+
					to_string(m_totalrows)+"x"+to_string(m_totalcols)+")");
		m_outcols.push_back(map.cols);
		cols += map.cols;
	}
	if(m_totalcols != cols)
		throw RUNTIME_ERROR("Error, mismatch in number of cols from input "
				"files (prefix "+tallpr+")");

	// check masks
	cols = 0;
	for(int ii=0; cols!=m_totalcols; ii++) {
		auto mask = readMRImage(maskpr+to_string(ii)+".nii.gz");
		for(FlatIter<int> mit(mask); !mit.eof(); ++mit) {
			if(*mit != 0)
				cols++;
		}

		// should exactly match up
		if(cols > m_totalcols)
			throw RUNTIME_ERROR("Error, mismatch in number of cols from input "
					"masks (prefix "+maskpr+")");
	}
	return 0;
}

/**
 * @brief Creates two sets of matrices from a set of input images. The matrices
 * (images) are ordered in column major order. In each column the mask is loaded
 * then each image in the column is loaded and the masked timepoints extracted.
 *
 * The order of reading from filenames is essentially:
 * time 0: 0246
 * time 1: 1357
 *
 * Masks correspond to each column so the number of masks should be = to number
 * masknames. Note that if no mask is provided, one will be generated from the
 * set of non-zero variance timeseries in the first input image in the column.
 *
 * This file writes matrices called /tall_# and /wide_#. Tall matrices have
 * the entire concatinated timeseries for a limited set of spacial locations,
 * Wide matrices have entire concatinated spacial signals for a limited number
 * of timepoints.
 *
 * @param timeblocks Number of timeseries to concatinate (concatined
 * time-series are adjacent in filenames vector)
 * @param spaceblocks Number of images to concatinate spacially. Unless PSD is
 * done, these images should have matching tasks
 * @param masknames Files matching columns of in the filenames matrix. That
 * indicate voxels to include
 * @param filenames Files to read in, images are stored in column (time)-major
 * order
 * @param normts Normalize each column before writing
 *
 * @return 0 if succesful, -1 if read failure, -2 if write failure
 */
int MatrixReorg::createMats(size_t timeblocks, size_t spaceblocks,
		const std::vector<std::string>& masknames,
		const std::vector<std::string>& filenames, bool normts)
{
	vector<int> inrows;
	vector<int> incols;
	std::string info = m_prefix + ".txt";
	std::string tallpr = m_prefix + "_tall_";
	std::string maskpr = m_prefix + "_mask_";

	if(m_verbose) {
		cerr << "Information File: " << info << endl;
		cerr << "Tall Matrix Prefix: " << tallpr << endl;
		cerr << "Mask Prefix:        " << maskpr << endl;
	}

	/* Determine Size:
	 *
	 * First Run Down the Top Row and Left Column to determine the number of
	 * rows in each block of rows and number of columns in each block of
	 * columns, also create masks in the output folder
	 */
	inrows.resize(timeblocks);
	incols.resize(spaceblocks);
	std::fill(inrows.begin(), inrows.end(), 0);
	std::fill(incols.begin(), incols.end(), 0);

	// Figure out the number of cols in each block, and create masks
	ptr<MRImage> mask;
	m_totalcols = 0;
	for(size_t sb = 0; sb<spaceblocks; sb++) {
		if(sb < masknames.size()) {
			mask = readMRImage(masknames[sb]);
		} else {
			auto img = readMRImage(filenames[sb*timeblocks+0]);
			mask = dPtrCast<MRImage>(binarize(varianceT(img), 0));
		}

		mask->write(maskpr+to_string(sb)+".nii.gz");
		for(FlatIter<int> it(mask); !it.eof(); ++it) {
			if(*it != 0)
				incols[sb]++;
		}
		if(incols[sb] == 0) {
			throw INVALID_ARGUMENT("Error, input mask for column " +
					to_string(sb) + " has no non-zero pixels");
		}
		m_totalcols += incols[sb];
	}

	// Figure out number of rows in each block
	m_totalrows = 0;
	for(size_t tb = 0; tb<timeblocks; tb++) {
		auto img = readMRImage(filenames[0*timeblocks+tb]);
		inrows[tb] += img->tlen();
		m_totalrows += inrows[tb];
	}

	if(m_verbose) {
		cerr << "Row/Time  Blocks: " << timeblocks << endl;
		cerr << "Col/Space Blocks: " << spaceblocks << endl;
		cerr << "Total Rows/Timepoints: " << m_totalrows<< endl;
		cerr << "Total Cols/Voxels:     " << m_totalcols << endl;
	}
	ofstream ofs(info);
	ofs << m_totalrows << " " << m_totalcols << endl;
	ofs.close();

	/*
	 * Break rows and columns into digestable sizes, and don't allow chunks to
	 * cross lines between images (this will make loading the matrices easier)
	 * break up output blocks of cols into chunks that 1) don't cross images
	 * and 2) with have fewer elements than m_maxdoubles
	 */
	m_outcols.resize(1, 0); // number of cols per block
	int blockind = 0;
	int blocknum = 0;
	if(m_totalrows > m_maxdoubles) {
		throw INVALID_ARGUMENT("maxdoubles is not large enough to hold a "
				"single full row!");
	}

	if(m_verbose) cerr << "Creating Matrices"<<endl;
	for(int cc=0; cc<m_totalcols; cc++) {
		if(blockind == incols[blocknum]) {
			// open file, create with proper size
			MemMap tfile(tallpr+to_string(m_outcols.size()-1), 2*sizeof(size_t)+
					m_outcols.back()*m_totalrows*sizeof(double), true);
			((size_t*)tfile.data())[0] = m_totalrows; // nrows
			((size_t*)tfile.data())[1] = m_outcols.back(); // ncols

			// if the index in the block would put us in a different image, then
			// start a new out block, and new in block
			blockind = 0;
			blocknum++;
			m_outcols.push_back(0);
		} else if((m_outcols.back()+1)*m_totalrows > m_maxdoubles) {
			MemMap tfile(tallpr+to_string(m_outcols.size()-1), 2*sizeof(size_t)+
					m_outcols.back()*m_totalrows*sizeof(double), true);
			((size_t*)tfile.data())[0] = m_totalrows; // nrows
			((size_t*)tfile.data())[1] = m_outcols.back(); // ncols

			// If this col won't fit in the current block of cols, start a new one,
			m_outcols.push_back(0);
		}
		blockind++;
		m_outcols.back()++;
	}
	{ // Create Last File
	MemMap tfile(tallpr+to_string(m_outcols.size()-1), 2*sizeof(size_t)+
			m_outcols.back()*m_totalrows*sizeof(double), true);
	((size_t*)tfile.data())[0] = m_totalrows; // nrows
	((size_t*)tfile.data())[1] = m_outcols.back(); // ncols
	}

	/*
	 * Fill tall matrices by breaking images along block cols specified
	 * in m_outcols
	 */
	if(m_verbose) cerr << "Filling Matrices"<<endl;
	if(normts) cerr<<"Normalizing Timeseries"<<endl;
	int img_glob_row = 0;
	int img_glob_col = 0;
	int img_oblock_col = 0;
	MatMap datamap;
	for(size_t sb = 0; sb<spaceblocks; sb++) {
		if(m_verbose) cerr<<"Mask Group: "<<maskpr+to_string(sb)+".nii.gz"<<endl;
		auto mask = readMRImage(maskpr+to_string(sb)+".nii.gz");

		img_glob_row = 0;
		for(size_t tb = 0; tb<timeblocks; tb++) {
			auto img = readMRImage(filenames[sb*timeblocks+tb]);
			if(m_verbose) cerr<<"Image: "<<filenames[sb*timeblocks+tb]<<endl;

			if(!img->matchingOrient(mask, false, true))
				throw INVALID_ARGUMENT("Mismatch in mask/image size in col:"+
						to_string(sb)+", row:"+to_string(tb));
			if(img->tlen() != inrows[tb])
				throw INVALID_ARGUMENT("Mismatch in time-length in col:"+
						to_string(sb)+", row:"+to_string(tb));

			double mean = 0, sd = 0;
			int cc, colbl;
			int tlen = img->tlen();
			Vector3DIter<double> it(img);
			NDIter<double> mit(mask);

			/*
			 * fill mat[img_glob_row:img_glob_row+tlen,0:bcols],
			 * Start with invalid and load new columns as needed
			 * cc iterates over the columns in the block, colbl indicates the
			 * index of the block in the overall scheme
			 * rows are global, cols are local to block
			 */
			for(cc=-1, colbl=img_oblock_col-1; !it.eof(); ++it, ++mit) {
				if(*mit != 0) {
					// If cc is invalid, open
					if(cc < 0 || cc >= m_outcols[colbl]) {
						cc = 0;
						colbl++;
						datamap.open(tallpr+to_string(colbl));
						if(m_verbose) cerr<<"Writing to: "<<tallpr+to_string(colbl)
								<<" at row "<<to_string(img_glob_row)<<endl;
						if(datamap.rows != m_totalrows || datamap.cols != m_outcols[colbl]) {
							throw INVALID_ARGUMENT("Unexpected size in input "
									+ tallpr+to_string(colbl));
						}
					}

					// Fill, Calculate Mean/SD and normalize
					mean = 0;
					sd = 0;
					for(size_t tt=0; tt<tlen; tt++) {
						datamap.mat(tt+img_glob_row, cc) = it[tt];
						mean += it[tt];
						sd += it[tt]*it[tt];
					}
					sd = sqrt(sample_var(tlen, mean, sd));
					mean /= tlen;
					if(normts) {
						if(sd > 0)
							datamap.mat.block(img_glob_row, cc, tlen, 1) =
								(datamap.mat.block(img_glob_row, cc, tlen, 1).array()-mean)/sd;
						else
							datamap.mat.block(img_glob_row, cc, tlen, 1).setZero();
					}
					cc++;
				}
			}

			assert(cc == m_outcols[colbl]);
			datamap.close();
			if(m_verbose) cerr<<"Image Done: "<<filenames[sb*timeblocks+tb]<<endl;

			// Increment Global Row by Input Block Size (same as image rows)
			img_glob_row += inrows[tb];
		}

		// Increment Global Col by Input Block Size (same as image cols)
		img_glob_col += incols[sb];

		// Increment Output block col to correspond to the next image
		for(int ii=0; ii != incols[sb]; )
			ii += m_outcols[img_oblock_col++];
		if(m_verbose) cerr<<"Mask Done: "<<maskpr+to_string(sb)+".nii.gz"<<endl;
	}

	return 0;
}

/**
 * @brief Fill a matrix (nrows x ncols) at the memory location provided by
 * rawdata. Each nonzero pixel in the mask corresponds to a column in rawdata,
 * and each timepoint corresponds to a row.
 *
 * @param rawdata Data, which should already be allocated, size nrows*ncols
 * @param nrows Number of rows in rawdata
 * @param ncols Number of cols in rawdata
 * @param img Image to read
 * @param mask Mask to use
 *
 * @return 0 if successful
 */
void fillMat(double* rawdata, size_t nrows, size_t ncols,
		ptr<const MRImage> img, ptr<const MRImage> mask)
{
	if(!mask->matchingOrient(img, false, true))
		throw INVALID_ARGUMENT("Mask and Image have different orientation or size!");
	if(nrows != img->tlen())
		throw INVALID_ARGUMENT("Input image tlen != nrows\n");

	// Create View of Matrix, and fill with PSD in each column
	Eigen::Map<MatrixXd> mat(rawdata, nrows, ncols);

	Vector3DConstIter<double> iit(img);
	NDConstIter<int> mit(mask);
	size_t cc=0;
	for(; !mit.eof() && !iit.eof(); ++iit, ++mit) {
		if(*mit != 0)
			continue;

		if(cc >= ncols)
			throw INVALID_ARGUMENT("masked pixels != ncols");

		for(int rr=0; rr<nrows; rr++)
			mat(rr, cc) =  iit[rr];

		// Only Increment Columns for Non-Zero Mask Pixels
		++cc;
	}

	if(cc != nrows)
		throw INVALID_ARGUMENT("masked pixels != ncols");
}
void MatrixReorg::preMult(Eigen::Ref<MatrixXd> out,
		const Eigen::Ref<const MatrixXd> in, bool transpose) const
{
	if(!transpose) {

		// Q = BA, Q = [BA_0 BA_1 ... ]
		if(out.rows() != in.rows() || out.cols() != cols() || rows() != in.cols()) {
			throw INVALID_ARGUMENT("Input arguments are non-conformant for "
					"matrix multiplication");
		}
		out.setZero();
		for(size_t cc=0, bb=0; cc<cols(); bb++) {
			// Load Block
			MatMap block(tallMatName(bb));

			// Multiply By Input
			out.middleCols(cc, m_outcols[bb]) = in*block.mat;
			cc += m_outcols[bb];
		}
	} else {
		// Q = BA^T, Q = [BA^T_1 ... ]
		if(out.rows() != in.rows() || out.cols() != rows() || cols() != in.cols()) {
			throw INVALID_ARGUMENT("Input arguments are non-conformant for "
					"matrix multiplication");
		}
		out.resize(in.rows(), rows());
		out.setZero();
		for(size_t cc=0, bb=0; cc<cols(); bb++) {
			// Load Block
			MatMap block(tallMatName(bb));

			// Multiply By Input
			out += in.middleCols(cc, m_outcols[bb])*block.mat.transpose();
			cc += m_outcols[bb];
		}
	}
};

void MatrixReorg::postMult(Eigen::Ref<MatrixXd> out,
		const Eigen::Ref<const MatrixXd> in, bool transpose) const
{
	if(!transpose) {
		if(out.rows() != rows() || out.cols() != in.cols() || cols() != in.rows()) {
			throw INVALID_ARGUMENT("Input arguments are non-conformant for "
					"matrix multiplication");
		}

		// Q = AB, Q = SUM(A_1B_1 A_2B_2 ... )
		out.setZero();
		for(size_t cc=0, bb=0; cc<cols(); bb++) {
			// Load Block
			MatMap block(tallMatName(bb));

			// Multiply By Input
			out += block.mat*in.middleRows(cc, m_outcols[bb]);
			cc += m_outcols[bb];
		}
	} else {
		if(out.rows() != cols() || out.cols() != in.cols() || rows() != in.rows()) {
			throw INVALID_ARGUMENT("Input arguments are non-conformant for "
					"matrix multiplication");
		}

		// Q = A^TB, Q = [A^T_1B ... ]
		out.setZero();
		for(size_t cc=0, bb=0; cc<cols(); bb++) {
			// Load Block
			MatMap block(tallMatName(bb));

			// Multiply By Input
			out.middleRows(cc, m_outcols[bb]) = block.mat.transpose()*in;
			cc += m_outcols[bb];
		}
	}
};


/**
 * @brief Fill a matrix, pointed to by rawdata with the Power-Spectral-Density
 * of each timeseries in img. Each column corresponds to an masked spatial
 * location in mask/img.
 *
 * @param rawdata Matrix to fill, should be allocated size nrows*ncols
 * @param nrows Number of rows in output data, must be >= img->tlen()
 * @param ncols Number of cols in output data, must be = # nonzer points in mask
 * @param img Input image, to fill data from
 * @param mask Mask determining whether points should be included in the matrix
 *
 * @return 0 if successful
 */
void fillMatPSD(double* rawdata, size_t nrows, size_t ncols,
		ptr<const MRImage> img, ptr<const MRImage> mask)
{
	if(!mask->matchingOrient(img, false, true))
		throw INVALID_ARGUMENT("Mask and Image have different orientation or size!");

	if(img->tlen() >= nrows)
		throw INVALID_ARGUMENT("fillMatPSD: nrows < tlen");

	// Create View of Matrix, and fill with PSD in each column
	Eigen::Map<MatrixXd> mat(rawdata, nrows, ncols);

	fftw_plan fwd;
	auto ibuffer = (double*)fftw_malloc(sizeof(double)*mat.rows());
	auto obuffer = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*mat.rows());
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

		if(cc >= ncols)
			throw INVALID_ARGUMENT("Number of masked pixels != ncols");

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

	if(cc != nrows)
		throw INVALID_ARGUMENT("Number of masked pixels != ncols");
}

GICAfmri::GICAfmri(std::string pref)
{
	m_pref = pref;
	maxmem = 4; //gigs
	verbose = 0;
	m_status = 0; //unitialized
	spatial = false; // spatial maps
	estrank = 100;
	poweriters = 1;
}

/**
 * @brief Compute ICA for the given group, defined by tcat x scat images
 * laid out in column major ordering.
 *
 * The basic idea is to split the rows into digesteable chunks, then
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
 * @param tcat Number of fMRI images to append in time direction
 * @param scat Number of fMRI images to append in space direction
 * @param masks Masks, one per spaceblock (columns of matching space)
 * @param inputs Files in time-major order, [s0t0 s0t1 s0t2 s1t0 s1t1 s1t2]
 * where s0 means 0th space-appended image, and t0 means the same for time
 */
void GICAfmri::computeICA()
{
	m_status = -1;

	MatrixReorg reorg(m_pref, (size_t)0.5*maxmem*(1<<27), verbose);
	cerr<<"Chcking matrices at prefix \""<<m_pref<<"\"...";
	int status = reorg.checkMats();
	if(status != 0)
		throw RUNTIME_ERROR("Error while loading existing 2D Matrices");
	cerr<<"Done\n";

	MatrixXd U, V;
	MatrixXd* Up = NULL;
	MatrixXd* Vp = NULL;

	if(spatial)
		Vp = &V;
	else
		Up = &U;

	VectorXd E = onDiskSVD(reorg, estrank, poweriters, Up, Vp);

	size_t rank = 0;
	double totalvar = E.sum();
	double var = 0;
	for(rank=0; rank<E.size(); rank++) {
		if(var > totalvar*varthresh)
			break;
		var += var + E[rank];
	}

	cerr << "SVD Rank: " << rank << endl;
	if(rank == 0) {
		throw INVALID_ARGUMENT("Error arguments have been set in such a "
				"way that no components will be selected. Try increasing "
				"your variance threshold or number of components");
	}

	if(spatial) {
		cerr<<"Performing Spatial ICA ("<<V.rows()<<"x"<<rank<<")"<<endl;
		MatMap tmpmat;
		tmpmat.create(m_pref+"_SICA", V.rows(), rank);
		tmpmat.mat = ica(V.leftCols(rank));
		cerr << "Done" << endl;
	} else {
		cerr<<"Performing Temporal ICA ("<<U.rows()<<"x"<<rank<<")"<<endl;
		MatMap tmpmat;
		tmpmat.create(m_pref+"_TICA", U.rows(), rank);
		tmpmat.mat = ica(U.leftCols(rank));
	}

	m_status = 1;
}

/**
 * @brief Compute ICA for the given group, defined by tcat x scat images
 * laid out in column major ordering.
 *
 * The basic idea is to split the rows into digesteable chunks, then
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
 * @param tcat Number of fMRI images to append in time direction
 * @param scat Number of fMRI images to append in space direction
 * @param masks Masks, one per spaceblock (columns of matching space)
 * @param inputs Files in time-major order, [s0t0 s0t1 s0t2 s1t0 s1t1 s1t2]
 * where s0 means 0th space-appended image, and t0 means the same for time
 * @param normts normalize
 */
void GICAfmri::compute(size_t tcat, size_t scat, vector<string> masks,
		vector<string> inputs)
{
	m_status = -1;

	size_t ndoubles = (size_t)(0.5*maxmem*(1<<27));
	MatrixReorg reorg(m_pref, ndoubles, verbose);
	// Don't use more than half of memory on each block of rows
	cerr << "Reorganizing data into matrices..."<<endl;
	int status = reorg.createMats(tcat, scat, masks, inputs, normts);
	if(status != 0)
		throw RUNTIME_ERROR("Error while reorganizing data into 2D Matrices");
	cerr<<"Done"<<endl;

	computeICA();
	computeSpatialMaps();
}

void GICAfmri::computeSpatialMaps()
{
	if(spatial) {
		cerr << "Spatial ICA" << endl;
		MatMap ics(m_pref+"_SICA");
		// Re-associate each column with a spatial signal (map)
		// Columns of ics correspond to rows of the wide matrices/voxels
		int npt = 0; // number of points in the current image
		int nptii = 0; // iterate through total points in all images
		int mm = 0; // current map
		int comp = 0; // current Independent Component
		int rr=0; // Current Row/Sample of IC
		for(size_t fullrows=0; fullrows != ics.rows; fullrows += npt) {
			for(mm=0; nptii<ics.rows; nptii+=npt, ++mm) {
				auto mask = readMRImage(m_pref+"_mask_"+to_string(mm)+".nii.gz");
				auto out = mask->copyCast(FLOAT32);
				FlatIter<int> mit(mask);
				FlatIter<double> oit(out);

				// count cols
				npt = 0;
				for(mit.goBegin(); !mit.eof(); ++mit) {
					if(*mit != 0)
						npt++;
				}

				// Iterate through Components
				for(comp=0; comp<ics.cols; comp++) {
					for(rr=0, mit.goBegin(),oit.goBegin(); !mit.eof(); ++mit, ++oit) {
						if(*mit != 0) {
							oit.set(ics.mat(rr, comp));
							rr++;
						}
					}
					out->write(m_pref+"_map_m"+to_string(mm)+"_c"+to_string(comp)+
							".nii.gz");
				}
			}
			if(nptii > ics.rows) {
				throw RUNTIME_ERROR("Error:\n"+m_pref+"_SICA\nSize does not match "
						"input masks");
			}

		}

		// TODO Mixture Model for T-Score
	} else {
		cerr << "Regressing Temporal Components" << endl;
		// Regress each column with each input timeseries
		MatMap ics;
		if(ics.open(m_pref+"_TICA") != 0)
			throw RUNTIME_ERROR(m_pref+"_TICA missing!");

		// need to compute the CDF for students_t_cdf
		const double MAX_T = 2000;
		const double STEP_T = 0.1;
		StudentsT distrib(ics.rows-1, STEP_T, MAX_T);
		cerr << "Computing Inverse of ICs ("<<ics.mat.rows()<<"x"
			<<ics.mat.cols()<<")"<<endl;
		MatrixXd Xinv = pseudoInverse(ics.mat);
		VectorXd Cinv = pseudoInverse(ics.mat.transpose()*ics.mat).diagonal();

		// Iterate through mask as we iterate through the columns of the ICs
		ptr<MRImage> mask;
		ptr<MRImage> out;
		FlatIter<int> mit;
		Vector3DIter<double> oit;

		// Create 4D Image matching mask
		size_t odim[4] = {0,0,0, ics.cols};

		int ii; // iterate through tall matrices
		int cc = 0; // iterate through columns of tall matrix
		int mm = -1; // iterate through masks
		RegrResult result;
		/*
		 * Convert statistics of each column of each tall matrix back to
		 * spatial position
		 */
		MatMap tall;
		for(ii=0; tall.open(m_pref+"_tall_"+to_string(ii)) == 0; ii++) {

			// Iterate through columns of tall matrix
			for(cc=0; cc<tall.cols; ) {
				// Match iterations of tall columns with mask iteration
				if(!mask || mit.eof()) {
					cerr << "Loading Mask" << endl;
					if(cc != 0) {
						throw RUNTIME_ERROR("Error Tall Matrix:\n"+m_pref+
								"_tall_"+to_string(ii)+"\nMismatches mask:"+
								m_pref+"_mask_"+to_string(mm));
					}

					// If there is currently an open output image, write
					if(out)
						out->write(m_pref+"_tmap_m"+to_string(mm)+".nii.gz");
					mm++;

					// Read Next Mask
					cerr<<"Reading mask ("<<mm<<") = "<<m_pref<<"_mask_"
							<<mm<<".nii.gz"<<endl;
					mask = readMRImage(m_pref+"_mask_"+to_string(mm)+".nii.gz");
					if(mask->ndim() != 3)
						throw RUNTIME_ERROR("Error input mask is not 3D!");
					for(size_t dd=0; dd<3; dd++)
						odim[dd] = mask->dim(dd);
					odim[3] = ics.cols;

					out = dPtrCast<MRImage>(mask->createAnother(4,
								odim, FLOAT32));
					mit.setArray(mask);
					oit.setArray(out);
					mit.goBegin();
					oit.goBegin();
				}

				if(*mit) {
					// regress each component with current column of tall
					regress(&result, tall.mat.col(cc), ics.mat,
							Cinv, Xinv, distrib);
					for(size_t comp=0; comp<ics.cols; comp++)
						oit.set(comp, result.t[comp]);
					cc++;
				} else {
					for(size_t comp=0; comp<ics.cols; comp++)
						oit.set(comp, 0);
				}

				++mit;
				++oit;
			}
		}

		// If there is currently an open output image, write as multiple, last
		if(out)
			out->write(m_pref+"_tmap_m"+to_string(mm)+".nii.gz");
		cerr << "Done with Regression" << endl;
	}
}


} // NPL

