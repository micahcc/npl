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
 * matrices. 
 *
 *****************************************************************************/

#include "ica_helpers.h"

#include <Eigen/Dense>
#include <Eigen/IterativeSolvers>

#include <iostream>
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
 * Macro Level Functions for Performing Large Scale ICA Analysis
 ****************************************************************************/

/**
 * @brief Helper function for large-scale ICA analysis. This takes
 * a working directory, which should already have 'mat_#' files with data
 * (one per element of nrows) and orthogonalizes the data to produce a
 * set of variables which are orthogonal.
 *
 * This assumes that the input is a set of matrices which will be concat'd in
 * the row (time) direction. By default it is assumed that the columns
 * represent dimensions and the rows samples. This means that the output
 * of Xorth will normally have one row for every row of the concatinated
 * matrices and far fewer columns. If rowdims is true then Xorth will have one
 * row for each of the column (which is the same for all images), and fewer
 * columns than the original matrices had rows.
 *
 * @param workdir Directory which should have mat_0, mat_1, ... up to
 * nrows.size()-1
 * @param evthresh Threshold for percent of variance to account for in the
 * original data when reducing dimensions to produce Xorth. This is determines
 * the number of dimensions to keep.
 * @param lancbasis Number of starting basis vectors to initialize the
 * BandLanczos algorithm with. If this is <= 1, one dimension of of XXT will be
 * used.
 * @param maxrank Maximum number of dimensions to keep in Xorth
 * @param rowdims Perform reduction on the original inputs rows. Note that the
 * output will still be in column format, where each columns is a dimension,
 * but the input will be treated such that each row is dimension. This makes it
 * easier to perform ICA.
 * @param nrows Number of rows in each block of rows
 * @param XXT Covariance (X*X.transpose())
 * @param Xorth Output orthogonal version of X
 *
 * @return
 */
int timecat_orthog(std::string workdir, double evthresh,
		int lancbasis, int maxrank, bool rowdims, const std::vector<int>& ncols,
		const MatrixXd& XXT, MatrixXd& Xorth)
{
	// Compute EigenVectors (U)
	Eigen::BandLanczosSelfAdjointEigenSolver<double> eig;
	eig.setTraceStop(evthresh);
	eig.setRank(maxrank);
	eig.compute(XXT, lancbasis);

	if(eig.info() == Eigen::NoConvergence) {
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
			string fn = workdir+"/mat_"+to_string(ii);
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
 * @param workdir Directory which should have mat_0, mat_1, ... up to
 * ncols.size()-1
 * @param evthresh Threshold for percent of variance to account for in the
 * original data when reducing dimensions to produce Xorth. This is determines
 * the number of dimensions to keep.
 * @param lancbasis Number of starting basis vectors to initialize the
 * BandLanczos algorithm with. If this is <= 1, one dimension of of XXT will be
 * used.
 * @param maxrank Maximum number of dimensions to keep in Xorth
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
int spcat_orthog(std::string workdir, double evthresh,
		int lancbasis, int maxrank, bool rowdims, const std::vector<int>& ncols,
		const MatrixXd& XXT, MatrixXd& Xorth)
{
	// Compute EigenVectors (U)
	Eigen::BandLanczosSelfAdjointEigenSolver<double> eig;
	eig.setTraceStop(evthresh);
	eig.setRank(maxrank);
	eig.compute(XXT, lancbasis);

	if(eig.info() == Eigen::NoConvergence) {
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
			string fn = workdir+"/mat_"+to_string(ii);
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
 * @param workdir Directory to create mat_* files and mask_* files in
 * @param evthresh Threshold for eigenvalues of XXT and singular values. Scale
 * is a ratio from 0 to 1 relative to the total sum of eigenvalues/variance.
 * Infinity will the BandLanczos Algorithm to run to completion and only
 * singular values > sqrt(epsilon) to be kept
 * @param lancbasis Basis size of BandLanczos Algorithm. This is used as the
 * seed for the Krylov Subspace.
 * @param maxrank Maximum rank of reduced dimension PCA
 * @param spatial Whether to do spatial ICA. Warning this is much more memory
 * and CPU intensive than PSD/Time ICA.
 *
 * @return 0 if successul
 */
int spcat_ica(bool psd, const vector<string>& imgnames,
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
		if(psd) {
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

		string fn = workdir+"/mat_"+to_string(ii);
		MemMap mmap(fn, ncols[ii]*nrows+sizeof(double), true);
		if(mmap.size() <= 0)
			return -1;

		double* ptr = (double*)mmap.data();
		if(psd) {
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
		string fn = workdir+"/mat_"+to_string(ii);
		MemMap mmap(fn, ncols[ii]*nrows+sizeof(double), false);
		if(mmap.size() <= 0)
			return -1;

		// Sum up XXT from each chunk
		Eigen::Map<MatrixXd> X((double*)mmap.data(), nrows, ncols[ii]);
		XXT += X*X.transpose();
	}

	// Each column is a regressor
	MatrixXd regressors;
	if(spcat_orthog(workdir, evthresh, lancbasis, maxrank,
				spatial, ncols, XXT, regressors) != 0) {
		return -1;
	}

	// Now Perform ICA
	MatrixXd X_ics = ica(regressors);

	// Create Maps
	// TODO

	return 0;
}

MatrixReorg::MatrixReorg(std::string prefix, size_t maxdoubles, bool verbose)
{
	m_prefix = prefix;
	m_maxdoubles = maxdoubles;
	m_verbose = verbose;
};

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
 *
 * @return 0 if succesful, -1 if read failure, -2 if write failure
 */
int MatrixReorg::createMats(size_t timeblocks, size_t spaceblocks, 
		const std::vector<std::string>& masknames, 
		const std::vector<std::string>& filenames)
{
	// size_t m_totalrows;
	// size_t m_totalcols;
	// std::string m_prefix;
	// size_t m_maxdoubles;
	// bool m_verbose;
	// vector<int> m_inrows;
	// vector<int> m_incols;
	// vector<int> m_outrows;
	// vector<int> m_outcols;
	std::string tallpr = m_prefix + "_tall_";
	std::string widepr = m_prefix + "_wide_";
	std::string maskpr = m_prefix + "_mask_";

	if(m_verbose) {
		cerr << "Tall Matrix Prefix: " << tallpr << endl;
		cerr << "Wide Matrix Prefix: " << widepr << endl;
		cerr << "Mask Prefix:        " << maskpr << endl;
	}

	/* Determine Size:
	 *
	 * First Run Down the Top Row and Left Column to determine the number of
	 * rows in each block of rows and number of columns in each block of
	 * columns, also create masks in the output folder
	 */
	m_inrows.resize(timeblocks);
	m_incols.resize(spaceblocks);
	std::fill(m_inrows.begin(), m_inrows.end(), 0);
	std::fill(m_incols.begin(), m_incols.end(), 0);

	m_totalrows = 0;
	m_totalcols = 0;

	// Figure out the number of cols in each block, and create masks
	ptr<MRImage> mask;
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
				m_incols[sb]++;
		}
		if(m_incols[sb] == 0) {
			throw INVALID_ARGUMENT("Error, input mask for column " +
					to_string(sb) + " has no non-zero pixels");
		}
		m_totalcols += m_incols[sb];
	}

	// Figure out number of rows in each block
	for(size_t tb = 0; tb<timeblocks; tb++) {
		auto img = readMRImage(filenames[0*timeblocks+tb]);

		m_inrows[tb] += img->tlen();
		m_totalrows += m_inrows[tb];
	}

	if(m_verbose) {
		cerr << "Row/Time  Blocks: " << timeblocks << endl;
		cerr << "Col/Space Blocks: " << spaceblocks << endl;
		cerr << "Total Rows/Timepoints: " << m_totalrows<< endl;
		cerr << "Total Cols/Voxels:     " << m_totalcols << endl;
	}

	/*
	 * Break rows and columns into digestable sizes, and don't allow chunks to
	 * cross lines between images (this will make loading the matrices easier)
	 */
	// for wide matrices
	m_outrows.resize(1); m_outrows[0] = 0; // number of rows per block
	std::vector<int> ob_srow(1, 0); // starting global row for each block

	// for tall matrices
	m_outcols.resize(1); m_outcols[0] = 0; // number of cols per block
	std::vector<int> ob_scol(1, 0); // starting global col for each block
	int blockind = 0;
	int blocknum = 0;

	// wide matrices, create files
	// break up output blocks of rows into chunks that 1) don't cross images 
	// and 2) with have fewer elements than m_maxdoubles
	if(m_totalcols > m_maxdoubles) {
		throw INVALID_ARGUMENT("maxdoubles is not large enough to hold a "
				"single full row!");
	}
	blockind = 0;
	blocknum = 0;
	for(int rr=0; rr<m_totalrows; rr++) {
		if(blockind == m_inrows[blocknum]) {
			// open file, create with proper size
			MemMap wfile(widepr+to_string(m_outrows.size()-1), 2*sizeof(size_t)+
					m_outrows.back()*m_totalcols*sizeof(double), true);
			((size_t*)wfile.data())[0] = m_outrows.back(); // nrows
			((size_t*)wfile.data())[1] = m_totalcols; // nrows

			// if the index in the block would put us in a different image, then
			// start a new out block
			blockind = 0;
			blocknum++;
			m_outrows.push_back(0);
			ob_srow.push_back(rr);
		} else if((m_outrows.back()+1)*m_totalcols > m_maxdoubles) {
			// open file, create with proper size
			MemMap wfile(widepr+to_string(m_outrows.size()-1), 2*sizeof(size_t)+
					m_outrows.back()*m_totalcols*sizeof(double), true);
			((size_t*)wfile.data())[0] = m_outrows.back(); // nrows
			((size_t*)wfile.data())[1] = m_totalcols; // ncols

			// If this row won't fit in the current block of rows, start a new one, 
			m_outrows.push_back(0);
			ob_srow.push_back(rr);
		}
		blockind++;
		m_outrows.back()++;
	}
	{ // Create Last File
	MemMap wfile(widepr+to_string(m_outrows.size()-1), 2*sizeof(size_t)+
			m_outrows.back()*m_totalcols*sizeof(double), true);
	((size_t*)wfile.data())[0] = m_outrows.back(); // nrows
	((size_t*)wfile.data())[1] = m_totalcols; // nrows
	}

	// tall matrices, create files
	// break up output blocks of cols into chunks that 1) don't cross images 
	// and 2) with have fewer elements than m_maxdoubles
	if(m_totalrows > m_maxdoubles) {
		throw INVALID_ARGUMENT("maxdoubles is not large enough to hold a "
				"single full column!");
	}
	blockind = 0;
	blocknum = 0;
	for(int cc=0; cc<m_totalcols; cc++) {
		if(blockind == m_incols[blocknum]) {
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
			ob_scol.push_back(cc);
		} else if((m_outcols.back()+1)*m_totalrows > m_maxdoubles) {
			MemMap tfile(tallpr+to_string(m_outcols.size()-1), 2*sizeof(size_t)+
					m_outcols.back()*m_totalrows*sizeof(double), true);
			((size_t*)tfile.data())[0] = m_totalrows; // nrows
			((size_t*)tfile.data())[1] = m_outcols.back(); // ncols

			// If this col won't fit in the current block of cols, start a new one, 
			m_outcols.push_back(0);
			ob_scol.push_back(cc);
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

	// Fill Tall and Wide Matrices by breaking up images along block rows and
	// block cols specified in m_outcols and m_outrows. 
	int img_glob_row = 0;
	int img_glob_col = 0;
	int img_oblock_row = 0;
	int img_oblock_col = 0;
	MatMap datamap;
	for(size_t sb = 0; sb<spaceblocks; sb++) {
		auto mask = readMRImage(maskpr+to_string(sb)+".nii.gz");

		img_glob_row = 0;
		img_oblock_row = 0;
		for(size_t tb = 0; tb<timeblocks; tb++) {
			auto img = readMRImage(filenames[sb*timeblocks+tb]);

			if(!img->matchingOrient(mask, false, true))
				throw INVALID_ARGUMENT("Mismatch in mask/image size in col:"+
						to_string(sb)+", row:"+to_string(tb));
			if(img->tlen() != m_inrows[tb])
				throw INVALID_ARGUMENT("Mismatch in time-length in col:"+
						to_string(sb)+", row:"+to_string(tb));

			int rr, cc, colbl, rowbl, tt;
			int tlen = img->tlen();
			Vector3DIter<double> it(img);
			NDIter<double> mit(mask);

			// Tall Matrix, fill mat[img_glob_row:img_glob_row+tlen,0:bcols],
			// Start with invalid and load new columns as needed
			// cc iterates over the columns in the block, colbl indicates the
			// index of the block in the overall scheme
			for(cc=-1, colbl=img_oblock_col-1; !it.eof(); ++it, ++mit) {
				if(*mit != 0) {
					// If cc is invalid, open
					if(cc < 0 || cc >= m_outcols[colbl]) {
						cc = 0;
						colbl++;
						datamap.open(tallpr+to_string(colbl));
						if(datamap.rows != m_totalrows || datamap.cols != m_outcols[colbl]) {
							throw INVALID_ARGUMENT("Unexpected size in input " 
									+ tallpr+to_string(colbl));
						}
					}

					// rows are global, cols are local to block
					for(size_t tt=0; tt<tlen; tt++) {
						datamap.mat(tt+img_glob_row, cc) = it[tt];
					}
					cc++;
				}
			}

			assert(cc == m_outcols[colbl]);
			datamap.close();

			// Wide Matrix, fill mat[0:brows,img_glob_col:img_glob_col+nvox],
			// Start with invalid and load new columns as needed
			// rr iterates over the rows in the block
			// rowbl is the index of the block in the set of output blocks
			// tt is the index row position in the image
			for(rr=-1, rowbl=img_oblock_row-1, tt=0; tt < tlen; rr++, tt++){
				if(rr < 0 || rr >= m_outrows[rowbl]) {
					rr = 0;
					rowbl++;
					datamap.open(widepr+to_string(rowbl));
					if(datamap.rows != m_outrows[rowbl] || datamap.cols !=
							m_totalcols) {
						throw INVALID_ARGUMENT("Unexpected size in input " 
								+ tallpr+to_string(rowbl));
					}
				}

				it.goBegin(), mit.goBegin();
				for(int cc=img_glob_col; !it.eof(); ++it, ++mit) {
					if(*mit != 0)
						datamap.mat(rr, cc++) = it[tt];
				}
			}

			assert(rr == m_outrows[rowbl]);
			datamap.close();

			// Increment Global Row by Input Block Size (same as image rows)
			img_glob_row += m_inrows[tb];

			// Increment Output block row to correspond to the next image
			for(int ii=0; ii != m_inrows[tb]; )
				ii += m_outrows[img_oblock_row++];
		}

		// Increment Global Col by Input Block Size (same as image cols)
		img_glob_col += m_incols[sb];

		// Increment Output block col to correspond to the next image
		for(int ii=0; ii != m_incols[sb]; )
			ii += m_outcols[img_oblock_col++];
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
int fillMat(double* rawdata, size_t nrows, size_t ncols,
		ptr<const MRImage> img, ptr<const MRImage> mask)
{
	if(!mask->matchingOrient(img, false, true)) {
		cerr << "Mask and Image have different orientation or size!" << endl;
		return -1;
	}
	if(nrows != img->tlen()) {
		cerr << "Input image tlen != nrows\n";
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

		if(cc >= ncols) {
			cerr << "Input image has non-masked pixels != ncols\n";
			return -1;
		}

		for(int rr=0; rr<nrows; rr++)
			mat(rr, cc) =  iit[rr];

		// Only Increment Columns for Non-Zero Mask Pixels
		++cc;
	}

	if(cc != nrows) {
		cerr << "Input image has non-masked pixels != ncols\n";
		return -1;
	}
	return 0;
}

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
int fillMatPSD(double* rawdata, size_t nrows, size_t ncols,
		ptr<const MRImage> img, ptr<const MRImage> mask)
{
	if(!mask->matchingOrient(img, false, true)) {
		cerr << "Mask and Image have different orientation or size!" << endl;
		return -1;
	}

	if(img->tlen() >= nrows) {
		cerr << "fillMatPSD: nrows < tlen" << endl;
		return -1;
	}

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

		if(cc >= ncols) {
			cerr << "Input image has non-masked pixels != ncols\n";
			return -1;
		}

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

	if(cc != nrows) {
		cerr << "Input image has non-masked pixels != ncols\n";
		return -1;
	}
	return 0;
}

} // NPL

