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
VectorXd onDiskSVD(const MatrixReorg& A, int minrank, size_t poweriters,
		double varthresh, MatrixXd* U, MatrixXd* V)
{
	// Algorithm 4.4
	MatrixXd Yc;
	MatrixXd Yhc;
	MatrixXd Qtmp;
	MatrixXd Qhat;
	MatrixXd Omega;

	// From the Original Algorithm A -> At everywhere
	Yc.resize(A.cols(), minrank);
	Yhc.resize(A.rows(), minrank);
	Omega.resize(A.rows(), minrank);

	fillGaussian<MatrixXd>(Omega);
	A.postMult(Yc, Omega, true);

	Eigen::HouseholderQR<MatrixXd> qr(Yc);
	Qtmp = qr.householderQ()*MatrixXd::Identity(A.cols(), minrank);
	Eigen::HouseholderQR<MatrixXd> qrh;
	cerr << "Power Iteration: ";
	for(size_t ii=0; ii<poweriters; ii++) {
		cerr<<ii<<" ";
		A.postMult(Yhc, Qtmp);
		qrh.compute(Yhc);
		Qhat = qrh.householderQ()*MatrixXd::Identity(A.rows(), minrank);
		A.postMult(Yc, Qhat, true);
		qr.compute(Yc);
		Qtmp = qr.householderQ()*MatrixXd::Identity(A.cols(), minrank);
	}
	cerr << "Done" << endl;

	// Form B = Q* x A
	cerr<<"Making Low Rank Approximation ("<<Qtmp.cols()<<"x"<<A.rows()<<")"<<endl;
	MatrixXd B(Qtmp.cols(), A.rows());;
	A.preMult(B, Qtmp.transpose(), true);
	Eigen::JacobiSVD<MatrixXd> smallsvd(B, Eigen::ComputeThinU | Eigen::ComputeThinV);

	// Estimate Rank
	size_t rank = 0;
	double totalvar = smallsvd.singularValues().sum();
	double var = 0;
	for(rank=0; rank<smallsvd.singularValues().size(); rank++) {
		if(var > totalvar*varthresh)
			break;
		var += smallsvd.singularValues()[rank];
	}
	cerr << "SVD Rank: " << rank << endl;
	if(rank == 0) {
		throw INVALID_ARGUMENT("Error arguments have been set in such a "
				"way that no components will be selected. Try increasing "
				"your variance threshold or number of components");
	}

	cerr<<"Filling U,E,V matrices"<<endl;
	VectorXd E(rank);
	E = smallsvd.singularValues().head(rank);
	if(V)
		*V = Qtmp*smallsvd.matrixU().leftCols(rank);
	if(U)
		*U = smallsvd.matrixV().leftCols(rank);

	return E;
}

/**
 * @brief Estimates noise in columns of A by computing the difference between
 * the columns of A and the predicted columns from UEVt.
 *
 * @param A Matrix with noisy full data
 * @param U U matrix in SVD
 * @param E E diagonal matrix in SVD
 * @param V V right singular vectors in SVD
 *
 * @return Vector of errors, one for each column in A
 */
VectorXd estnoise(const MatrixReorg& A,
		Ref<MatrixXd> U, Ref<VectorXd> E, Ref<MatrixXd> V)
{
	VectorXd out(A.cols());
	size_t oc = 0;
	for(size_t cg = 0; cg < A.tallMatCols().size(); cg++) {
		MatMap m(A.tallMatName(cg));
		if(A.tallMatCols()[cg] != m.cols())
			throw INVALID_ARGUMENT("Columns mismatch between read data and "
					"previously written data");
		for(size_t cc=0; cc<m.cols(); cc++, oc++)
			out[oc] = (m.mat.col(cc)-U*E.asDiagonal()*V.row(oc).transpose()).norm();
	}
	return out;
}


/**
 * @brief Computes the SVD from XXt using the JacobiSVD
 *
 * @param A MatrixReorg object that can be used to load images on disk
 * @param U Output U matrix, if null then ignored
 * @param V Output V matrix, if null then ignored
 *
 * @return Vector of singular values
 */
VectorXd covSVD(const MatrixReorg& A, double varthresh, MatrixXd* U, MatrixXd* V)
{
	if(!U && V){
		throw INVALID_ARGUMENT("U must be nonnull to compute V!");
	}

	if(varthresh < 0 || varthresh > 1)
		varthresh = 0.9;

	int rank = 0; // Rank of Singular Value Decomp

	// Create AA* matrix
	MatrixXd AAt(A.rows(), A.rows());
	for(size_t cc = 0; cc < A.ntall(); cc++) {
		MatMap m(A.tallMatName(cc));
		AAt += m.mat*m.mat.transpose();
	}

	/*
	 * Perform SVD
	 */
	Eigen::SelfAdjointEigenSolver<MatrixXd> eig(AAt);
	const auto& evecs = eig.eigenvectors();
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

	VectorXd singvals(rank);
	for(size_t ii=0; ii<rank; ii++)
		singvals[ii] = sqrt(evals[AAt.rows()-ii-1]);

	if(U) {
		U->resize(A.rows(), rank);
		for(size_t ii=0; ii<rank; ii++)
			U->col(ii) = evecs.col(AAt.rows()-ii-1);
	}

	// V = A^TUE^-1
	if(V) {
		V->resize(A.cols(), rank);
		A.postMult(*V, *U, true);
		for(size_t cc=0; cc<rank; cc++)
			V->col(cc) /= singvals[cc];
	}

	return singvals;
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
		if(m_totalrows != map.rows())
			throw RUNTIME_ERROR("Error, mismatch in file size ("+
					to_string(map.rows())+"x"+to_string(map.cols())+" vs "+
					to_string(m_totalrows)+"x"+to_string(m_totalcols)+")");
		m_outcols.push_back(map.cols());
		cols += map.cols();
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
			cerr<<"Reading mask in column "<<sb<< " from "
				<<masknames[sb]<<endl;
			mask = readMRImage(masknames[sb]);
		} else {
			cerr<<"Creating mask in column "<<sb<< " from "
				<<filenames[sb*timeblocks+0]<<endl;
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
		auto img = readMRImage(filenames[0*timeblocks+tb], false, true);
		inrows[tb] += img->tlen();
		m_totalrows += inrows[tb];
		if(m_verbose)
			cerr << "rows += "<<inrows[tb]<<" = " <<m_totalrows<<endl;
	}

	if(m_verbose) {
		cerr << endl;
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

	if(m_verbose) cerr << "Creating Blank Matrices"<<endl;
	MatMap datamap;
	for(int cc=0; cc<m_totalcols; cc++) {
		if(blockind == incols[blocknum]) {
			// open file, create with proper size
			datamap.create(tallMatName(m_outcols.size()-1), m_totalrows,
					m_outcols.back());
			// if the index in the block would put us in a different image, then
			// start a new out block, and new in block
			blockind = 0;
			blocknum++;
			m_outcols.push_back(0);
		} else if((m_outcols.back()+1)*m_totalrows > m_maxdoubles) {
			datamap.create(tallMatName(m_outcols.size()-1), m_totalrows,
					m_outcols.back());
			// If this col won't fit in the current block of cols, start a new one,
			m_outcols.push_back(0);
		}
		blockind++;
		m_outcols.back()++;
	}
	// Final File
	datamap.create(tallMatName(m_outcols.size()-1), m_totalrows,
			m_outcols.back());

	/*
	 * Fill tall matrices by breaking images along block cols specified
	 * in m_outcols
	 */
	if(m_verbose) {
		cerr << "Filling Matrices"<<endl;
		if(normts) cerr<<"Normalizing Individual Timeseries"<<endl;
	}
	int img_glob_row = 0;
	int img_glob_col = 0;
	int img_oblock_col = 0;
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
						if(datamap.rows() != m_totalrows ||
								datamap.cols() != m_outcols[colbl]) {
							throw INVALID_ARGUMENT("Unexpected size in input "
									+ tallpr+to_string(colbl)+" expected "+
									to_string(m_totalrows)+"x"+
									to_string(m_outcols[colbl])+", found "+
									to_string(datamap.rows())+"x"+
									to_string(datamap.cols()));
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

GICAfmri::GICAfmri(std::string pref)
{
	m_pref = pref;
	varthresh = 0.9;
	minrank = 200;
	poweriters = 2;
	maxmem = 4; //gigs
	verbose = false;
	spatial = true;
	normts = true;
	minfreq = 0.01;
	maxfreq = 0.5;
	fullsvd = false;
}

void GICAfmri::createMatrices(size_t tcat, size_t scat, vector<string> masks,
		vector<string> inputs)
{
	cerr << "Reorganizing data into matrices..."<<endl;
	size_t ndoubles = (size_t)(0.5*maxmem*(1<<27));
	new (&m_A) MatrixReorg(m_pref, ndoubles, verbose);
	int status = m_A.createMats(tcat, scat, masks, inputs, normts);
	if(status != 0)
		throw RUNTIME_ERROR("Error while reorganizing data into 2D Matrices");
	cerr<<"Done"<<endl;
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
	createMatrices(tcat, scat, masks, inputs);

	if(fullsvd)
		m_E = covSVD(m_A, varthresh, &m_U, &m_V);
	 else
		m_E = onDiskSVD(m_A, minrank, poweriters, varthresh, &m_U, &m_V);

	if(spatial)
		 computeSpatialICA();
	else
		 computeTemporaICA();
}

void GICAfmri::computeTemporaICA()
{
	/*
	 * Compute ICA and spatial maps
	 */
	size_t odim[4];
	cerr<<"Performing Temporal ICA ("<<m_U.rows()<<"x"<<m_U.cols()<<")"<<endl;
	MatrixXd ics = ica(m_U);

	cerr << "Regressing Temporal Components" << endl;
	/*
	 * Initialize Regression Variables
	 */
	const double MAX_T = 2000;
	const double STEP_T = 0.1;
	StudentsT distrib(ics.rows()-1, STEP_T, MAX_T);
	MatrixXd Xinv = pseudoInverse(ics);
	VectorXd Cinv = pseudoInverse(ics.transpose()*ics).diagonal();
	RegrResult result;

	/*
	 * Convert statistics of each column of each tall matrix back to
	 * spatial position
	 * Iterate through mask as we iterate through the columns of the ICs
	 * Create 4D Image matching mask
	 */
	size_t tc = 0; // tall column
	size_t matn = 0; // matrix number (tall matrices)
	size_t maskn = 0; // Mask number
	MatMap tall(m_A.tallMatName(matn));
	for(size_t cc=0; cc < m_A.cols(); maskn++) {
		auto mask = readMRImage(m_pref+"_mask_"+to_string(maskn)+".nii.gz");
		for(size_t ii=0; ii<3; ii++)
			odim[ii] = mask->dim(ii);
		odim[3] = ics.cols();

		auto tmap = mask->copyCast(4, odim, FLOAT32);
		auto pmap = mask->copyCast(4, odim, FLOAT32);
		auto bmap = mask->copyCast(4, odim, FLOAT32);
		NDIter<int> mit(mask);
		Vector3DIter<double> pit(pmap), tit(tmap), bit(bmap);

		// Iterate Through pixels of mask
		for(; !mit.eof(); ++mit, ++pit, ++tit, ++bit) {
			if(*mit == 0) {
				for(size_t comp=0; comp<ics.cols(); comp++) {
					tit.set(comp, 0);
					pit.set(comp, 0.5);
				}
			}
			// Load the next block of columns as necessary
			if(tc >= tall.cols()) {
				tall.open(m_A.tallMatName(++matn));
				tc = 0;
			}

			// Perform regression
			regress(&result, tall.mat.col(tc), ics, Cinv, Xinv, distrib);
			for(size_t comp=0; comp<ics.cols(); comp++) {
				tit.set(comp, result.t[comp]);
				pit.set(comp, result.p[comp]);
				bit.set(comp, result.bhat[comp]);
			}
			cc++;
			tc++;
		}
		// write output matching mask
		tmap->write(m_pref+"_tmap_m"+to_string(maskn)+".nii.gz");
		pmap->write(m_pref+"_pmap_m"+to_string(maskn)+".nii.gz");
		bmap->write(m_pref+"_bmap_m"+to_string(maskn)+".nii.gz");
	}

	cerr << "Done with Regression" << endl;
}

void GICAfmri::computeSpatialICA()
{
	cerr<<"Performing Spatial ICA ("<<m_V.rows()<<"x"<<m_V.cols()<<")"<<endl;
	MatrixXd ics = ica(m_V);
	cerr << "Done" << endl;

	// Convert each signal matrix into a t-score
	cerr<<"Estimating Noise"<<endl;
	VectorXd sigma = estnoise(m_A, m_U, m_E, m_V);
	const double MAX_T = 2000;
	const double STEP_T = 0.1;
	StudentsT distrib(ics.rows()-1, STEP_T, MAX_T);

	// Re-associate each column with a spatial signal (map)
	// Columns of ics correspond to rows of the wide matrices/voxels
	cerr<<"Computing T-Maps"<<endl;
	size_t odim[4];
	for(size_t rr=0, maskn = 0; rr < ics.rows(); maskn++) {
		auto mask = readMRImage(m_pref+"_mask_"+to_string(maskn)+".nii.gz");
		for(size_t ii=0; ii<3; ii++)
			odim[ii] = mask->dim(ii);
		odim[3] = ics.cols();

		auto tmap = mask->copyCast(4, odim, FLOAT32);
		auto pmap = mask->copyCast(4, odim, FLOAT32);
		auto bmap = mask->copyCast(4, odim, FLOAT32);
		NDIter<int> mit(mask);
		Vector3DIter<double> pit(pmap), tit(tmap), bit(bmap);

		// Iterate Through pixels of mask
		for(; !mit.eof(); ++mit, ++pit, ++tit, ++bit) {
			if(*mit == 0) {
				for(size_t comp=0; comp<ics.cols(); comp++) {
					tit.set(comp, 0);
					pit.set(comp, 0.5);
				}
			}
			// Iterate through Components
			for(size_t comp=0; comp<ics.cols(); comp++) {
				double b = ics(rr, comp);
				double t = sigma[rr] == 0 ? 0 : b/sigma[rr];
				double p = distrib.cdf(t);
				if(p > 0.5) p = 1-p;
				p *= 2;
				tit.set(comp, t);
				pit.set(comp, p);
				bit.set(comp, b);
			}
			rr++;
		}
		// write output matching mask
		tmap->write(m_pref+"_tmap_m"+to_string(maskn)+".nii.gz");
		pmap->write(m_pref+"_pmap_m"+to_string(maskn)+".nii.gz");
		bmap->write(m_pref+"_bmap_m"+to_string(maskn)+".nii.gz");
	}
	cerr<<"Done with Spatial ICA!"<<endl;
}

} // NPL

