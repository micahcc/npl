/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file fmri_inference.cpp Tools for performing ICA, including rewriting images as
 * matrices. All the main functions for real world ICA.
 *
 *****************************************************************************/

#include "fmri_inference.h"

#include <Eigen/Dense>
#include <Eigen/IterativeSolvers>

#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>

#include "fftw3.h"

#include "utility.h"
#include "npltypes.h"
#include "ndarray_utils.h"
#include "mrimage.h"
#include "nplio.h"
#include "iterators.h"
#include "statistics.h"
#include <signal.h>

using namespace std;

namespace npl {

std::string sica_name(std::string m_pref) {
	return m_pref+"_SpaceIC";
};
std::string tica_name(std::string m_pref) {
	return m_pref+"_TimeIC";
};
std::string sw_name(std::string m_pref) {
	return m_pref+"_SpaceW";
};
std::string tw_name(std::string m_pref) {
	return m_pref+"_TimeW";
};
std::string tmap_name(std::string m_pref, size_t ii) {
	return m_pref+"_tmap_m"+std::to_string(ii)+".nii.gz";
};
std::string pmap_name(std::string m_pref, size_t ii) {
	return m_pref+"_pmap_m"+std::to_string(ii)+".nii.gz";
};
std::string zmap_name(std::string m_pref, size_t ii) {
	return m_pref+"_zmap_m"+std::to_string(ii)+".nii.gz";
};
std::string bmap_name(std::string m_pref, size_t ii) {
	return m_pref+"_bmap_m"+std::to_string(ii)+".nii.gz";
};
std::string tplot_name(std::string m_pref, size_t ii) {
	return m_pref+"_tplot_c"+std::to_string(ii)+".svg";
};
std::string U_name(std::string m_pref) {
	return m_pref+"_Umat";
};
std::string V_name(std::string m_pref) {
	return m_pref+"_Vmat";
};
std::string E_name(std::string m_pref) {
	return m_pref+"_Evec";
};

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
 * @param A Input matrix (made up of many tall on disk matrices)
 * @param minrank Minimum rank to estimate
 * @param poweriters Number of power iterations to perform during computation
 * @param varthresh stop after the eigenvalues reach this ratio of the maximum
 * @param cvarthresh stop after the sum of eigenvalues reaches this ratio of total
 * @param U Output U matrix, if null then ignored
 * @param V Output V matrix, if null then ignored
 *
 * @return Vector of singular values
 */
VectorXd onDiskSVD(const MatrixReorg& A, int minrank, size_t poweriters,
		double varthresh, double cvarthresh, MatrixXd* U, MatrixXd* V)
{
	// Algorithm 4.4
	// From the Original Algorithm A -> At everywhere
	cerr<<"Allocating 2x "<<A.cols()<<"x"<<minrank<<" and 3x "<<A.rows()<<"x"<<minrank<<endl;
	MatrixXd Qtmp(A.cols(), minrank);

	{
	MatrixXd Yc(A.cols(), minrank);
	MatrixXd Yhc(A.rows(), minrank);
	MatrixXd Qhat(A.rows(), minrank);
	MatrixXd Omega(A.rows(), minrank);
	cerr<<"Done"<<endl;

	if(varthresh < 0 || varthresh > 1)
		varthresh = 0.1;
	if(cvarthresh < 0 || cvarthresh > 1)
		cvarthresh = 0.9;

	cerr<<"Gaussian Projecting"<<endl;
	fillGaussian<MatrixXd>(Omega);
	A.postMult(Yc, Omega, true);
	cerr<<"Done"<<endl;

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
	}

	// Form B = Q* x A
	cerr<<"Making Low Rank Approximation ("<<Qtmp.cols()<<"x"<<A.rows()<<")"<<endl;
	MatrixXd B(Qtmp.cols(), A.rows());;
	A.preMult(B, Qtmp.transpose(), true);
	Eigen::JacobiSVD<MatrixXd> smallsvd(B, Eigen::ComputeThinU | Eigen::ComputeThinV);
	const auto& svals = smallsvd.singularValues();

	// Estimate Rank
	size_t rank = 0;
	double maxval = smallsvd.singularValues()[0];
	double totalvar = smallsvd.singularValues().sum();
	double var = 0;
	for(rank=0; rank<smallsvd.singularValues().size(); rank++) {
		if(var > totalvar*cvarthresh || svals[rank] < varthresh*maxval)
			break;
		var += svals[rank];
	}
	cerr<<"SVD Rank: "<<rank<<endl;
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
 * Let A = UEVt, and VW = S and Vt = WS^t and A=UEWS^t, and form a regression
 * using Y = XB, with Y=A, X=UEW and B=S^t, then
 * inv(XtX) = inv(WtEUtUEW)
 * inv(XtX) = inv(WtE^2W)
 * inv(XtX) = WtE^-2W
 *
 * B(COMP, OUTNUM)
 * ssres = (UEV-A)^2_outnum
 * SIGMAHAT(OUTNUM) = ssres/(SAMPLES-REGRESSORS)
 * STDERR(OUTNUM,REGRESSOR) = beta(COMP, OUTNUM)*inv(XtX)(comp, comp)
 *
 * Standard error = sqrt(sigmahat*C^-1_ii)
 * C^1 = ((UEW)^t(UEW))^1
 * C^1 = (WtE^2W))^1
 * C^1 = WtE^-2W
 *
 * B(COMP,OUTNUM) = IC(OUTNUM, COMP)
 *
 * @param A Matrix with noisy full data
 * @param U U matrix in SVD
 * @param E E diagonal matrix in SVD
 * @param V V right singular vectors in SVD
 *
 * @return Vector of errors, one for each column in A
 */
MatrixXd icsToT(const MatrixReorg& A,
		Ref<MatrixXd> U, Ref<VectorXd> E, Ref<MatrixXd> V,
		Ref<MatrixXd> W, Ref<MatrixXd> ics)
{
	// sigmahat = (A - UEV^t), for each column there is a sigma hat
	size_t nreg = U.cols();
	size_t npoints = U.rows();
	VectorXd Cinv = (W.transpose()*(E.array().pow(-2)).matrix().asDiagonal()*W).diagonal();
	MatrixXd out = ics;
	size_t oc = 0;
	for(size_t cg = 0; cg < A.tallMatCols().size(); cg++) {
		MatMap m(A.tallMatName(cg), false);
		if(A.tallMatCols()[cg] != m.cols())
			throw INVALID_ARGUMENT("Columns mismatch between read data and "
					"previously written data");
		for(size_t cc=0; cc<m.cols(); cc++, oc++) {
			double ssres = (m.mat.col(cc)-U*E.asDiagonal()*V.row(oc).
					transpose()).squaredNorm();
			double sigmahat = ssres/(npoints-nreg);
			for(size_t comp=0; comp < ics.cols(); comp++) {
				double std_err = sqrt(sigmahat*Cinv[comp]);
				double beta = ics(oc, comp);
				double t = beta/std_err;
				out(oc, comp) = t;
			}
		}
	}
	return out;
}


/**
 * @brief Computes the SVD from XXt using the JacobiSVD
 *
 * @param A MatrixReorg object that can be used to load images on disk
 * @param varthresh stop after the eigenvalues reach this ratio of the maximum
 * @param cvarthresh stop after the sum of eigenvalues reaches this ratio of total
 * @param maxrank maximum rank to use
 * @param U Output U matrix, if null then ignored
 * @param V Output V matrix, if null then ignored
 *
 * @return Vector of singular values
 */
VectorXd covSVD(const MatrixReorg& A, double varthresh, double cvarthresh,
		size_t maxrank, MatrixXd* U, MatrixXd* V)
{
	if(!U && V)
		throw INVALID_ARGUMENT("U must be nonnull to compute V!");

	if(varthresh < 0 || varthresh > 1)
		varthresh = 0.1;
	if(cvarthresh < 0 || cvarthresh > 1)
		cvarthresh = 0.9;

	int rank = 0; // Rank of Singular Value Decomp

	// Create AA* matrix
	cerr<<"Computing AAt ("<<A.rows()<<" x "<<A.rows()<<")"<<endl;
	MatrixXd AAt(A.rows(), A.rows());
	AAt.setZero();
	for(size_t cc = 0; cc < A.ntall(); cc++) {
		cerr<<(cc+1)<<"/"<<A.ntall();
		MatMap m(A.tallMatName(cc), false);
		cerr<<" sz="<<m.mat.rows()<<"x"<<m.mat.cols()<<endl;
		AAt += m.mat*m.mat.transpose();
	}
	cerr<<"Done"<<endl;

	if(maxrank == 0) maxrank = AAt.rows();

	/*
	 * Perform SVD
	 */
	cerr<<"Computing Eigensolutions to AAT"<<endl;
	Eigen::SelfAdjointEigenSolver<MatrixXd> eig(AAt);
	const auto& evecs = eig.eigenvectors();
	const auto& evals = eig.eigenvalues();
	cerr<<"Done"<<endl;

	// Compute Rank
	cerr<<"Reducing rank"<<endl;
	double maxvar = sqrt(evals[AAt.rows()-1]);
	double totalvar = 0;
	double sum = 0;
	for(size_t ii=0; ii<maxrank; ii++) {
		if(evals[AAt.rows()-ii-1] > std::numeric_limits<double>::epsilon())
			totalvar += sqrt(evals[AAt.rows()-ii-1]);
	}

	for(size_t ii=0; ii<maxrank; ii++) {
		if(evals[AAt.rows()-ii-1] > std::numeric_limits<double>::epsilon()) {
			double v = sqrt(evals[AAt.rows()-ii-1]);
			if(sum < totalvar*cvarthresh && v > varthresh*maxvar)
				rank++;
			else
				break;
			sum += v;
		}
	}
	cerr<<"SVD Rank: "<<rank<<endl;
	if(rank <= 0)
		throw RUNTIME_ERROR("Error rank found to be zero!");

	VectorXd singvals(rank);
	for(size_t ii=0; ii<rank; ii++)
		singvals[ii] = sqrt(evals[AAt.rows()-ii-1]);

	if(U) {
		cerr<<"Computing U ("<<A.rows()<<"x"<<rank<<")"<<endl;
		U->resize(A.rows(), rank);
		for(size_t ii=0; ii<rank; ii++)
			U->col(ii) = evecs.col(AAt.rows()-ii-1);
		cerr<<"Done"<<endl;
	}

	// V = A^TUE^-1
	if(V) {
		cerr<<"Computing V ("<<A.cols()<<"x"<<rank<<")"<<endl;
		V->resize(A.cols(), rank);
		A.postMult(*V, *U, true);
		for(size_t cc=0; cc<rank; cc++)
			V->col(cc) /= singvals[cc];
		cerr<<"Done"<<endl;
	}

	cerr<<"Done"<<endl;
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

	ifstream ifs(info_name());
	ifs >> m_totalrows >> m_totalcols;
	if(m_verbose) {
		cerr << "Information File: " << info_name() << endl;
		cerr << "First Tall Matrix: " << tall_name(0) << endl;
		cerr << "First Mask:        " << mask_name(0) << endl;
		cerr << "Total Rows/Timepoints: " << m_totalrows<< endl;
		cerr << "Total Cols/Voxels:     " << m_totalcols << endl;
	}
	if(m_totalcols == 0 || m_totalrows == 0)
		throw RUNTIME_ERROR("Error zero size input from "+info_name());

	/* Read First Tall and Wide to get totalrows and totalcols */
	MatMap map;
	int cols = 0;
	for(size_t bb=0; cols < m_totalcols; bb++) {
		map.open(tall_name(bb), false);
		if(m_totalrows != map.rows())
			throw RUNTIME_ERROR("Error, mismatch in file size ("+
					to_string(map.rows())+"x"+to_string(map.cols())+" vs "+
					to_string(m_totalrows)+"x"+to_string(m_totalcols)+")");
		m_outcols.push_back(map.cols());
		cols += map.cols();
	}
	if(m_totalcols != cols)
		throw RUNTIME_ERROR("Error, mismatch in number of cols from input "
				"files");

	// check masks
	cols = 0;
	for(int ii=0; cols!=m_totalcols; ii++) {
		auto mask = readMRImage(mask_name(ii));
		for(FlatIter<int> mit(mask); !mit.eof(); ++mit) {
			if(*mit != 0)
				cols++;
		}

		// should exactly match up
		if(cols > m_totalcols)
			throw RUNTIME_ERROR("Error, mismatch in number of cols from input "
					"masks");
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

	if(m_verbose) {
		cerr << "Information File: " << info_name() << endl;
		cerr << "First Tall Matrix: " << tall_name(0) << endl;
		cerr << "First Mask:        " << mask_name(0) << endl;
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

		mask->write(mask_name(sb));
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
		cerr << "Row/Time  Blocks: " << timeblocks << endl;
		cerr << "Col/Space Blocks: " << spaceblocks << endl;
		cerr << "Total Rows/Timepoints: " << m_totalrows<< endl;
		cerr << "Total Cols/Voxels:     " << m_totalcols << endl;
	}
	ofstream ofs(info_name());
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
		if(m_verbose) cerr<<"Mask Group: "<<mask_name(sb)<<endl;
		auto mask = readMRImage(mask_name(sb));

		img_glob_row = 0;
		for(size_t tb = 0; tb<timeblocks; tb++) {
			if(m_verbose) cerr<<"Image: "<<filenames[sb*timeblocks+tb]<<endl;
			auto img = readMRImage(filenames[sb*timeblocks+tb]);
			if(m_verbose) cerr<<"Done Loading"<<endl;

			if(!img->matchingOrient(mask, false, true))
				throw INVALID_ARGUMENT("Mismatch in mask/image size in col:"+
						to_string(sb)+", row:"+to_string(tb));
			if(img->tlen() != inrows[tb])
				throw INVALID_ARGUMENT("Mismatch in time-length in col:"+
						to_string(sb)+", row:"+to_string(tb));

			double mean = 0, sd = 0;
			int cc, colbl;
			int64_t tlen = img->tlen();
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
						datamap.open(tall_name(colbl), true);
						if(m_verbose) cerr<<"Writing to: "<<tall_name(colbl)
								<<" at row "<<to_string(img_glob_row)<<endl;
						if(datamap.rows() != m_totalrows ||
								datamap.cols() != m_outcols[colbl]) {
							throw INVALID_ARGUMENT("Unexpected size in input "
									+ tall_name(colbl)+" expected "+
									to_string(m_totalrows)+"x"+
									to_string(m_outcols[colbl])+", found "+
									to_string(datamap.rows())+"x"+
									to_string(datamap.cols()));
						}
					}

					// Fill, Calculate Mean/SD and normalize
					mean = 0;
					sd = 0;
//					if(m_verbose) cerr<<"Loading ["<<img_glob_row<<"-"
//						<<(img_glob_row+tlen)<<","<<cc<<"]"<<endl;
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
		if(m_verbose) cerr<<"Mask Done: "<<mask_name(sb)<<endl;
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
			MatMap block(tallMatName(bb), false);

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
			MatMap block(tallMatName(bb), false);

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
			MatMap block(tallMatName(bb), false);

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
			MatMap block(tallMatName(bb), false);
			cerr<<"Multiplying "<<block.mat.rows()<<"x"<<block.mat.cols()<<" by "
				<<in.rows()<<"x"<<in.cols()<<endl;

			// Multiply By Input
			out.middleRows(cc, m_outcols[bb]) = block.mat.transpose()*in;
			cc += m_outcols[bb];
		}
	}
};

void gicaCreateMatrices(size_t tcat, size_t scat, vector<string> masks,
		vector<string> inputs, std::string prefix, double maxmem, bool normts,
		bool verbose)
{
	cerr << "Reorganizing data into matrices..."<<endl;
	size_t ndoubles = (size_t)(0.5*maxmem*(1<<27));
	MatrixReorg A(prefix, ndoubles, verbose);
	int status = A.createMats(tcat, scat, masks, inputs, normts);
	if(status != 0)
		throw RUNTIME_ERROR("Error while reorganizing data into 2D Matrices");
	cerr<<"Done Reorganizing"<<endl;
}

void gicaReduceFull(std::string inpref, std::string outpref, double varthresh,
		double cvarthresh, size_t maxrank, bool verbose)
{
	MatrixXd U, E, V;
	MatrixReorg A(inpref, -1, verbose);
	A.checkMats();

	cerr<<"Running XXt SVD"<<endl;
	cerr<<"A = "<<A.rows()<<" x "<<A.cols()<<endl;
	cerr<<"Varthresh = "<<varthresh<<endl;
	cerr<<"Cumulative Varthresh = "<<cvarthresh<<endl;
	E = covSVD(A, varthresh, cvarthresh, maxrank, &U, &V);
	cerr<<"Done"<<endl;
	MatMap writer;

	cerr<<"Writing "<<E_name(outpref)<<" ("<<E.rows()<<"x"<<E.cols()<<")"<<endl;
	writer.create(E_name(outpref), E.rows(), E.cols());
	writer.mat = E;

	cerr<<"Writing "<<U_name(outpref)<<" ("<<U.rows()<<"x"<<U.cols()<<")"<<endl;
	writer.create(U_name(outpref), U.rows(), U.cols());
	writer.mat = U;
	U.resize(0,0);

	cerr<<"Writing "<<V_name(outpref)<<" ("<<V.rows()<<"x"<<V.cols()<<")"<<endl;
	writer.create(V_name(outpref), V.rows(), V.cols());
	writer.mat = V;
	V.resize(0,0);
	writer.close();

	if(verbose)
		cerr<<"Singular Values:\n" << E.transpose()<<endl;
}

void gicaReduceProb(std::string inpref, std::string outpref, double varthresh,
		double cvarthresh, size_t rank, size_t poweriters, bool verbose)
{
	MatrixXd U, E, V;
	MatrixReorg A(inpref, -1, verbose);
	A.checkMats();

	cerr<<"Running On Disk SVD"<<endl;
	cerr<<"A = "<<A.rows()<<" x "<<A.cols()<<endl;
	cerr<<"Rank = "<<rank<<endl;
	cerr<<"PowerIters = "<<poweriters<<endl;
	cerr<<"Varthresh = "<<varthresh<<endl;
	cerr<<"Cumulative Varthresh = "<<cvarthresh<<endl;
	E = onDiskSVD(A, rank, poweriters, varthresh, cvarthresh, &U, &V);
	cerr<<"Done"<<endl;
	MatMap writer;

	cerr<<"Writing "<<E_name(outpref)<<" ("<<E.rows()<<"x"<<E.cols()<<")"<<endl;
	writer.create(E_name(outpref), E.rows(), E.cols());
	writer.mat = E;

	cerr<<"Writing "<<U_name(outpref)<<" ("<<U.rows()<<"x"<<U.cols()<<")"<<endl;
	writer.create(U_name(outpref), U.rows(), U.cols());
	writer.mat = U;
	U.resize(0,0);

	cerr<<"Writing "<<V_name(outpref)<<" ("<<V.rows()<<"x"<<V.cols()<<")"<<endl;
	writer.create(V_name(outpref), V.rows(), V.cols());
	writer.mat = V;
	V.resize(0,0);
	writer.close();

	if(verbose)
		cerr<<"Singular Values:\n" << E.transpose()<<endl;
}

void gicaTemporalICA(std::string reorgpref, std::string reducepref,
		std::string outpref, bool verbose)
{
	// load tall matrices
	MatrixReorg A(reorgpref, -1, verbose);
	A.checkMats();

	// Load U, V, E matrices
	MatrixXd U, V, W, ics;
	VectorXd E;
	{
		MatMap reader;
		reader.open(U_name(reducepref), false);
		U = reader.mat;
		reader.open(V_name(reducepref), false);
		V = reader.mat;
		reader.open(E_name(reducepref), false);
		E = reader.mat;
	}

	/*
	 * Compute ICA and explained variance
	 */
	size_t odim[4];
	cerr<<"Performing Temporal ICA ("<<U.rows()<<"x"<<U.cols()<<")"<<endl;

	W.resize(U.cols(), U.cols());
	ics = symICA(U, &W);
	MatMap writer;
	writer.create(tica_name(outpref), ics.rows(), ics.cols());
	writer.mat = ics;
	writer.create(tw_name(outpref), W.rows(), W.cols());
	writer.mat = W;
	cerr << "Done" << endl;

	// A = UEVt // UW = S // U = SWt // A = SWtEVt
	// B = WtEVt, BBt = WtEEW
	// X = S, XtX = StS
	// Note that B here could be used instead of regresing each ...
	// DVar = 2*BBt(c,:)*XtX(:,c)-XtX(c,c)*BB(c,c)-2*B(c,:)*YtX(:,c)
	VectorXd expvar(ics.cols());
	MatrixXd XtX = ics.transpose()*ics;
	MatrixXd BBt = W.transpose()*E.asDiagonal()*E.asDiagonal()*W;
	MatrixXd B = W.transpose()*E.asDiagonal()*V.transpose();
	MatrixXd YtX(A.cols(), ics.cols());
	A.postMult(YtX, ics, true);
	for(size_t c=0; c<ics.cols(); c++){
		expvar[c] = 2*BBt.row(c)*XtX.col(c)-XtX(c,c)*BBt(c,c)-
			2*B.row(c)*YtX.col(c);
	}

	// Create Sorted Lookup for explained variance
	vector<int> sorted(expvar.rows());
	for(size_t ii=0; ii<sorted.size(); ii++) sorted[ii] = ii;
	std::sort(sorted.begin(), sorted.end(),
			[&expvar](int i, int j) { return expvar[i] > expvar[j]; });

	cerr << "Explained Variance:"<<endl;
	for(size_t cc=0; cc<ics.cols(); cc++)
		cerr<<sorted[cc]<<" "<<expvar[sorted[cc]]<<endl;

	cerr << "Regressing Temporal Components" << endl;
	/*
	 * Initialize Regression Variables
	 */
	cerr<<"Setting up Regression"<<endl;
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
	size_t matn = 0; // matrix number (tall matrices)
	size_t maskn = 0; // Mask number

	MatMap tall(A.tallMatName(matn), false);
	MatrixXd tvalues(A.cols(), ics.cols());
	cerr<<"Regressing full dataset"<<endl;
	for(size_t cc=0, tc=0; cc < A.cols(); maskn++) {
		cerr<<"Subject Column: "<<maskn<< " Full Column: "<<cc
			<<" TallMat Column: "<<tc<<endl;
		auto mask = readMRImage(A.mask_name(maskn));
		for(size_t ii=0; ii<3; ii++)
			odim[ii] = mask->dim(ii);
		odim[3] = ics.cols();

		auto tmap = mask->copyCast(4, odim, FLOAT32);
		auto bmap = mask->copyCast(4, odim, FLOAT32);
		NDIter<int> mit(mask);
		Vector3DIter<double> tit(tmap), bit(bmap);

		// Iterate Through pixels of mask
		for(; !mit.eof(); ++mit, ++tit, ++bit) {
			if(*mit == 0) {
				for(size_t comp=0; comp<ics.cols(); comp++) {
					tit.set(comp, 0);
					bit.set(comp, 0);
				}
			} else {
				// Load the next block of columns as necessary
				if(tc >= tall.cols()) {
					tall.open(A.tallMatName(++matn));
					tc = 0;
				}

				// Perform regression
				regress(&result, tall.mat.col(tc), ics, Cinv, Xinv, distrib);
				for(size_t comp=0; comp<ics.cols(); comp++) {
					size_t trcomp = sorted[comp];
					tit.set(comp, result.t[trcomp]);
					bit.set(comp, result.bhat[trcomp]);
					tvalues(cc, comp) = result.t[trcomp];
				}
				tc++;
				cc++;
			}
		}
		// write output matching mask
		tmap->write(tmap_name(reducepref, maskn));
		bmap->write(bmap_name(reducepref, maskn));
	}

	// don't do this right now, its not working properly
//	computeProb(ics.cols(), tvalues);
}

void gicaSpatialICA(std::string reorgpref, std::string reducepref,
		std::string outpref, bool verbose)
{
	// load tall matrices
	MatrixReorg A(reorgpref, -1, verbose);
	A.checkMats();

	// Load U, V, E matrices
	MatrixXd U, V, W, ics;
	VectorXd E;
	{
		MatMap reader;
		reader.open(U_name(reducepref), false);
		U = reader.mat;
		reader.open(V_name(reducepref), false);
		V = reader.mat;
		reader.open(E_name(reducepref), false);
		E = reader.mat;
	}

	/*
	 * Compute ICA and explained variance
	 */
	cerr<<"Performing Spatial ICA ("<<V.rows()<<"x"<<V.cols()<<")"<<endl;
	W.resize(V.cols(), V.cols());
	ics = symICA(V, &W);
	MatMap writer;
	writer.create(sica_name(outpref), ics.rows(), ics.cols());
	writer.mat = ics;
	writer.create(sw_name(outpref), W.rows(), W.cols());
	writer.mat = W;
	cerr << "Done" << endl;

	// Convert each signal matrix into a t-score
	// A = UEVt, VW = X, WXt = Vt, A = UEWXt, A = UEWB
	cerr<<"Estimating Noise"<<endl;
	MatrixXd tvalues = icsToT(A, U, E, V, W, ics);

	// Explained variance
	// A = UEVt, VW = S, WSt = Vt, A = UEWSt
	// X = UEW, B = St
	// BBt = StS
	// XtX = WtEEW
	// 2*BBt(c,:)*XtX(:,c)-XtX(c,c)*BB(c,c)-2*B(c,:)*YtX(:,c)
	MatrixXd BBt = ics.transpose()*ics;
	MatrixXd XtX = W.transpose()*E.asDiagonal()*E.asDiagonal()*W;
	MatrixXd X = U*E.asDiagonal()*W;
	MatrixXd YtX(A.cols(), W.cols());
	A.postMult(YtX, X, true);
	VectorXd expvar(ics.cols());
	for(size_t cc=0; cc<ics.cols(); cc++)
		expvar[cc] = 2*BBt.row(cc)*XtX.col(cc) - XtX(cc,cc)*BBt(cc,cc) -
			2*ics.col(cc).transpose()*YtX.col(cc);

	// Create Sorted Lookup for explained variance
	vector<int> sorted(ics.cols());
	for(size_t ii=0; ii<sorted.size(); ii++) sorted[ii] = ii;
	std::sort(sorted.begin(), sorted.end(),
			[&expvar](int i, int j) { return expvar[i] > expvar[j]; });

	cerr << "Explained Variance:"<<endl;
	for(size_t cc=0; cc<ics.cols(); cc++)
		cerr<<sorted[cc]<<" "<<expvar[sorted[cc]]<<endl;

	/*
	 * Re-associate each column with a spatial signal (map)
	 * Columns of ics correspond to rows of the wide matrices/voxels
	 */
	cerr<<"Computing T-Maps"<<endl;
	size_t odim[4];
	for(size_t rr=0, maskn = 0; rr < ics.rows(); maskn++) {
		auto mask = readMRImage(A.mask_name(maskn));
		for(size_t ii=0; ii<3; ii++)
			odim[ii] = mask->dim(ii);
		odim[3] = ics.cols();

		auto tmap = mask->copyCast(4, odim, FLOAT32);
		auto bmap = mask->copyCast(4, odim, FLOAT32);
		NDIter<int> mit(mask);
		Vector3DIter<double> tit(tmap), bit(bmap);

		// Iterate Through pixels of mask
		for(; !mit.eof(); ++mit, ++tit, ++bit) {
			if(*mit == 0) {
				for(size_t comp=0; comp<ics.cols(); comp++) {
					tit.set(comp, 0);
					bit.set(comp, 0);
				}
			} else {
				// Iterate through Components
				for(size_t comp=0; comp<ics.cols(); comp++) {
					size_t trcomp = sorted[comp];
					double b = ics(rr, trcomp);
					double t = tvalues(rr, trcomp);
					tit.set(comp, t);
					bit.set(comp, b);
				}
				rr++;
			}
		}
		// write output matching mask
		tmap->write(tmap_name(outpref, maskn));
		bmap->write(bmap_name(outpref, maskn));
	}

// Probabilities are a bit a iffy at this point, better to do this later
//	computeProb(ics.cols(), tvalues);
	cerr<<"Done with Spatial ICA!"<<endl;
}
//
///**
// * @brief Converts T-Values to probabilities and z-scores. Note that the input
// * "tvalues" will actually be centered on the mode of the distribution.
// *
// * TODO specialized version constraining the mode to modes to be outside the mean
// *
// * @param ncomp
// * @param tvalues
// */
//void GICAfmri::computeProb(size_t ncomp, Ref<MatrixXd> tvalues)
//{
//	cerr<<"Fitting T-maps"<<endl;
//	MatrixXd mu(3, ncomp);
//	MatrixXd sd(3, ncomp);
//	VectorXd prior(3);
//
//	for(size_t comp=0; comp<ncomp; comp++) {
//		gaussGammaMixtureModel(tvalues.col(comp), mu.col(comp), sd.col(comp),
//				prior, tplot_name(comp));
//	}
//
//	// now convert the t-scores to p-scores
//	for(size_t cc=0, maskn=0; cc<m_A.cols(); cc++, maskn++) {
//		auto tmap = readMRImage(tmap_name(maskn));
//		auto mask = readMRImage(mask_name(maskn));
//		auto pmap = tmap->copy();
//		auto zmap = tmap->copy();
//		for(Vector3DIter<double> tit(tmap), pit(pmap), zit(zmap), mit(mask);
//					!tit.eof(); ++mit, ++tit, ++pit, ++zit) {
//			if(mit[0] == 0)
//				continue;
//			for(size_t comp=0; comp<ncomp; comp++) {
//				double z = (tit[comp]-mu(1,comp))/sd(1,comp);
//				double p = gaussianCDF(0, 1, z);
//				if(p > 0.5) p = 1-p;
//				p = prior[1]*p/2;
//				p = 1-p;
//				zit.set(comp, z);
//				pit.set(comp, p);
//			}
//			cc++;
//		}
//		zmap->write(zmap_name(maskn));
//		pmap->write(pmap_name(maskn));
//	}
//	cerr << "Done with Regression" << endl;
//}

/**
 * @brief Performs general linear model analysis on a 4D image.
 *
 * @param fmri Input 4D image.
 * @param Input X Regressors
 * @param bimg Output betas for each voxel (should have same
 * @param Timg
 * @param pimg
 */
void fmriGLM(ptr<const MRImage> fmri, const MatrixXd& X,
		ptr<MRImage> bimg, ptr<MRImage> Timg, ptr<MRImage> pimg)
{
	if(fmri->tlen() != fmri->dim(3))
		throw INVALID_ARGUMENT("fMRI must be 4D!");
	if(fmri->tlen() <= 2)
		throw INVALID_ARGUMENT("fMRI must have at least 2 timepoints!");
	if(fmri->tlen() != X.rows())
		throw INVALID_ARGUMENT("fMRI must have same number of timepoints as X "
				"has rows!");
	if(X.cols() < 1)
		throw INVALID_ARGUMENT("X (regressors) must have at least 1 column!");
	for(size_t ii=0; ii<3; ii++) {
		if(bimg->dim(ii) != fmri->dim(ii))
			throw INVALID_ARGUMENT("beta image must have matching size to fMRI!");
		if(pimg->dim(ii) != fmri->dim(ii))
			throw INVALID_ARGUMENT("p image must have matching size to fMRI!");
		if(Timg->dim(ii) != fmri->dim(ii))
			throw INVALID_ARGUMENT("T image must have matching size to fMRI!");
	}
	if(X.cols() != bimg->tlen())
		throw INVALID_ARGUMENT("X (regressors) must have at same number of "
				"columns as beta image has vector elements (volumes)!");
	if(X.cols() != Timg->tlen())
		throw INVALID_ARGUMENT("X (regressors) must have at same number of "
				"columns as T image has vector elements (volumes)!");
	if(X.cols() != pimg->tlen())
		throw INVALID_ARGUMENT("X (regressors) must have at same number of "
				"columns as p image has vector elements (volumes)!");
	if(!fmri->matchingOrient(bimg, false, false))
		throw INVALID_ARGUMENT("beta image must have matching orientation to fMRI!");
	if(!fmri->matchingOrient(pimg, false, false))
		throw INVALID_ARGUMENT("p image must have matching orientation to fMRI!");
	if(!fmri->matchingOrient(Timg, false, false))
		throw INVALID_ARGUMENT("T image must have matching orientation to fMRI!");

	size_t tlen = fmri->tlen();
	vector<size_t> osize(4);
	for(size_t ii=0; ii<3; ii++)
		osize[ii] = fmri->dim(ii);
	osize[3] = X.cols();

    // Cache Reused Vectors
    MatrixXd Xinv = pseudoInverse(X);
    VectorXd covInv = pseudoInverse(X.transpose()*X).diagonal();

    const double MAX_T = 100;
    const double STEP_T = 0.1;
    StudentsT student_cdf(X.rows()-1, STEP_T, MAX_T);

    // regress each timesereies
    Vector3DConstIter<double> it(fmri);
    Eigen::VectorXd y(tlen);
    vector<int64_t> ind(3);
	RegrResult ret;
    for(Vector3DIter<double> bit(bimg), tit(Timg), pit(pimg); !it.eof();
			++tit, ++it, ++bit, ++pit) {

        // copy to signal
        for(size_t tt=0; tt<tlen; ++tt)
            y[tt] = it[tt];

		// perform regression
		regress(&ret, y, X, covInv, Xinv, student_cdf);

		for(size_t ii=0; ii<X.cols(); ii++) {
			bit.set(ii, ret.bhat[ii]);
			tit.set(ii, ret.t[ii]);
			pit.set(ii, ret.p[ii]);
		}
    }
}

/**
 * @brief Lowpass smooth transition.
 *
 * \frac{G_0^2}{1+(\frac{\omega}{\omega_c})^{2n}}
 * G_0 is the zero frequency gain
 * \omega_c is the center frequency
 *
 * \frac{1}{1+(\frac{\omega}{\omega_c})^{2n}}
 *
 * @param middle
 * @param w the order of the butterworth filter
 *
 * @return
 */
double lp_transition(double f, double cutoff, double cuton, int w)
{
	(void)cuton;
	return 1.0/(1.0+pow(f/cutoff, 2*w));
}

/**
 * @brief High pass smooth transition.
 *
 * @param middle
 * @param width
 *
 * @return
 */
double hp_transition(double f, double cutoff, double cuton, int w)
{
	(void)cutoff;
	return 1-1.0/(1.0+pow(-f/cuton, 2*w));
}

/**
 * @brief Band pass smooth transition.
 *
 * @param middle
 * @param width
 *
 * @return
 */
double bp_transition(double f, double cutoff, double cuton, int w)
{
	return lp_transition(f, cutoff, cuton, w) +
		hp_transition(f, cutoff, cuton, w) - 1;
}

/**
 * @brief Band Stop smooth transition.
 *
 * @param middle
 * @param width
 *
 * @return
 */
double bs_transition(double f, double cutoff , double cuton, int w)
{
	return lp_transition(f, cutoff, cuton, w) +
		hp_transition(f, cutoff, cuton, w);
}

/**
 * @brief Takes the FFT of each line of the image, performs bandpass filtering
 * on the line and then invert FFTs and writes back to the input image.
 *
 * @param inimg Input image
 * @param cuton Minimum frequency (may be 0)
 * @param cutoff Maximum frequency in band (may be INFINITY)
 */
void fmriBandPass(ptr<MRImage> inimg, double cuton, double cutoff)
{
	const int BUTTERWORTH_ORDER = 2;

	if(inimg->ndim() != 4)
		throw INVALID_ARGUMENT("Input image to timeFilter is not 4D!");

	size_t tlen = inimg->tlen();
	double Fs = 1./inimg->spacing(3);

	int psize = round2(inimg->tlen()); // padded data size

	auto rbuffer = (double*)fftw_malloc(sizeof(double)*psize);
	auto ibuffer = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*psize);

	fftw_plan fwd = fftw_plan_dft_r2c_1d(psize, rbuffer, ibuffer, FFTW_MEASURE);
	fftw_plan rev = fftw_plan_dft_c2r_1d(psize, ibuffer, rbuffer, FFTW_MEASURE);

	double(*smoothfunc)(double, double, double, int) = NULL;
	bool pos_valid = !(cuton < 0 || std::isnan(cuton) || std::isinf(cuton));
	bool neg_valid = !(cutoff < 0 || std::isnan(cutoff) || std::isinf(cutoff));

	if(!pos_valid && !neg_valid) {
		throw INVALID_ARGUMENT("Error there appears to be invalid negative frequency "
			"(low-pass) transitions and invalid positve (high-pass)");
	}

	//high pass filter
	if(pos_valid && !neg_valid) {
		cerr << "High Pass Filtering, Min Freq: " << cuton << endl;
		smoothfunc = hp_transition;
	//low pass filter
	} else if(!pos_valid && neg_valid) {
		cerr << "Low Pass Filtering, Max Freq: " << cutoff << endl;
		smoothfunc = lp_transition;
	//band stop
	} else if(cutoff < cuton) {
		cerr << "Band Stop Filtering: [" << cutoff << ", " << cuton
			<< "]" << endl;
		smoothfunc = bs_transition;
	//band pass
	} else if(cuton < cutoff) {
		cerr << "Band Pass Filtering: [" << cuton << ", " << cutoff
			<< "]" << endl;
		smoothfunc = bp_transition;
	}

	cerr << "Frequencies in FFT: " << 0 << " - " << Fs/2.0
		<< "hz" << endl;

	//Create the 1-D filter
	double T = (double)psize*inimg->spacing(3);
	for(Vector3DIter<double> it(inimg); !it.eof(); ++it) {
		// Copy from input
		for(size_t tt = 0 ; tt < tlen; tt++)
			rbuffer[tt] = it[tt];

		// Zero Pad
		for(size_t tt = tlen; tt < psize; tt++)
			rbuffer[tt] = 0;

		// fourier transform
		fftw_execute(fwd);

		// Positive Frequencies Only
		for(size_t ii=0; ii<psize/2+1; ii++) {
			double ff = (double)ii/T;
			cdouble_t tmp(ibuffer[ii][0], ibuffer[ii][1]);
			tmp *= smoothfunc(ff, cutoff, cuton, BUTTERWORTH_ORDER)/psize;
			ibuffer[ii][0] = tmp.real();
			ibuffer[ii][1] = tmp.imag();
		}

		// inverse fourier transform
		fftw_execute(rev);

		// Copy Back Out
		for(size_t tt = 0 ; tt < tlen; tt++)
			it.set(tt, rbuffer[tt]);
	}
};

/**
 * @brief Regresses out the given variables, creating time series which are
 * uncorrelated with X
 *
 * @param inout Input 4D image
 * @param X matrix of covariates
 *
 */
ptr<MRImage> regressOut(ptr<const MRImage> inimg, const MatrixXd& X)
{
	// Add Augmented version with intercept
	MatrixXd Xaug(X.rows(), X.cols()+1);
	Xaug.leftCols(X.cols()) = X;
	Xaug.rightCols(1).setOnes();

	const double DELTA = 1e-20;
	size_t tlen = inimg->tlen();
	if(tlen != Xaug.rows()) {
		throw INVALID_ARGUMENT("Error, input design matrix must have matching "
				"number of rows to input fMRI");
	}
	MatrixXd cinv = pseudoInverse(X.transpose()*X);
	MatrixXd prebeta = cinv*X.transpose();
	VectorXd y(X.rows());
	VectorXd beta(X.cols());

	//allocate output
	auto out = dPtrCast<MRImage>(inimg->copyCast(FLOAT32));

	// Create Iterators
	Vector3DIter<double> oit(out);
	Vector3DConstIter<double> iit(inimg);

	// Iterate through and regress each line
	for(iit.goBegin(), oit.goBegin(); !iit.eof(); ++iit, ++oit) {

		// statistics
		double minv = std::numeric_limits<double>::max();
		double maxv = std::numeric_limits<double>::min();

		for(size_t tt=0; tt<tlen; ++tt) {
			minv = std::min(minv, iit[tt]);
			maxv = std::max(maxv, iit[tt]);
			y[tt] = iit[tt];
		}

		// b = (XtX)^-1Xty
		// X*((XtX)^-1Xty)
		// y
		if(fabs(maxv - minv) > DELTA) {
			beta = prebeta*y;
			y -= X*beta;
		} else
			y.setZero();

		for(size_t tt=0; tt<tlen; ++tt) {
			oit.set(tt, y[tt]);
		}
	}

	return out;
}

/**
 * @brief Creates a matrix of timeseries, then perfrorms principal components
 * analysis on it to reduce the number of timeseries to outsz. Each unique
 * non-zero label in the input image will be considered a group of measurements
 * which will be reduced together. Thus if there are labels 0,1,2 there will be
 * 2 columns in the output.
 *
 * Note that labelmap and fmri should be in the same pixel space (except for
 * dimension 3)
 *
 * @param fmri 		FMRI image with timeseres to extract
 * @param labelmap	Labelmap used to identify relevent input timeseries
 *
 * @return Matrix of time-series which were reduced, 1 column per
 * label group, since each label group gets reduced to an average
 */
MatrixXd extractLabelAVG(ptr<const MRImage> fmri,
		ptr<const MRImage> labelmap)
{
	// Check Inputs
	if(!fmri->matchingOrient(labelmap, false, true))
		throw INVALID_ARGUMENT("Input image orientations do not match!");

	size_t tlen = fmri->tlen();
	if(tlen <= 1)
		throw INVALID_ARGUMENT("Input image not 4D!");

	// Create output regressors (all non-zero labels)
	MatrixXd X;
	auto labels = getLabels(labelmap);
	if(labels.count(0) > 0)
		X.resize(tlen, labels.size()-1);
	else
		X.resize(tlen, labels.size());
	X.setZero();

	// Create Map of labels->columns
	std::map<int64_t, int64_t> labelToCol;
	{
	auto it = labels.begin();
	for(size_t ii=0; it != labels.end(); ++it) {
		if(*it != 0)
			labelToCol[*it] = ii++;
	}
	}

	// Create Local Variables
	Vector3DConstIter<double> it(fmri);
	NDConstIter<int> lit(labelmap);
	for(it.goBegin(), lit.goBegin(); !lit.eof(); ++lit, ++it) {
		if(*lit != 0) {
			size_t col = labelToCol[*lit];
			for(size_t tt=0; tt<tlen; tt++)
				X(tt, col) += it[tt];
		}
	}

	X /= tlen;
	return X;
}

/**
 * @brief Creates a matrix of timeseries, then perfrorms principal components
 * analysis on it to reduce the number of timeseries to outsz. Each unique
 * non-zero label in the input image will be considered a group of measurements
 * which will be reduced together. Thus if there are labels 0,1,2 there will be
 * 2*outsz columns in the output.
 *
 * Note that labelmap and fmri should be in the same pixel space (except for
 * dimension 3)
 *
 * @param fmri 		FMRI image with timeseres to extract
 * @param labelmap	Labelmap used to identify relevent input timeseries
 * @param outsz		Number of output timeseres ti append to design
 *
 * @return Matrix of time-series which were reduced, 1 block of outsz for each
 * label group, since each label group gets reduced to outsz leading components
 */
MatrixXd extractLabelPCA(ptr<const MRImage> fmri,
		ptr<const MRImage> labelmap, size_t outsz)
{
	// Check Inputs
	if(!fmri->matchingOrient(labelmap, false, true))
		throw INVALID_ARGUMENT("Input image orientations do not match!");

	int tlen = fmri->tlen();
	if(tlen <= 1)
		throw INVALID_ARGUMENT("Input image is not 4D!");

	// Need to create a block matrix where each block corresponds to 1 label
	// group. To do this, the number of members of each group is computed and
	// then a map to the first element of each column block is calculated

	// Create Map of labels->size
	std::map<int64_t, int64_t> labelToCol;
	for(FlatConstIter<int64_t> it(labelmap); !it.eof(); ++it) {
		if(*it != 0) {
			auto ret = labelToCol.insert({*it,0});
			ret.first->second++;
		}
	}

	cerr << "Counts: " <<endl;
	for(auto& p : labelToCol)
		cerr<<p.first<<"->"<<p.second<<endl;
	cerr<<endl;

	size_t cumu = 0;
	for(auto& p : labelToCol) {
		size_t prev = cumu;
		cumu += p.second;
		p.second = prev;
	}

	cerr << "Offsets: " <<endl;
	for(auto& p : labelToCol)
		cerr<<p.first<<"->"<<p.second<<endl;
	cerr<<endl;

	cerr<<"Total cols: " <<cumu<<endl;
	MatrixXd X(tlen, cumu);

	// Create Local Variables
	Vector3DConstIter<double> it(fmri);
	NDConstIter<int> lit(labelmap);

	// Fill Matrix with X, note that each time we find a particular label, we
	// use the current value in labelToCol then incremente it, so that next
	// visitation to the label produces the next volumn, and so on
	double mean, stddev;
	for(it.goBegin(), lit.goBegin(); !lit.eof(); ++lit, ++it) {
		if(*lit != 0) {
			mean = 0;
			stddev = 0;
			size_t col = labelToCol[*lit];
			labelToCol[*lit]++;

			// Compute Mean and Fill Column of X
			for(int tt = 0 ; tt<tlen; tt++) {
				X(tt, col) = it[tt];
				mean += it[tt];
				stddev += it[tt]*it[tt];
			}
			stddev = sqrt(sample_var(tlen, mean, stddev));
			mean /= tlen;

			//skip invalid timeseries (left as zeros)
			if(std::isnan(mean) || std::isinf(mean) || stddev == 0)
				X.col(col).setZero();
			else
				X.col(col) = (X.col(col).array()-mean)/stddev;
		}
	}

	std::cout << "PCA" << endl;
	X = rpiPCA(X, 1, outsz);
	std::cout << "Done" << endl;

	return X;
}

/**
 * @brief Creates a matrix of timeseries, then perfrorms independent components
 * analysis on it to reduce the number of timeseries to outsz. Each unique
 * non-zero label in the input image will be considered a group of measurements
 * which will be reduced together. Thus if there are labels 0,1,2 there will be
 * 2*outsz columns in the output.
 *
 * Note that labelmap and fmri should be in the same pixel space (except for
 * dimension 3)
 *
 * @param fmri 		FMRI image with timeseres to extract
 * @param labelmap	Labelmap used to identify relevent input timeseries
 * @param outsz		Number of output timeseres ti append to design
 *
 * @return Matrix of time-series which were reduced, 1 block of outsz for each
 * label group, since each label group gets reduced to outsz leading components
 */
MatrixXd extractLabelICA(ptr<const MRImage> fmri,
		ptr<const MRImage> labelmap, size_t outsz)
{
	// Check Inputs
	if(!fmri->matchingOrient(labelmap, false, true))
		throw INVALID_ARGUMENT("Input image orientations do not match!");

	int tlen = fmri->tlen();
	if(tlen <= 1)
		throw INVALID_ARGUMENT("Input image is not 4D!");

	// Need to create a block matrix where each block corresponds to 1 label
	// group. To do this, the number of members of each group is computed and
	// then a map to the first element of each column block is calculated

	// Create Map of labels->size
	std::map<int64_t, int64_t> labelToCol;
	size_t tc = 0;
	for(FlatConstIter<int64_t> it(labelmap); !it.eof(); ++it) {
		if(*it != 0) {
			auto ret = labelToCol.insert({*it,0});
			ret.first->second++;
			tc++;
		}
	}

	cerr << "Counts: " <<endl;
	for(auto& p : labelToCol)
		cerr<<p.first<<"->"<<p.second<<endl;
	cerr<<endl;

	size_t cumu = 0;
	for(auto& p : labelToCol) {
		size_t prev = cumu;
		cumu += p.second;
		p.second = prev;
	}

	cerr << "Offsets: " <<endl;
	for(auto& p : labelToCol)
		cerr<<p.first<<"->"<<p.second<<endl;
	cerr<<endl;

	cerr<<"Total cols: " <<cumu<<endl;
	MatrixXd X(tlen, cumu);

	// Create Local Variables
	Vector3DConstIter<double> it(fmri);
	NDConstIter<int> lit(labelmap);

	// Fill Matrix with X, note that each time we find a particular label, we
	// use the current value in labelToCol then incremente it, so that next
	// visitation to the label produces the next volumn, and so on
	double mean, stddev;
	for(it.goBegin(), lit.goBegin(); !lit.eof(); ++lit, ++it) {
		if(*lit != 0) {
			tc--;
			mean = 0;
			stddev = 0;
			size_t col = labelToCol[*lit];
			labelToCol[*lit]++;

			// Compute Mean and Fill Column of X
			for(int tt = 0 ; tt<tlen; tt++) {
				X(tt, col) = it[tt];
				mean += it[tt];
				stddev += it[tt]*it[tt];
			}
			stddev = sqrt(sample_var(tlen, mean, stddev));
			mean /= tlen;

			//skip invalid timeseries (left as zeros)
			if(std::isnan(mean) || std::isinf(mean) || stddev == 0)
				X.col(col).setZero();
			else
				X.col(col) = (X.col(col).array()-mean)/stddev;
		}
	}

	std::cout << "PCA" << endl;
	X = rpiPCA(X, 1, outsz);
	std::cout << "Done" << endl;

	std::cout << "ICA...";
	X = symICA(X);
	std::cout << "Done" << endl;

	return X;
}

void openRWBusError(int sig, siginfo_t* inf, void*)
{
	ostringstream oss;
	oss<<"Signal"<<sig<<" caught during rw-open at "<<inf->si_addr<<endl;
	throw RUNTIME_ERROR(oss.str());
}

void openROBusError(int sig, siginfo_t* inf, void*)
{
	ostringstream oss;
	oss<<"Signal"<<sig<<" caught during ro-open at "<<inf->si_addr<<endl;
	throw RUNTIME_ERROR(oss.str());
}

void createBusError(int sig, siginfo_t* inf, void*)
{
	ostringstream oss;
	oss<<"Signal"<<sig<<" caught during create at "<<inf->si_addr<<endl;
	throw RUNTIME_ERROR(oss.str());
}

void externalBusError(int sig, siginfo_t* inf, void*)
{
	ostringstream oss;
	oss<<"Signal"<<sig<<" caught outside at "<<inf->si_addr<<endl;
	throw RUNTIME_ERROR(oss.str());
}


/**
 * @brief Open an existing file as a memory map. The file must already
 * exist. If writeable is true then the file will be opened for reading
 * and writing, by default write is off. Note the same file should not
 * be simultaneously written two by two separate processes.
 *
 * @param filename File to open
 * @param writeable whether writing is allowed
 */
void MatMap::open(std::string filename, bool writeable)
{
//	struct sigaction act;
//
//	if(writeable) act.sa_sigaction = openRWBusError;
//	else act.sa_sigaction = openROBusError;
//	sigaction(SIGBUS, &act, NULL);
//	sigaction(SIGSEGV, &act, NULL);
//	sigaction(SIGABRT, &act, NULL);
	if(datamap.openExisting(filename, writeable, false) < 0)
		throw INVALID_ARGUMENT("Error opening "+filename+" as "+
				(writeable ? "rw-":"ro-")+"memmap");

	size_t* nrowsptr = (size_t*)datamap.data();
	size_t* ncolsptr = nrowsptr+1;
	double* dataptr = (double*)((size_t*)datamap.data()+2);

	if(datamap.size() < 2*sizeof(size_t))
		throw INVALID_ARGUMENT("Error! Mapped Dataset is less than the "
				"size of two size_t's");
	if(datamap.size() != (*nrowsptr)*(*ncolsptr)*sizeof(double)+
			2*sizeof(size_t))
		throw INVALID_ARGUMENT("Error! Mismatch in map size ("+
				std::to_string(datamap.size())+") and size in file ("+
				std::to_string(*nrowsptr)+", "+std::to_string(*ncolsptr)+")");

	m_rows = *nrowsptr;
	m_cols = *ncolsptr;
	new (&this->mat) Eigen::Map<MatrixXd>(dataptr, m_rows, m_cols);

//	act.sa_sigaction = externalBusError;
//	sigaction(SIGBUS, &act, NULL);
//	sigaction(SIGSEGV, &act, NULL);
//	sigaction(SIGABRT, &act, NULL);
};

/**
 * @brief Open an new file as a memory map. ANY OLD FILE WILL BE DELETED
 * The file is always opened for writing and reading. Note the same file
 * should not be simultaneously written two by two separate processes.
 *
 * @param filename File to open
 * @param rows Number of rows in matrix file
 * @param cols number of columns in matrix file
 */
void MatMap::create(std::string filename, size_t newrows, size_t newcols)
{
//	struct sigaction act;
//	act.sa_sigaction = createBusError;
//	sigaction(SIGBUS, &act, NULL);
//	sigaction(SIGSEGV, &act, NULL);
//	sigaction(SIGABRT, &act, NULL);
//
	m_rows = newrows;
	m_cols = newcols;
	if(datamap.openNew(filename, 2*sizeof(size_t)+
				m_rows*m_cols*sizeof(double)) < 0)
		throw INVALID_ARGUMENT("Error creating "+filename+" as rw-memmap");

	size_t* nrowsptr = (size_t*)datamap.data();
	size_t* ncolsptr = nrowsptr+1;
	double* dataptr = (double*)((size_t*)datamap.data()+2);

	*nrowsptr = m_rows;
	*ncolsptr = m_cols;
	new (&this->mat) Eigen::Map<MatrixXd>(dataptr, m_rows, m_cols);

//	act.sa_sigaction = externalBusError;
//	sigaction(SIGBUS, &act, NULL);
//	sigaction(SIGSEGV, &act, NULL);
//	sigaction(SIGABRT, &act, NULL);
};


} // NPL

