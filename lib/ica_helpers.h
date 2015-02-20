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
 * @file ica_helpers.h Tools for performing ICA, including rewriting images as
 * matrices.
 *
 *****************************************************************************/
#ifndef ICA_HELPERS_H
#define ICA_HELPERS_H

#include <Eigen/Dense>
#include <Eigen/IterativeSolvers>
#include <string>
#include <vector>

#include "npltypes.h"
#include "mrimage.h"
#include "utility.h"
#include "macros.h"

namespace npl {

class MatrixReorg;

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
 * @param A MatrixReorg object that can be used to load images on disk
 * @param rank Number of output columns (output rank)
 * @param poweriters Number of power iterations to perform. 2 passes over A are
 * required for each iteration, but iteration drives error to 0 at an
 * exponential rate
 * @param U Output U matrix, if null then ignored
 * @param V Output V matrix, if null then ignored
 *
 * @return Vector of singular values
 */
VectorXd onDiskSVD(const MatrixReorg& A,
		int rank, size_t poweriters, double varthresh,
		MatrixXd* U=NULL, MatrixXd* V=NULL);

/**
 * @brief Computes the SVD from XXt using the JacobiSVD
 *
 * @param A MatrixReorg object that can be used to load images on disk
 * @param U Output U matrix, if null then ignored
 * @param V Output V matrix, if null then ignored
 *
 * @return Vector of singular values
 */
VectorXd covSVD(const MatrixReorg& A, double varthresh, MatrixXd* U, MatrixXd* V);

class MatMap
{
public:
	MatMap() : mat(NULL, 0, 0)
	{
	};

	MatMap(std::string filename) : mat(NULL, 0, 0)
	{
		open(filename);
	};

	int open(std::string filename)
	{
		if(datamap.openExisting(filename, true) < 0)
			return -1;

		size_t* nrowsptr = (size_t*)datamap.data();
		size_t* ncolsptr = nrowsptr+1;
		double* dataptr = (double*)((size_t*)datamap.data()+2);

		m_rows = *nrowsptr;
		m_cols = *ncolsptr;
		new (&this->mat) Eigen::Map<MatrixXd>(dataptr, m_rows, m_cols);
		return 0;
	};

	int create(std::string filename, size_t newrows, size_t newcols)
	{
		m_rows = newrows;
		m_cols = newcols;
		if(datamap.openNew(filename, 2*sizeof(size_t)+
					m_rows*m_cols*sizeof(double)) < 0)
			return -1;

		size_t* nrowsptr = (size_t*)datamap.data();
		size_t* ncolsptr = nrowsptr+1;
		double* dataptr = (double*)((size_t*)datamap.data()+2);

		*nrowsptr = m_rows;
		*ncolsptr = m_cols;
		new (&this->mat) Eigen::Map<MatrixXd>(dataptr, m_rows, m_cols);
		return 0;
	};

	void close()
	{
		datamap.close();
	};

	bool isopen()
	{
		return datamap.isopen();
	};

	const size_t& rows() const { return m_rows; };
	const size_t& cols() const { return m_cols; };

	Eigen::Map<MatrixXd> mat;
private:
	MemMap datamap;
	size_t m_rows;
	size_t m_cols;
};

/**
 * @brief Reorganizes input images into tall and wide matrices (matrices that
 * span the total rows and cols, respectively).
 *
 * Only use tall mats, I think I might drop wide mats eventually
 * TODO allow for JUST tall or JUST wide construction
 */
class MatrixReorg
{
public:

	/**
	 * @brief Constructor
	 *
	 * @param prefix Prefix for matrix files to be written
	 * PREFIX_tall_[0-9]* and  * PREFIX_wide_[0-9]*
	 * @param maxd Maximum number of doubles to include in a block, this
	 * should be sized to fit into memory
	 * @param verbose Print information
	 */
	MatrixReorg(std::string prefix="", size_t maxd=(1<<30), bool verbose=true);

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
	 * @param normts whether to normalize the timeseries
	 *
	 * @return 0 if succesful, -1 if read failure, -2 if write failure
	 */
	int createMats(size_t timeblocks, size_t spaceblocks,
			const std::vector<std::string>& masknames,
			const std::vector<std::string>& filenames, bool normts = true);

	/**
	 * @brief Loads existing matrices by first reading ${prefix}_tall_0,
	 * ${prefix}_wide_0, and ${prefix}_mask_*, and checking that all the dimensions
	 * can be made to match (by loading the appropriate number of matrices/masks).
	 *
	 * @return 0 if succesful, -1 if read failure, -2 if write failure
	 */
	int checkMats();

//	inline int nwide() const { return m_outrows.size(); };
	inline int ntall() const { return m_outcols.size(); };

	inline const vector<int>& tallMatCols() const
	{
		return m_outcols;
	};

	inline int tallMatRows() const
	{
		return m_totalrows;
	};

//	inline const vector<int>& wideMatRows() const
//	{
//		return m_outrows;
//	};
//
//	inline int wideMatCols() const
//	{
//		return m_totalcols;
//	};
//
	inline std::string tallMatName(size_t ii) const
	{
		return m_prefix+"_tall_"+std::to_string(ii);
	};

//	inline std::string wideMatName(size_t ii) const
//	{
//		return m_prefix+"_wide_"+std::to_string(ii);
//	};
//
	inline std::string inColMaskName(size_t ii) const
	{
		return m_prefix+"_mask_"+std::to_string(ii)+".nii.gz";
	};

	void preMult(Eigen::Ref<MatrixXd> out, const Eigen::Ref<const MatrixXd> in,
			bool transpose = false) const;

	void postMult(Eigen::Ref<MatrixXd> out, const Eigen::Ref<const MatrixXd> in,
			bool transpose = false) const;

	inline int rows() const { return m_totalrows; };
	inline int cols() const { return m_totalcols; };

	private:
	int m_totalrows;
	int m_totalcols;
	std::string m_prefix;
	bool m_verbose;
//	vector<int> m_outrows;
	vector<int> m_outcols;
	size_t m_maxdoubles;
};

/**
 * @brief Perform Group ICA on multiple fMRI Images
 */
class GICAfmri
{
public:
	GICAfmri(std::string pref) ;

	/**
	 * @brief Cutoff for explained variance in PCA
	 */
	double varthresh;

	/**
	 * @brief Minimum rank to estimate the full matrix with
	 */
	int minrank;

	/**
	 * @brief Number of power iterations, usually 2-3 are sufficient
	 */
	size_t poweriters;

	/**
	 * @brief Maximum number of gigabytes of memory to use
	 */
	double maxmem;

	/**
	 * @brief Print more information
	 */
	int verbose;

	/**
	 * @brief Whether to perform spatial ICA (rather than temporal)
	 */
	bool spatial;

	/**
	 * @brief Do not use the randomized method of matrix reduction and
	 * instead compute the full svd, then find the non-zero singular values
	 * from that. This is much slower but is useful for testing
	 */
	bool fullsvd;

	/**
	 * @brief Normalize individual subjects' time-series
	 */
	bool normts;

	/**
	 * @brief Minimum frequency to keep in single fMRI, subject timeseries will
	 * be bandpass filtered with this cuton frequency
	 */
	double minfreq;

	/**
	 * @brief Maximum frequency to keep in single fMRI, subject timeseries will
	 * be bandpass filtered with this cuton frequency
	 */
	double maxfreq;

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
	void compute(size_t tcat, size_t scat, std::vector<std::string> masks,
			vector<std::string> inputs);

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
	void compute(std::string mask, std::string input)
	{
		std::vector<std::string> tmask;
		if(!mask.empty())
			tmask.push_back(mask);
		std::vector<std::string> tinput(1, input);
		compute(1, 1, tmask, tinput);
	}

private:
	std::string m_pref;

	MatrixXd m_U, m_V;
	VectorXd m_E;

	MatrixReorg m_A;

	size_t svd_help(std::string inname, std::string usname,
			std::string vname);

	void computeSpatialICA();
	void computeTemporaICA();
	void createMatrices(size_t tcat, size_t scat, vector<std::string> masks,
			vector<std::string> inputs);
	void computeProb(size_t ncomp, Ref<MatrixXd> tvalues);

};

}

#endif //ICA_HELPERS_H
