/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file fmri_inference.h Tools for performing ICA, including rewriting images as
 * matrices. This file should really be fmri inference helpers or something
 * along that line, because it also include GLM helpers.
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
 * @param varthresh stop after the eigenvalues reach this ratio of the maximum
 * @param cvarthresh stop after the sum of eigenvalues reaches this ratio of total
 * @param U Output U matrix, if null then ignored
 * @param V Output V matrix, if null then ignored
 *
 * @return Vector of singular values
 */
VectorXd onDiskSVD(const MatrixReorg& A,
		int rank, size_t poweriters, double varthresh, double cvarthresh,
		MatrixXd* U=NULL, MatrixXd* V=NULL);

/**
 * @brief Create on disk matrices based on an array of input images. The array
 * will be concatinated in time tcat time and space scat time, with time as the
 * faster dimension.
 *
 * @param tcat Number of images to concatinate in time (faster moving in inputs)
 * @param scat Number of images to concatinate in space (slower moving in inputs)
 * @param masks Array of all image masks, 1 per spatial concatination
 * @param inputs Input files, should be tcat*scat in size
 * @param prefix Prefix of output matrices. Will create prefix_tall_0,
 * prefix_tall_1 ...
 * @param maxmem Maximum memory per individual tall file, this is used to limit
 * the loaded memory at a time. There will be more file overhead for smaller
 * memory (though not much)
 * @param normts Whether to normalize the timeseries, within each input image
 * before  writing to flat matrices
 * @param verbose Print more information during reorganization
 */
void gicaCreateMatrices(size_t tcat, size_t scat, vector<std::string> masks,
		vector<std::string> inputs, std::string prefix, double maxmem, bool normts,
		bool verbose);

/**
 * @brief Compute PCA for the given group, defined
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
 * @param inpref input prefix of reorganized data matrices
 * @param outpref output prefix for U,E,V matrices
 * @param varthresh threshold for singular values, everything smaller than
 * this ratio of the leading singular value will be treated as zero
 * @param cvarthresh threshold for singular values, everything after this
 * ratio of the total sum of singular values will be treated as zero
 * @param maxrank maximum rank to use in reduction
 * @param verbose whether to print debuging information
 */
void gicaReduceFull(std::string inpref, std::string outpref, double varthresh,
		double cvarthresh, size_t maxrank, bool verbose);

/**
 * @brief Compute PCA for the given group, defined
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
 * @param inpref input prefix of reorganized data matrices
 * @param outpref output prefix for U,E,V matrices
 * @param rank number of rows to estimate full matrix
 * @param poweriters number of power iterations to perform do better
 * distringuish between close singular values 2-3 are probably sufficient for
 * most cases
 * @param varthresh threshold for singular values, everything smaller than
 * this ratio of the leading singular value will be treated as zero
 * @param cvarthresh threshold for singular values, everything after this
 * ratio of the total sum of singular values will be treated as zero
 * @param verbose whether to print debuging information
 */
void gicaReduceProb(std::string inpref, std::string outpref, double varthresh,
		double cvarthresh, size_t rank, size_t poweriters, bool verbose);

/**
 * @brief Perform ICA with each dimension as a separate timeseries. This is
 * essentially unmixing in space and producing independent timeseries.
 *
 * @param reorgpref Prefix of tall input file *_tall_[0-9]*
 * @param reducepref Prefix input files *Umat, *Vmat *Evec files
 * @param outpref Output file prefix *_SpaceIC *TimeIC etc
 * @param verbose Whether to print more debugging information
 */
void gicaTemporalICA(std::string reorgpref, std::string reducepref,
		std::string outpref, bool verbose);

/**
 * @brief Perform ICA with each dimension as a separate timeseries. This is
 * essentially unmixing in space and producing independent timeseries.
 *
 * @param reorgpref Prefix of tall input file *_tall_[0-9]*
 * @param reducepref Prefix input files *Umat, *Vmat *Evec files
 * @param outpref Output file prefix *_SpaceIC *TimeIC etc
 * @param verbose Whether to print more debugging information
 */
void gicaSpatialICA(std::string reorgpref, std::string reducepref,
		std::string outpref, bool verbose);

/**
 * @brief Computes the SVD from XXt using the JacobiSVD
 *
 * @param A MatrixReorg object that can be used to load images on disk
 * @param varthresh stop after the eigenvalues reach this ratio of the maximum
 * @param cvarthresh stop after the sum of eigenvalues reaches this ratio of total
 * @param maxrank use no more than the given rank when reducing
 * @param U Output U matrix, if null then ignored
 * @param V Output V matrix, if null then ignored
 *
 * @return Vector of singular values
 */
VectorXd covSVD(const MatrixReorg& A, double varthresh, double cvarthresh,
		size_t maxrank, MatrixXd* U, MatrixXd* V);

class MatMap
{
public:
	/**
	 * @brief default constructor no file is opened
	 */
	MatMap() : mat(NULL, 0, 0)
	{
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
	MatMap(std::string filename, size_t rows, size_t cols) : mat(NULL, 0, 0)
	{
		create(filename, rows, cols);
	};

	/**
	 * @brief Open an existing file as a memory map. The file must already
	 * exist. If writeable is true then the file will be opened for reading
	 * and writing, by default write is off. Note the same file should not
	 * be simultaneously written two by two separate processes.
	 *
	 * @param filename File to open
	 * @param writeable whether writing is allowed
	 */
	MatMap(std::string filename, bool writeable=false) : mat(NULL, 0, 0)
	{
		open(filename, writeable);
	};

	~MatMap() { close(); };

	/**
	 * @brief Open an existing file as a memory map. The file must already
	 * exist. If writeable is true then the file will be opened for reading
	 * and writing, by default write is off. Note the same file should not
	 * be simultaneously written two by two separate processes.
	 *
	 * @param filename File to open
	 * @param writeable whether writing is allowed
	 */
	void open(std::string filename, bool writeable = false);

	/**
	 * @brief Open an new file as a memory map. ANY OLD FILE WILL BE DELETED
	 * The file is always opened for writing and reading. Note the same file
	 * should not be simultaneously written two by two separate processes.
	 *
	 * @param filename File to open
	 * @param newrows Number of rows in matrix file
	 * @param newcols number of columns in matrix file
	 */
	void create(std::string filename, size_t newrows, size_t newcols);

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

	inline std::string tallMatName(size_t ii) const
	{
		return m_prefix+"_tall_"+std::to_string(ii);
	};

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

	std::string mask_name(size_t ii) const
	{
		return m_prefix+"_mask_"+std::to_string(ii)+".nii.gz";
	};

	std::string info_name() const
	{
		return m_prefix+".info";
	};
	std::string tall_name(size_t ii) const
	{
		return m_prefix+"_tall_"+std::to_string(ii);
	};

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
 * @brief Performs general linear model analysis on a 4D image.
 *
 * @param fmri Input 4D image.
 * @param X Regressors
 * @param bimg Output betas for each voxel (should have same
 * @param Timg Output t-score image
 * @param pimg Output p-valeu image
 */
void fmriGLM(ptr<const MRImage> fmri, const MatrixXd& X,
		ptr<MRImage> bimg, ptr<MRImage> Timg, ptr<MRImage> pimg);


/**
 * @brief Takes the FFT of each line of the image, performs bandpass filtering
 * on the line and then invert FFTs and writes back to the input image.
 *
 * @param inimg Input image
 * @param cuton Minimum frequency (may be 0)
 * @param cutoff Maximum frequency in band (may be INFINITY)
 */
void fmriBandPass(ptr<MRImage> inimg, double cuton, double cutoff);

/**
 * @brief Regresses out the given variables, creating time series which are
 * uncorrelated with X
 *
 * @param inimg Input 4D image
 * @param X matrix of covariates
 *
 */
ptr<MRImage> regressOut(ptr<const MRImage> inimg, const MatrixXd& X);

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
		ptr<const MRImage> labelmap);

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
		ptr<const MRImage> labelmap, size_t outsz);

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
MatrixXd extractLabelICA(ptr<const MRImage> fmri,
		ptr<const MRImage> labelmap, size_t outsz);

} // NPL

#endif //ICA_HELPERS_H
