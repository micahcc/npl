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

namespace npl {

/**
 * @brief Helper function for large-scale ICA analysis. This takes
 * a working directory, which should already have 'mat_#' files with data
 * (one per element of ncols) and orthogonalizes the data to produce a
 * set of variables variables which are orthogonal.
 *
 * This assumes that the input is a set of matrices which will be concat'd in
 * the row (spatial) direction. By default it is assumed that the columns
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
		const MatrixXd& XXT, MatrixXd& Xorth);

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
int spcat_ica(bool psd, const std::vector<std::string>& imgnames,
		const std::vector<std::string>& masknames, std::string workdir, double evthresh,
		int lancbasis, int maxrank, bool spatial);

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
		ptr<const MRImage> img, ptr<const MRImage> mask);

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
		ptr<const MRImage> img, ptr<const MRImage> mask);

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
	MatrixReorg(std::string prefix, size_t maxd=(1<<30), bool verbose=true);

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
	int createMats(size_t timeblocks, size_t spaceblocks, 
			const std::vector<std::string>& masknames, 
			const std::vector<std::string>& filenames);

	size_t m_totalrows;
	size_t m_totalcols;
	std::string m_prefix;
	size_t m_maxdoubles;
	bool m_verbose;
	vector<int> m_inrows;
	vector<int> m_incols;
	vector<int> m_outrows;
	vector<int> m_outcols;
};

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

	void open(std::string filename)
	{
		datamap.openExisting(filename);
		
		size_t* nrowsptr = (size_t*)datamap.data();
		size_t* ncolsptr = nrowsptr+1;
		double* dataptr = (double*)((size_t*)datamap.data()+2);
		
		rows = *nrowsptr;
		cols = *ncolsptr;
		new (&this->mat) Eigen::Map<MatrixXd>(dataptr, rows, cols);
	};

	void close()
	{
		datamap.close();
	};

	bool isopen()
	{
		return datamap.isopen();
	};

	Eigen::Map<MatrixXd> mat;
	size_t rows;
	size_t cols;
private:
	MemMap datamap;
};


}

#endif //ICA_HELPERS_H
