/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file mrimage.h
 * @brief This file contains the definition for MRImage and its derived types.
 * The derived types are templated over dimensionality and pixel type.
 ******************************************************************************/

#ifndef NDIMAGE_H
#define NDIMAGE_H

#include "npltypes.h"
#include "ndarray.h"
#include "nifti.h"

#include "zlib.h"

#include <string>
#include <iomanip>
#include <cassert>
#include <memory>
#include <map>

namespace npl {

using std::vector;

enum SliceOrderT {UNKNOWN_SLICE=0, SEQ=1, RSEQ=2, ALT=3, RALT=4, ALT_SHFT=5,
	RALT_SHFT=6};

enum CoordinateT {NOFORM=0, QFORM=1, SFORM=2};

enum BoundaryConditionT {ZEROFLUX=0, CONSTZERO=1, WRAP=2};

class MRImage;

/*****************************************************************************
 * Helper Function to Create MRImage's
 ****************************************************************************/
/**
 * \addtogroup MRImageUtilities
 * @{
 */

/**
 * @brief Creates a new MRImage with dimensions set by ndim, and size set by
 * size. Output pixel type is decided by type variable.
 *
 * @param ndim number of image dimensions
 * @param size size of image, in each dimension
 * @param type Pixel type npl::PixelT
 *
 * @return New image, default orientation
 */
ptr<MRImage> createMRImage(size_t ndim, const size_t* size, PixelT type);

/**
 * @brief Creates a new MRImage with dimensions set by ndim, and size set by
 * size. Output pixel type is decided by type variable.
 *
 * @param dim size of image, in each dimension, number of dimensions decied by
 * length of size vector
 * @param type Pixel type npl::PixelT
 *
 * @return New image, default orientation
 */
ptr<MRImage> createMRImage(const std::vector<size_t>& dim, PixelT type);

/**
 * @brief Creates a new MRImage with dimensions set by ndim, and size set by
 * size. Output pixel type is decided by type variable.
 *
 * @param ndim number of image dimensions
 * @param size size of image, in each dimension
 * @param type Pixel type npl::PixelT
 * @param ptr Pointer to data block
 * @param deleter function to delete data block
 *
 * @return New image, default orientation
 */
ptr<MRImage> createMRImage(size_t ndim, const size_t* size, PixelT type,
				void* ptr, std::function<void(void*)> deleter);

/**
 * @brief Creates a new MRImage with dimensions set by ndim, and size set by
 * size. Output pixel type is decided by type variable.
 *
 * @param dim size of image, in each dimension, number of dimensions decied by
 * length of size vector
 * @param type Pixel type npl::PixelT
 * @param ptr Pointer to data block
 * @param deleter function to delete data block
 *
 * @return New image, default orientation
 */
ptr<MRImage> createMRImage(const std::vector<size_t>& dim, PixelT type,
				void* ptr, std::function<void(void*)> deleter);

/**
 * @brief Writes out information about an MRImage
 *
 * @param out Output ostream
 * @param img Image to write information about
 *
 * @return More ostream
 */
std::ostream& operator<<(std::ostream &out, const MRImage& img);

/** @} MRImageUtilities */

/******************************************************************************
 * Classes.
 ******************************************************************************/

/**
 * @brief MRImage can basically be used like an NDArray, with the addition
 * of orientation related additions.
 */
class MRImage : public virtual NDArray
{
public:
	MRImage() : m_freqdim(-1), m_phasedim(-1), m_slicedim(-1),
	m_coordinate(NOFORM), m_slice_duration(0), m_slice_start(-1),
	m_slice_end(-1), m_slice_order(UNKNOWN_SLICE) {} ;

	/********************************************
	 * Orientation Functions/Variables
	 *******************************************/

	/**
	 * @brief Default orientation (dir=ident, space=1 and origin=0), also resizes
	 * them. So this could be called without first initializing size.
	 */
	void orientDefault();

	/**
	 * @brief Returns true if the image has a valid orientation
	 *
	 * @return True if the image has a valid orientation
	 */
	bool isOriented() { return m_coordinate != NOFORM; };

	/**
	 * @brief Update the orientation of the pixels in RAS space.
	 *
	 * @param neworig New Origin.
	 * @param newspace New Spacing.
	 * @param newdir New Direction
	 * @param reinit Whether to reset everything to Identity/0 before applying.
	 * You may want to do this if theinput matrices/vectors differ in dimension
	 * from this image.
	 * @param coord coordinate system this refers to. Eventually this might
	 * include more advanced options (scanner,anat etc). For now it is just
	 * QFORM or SFORM
	 */
	void setOrient(const VectorXd& neworig, const VectorXd& newspace,
					const MatrixXd& newdir, bool reinit = true,
					CoordinateT coord = QFORM);

	/**
	 * @brief Returns reference to a value in the direction matrix.
	 * Each row indicates the direction of the grid in
	 * RAS coordinates. This is the rotation of the Index grid.
	 *
	 * @param row Row to access
	 * @param col Column to access
	 *
	 * @return Element in direction matrix
	 */
	const double& direction(int64_t row, int64_t col) const;

	/**
	 * @brief Returns reference to a value in the inverse direction matrix.
	 * Each row indicates the direction of the grid in
	 * RAS coordinates. This is the rotation of the Index grid.
	 *
	 * @param row Row to access
	 * @param col Column to access
	 *
	 * @return Element in direction matrix
	 */
	const double& invdirection(int64_t row, int64_t col) const;

	/**
	 * @brief Returns reference to the direction matrix.
	 * Each row indicates the direction of the grid in
	 * RAS coordinates. This is the rotation of the Index grid.
	 *
	 * @return Direction matrix
	 */
	const MatrixXd& getDirection() const;

	/**
	 * @brief Updates orientation information. If reinit is given then it will first
	 * set direction to the identity. otherwise old values will be left. After this
	 * the first min(DIMENSION,dir.rows()) columns and min(DIMENSION,dir.cols())
	 * columns will be copies into the image direction matrix.
	 *
	 * @param newdir Input direction/rotation
	 * @param reinit Whether to reset everything to Identity/0 before applying.
	 * You may want to do this if theinput matrices/vectors differ in dimension
	 * from this image.
	 * @param coord coordinate system this refers to. Eventually this might
	 * include more advanced options (scanner,anat etc). For now it is just
	 * QFORM or SFORM
	 */
	void setDirection(const MatrixXd& newdir, bool reinit, CoordinateT coord = QFORM);

	/**
	 * @brief Returns reference to a value in the origin vector. This is the
	 * physical point that corresponds to index 0.
	 *
	 * @param row Row to access
	 *
	 * @return Element in origin vector
	 */
	double& origin(int64_t row);

	/**
	 * @brief Returns reference to a value in the origin vector. This is the
	 * physical point that corresponds to index 0.
	 *
	 * @param row Row to access
	 *
	 * @return Element in origin vector
	 */
	const double& origin(int64_t row) const;

	/**
	 * @brief Returns const reference to the origin vector. This is the physical
	 * point that corresponds to index 0.
	 *
	 * @return Origin vector
	 */
	const VectorXd& getOrigin() const;

	/**
	 * @brief Sets the origin vector. This is the physical
	 * point that corresponds to index 0. Note that min(current, new) elements
	 * will be copied
	 *
	 * @param neworigin the new origin vector to copy.
	 * @param reinit Whether to reset everything to Identity/0 before applying.
	 * You may want to do this if theinput matrices/vectors differ in dimension
	 * from this image.
	 *
	 */
	void setOrigin(const VectorXd& neworigin, bool reinit,
			CoordinateT coord = QFORM);

	/**
	 * @brief Returns reference to a value in the spacing vector. This is the
	 * physical distance between adjacent indexes.
	 *
	 * @param row Row to access
	 *
	 * @return Element in spacing vector
	 */
	double& spacing(int64_t row);

	/**
	 * @brief Returns reference to a value in the spacing vector. This is the
	 * physical distance between adjacent indexes.
	 *
	 * @param row Row to access
	 *
	 * @return Element in spacing vector
	 */
	const double& spacing(int64_t row) const;

	/**
	 * @brief Returns const reference to the spacing vector. This is the
	 * physical distance between adjacent indexes.
	 *
	 * @return Spacing vector
	 */
	const VectorXd& getSpacing() const;

	/**
	 * @brief Sets the spacing vector. This is the physical
	 * point that corresponds to index 0. Note that min(current, new) elements
	 * will be copied
	 *
	 * @param newspacing the new spacing vector to copy.
	 * @param reinit Set the whole vector to 1s first. This might be useful if you
	 * are setting fewer elements than dimensions
	 *
	 */
	void setSpacing(const VectorXd& newspacing, bool reinit);

	/**********************************
	 * Orientation Transform Functions
	 *********************************/
	/*
	 * @brief Converts an index in pixel space to RAS, physical/time coordinates.
	 * If len < dimensions, additional dimensions are assumed to be 0. If len >
	 * dimensions then additional values are ignored, and only the first DIM
	 * values will be transformed and written to ras.
	 *
	 * @param len Length of xyz/ras arrays.
	 * @param xyz Array in xyz... coordinates (maybe as long as you want).
	 * @param ras Corresponding coordinate
	 *
	 * @return 0
	 */
	virtual int indexToPoint(size_t len, const int64_t* xyz, double* ras) const=0;

	/**
	 * @brief Converts an index in pixel space to RAS, physical/time coordinates.
	 * If len < dimensions, additional dimensions are assumed to be 0. If len >
	 * dimensions then additional values are ignored, and only the first DIM
	 * values will be transformed and written to ras.
	 *
	 * @param len Length of xyz/ras arrays.
	 * @param xyz Array in xyz... coordinates (maybe as long as you want).
	 * @param ras Corresponding coordinate
	 *
	 * @return 0
	 */
	virtual int indexToPoint(size_t len, const double* xyz, double* ras) const=0;

	/**
	 * @brief Converts a point in RAS coordinate system to index.
	 * If len < dimensions, additional dimensions are assumed to be 0. If len >
	 * dimensions then additional values are ignored, and only the first DIM
	 * values will be transformed and written to ras.
	 *
	 * @param len Length of xyz/ras arrays.
	 * @param ras Array in RAS... coordinates (may be as long as you want).
	 * @param xyz Corresponding coordinate
	 *
	 * @return 0
	 */
	virtual int pointToIndex(size_t len, const double* ras, double* xyz) const=0;

	/**
	 * @brief Converts a point in RAS coordinate system to index.
	 * If len < dimensions, additional dimensions are assumed to be 0. If len >
	 * dimensions then additional values are ignored, and only the first DIM
	 * values will be transformed and written to ras.
	 *
	 * @param len Length of xyz/ras arrays.
	 * @param ras Array in RAS... coordinates (may be as long as you want).
	 * @param index Corresponding coordinate, rounded to nearest integer
	 *
	 * @return 0
	 */
	virtual int pointToIndex(size_t len, const double* ras, int64_t* index) const=0;

	/**
	 * @brief Convert a vector in index coordinates to a vector in ras
	 * coordinates. Vector is simply multiplied by the internal rotation
	 * matrix.
	 *
	 * @param len Length of input vector (may be different than dimension -
	 * extra values will be ignored, missing values will be assumed zero)
	 * @param xyz Input vector in index space ijk....
	 * @param ras Output vector in physical space. This is the product of the
	 * input vector and rotation matrix
	 *
	 * @return Success
	 */
	virtual int orientVector(size_t len, const double* xyz, double* ras) const=0;

	/**
	 * @brief Convert a vector in index coordinates to a vector in ras
	 * coordinates. Vector is simply multiplied by the internal rotation
	 * matrix.
	 *
	 * @param len Length of input vector (may be different than dimension -
	 * extra values will be ignored, missing values will be assumed zero)
	 * @param ras Input vector in physical space.
	 * @param xyz Output vector in index space ijk. This is the product of the
	 * input vector and inverse rotation matrix

	 *
	 * @return Success
	 */
	virtual int disOrientVector(size_t len, const double* ras, double* xyz) const=0;

	/**
	 * @brief Returns true if the point is within the field of view of the
	 * image. Note, like all coordinates pass to MRImage, if the array given
	 * differs from the dimensions of the image, then the result will either
	 * pad out zeros and ignore extra values in the input array.
	 *
	 * @param len Length of RAS array
	 * @param ras Array of Right-handed coordinates Right+, Anterior+, Superior+
	 *
	 * @return Whether the point would round to a voxel inside the image.
	 */
	virtual bool pointInsideFOV(size_t len, const double* ras) const=0;

	/**
	 * @brief Returns true if the constinuous index is within the field of
	 * view of the image. Note, like all coordinates pass to MRImage, if the
	 * array given differs from the dimensions of the image, then the result
	 * will either pad out zeros and ignore extra values in the input array.

	 *
	 * @param len Length of xyz array
	 * @param xyz Array of continouos indices
	 *
	 * @return Whether the index would round to a voxel inside the image.
	 */
	virtual bool indexInsideFOV(size_t len, const double* xyz) const=0;

	/**
	 * @brief Returns true if the constinuous index is within the field of
	 * view of the image. Note, like all coordinates pass to MRImage, if the
	 * array given differs from the dimensions of the image, then the result
	 * will either pad out zeros and ignore extra values in the input array.
	 *
	 * @param len Length of xyz array
	 * @param xyz Array of indices
	 *
	 * @return Whether the index is inside the image
	 */
	virtual bool indexInsideFOV(size_t len, const int64_t* xyz) const=0;

	/**
	 * @brief Returns true of the other image has matching orientation as this.
	 * If checksize = true, then it will also check the size of the two images
	 * and return true if both orientation and size match, and false if they
	 * don't.
	 *
	 * @param other MRimage to compare.
	 * @param checkdim Whether to enforce identical dimensionality. If this is
	 * false then the first min(D1,D2) dimensions will be checked, if this is true
	 * then mismatched dimensionality will cause this to return a false
	 * @param checksize Whether to enforce identical size as well as orientation
	 *
	 * @return True if the two images have matching orientation information.
	 */
	virtual bool matchingOrient(ptr<const MRImage> other, bool checkdim,
			bool checksize, double tol = 0.01) const;

	/**
	 * @brief Returns true if the image is isotropic (same spacing in all
	 * dimensions). This can be looseened to checking only the first 3 dims
	 * with only3d = true. Tolerence in the absolute maximum difference between
	 * the first dim and any other dim.
	 *
	 * @param only3d Only check spacing in the first 3 dimensions (default)
	 * @param tol Tolerence in the absolute maximum difference between
	 * the first dim and any other dim.
	 *
	 * @return True if the image is roughly isotropic
	 */
	virtual bool isIsotropic(bool only3d = true, double tol = 0.01) const;

	/********************************************
	 * Output Functions
	 *******************************************/

	/**
	 * @brief Write the image to a nifti file.
	 *
	 * @param filename Filename
	 * @param version Version of nifti to use
	 *
	 * @return 0 if successful
	 */
	virtual int write(std::string filename, double version = 1) const = 0;

	/********************************************
	 * Copying/Pointer Functions
	 *******************************************/

	/**
	 * @brief Returns a pointer to self
	 *
	 * @return this
	 */
	ptr<MRImage> getPtr()  {
			return dPtrCast<MRImage>(shared_from_this());
	};

	/**
	 * @brief Returns a constant pointer to self
	 *
	 * @return this
	 */
	ptr<const MRImage> getConstPtr() const {
			return dPtrCast<const MRImage>(shared_from_this());
	};


	/**
	 * @brief Create a copy of this image. This is identical to copy() but will
	 * return a pointer to an image rather than an NDArray.
	 *
	 * @return  Pointer to deep-copied image.
	 */
	virtual ptr<MRImage> cloneImage() const = 0;


	/**
	 * @brief Performs a deep copy of the entire image and all metadata.
	 *
	 * @return Copied image (as NDarray pointer)
	 */
	virtual ptr<NDArray> copy() const = 0;

	/**
	 * @brief Creates an identical array, but does not initialize pixel values.
	 *
	 * @return New array.
	 */
	virtual ptr<NDArray> createAnother() const = 0;

	/**
	 * @brief Create a new array that is the same underlying type as this.
	 * If this is an image then it will also copy the metdata, but NOT the
	 * pixels.
	 *
	 * @param newdims Number of dimensions in copied output
	 * @param newsize Size of output, this array should be of size newdims
	 * @param newtype Type of pixels in output array
	 *
	 * @return Image with identical orientation but different size and pixeltype
	 */
	virtual ptr<NDArray> createAnother(size_t newdims, const size_t* newsize,
					PixelT newtype) const = 0;

	/**
	 * @brief Create a new array that is the same underlying type as this, but
	 * with a different pixel type.
	 *
	 * @param newtype Type of pixels in output array
	 *
	 * @return Image with identical orientation and size but different pixel
	 * type
	 */
	virtual ptr<NDArray> createAnother(PixelT newtype) const = 0;

	/**
	 * @brief Create a new array that is the same underlying type as this,
	 * and same pixel type and orientation as this, but with a different
	 * size.
	 *
	 * @param newdims Number of dimensions in output array
	 * @param newsize Input array of length newdims that gives the size of
	 *                output array,
	 *
	 * @return Image with identical orientation and pixel type but different
	 * size from this
	 */
	virtual ptr<NDArray> createAnother(size_t newdims,
					const size_t* newsize) const = 0;


	/**
	 * @brief Create a new image that is a copy of the input, possibly with new
	 * dimensions and pixeltype. The new image will have all overlapping pixels
	 * copied from the old image.
	 *
	 * @param newdims Number of dimensions in output image
	 * @param newsize Size of output image
	 * @param newtype Type of pixels in output image
	 *
	 * @return Image with overlapping sections cast and copied from 'in'
	 */
	virtual ptr<NDArray> copyCast(size_t newdims, const size_t* newsize,
					PixelT newtype) const = 0;

	/**
	 * @brief Create a new image that is a copy of the input, with same dimensions
	 * but pxiels cast to newtype. The new image will have all overlapping pixels
	 * copied from the old image.
	 *
	 * @param newtype Type of pixels in output image
	 *
	 * @return Image with overlapping sections cast and copied from 'in'
	 */
	virtual ptr<NDArray> copyCast(PixelT newtype) const = 0;

	/**
	 * @brief Create a new image that is a copy of the input, possibly with new
	 * dimensions or size. The new image will have all overlapping pixels
	 * copied from the old image. The new image will have the same pixel type as
	 * the input image
	 *
	 * @param newdims Number of dimensions in output image
	 * @param newsize Size of output image
	 *
	 * @return Image with overlapping sections cast and copied from 'in'
	 */
	virtual ptr<NDArray> copyCast(size_t newdims,
					const size_t* newsize) const = 0;

	/**
	 * @brief extracts a region of this image. Zeros in the size variable
	 * indicate dimension to be removed.
	 *
	 * @param len     Length of index/newsize arrays
	 * @param index   Index to start copying from.
	 * @param size Size of output image. Note length 0 dimensions will be
	 * removed, while length 1 dimensions will be left.
	 *
	 * @return Image with overlapping sections cast and copied from 'in'
	 */
	virtual ptr<NDArray> extractCast(size_t len, const int64_t* index,
					const size_t* size) const = 0;

	/**
	 * @brief extracts a region of this image. Zeros in the size variable
	 * indicate dimension to be removed.
	 *
	 * @param len     Length of index/size arrays
	 * @param size Size of output image. Note length 0 dimensions will be
	 * removed, while length 1 dimensions will be left.
	 *
	 * @return Image with overlapping sections cast and copied from 'in'
	 */
	virtual ptr<NDArray> extractCast(size_t len, const size_t* size) const = 0;

	/**
	 * @brief extracts a region of this image. Zeros in the size variable
	 * indicate dimension to be removed.
	 *
	 * @param len     Length of index/size arrays
	 * @param index   Index to start copying from.
	 * @param size Size of output image. Note length 0 dimensions will be
	 * removed, while length 1 dimensions will be left.
	 * @param newtype Pixel type of output image.
	 *
	 * @return Image with overlapping sections cast and copied from 'in'
	 */
	virtual ptr<NDArray> extractCast(size_t len,
					const int64_t* index, const size_t* size, PixelT newtype) const = 0;

	/**
	 * @brief extracts a region of this image. Zeros in the size variable
	 * indicate dimension to be removed.
	 *
	 * @param len     Length of index/size arrays
	 * @param size Size of output image. Note length 0 dimensions will be
	 * removed, while length 1 dimensions will be left.
	 * @param newtype Pixel type of output image.
	 *
	 * @return Image with overlapping sections cast and copied from 'in'
	 */
	virtual ptr<NDArray> extractCast(size_t len, const size_t* size,
					PixelT newtype) const = 0;


	/**
	 * @brief Copies metadata from another image. This includes slice timing,
	 * anything read from nifti files, spacing, orientation etc, but NOT
	 * pixel data, size, and dimensionality.
	 *
	 * @param src Other image to copy from
	 */
	virtual void copyMetadata(ptr<const MRImage> src);


	//	virtual int unary(double(*func)(double,double)) const = 0;
	//	virtual int binOp(const MRImage* right, double(*func)(double,double), bool elevR) const = 0;

	/**********************************************************************
	 * Medical Image Specific
	 *********************************************************************/

	// < 0 indicate unset variables
	int m_freqdim;
	int m_phasedim;
	int m_slicedim;
	CoordinateT m_coordinate;

	/*
	 * nifti specific stuff, eventually these should be moved to a nifti
	 * image subclass
	 */

	void updateSliceTiming(double duration, int start, int end, SliceOrderT order);

	// raw values for slice data, < 0 indicate unset
	double m_slice_duration;
	int m_slice_start;
	int m_slice_end;

	// SEQ, RSEQ, ALT, RALT, ALT_SHFT, RALT_SHFT
	// SEQ (sequential):	slice_start .. slice_end
	// RSEQ (reverse seq):	slice_end .. slice_start
	// ALT (alternated):	slice_start, slice_start+2, .. slice_end|slice_end-1,
	//						slice_start+1 .. slice_end|slice_end-1
	// RALT (reverse alt):	slice_end, slice_end-2, .. slice_start|slice_start+1,
	//						slice_end-1 .. slice_start|slice_start+1
	// ALT_SHFT (siemens alt):slice_start+1, slice_start+3, .. slice_end|slice_end-1,
	//						slice_start .. slice_end|slice_end-1
	// RALT (reverse alt):	slice_end-1, slice_end-3, .. slice_start|slice_start+1,
	//						slice_end-2 .. slice_start|slice_start+1
	SliceOrderT m_slice_order;

	// each slice is given its relative time, with 0 as the first
	std::map<int64_t,double> m_slice_timing;

protected:
	virtual int writeNifti1Image(gzFile file) const = 0;
	virtual int writeNifti2Image(gzFile file) const = 0;

	/**
	 * @brief Direction Matrix. Each row indicates the direction of the grid in
	 * RAS coordinates. This is the rotation of the Index grid.
	 * Note that you should not set this directly,
	 */
	MatrixXd m_direction;

	/**
	 * @brief Inverse of Direction Matrix.
	 */
	MatrixXd m_inv_direction;

	/**
	 * @brief Spacing vector. Indicates distance between adjacent pixels in
	 * each dimension. Note that you should not set this directly, use
	 * setSpacing() instead because it will update the affine Matrix.
	 */
	VectorXd m_spacing;

	/**
	 * @brief Origin vector. Indicates the RAS coordinates of index [0,0,0,..]
	 * Note that you should not set this directly, use setOrigin() instead
	 * because it will update the affine Matrix.
	 */
	VectorXd m_origin;

	friend ptr<MRImage> readNiftiImage(gzFile file, bool verbose);
};

/**
 * @brief MRImageStore is a version of NDArray that has an orientation matrix.
 * Right now it also has additional data that is unique to nifti. Eventually
 * this class will be forked into a subclass, and this will only have the
 * orientation.
 *
 * @tparam D	Number of dimensions
 * @tparam T	Pixel type
 */
template <size_t D, typename T>
		class MRImageStore :  public virtual NDArrayStore<D,T>, public virtual MRImage
{

public:

	/*****************************************
	 * Constructors
	 ****************************************/
	/**
	 * @brief Constructor with initializer list. Orientation will be default
	 * (direction = identity, spacing = 1, origin = 0).
	 *
	 * @param a_args dimensions of input, the length of this initializer list
	 * may not be fully used if a_args is longer than D. If it is shorter
	 * then D then additional dimensions are left as size 1.
	 */
	MRImageStore(std::initializer_list<size_t> a_args);

	/**
	 * @brief Constructor with vector. Orientation will be default
	 * (direction = identity, spacing = 1, origin = 0).
	 *
	 * @param a_args dimensions of input, the length of this initializer list
	 * may not be fully used if a_args is longer than D. If it is shorter
	 * then D then additional dimensions are left as size 1.
	 */
	MRImageStore(const std::vector<size_t>& a_args);

	/**
	 * @brief Constructor with array of length len, Orientation will be default
	 * (direction = identity, spacing = 1, origin = 0).
	 *
	 * @param len Length of array 'size'
	 * @param size dimensions of input, the length of this initializer list
	 * may not be fully used if a_args is longer than D. If it is shorter
	 * then D then additional dimensions are left as size 1.
	 */
	MRImageStore(size_t len, const size_t* size);

	/**
	 * @brief Constructor which uses a preexsting array, to graft into the
	 * image. No new allocation will be performed, however ownership of the
	 * array will be taken, meaning it could be deleted anytime after this
	 * constructor completes.
	 *
	 * @param len Length of array 'size'
	 * @param size dimensions of input, the length of this initializer list
	 * may not be fully used if a_args is longer than D. If it is shorter
	 * then D then additional dimensions are left as size 1.
	 * @param ptr Pointer to data array, should be allocated with new, and
	 * size should be exactly sizeof(T)*size[0]*size[1]*...*size[len-1]
	 * @param deleter Function to use to delete (free) ptr
	 */
	MRImageStore(size_t len, const size_t* size, T* ptr,
					const std::function<void(void*)>& deleter) ;


	/**
	 * @brief Default constructor, uses identity for direction matrix, 1 for
	 * spacing and 0 for origin. Image size is 0
	 */
	MRImageStore();

	/*************************************************************************
	 * Coordinate Transform Functions
	 ************************************************************************/

	/**
	 * @brief Converts an index in pixel space to RAS, physical/time coordinates.
	 * If len < dimensions, additional dimensions are assumed to be 0. If len >
	 * dimensions then additional values are ignored, and only the first DIM
	 * values will be transformed and written to ras.
	 *
	 * @param len Length of xyz/ras arrays.
	 * @param xyz Array in xyz... coordinates (maybe as long as you want).
	 * @param ras Corresponding coordinate
	 *
	 * @return 0
	 */
	virtual int indexToPoint(size_t len, const int64_t* xyz, double* ras) const;

	/**
	 * @brief Converts an index in pixel space to RAS, physical/time coordinates.
	 * If len < dimensions, additional dimensions are assumed to be 0. If len >
	 * dimensions then additional values are ignored, and only the first DIM
	 * values will be transformed and written to ras.
	 *
	 * @param len Length of xyz/ras arrays.
	 * @param xyz Array in xyz... coordinates (maybe as long as you want).
	 * @param ras Corresponding coordinate
	 *
	 * @return 0
	 */
	virtual int indexToPoint(size_t len, const double* xyz, double* ras) const;

	/**
	 * @brief Converts a point in RAS coordinate system to index.
	 * If len < dimensions, additional dimensions are assumed to be 0. If len >
	 * dimensions then additional values are ignored, and only the first DIM
	 * values will be transformed and written to ras.
	 *
	 * @param len Length of xyz/ras arrays.
	 * @param ras Array in RAS... coordinates (may be as long as you want).
	 * @param xyz Corresponding coordinate
	 *
	 * @return 0
	 */
	virtual int pointToIndex(size_t len, const double* ras, double* xyz) const;

	/**
	 * @brief Converts a point in RAS coordinate system to index.
	 * If len < dimensions, additional dimensions are assumed to be 0. If len >
	 * dimensions then additional values are ignored, and only the first DIM
	 * values will be transformed and written to ras.
	 *
	 * @param len Length of xyz/ras arrays.
	 * @param ras Array in RAS... coordinates (may be as long as you want).
	 * @param index Corresponding coordinate, rounded to nearest integer
	 *
	 * @return 0
	 */
	virtual int pointToIndex(size_t len, const double* ras, int64_t* index) const;

	/**
	 * @brief Convert a vector in index coordinates to a vector in ras
	 * coordinates. Vector is simply multiplied by the internal rotation
	 * matrix.
	 *
	 * @param len Length of input vector (may be different than dimension -
	 * extra values will be ignored, missing values will be assumed zero)
	 * @param xyz Input vector in index space ijk....
	 * @param ras Output vector in physical space. This is the product of the
	 * input vector and rotation matrix
	 *
	 * @return Success
	 */
	virtual int orientVector(size_t len, const double* xyz, double* ras) const;

	/**
	 * @brief Convert a vector in index coordinates to a vector in ras
	 * coordinates. Vector is simply multiplied by the internal rotation
	 * matrix.
	 *
	 * @param len Length of input vector (may be different than dimension -
	 * extra values will be ignored, missing values will be assumed zero)
	 * @param ras Input vector in physical space.
	 * @param xyz Output vector in index space ijk. This is the product of the
	 * input vector and inverse rotation matrix

	 *
	 * @return Success
	 */
	virtual int disOrientVector(size_t len, const double* ras, double* xyz) const;

	/**
	 * @brief Returns true if the point is within the field of view of the
	 * image. Note, like all coordinates pass to MRImage, if the array given
	 * differs from the dimensions of the image, then the result will either
	 * pad out zeros and ignore extra values in the input array.
	 *
	 * @param len Length of RAS array
	 * @param ras Array of Right-handed coordinates Right+, Anterior+, Superior+
	 *
	 * @return Whether the point would round to a voxel inside the image.
	 */
	virtual bool pointInsideFOV(size_t len, const double* ras) const;

	/**
	 * @brief Returns true if the constinuous index is within the field of
	 * view of the image. Note, like all coordinates pass to MRImage, if the
	 * array given differs from the dimensions of the image, then the result
	 * will either pad out zeros and ignore extra values in the input array.

	 *
	 * @param len Length of xyz array
	 * @param xyz Array of continouos indices
	 *
	 * @return Whether the index would round to a voxel inside the image.
	 */
	virtual bool indexInsideFOV(size_t len, const double* xyz) const;

	/**
	 * @brief Returns true if the constinuous index is within the field of
	 * view of the image. Note, like all coordinates pass to MRImage, if the
	 * array given differs from the dimensions of the image, then the result
	 * will either pad out zeros and ignore extra values in the input array.
	 *
	 * @param len Length of xyz array
	 * @param xyz Array of indices
	 *
	 * @return Whether the index is inside the image
	 */
	virtual bool indexInsideFOV(size_t len, const int64_t* xyz) const;

	/**
	 * @brief Returns units of given dimension, note that this is prior to the
	 * direction matrix, so if there is oblique orientation you are really
	 * looking at a mix of units.
	 *
	 * @param d Dimension
	 *
	 * @return String describing units.
	 */
	std::string getUnits(size_t d) { return m_units[d]; };

	/********************************************
	 * Output Functions
	 *******************************************/
	/**
	 * @brief Write out nifti image with the current images data.
	 *
	 * @param filename
	 * @param version > 2 or < 2 to indicate whether to use nifti version 2
	 * or nifti version 1.
	 *
	 * @return Success if 0
	 */
	int write(std::string filename, double version) const;

	/**
	 * @brief Print information about the image
	 */
	void printSelf();

	/********************************************
	 * Copying/Pointer Functions
	 *******************************************/

	/**
	 * @brief Performs a deep copy of the entire image and all metadata.
	 *
	 * @return Copied image.
	 */
	virtual ptr<NDArray> copy() const;

	/**
	 * @brief Creates an identical array, but does not initialize pixel values.
	 *
	 * @return New array.
	 */
	virtual ptr<NDArray> createAnother() const;

	/**
	 * @brief Create a new array that is the same underlying type as this.
	 * If this is an image then it will also copy the metdata, but NOT the
	 * pixels.
	 *
	 * @param newdims Number of dimensions in copied output
	 * @param newsize Size of output, this array should be of size newdims
	 * @param newtype Type of pixels in output array
	 *
	 * @return Image with identical orientation but different size and pixeltype
	 */
	virtual ptr<NDArray> createAnother(size_t newdims, const size_t* newsize,
					PixelT newtype) const;

	/**
	 * @brief Create a new array that is the same underlying type as this, but
	 * with a different pixel type.
	 *
	 * @param newtype Type of pixels in output array
	 *
	 * @return Image with identical orientation and size but different pixel
	 * type
	 */
	virtual ptr<NDArray> createAnother(PixelT newtype) const;

	/**
	 * @brief Create a new array that is the same underlying type as this,
	 * and same pixel type and orientation as this, but with a different
	 * size.
	 *
	 * @param newdims Number of dimensions in output array
	 * @param newsize Input array of length newdims that gives the size of
	 *                output array,
	 *
	 * @return Image with identical orientation and pixel type but different
	 * size from this
	 */
	virtual ptr<NDArray> createAnother(size_t newdims,
					const size_t* newsize) const;

	/**
	 * @brief Create a new image that is a copy of the input, possibly with new
	 * dimensions and pixeltype. The new image will have all overlapping pixels
	 * copied from the old image.
	 *
	 * @param newdims Number of dimensions in output image
	 * @param newsize Size of output image
	 * @param newtype Type of pixels in output image
	 *
	 * @return Image with overlapping sections cast and copied from 'in'
	 */
	virtual ptr<NDArray> copyCast(size_t newdims, const size_t* newsize,
					PixelT newtype) const;

	/**
	 * @brief Create a new image that is a copy of the input, with same dimensions
	 * but pxiels cast to newtype. The new image will have all overlapping pixels
	 * copied from the old image.
	 *
	 * @param newtype Type of pixels in output image
	 *
	 * @return Image with overlapping sections cast and copied from 'in'
	 */
	virtual ptr<NDArray> copyCast(PixelT newtype) const;

	/**
	 * @brief Create a new image that is a copy of the input, possibly with new
	 * dimensions or size. The new image will have all overlapping pixels
	 * copied from the old image. The new image will have the same pixel type as
	 * the input image
	 *
	 * @param newdims Number of dimensions in output image
	 * @param newsize Size of output image
	 *
	 * @return Image with overlapping sections cast and copied from 'in'
	 */
	virtual ptr<NDArray> copyCast(size_t newdims, const size_t* newsize) const;

	/**
	 * @brief Create a new array that is a copy of the input, possibly with new
	 * dimensions or size. The new array will have all overlapping pixels
	 * copied from the old array. The new array will have the same pixel type as
	 * the input array
	 *
	 * @param len     Length of index/newsize arrays
	 * @param index   Index to start copying from.
	 * @param size Size of output image. Note length 0 dimensions will be
	 * removed, while length 1 dimensions will be left.
	 *
	 * @return Image with overlapping sections cast and copied from 'in'
	 */
	virtual ptr<NDArray> extractCast(size_t len, const int64_t* index,
					const size_t* size) const;

	/**
	 * @brief Create a new array that is a copy of the input, possibly with new
	 * dimensions or size. The new array will have all overlapping pixels
	 * copied from the old array. The new array will have the same pixel type as
	 * the input array. Index assumed to be [0,0,...], so the output image will
	 * start at the origin of this image.
	 *
	 * @param len     Length of index/size arrays
	 * @param size Size of output image. Note length 0 dimensions will be
	 * removed, while length 1 dimensions will be left.
	 *
	 * @return Image with overlapping sections cast and copied from 'in'
	 */
	virtual ptr<NDArray> extractCast(size_t len, const size_t* size) const;

	/**
	 * @brief Create a new array that is a copy of the input, possibly with new
	 * dimensions or size. The new array will have all overlapping pixels
	 * copied from the old array. The new array will have the same pixel type as
	 * the input array
	 *
	 * @param len     Length of index/size arrays
	 * @param index   Index to start copying from.
	 * @param size Size of output image. Note length 0 dimensions will be
	 * removed, while length 1 dimensions will be left.
	 * @param newtype Pixel type of output image.
	 *
	 * @return Image with overlapping sections cast and copied from 'in'
	 */

	virtual ptr<NDArray> extractCast(size_t len,
					const int64_t* index, const size_t* size, PixelT newtype) const;

	/**
	 * @brief Create a new array that is a copy of the input, possibly with new
	 * dimensions or size. The new array will have all overlapping pixels
	 * copied from the old array. The new array will have the same pixel type as
	 * the input array. Index assumed to be [0,0,...], so the output image will
	 * start at the origin of this image.
	 *
	 * @param len     Length of index/size arrays
	 * @param size Size of output image. Note length 0 dimensions will be
	 * removed, while length 1 dimensions will be left.
	 * @param newtype Pixel type of output image.
	 *
	 * @return Image with overlapping sections cast and copied from 'in'
	 */
	virtual ptr<NDArray> extractCast(size_t len, const size_t* size,
					PixelT newtype) const;

	/**
	 * @brief Create an exact copy of the current image object, and return
	 * a pointer to it.
	 *
	 * @return Pointer to exact duplicate of current image.
	 */
	ptr<MRImage> cloneImage() const;

protected:

	/**
	 * @brief Vector of units for each dimension
	 */
	std::string m_units[D];

	int writeNifti1Image(gzFile file) const;
	int writeNifti2Image(gzFile file) const;
	int writeNifti1Header(gzFile file) const;
	int writeNifti2Header(gzFile file) const;
	int writePixels(gzFile file) const;
	int writeJSON(gzFile file) const;
};
} // npl
#endif
