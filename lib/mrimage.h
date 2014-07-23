/*******************************************************************************
This file is part of Neuro Programs and Libraries (NPL), 

Written and Copyrighted by by Micah C. Chambers (micahc.vt@gmail.com)

The Neuro Programs and Libraries are free software: you can redistribute it
and/or modify it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

The Neural Programs and Libraries are distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along with
the Neural Programs Library.  If not, see <http://www.gnu.org/licenses/>.
*******************************************************************************/

#ifndef NDIMAGE_H
#define NDIMAGE_H

#include "ndarray.h"
#include "npltypes.h"
#include "matrix.h"

#include "zlib.h"

#include <string>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <memory>

namespace npl {

using std::vector;
using std::shared_ptr;

enum SliceOrderT {UNKNOWN_SLICE=0, SEQ=1, RSEQ=2, ALT=3, RALT=4, ALT_SHFT=5,
	RALT_SHFT=6};

enum BoundaryConditionT {ZEROFLUX=0, CONSTZERO=1, WRAP=2};

class MRImage;

// simply reads an image in its native type
shared_ptr<MRImage> readNiftiImage(gzFile file, bool verbose);
shared_ptr<MRImage> readMRImage(std::string filename, bool verbose = false);
shared_ptr<MRImage> createMRImage(const std::vector<size_t>& dims, PixelT);
//int writeMRImage(MRImage* img, std::string fn, bool nifti2 = false);

std::ostream& operator<<(std::ostream &out, const MRImage& img);

/**
 * @brief MRImage can basically be used like an NDArray, with the addition
 * of orientation related additions.
 */
class MRImage : public virtual NDArray
{
public:
	MRImage() : m_freqdim(-1), m_phasedim(-1), m_slicedim(-1), 
				m_slice_duration(0), m_slice_start(-1), 
				m_slice_end(-1), m_slice_order(UNKNOWN_SLICE) {} ;

	virtual MatrixP& spacing() = 0;
	virtual MatrixP& origin() = 0;
	virtual MatrixP& direction() = 0;
//	virtual MatrixP& affine() = 0;
//	virtual MatrixP& iaffine() = 0;
	virtual const MatrixP& spacing() const = 0;
	virtual const MatrixP& origin() const  = 0;
	virtual const MatrixP& direction() const = 0;
	virtual const MatrixP& affine() const = 0;
	virtual const MatrixP& iaffine() const = 0;

	virtual void setOrient(const MatrixP& orig, const MatrixP& space, 
			const MatrixP& dir, bool reinit) = 0;
	virtual void setOrigin(const MatrixP& orig, bool reinit = false) = 0;
	virtual void setSpacing(const MatrixP& space, bool reinit = false) = 0;
	virtual void setDirection(const MatrixP& dir, bool reinit = false) = 0;
	
	virtual int write(std::string filename, double version = 1) const = 0;
	
	virtual std::shared_ptr<MRImage> cloneImage() const = 0;
	
	// coordinate system conversion 
	
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
	 * @param xyz Corresponding coordinate, rounded to nearest integer
	 *
	 * @return 0
	 */
	virtual int pointToIndex(size_t len, const double* ras, int64_t* index) const=0;
	
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

//	virtual int unary(double(*func)(double,double)) const = 0;
//	virtual int binOp(const MRImage* right, double(*func)(double,double), bool elevR) const = 0;

	/*
	 * medical image specific stuff, eventually these should be moved to a 
	 * medical image subclass
	 */

	// each slice is given its relative time, with 0 as the first
	std::vector<double> m_slice_timing;
	
	// < 0 indicate unset variables
	int m_freqdim;
	int m_phasedim;
	int m_slicedim;

	/*
	 * nifti specific stuff, eventually these should be moved to a nifti 
	 * image subclass
	 */

	// raw values for slice data, < 0 indicate unset
	double m_slice_duration;
	int m_slice_start;
	int m_slice_end;

	// SEQ, RSEQ, ALT, RALT, ALT_SHFT, RALT_SHFT
	// SEQ (sequential): 	slice_start .. slice_end
	// RSEQ (reverse seq): 	slice_end .. slice_start
	// ALT (alternated): 	slice_start, slice_start+2, .. slice_end|slice_end-1,
	// 						slice_start+1 .. slice_end|slice_end-1
	// RALT (reverse alt): 	slice_end, slice_end-2, .. slice_start|slice_start+1,
	// 						slice_end-1 .. slice_start|slice_start+1
	// ALT_SHFT (siemens alt):slice_start+1, slice_start+3, .. slice_end|slice_end-1,
	// 						slice_start .. slice_end|slice_end-1
	// RALT (reverse alt): 	slice_end-1, slice_end-3, .. slice_start|slice_start+1,
	// 						slice_end-2 .. slice_start|slice_start+1
	SliceOrderT m_slice_order;

	friend std::ostream& operator<<(std::ostream &out, const MRImage& img);

protected:
	virtual int writeNifti1Image(gzFile file) const = 0;
	virtual int writeNifti2Image(gzFile file) const = 0;
	
	virtual void updateAffine() = 0;

	
	friend shared_ptr<MRImage> readNiftiImage(gzFile file, bool verbose);
};

/**
 * @brief MRImageStore is a version of NDArray that has an orientation matrix.
 * Right now it also has additional data that is unique to nifti. Eventually
 * this class will be forked into a subclass, and this will only have the 
 * orientation.
 *
 * @tparam D 	Number of dimensions
 * @tparam T	Pixel type
 */
template <size_t D, typename T>
class MRImageStore :  public virtual NDArrayStore<D,T>, public virtual MRImage
{

public:

	/**
	 * @brief Create an image with default orientation, of the specified size
	 *
	 * @param dim	number of image dimensions
	 * @param size	vector of size dim, with the image size
	 * @param orient orientation
	 */

	MRImageStore(std::initializer_list<size_t> a_args);
	MRImageStore(const std::vector<size_t>& a_args);
	
	/*************************************************************************
	 * Coordinate Transform Functions
	 ************************************************************************/
	void orientDefault();
	void updateAffine();
	void setOrient(const MatrixP& orig, const MatrixP& space, 
			const MatrixP& dir, bool reinit);
	void setOrigin(const MatrixP& orig, bool reinit = false);
	void setSpacing(const MatrixP& space, bool reinit = false);
	void setDirection(const MatrixP& dir, bool reinit = false);
	
	void printSelf();
	
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
	 * @param xyz Corresponding coordinate, rounded to nearest integer
	 *
	 * @return 0
	 */
	virtual int pointToIndex(size_t len, const double* ras, int64_t* index) const;
	
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
	 * @brief Returns a matrix (single column) with spacing information. The 
	 * number of rows is equal to the number of dimensions in the image.
	 *
	 * @return Reference to spacing matrix
	 */
	MatrixP& spacing() {return *((MatrixP*)&m_space); };
	
	/**
	 * @brief Returns a matrix (single column) with origin (RAS coordinate of 
	 * index 0,0,0). The number of rows is equal to the number of dimensions in
	 * the image.
	 *
	 * @return Reference to origin matrix
	 */
	MatrixP& origin() {return *((MatrixP*)&m_origin); };
	
	/**
	 * @brief Returns a square matrix with direction, which is the rotation off
	 * the indices to +R +A +S. 
	 *
	 * @return Reference to direction matrix
	 */
	MatrixP& direction() {return *((MatrixP*)&m_dir); };


//	MatrixP& affine() {return *((MatrixP*)&m_affine); };
//	MatrixP& iaffine() {return *((MatrixP*)&m_inv_affine); };
	
	/**
	 * @brief Returns a matrix (single column) with spacing information. The 
	 * number of rows is equal to the number of dimensions in the image.
	 *
	 * @return Reference to spacing matrix
	 */
	const MatrixP& spacing() const {return *((MatrixP*)&m_space); };
	
	/**
	 * @brief Returns a matrix (single column) with origin (RAS coordinate of 
	 * index 0,0,0). The number of rows is equal to the number of dimensions in
	 * the image.
	 *
	 * @return Reference to origin matrix
	 */
	const MatrixP& origin() const {return *((MatrixP*)&m_origin); };
	
	/**
	 * @brief Returns a square matrix with direction, which is the rotation off
	 * the indices to +R +A +S. 
	 *
	 * @return Reference to direction matrix
	 */
	const MatrixP& direction() const {return *((MatrixP*)&m_dir); };
	
	/**
	 * @brief Returns a square matrix that may be used to convert an index to 
	 * a point in the coordinate system of real RAS space (rather than index
	 * space).
	 *
	 * @return Reference to affine matrix
	 */
	const MatrixP& affine() const {return *((MatrixP*)&m_affine); };

	/**
	 * @brief Returns a square matrix that may be used to convert a point in
	 * the coordinate system of real RAS space to index space.
	 *
	 * @return Reference to inverse affine matrix
	 */
	const MatrixP& iaffine() const {return *((MatrixP*)&m_inv_affine); };

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
	 * @brief Create an exact copy of the current image object, and return
	 * a pointer to it.
	 *
	 * @return Pointer to exact duplicate of current image.
	 */
	std::shared_ptr<MRImage> cloneImage() const;
protected:
	// used to transform index to RAS (Right Handed Coordinate System)

	/**
	 * @brief Raw direction matrix
	 */
	Matrix<D,D> m_dir;

	/**
	 * @brief Raw spacing matrix
	 */
	Matrix<D,1> m_space;

	/**
	 * @brief Raw origin
	 */
	Matrix<D,1> m_origin;

	/**
	 * @brief String indicating units
	 */
	std::string m_units[D];
	
	/**
	 * @brief Matrix which transforms an index vector into a point vector
	 */
	Matrix<D+1,D+1> m_affine;


	/**
	 * @brief Matrix which transforms a space vector to an index vector.
	 */
	Matrix<D+1,D+1> m_inv_affine;
	
	int writeNifti1Image(gzFile file) const;
	int writeNifti2Image(gzFile file) const;
	int writeNifti1Header(gzFile file) const;
	int writeNifti2Header(gzFile file) const;
	int writePixels(gzFile file) const;
};

} // npl
#endif 
