/*******************************************************************************
This file is part of Neural Program Library (NPL), 

Written and Copyrighted by by Micah C. Chambers (micahc.vt@gmail.com)

The Neural Program Library is free software: you can redistribute it and/or
modify it under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your option)
any later version.

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
	
// Match Nifti Codes
enum PixelT {UNKNOWN_TYPE=0, UINT8=2, INT16=4, INT32=8, FLOAT32=16, COMPLEX64=32,
	FLOAT64=64, RGB24=128, INT8=256, UINT16=512, UINT32=768, INT64=1024,
	UINT64=1280, FLOAT128=1536, COMPLEX128=1792,
	COMPLEX256=2048, RGBA32=2304 };

enum SliceOrderT {UNKNOWN_SLICE=0, SEQ=1, RSEQ=2, ALT=3, RALT=4, ALT_SHFT=5, RALT_SHFT=6};

class MRImage;

// simply reads an image in its native type
MRImage* readNiftiImage(gzFile file, bool verbose);
MRImage* readMRImage(std::string filename, bool verbose = false);
MRImage* createMRImage(const std::vector<size_t>& dims, PixelT);
int writeMRImage(MRImage* img, std::string fn, bool nifti2 = false);

ostream& operator<<(ostream &out, const MRImage& img);

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

	virtual MatrixP& space() = 0;
	virtual MatrixP& origin() = 0;
	virtual MatrixP& direction() = 0;
	virtual MatrixP& affine() = 0;
	virtual MatrixP& iaffine() = 0;
	virtual const MatrixP& space() const = 0;
	virtual const MatrixP& origin() const  = 0;
	virtual const MatrixP& direction() const = 0;
	virtual const MatrixP& affine() const = 0;
	virtual const MatrixP& iaffine() const = 0;
	
	virtual PixelT type() const = 0;
	
	virtual int write(std::string filename, double version) const = 0;
	
	// iterators
	class iterator;
	class const_iterator;

	class iterator : public virtual NDArray::iterator {
		void index(std::vector<size_t>& ind) { 
			ind.assign(m_pos.begin(), m_pos.end());
		}
		void dblindex(std::vector<double>& ind) { 
			ind.resize(m_pos.size());
			for(size_t ii=0; ii<m_pos.size(); ii++)
				ind[ii] = m_pos[ii];
		}
		void point(std::vector<double>& ras) {
			dynamic_cast<MRImage*>(m_parent)->affine().mvproduct(m_pos, ras);
		}
	};
	
	class const_iterator : public virtual NDArray::const_iterator {
		void index(std::vector<size_t>& ind) { 
			ind.assign(m_pos.begin(), m_pos.end());
		}
		void dblindex(std::vector<double>& ind) { 
			ind.resize(m_pos.size());
			for(size_t ii=0; ii<m_pos.size(); ii++)
				ind[ii] = m_pos[ii];
		}
		void point(std::vector<double>& ras) {
			dynamic_cast<const MRImage*>(m_parent)->affine().mvproduct(m_pos, ras);
		}
	};
	
	virtual int indexToPoint(const std::vector<size_t>& xyz, std::vector<double>& ras) const = 0;
	virtual int indexToPoint(const std::vector<double>& xyz, std::vector<double>& ras) const = 0;
	virtual int pointToIndex(const std::vector<double>& ras, std::vector<double>& xyz) const = 0;

//	virtual std::shared_ptr<MRImage> cloneImage() const = 0;
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

	friend ostream& operator<<(ostream &out, const MRImage& img);

protected:
	virtual int writeNifti1Image(gzFile file) const = 0;
	virtual int writeNifti2Image(gzFile file) const = 0;
	
	virtual void updateAffine() = 0;
	
	friend MRImage* readNiftiImage(gzFile file, bool verbose);
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
template <int D, typename T>
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
	
	std::shared_ptr<NDArray> clone() const;

	/*************************************************************************
	 * Coordinate Transform Functions
	 ************************************************************************/
	void orientDefault();
	void updateAffine();
	void printSelf();
	PixelT type() const;
	
	int indexToPoint(const std::vector<size_t>& index, std::vector<double>& rast) const;
	int indexToPoint(const std::vector<double>& index, std::vector<double>& rast) const;
	int pointToIndex(const std::vector<double>& rast, std::vector<double>& index) const;

	//double& space(size_t d);
	//double& origin(size_t d);
	//double& direction(size_t d1, size_t d2);
	//double& affine(size_t d1, size_t d2);
	//const double& space(size_t d) const;
	//const double& origin(size_t d) const;
	//const double& direction(size_t d1, size_t d2) const;
	//const double& affine(size_t d1, size_t d2) const;
	
	MatrixP& space() {return *((MatrixP*)&m_space); };
	MatrixP& origin() {return *((MatrixP*)&m_origin); };
	MatrixP& direction() {return *((MatrixP*)&m_dir); };
	MatrixP& affine() {return *((MatrixP*)&m_affine); };
	MatrixP& iaffine() {return *((MatrixP*)&m_inv_affine); };
	const MatrixP& space() const {return *((MatrixP*)&m_space); };
	const MatrixP& origin() const {return *((MatrixP*)&m_origin); };
	const MatrixP& direction() const {return *((MatrixP*)&m_dir); };
	const MatrixP& affine() const {return *((MatrixP*)&m_affine); };
	const MatrixP& iaffine() const {return *((MatrixP*)&m_inv_affine); };

	std::string getUnits(size_t d1, double d2);

	int write(std::string filename, double version) const;
protected:
	// used to transform index to RAS (Right Handed Coordinate System)
	Matrix<D,D> m_dir;
	Matrix<D,1> m_space;
	Matrix<D,1> m_origin;
	std::string m_units[D];
	
	// chache of the affine index -> RAS (right handed coordiante system)
	Matrix<D+1,D+1> m_affine;
	Matrix<D+1,D+1> m_inv_affine;
	
	int writeNifti1Image(gzFile file) const;
	int writeNifti2Image(gzFile file) const;
	int writeNifti1Header(gzFile file) const;
	int writeNifti2Header(gzFile file) const;
	int writePixels(gzFile file) const;
};

} // npl
#endif 
