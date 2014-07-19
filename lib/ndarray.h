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

#ifndef NDARRAY_H
#define NDARRAY_H

#include "npltypes.h"

#include <cstddef>
#include <cmath>
#include <initializer_list>
#include <vector>
#include <cstdint>
#include <complex>
#include <cassert>
#include <memory>


namespace npl {

// Match Nifti Codes
enum PixelT {UNKNOWN_TYPE=0, UINT8=2, INT16=4, INT32=8, FLOAT32=16,
	COMPLEX64=32, FLOAT64=64, RGB24=128, INT8=256, UINT16=512, UINT32=768,
	INT64=1024, UINT64=1280, FLOAT128=1536, COMPLEX128=1792, COMPLEX256=2048,
	RGBA32=2304 };

/**
 * @brief Pure virtual interface to interact with an ND array
 */
class NDArray
{
public:
	/*
	 * get / set functions
	 */
	// Get Address

	virtual size_t ndim() const = 0;
	virtual size_t bytes() const = 0;
	virtual size_t elements() const = 0;
	virtual size_t dim(size_t dir) const = 0;
	virtual const size_t* dim() const = 0;

	// return type of stored value
	virtual PixelT type() const = 0;
	
//	virtual int opself(const NDArray* right, double(*func)(double,double), 
//			bool elevR) = 0;
//	virtual std::shared_ptr<NDArray> opnew(const NDArray* right, 
//			double(*func)(double,double), bool elevR) = 0;

	virtual void* __getAddr(std::initializer_list<int64_t> index) const = 0;
	virtual void* __getAddr(const int64_t* index) const = 0;
	virtual void* __getAddr(const std::vector<int64_t>& index) const = 0;
	virtual void* __getAddr(int64_t i) const = 0;
	virtual void* __getAddr(int64_t x, int64_t y, int64_t z, int64_t t) const = 0;
	
	virtual int64_t getLinIndex(std::initializer_list<int64_t> index) const = 0;
	virtual int64_t getLinIndex(const int64_t* index) const = 0;
	virtual int64_t getLinIndex(const std::vector<int64_t>& index) const = 0;
	virtual int64_t getLinIndex(int64_t x, int64_t y, int64_t z, int64_t t) const = 0;
	
	/**
	 * @brief This function just returns the number of elements in a theoretical
	 * fourth dimension (ignoring orgnaization of higher dimensions)
	 *
	 * @return number of elements in the 4th or greater dimensions
	 */
	virtual int64_t tlen() = 0;

protected:
	NDArray() {} ;

};


/**
 * @brief Basic storage unity for ND array. Creates a big chunk of memory.
 *
 * @tparam D dimension of array
 * @tparam T type of sample
 */
template <int D, typename T>
class NDArrayStore : public virtual NDArray
{
public:
	NDArrayStore(const std::vector<size_t>& dim);
	
	~NDArrayStore() { delete[] _m_data; };

	/* 
	 * get / set functions
	 */
	T& operator[](const std::vector<int64_t>& index);
	T& operator[](const int64_t* index);
	T& operator[](std::initializer_list<int64_t> index);
	T& operator[](int64_t pixel);
	
	const T& operator[](const std::vector<int64_t>& index) const;
	const T& operator[](const int64_t* index) const;
	const T& operator[](std::initializer_list<int64_t> index) const;
	const T& operator[](int64_t pixel) const;

	/* 
	 * General Information 
	 */
	virtual size_t ndim() const;
	virtual size_t bytes() const;
	virtual size_t elements() const;
	virtual size_t dim(size_t dir) const;
	virtual const size_t* dim() const;

	virtual void resize(size_t dim[D]);

	// return the pixel type
	virtual PixelT type() const;

	/* 
	 * Higher Level Operations
	 */
//	virtual int opself(const NDArray* right, double(*func)(double,double), 
//			bool elevR);
//	virtual std::shared_ptr<NDArray> opnew(const NDArray* right, 
//			double(*func)(double,double), bool elevR);
	
	inline virtual void* __getAddr(std::initializer_list<int64_t> index) const 
	{
		return &_m_data[getLinIndex(index)];
	};

	inline virtual void* __getAddr(const int64_t* index) const
	{
		return &_m_data[getLinIndex(index)];
	};

	inline virtual void* __getAddr(const std::vector<int64_t>& index) const
	{
		return &_m_data[getLinIndex(index)];
	};

	inline virtual void* __getAddr(int64_t i) const 
	{
		return &_m_data[i]; 
	};
	inline virtual void* __getAddr(int64_t x, int64_t y, int64_t z, int64_t t) const
	{
		return &_m_data[getLinIndex(x,y,z,t)]; 
	};

	virtual int64_t getLinIndex(std::initializer_list<int64_t> index) const;
	virtual int64_t getLinIndex(const int64_t* index) const;
	virtual int64_t getLinIndex(const std::vector<int64_t>& index) const;
	virtual int64_t getLinIndex(int64_t x, int64_t y, int64_t z, int64_t t) const;

	/**
	 * @brief This function just returns the number of elements in a theoretical
	 * fourth dimension (ignoring orgnaization of higher dimensions)
	 *
	 * @return number of elements in the 4th or greater dimensions
	 */
	virtual int64_t tlen() {
		if(D >= 3)
			return _m_stride[2];
		else 
			return 1;
	};

	protected:
	size_t _m_stride[D]; // steps between pixels

	T* _m_data;
	size_t _m_dim[D];	// overall image dimension

	void updateStrides();
	
};

/**
 * @brief Returns whether two NDArrays have the same dimensions, and therefore
 * can be element-by-element compared/operated on. elL is set to true if left
 * is elevatable to right (ie all dimensions match or are missing or are unary).
 * elR is the same but for the right. 
 *
 * Strictly R is elevatable if all dimensions that don't match are missing or 1
 * Strictly L is elevatable if all dimensions that don't match are missing or 1
 *
 * Examples of *elR = true (return false):
 *
 * left = [10, 20, 1]
 * right = [10, 20, 39]
 *
 * left = [10]
 * right = [10, 20, 39]
 *
 * Examples where neither elR or elL (returns true):
 *
 * left = [10, 20, 39]
 * right = [10, 20, 39]
 *
 * Examples where neither elR or elL (returns false):
 *
 * left = [10, 20, 9]
 * right = [10, 20, 39]
 *
 * left = [10, 1, 9]
 * right = [10, 20, 1]
 *
 * @param left	NDArray input
 * @param right NDArray input
 * @param elL Whether left is elevatable to right (see description of function)
 * @param elR Whether right is elevatable to left (see description of function)
 *
 * @return 
 */
bool comparable(const NDArray* left, const NDArray* right, 
		bool* elL = NULL, bool* elR = NULL)
{
	bool ret = true;

	bool rightEL = true;
	bool leftEL = true;

	for(size_t ii = 0; ii < left->ndim(); ii++) {
		if(ii < right->ndim()) {
			if(right->dim(ii) != left->dim(ii)) {
				ret = false;
				// if not 1, then R is not elevateable
				if(right->dim(ii) != 1)
					rightEL = false;
			}
		}
	}
	
	for(size_t ii = 0; ii < right->ndim(); ii++) {
		if(ii < left->ndim()) {
			if(right->dim(ii) != left->dim(ii)) {
				ret = false;
				// if not 1, then R is not elevateable
				if(left->dim(ii) != 1)
					leftEL = false;
			}
		}
	}
	
	if(ret) {
		leftEL = false;
		rightEL = false;
	}

	if(elL) *elL = leftEL;
	if(elR) *elR = rightEL;

	return ret;
}


#undef VIRTGETSET
#undef GETSET

} //npl

#endif
