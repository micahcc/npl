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


// virtual get and set function macro, ie
// VIRTGETSET(double, dbl); 
// generates:
// double getdbl(std::initializer_list<size_t> index) const;
// void setdbl(std::initializer_list<size_t> index, double newval) const;
// double getdbl(const std::vector<size_t>& index) const;
// void setdbl(const std::vector<size_t>& index, double newval) const;
#define VIRTGETSET(TYPE, GETFUNC, SETFUNC) \
	virtual TYPE GETFUNC(std::initializer_list<int64_t> index) const = 0; \
	virtual TYPE GETFUNC(const std::vector<int64_t>& index) const = 0; \
	virtual TYPE GETFUNC(int64_t index) const = 0; \
	virtual void SETFUNC(std::initializer_list<int64_t> index, TYPE) = 0; \
	virtual void SETFUNC(const std::vector<int64_t>& index, TYPE) = 0; \
	virtual void SETFUNC(int64_t index, TYPE) = 0; \

#define GETSET(TYPE, GETFUNC, SETFUNC) \
	TYPE GETFUNC(std::initializer_list<int64_t> index) const; \
	TYPE GETFUNC(const std::vector<int64_t>& index) const; \
	TYPE GETFUNC(int64_t index) const; \
	void SETFUNC(std::initializer_list<int64_t> index, TYPE); \
	void SETFUNC(const std::vector<int64_t>& index, TYPE); \
	void SETFUNC(int64_t index, TYPE); \

//	void SETFUNC(const int64_t* index, TYPE); 
//	TYPE GETFUNC(const int64_t* index) const; 
//	virtual TYPE GETFUNC(const int64_t* index) const = 0; 
//	virtual void SETFUNC(const int64_t* index, TYPE) = 0; 
//

namespace npl {

// Match Nifti Codes
enum PixelT {UNKNOWN_TYPE=0, UINT8=2, INT16=4, INT32=8, FLOAT32=16,
	COMPLEX64=32, FLOAT64=64, RGB24=128, INT8=256, UINT16=512, UINT32=768,
	INT64=1024, UINT64=1280, FLOAT128=1536, COMPLEX128=1792, COMPLEX256=2048,
	RGBA32=2304 };

template<typename T>
class NDAccess
{
public:
	NDAccess(std::shared_ptr<NDArray> in) : parent(in)
	{
		switch(in->type()) {
			case UINT8:
				castfunc = castor<uint8_t>;
				break;
			case INT8:
				castfunc = castor<int8_t>;
				break;
			case UINT16:
				castfunc = castor<uint16_t>;
				break;
			case INT16:
				castfunc = castor<int16_t>;
				break;
			case UINT32:
				castfunc = castor<uint32_t>;
				break;
			case INT32:
				castfunc = castor<int32_t>;
				break;
			case UINT64:
				castfunc = castor<uint64_t>;
				break;
			case INT64:
				castfunc = castor<int64_t>;
				break;
			case FLOAT32:
				castfunc = castor<float>;
				break;
			case FLOAT64:
				castfunc = castor<double>;
				break;
			case FLOAT128:
				castfunc = castor<long double>;
				break;
			case COMPLEX64:
				castfunc = castor<cfloat_t>;
				break;
			case COMPLEX128:
				castfunc = castor<cdouble_t>;
				break;
			case COMPLEX256:
				castfunc = castor<cquad_t>;
				break;
			case RGB24:
				castfunc = castor<rgb_t>;
				break;
			case RGBA32:
				castfunc = castor<rgba_t>;
				break;
			case UNKNOWN_TYPE:
				throw std::invalid_argument("Unknown type to BasicIter");
				break;
		}
	};

	/**
	 * @brief Works just like a function
	 *
	 * @return current value
	 */
	T operator()(std::initializer_list<int64_t> index)
	{
		return castfunc(parent->getAddr(index)); 
	};
	
private:
	template <typename U>
	static T castor(void* ptr)
	{
		return (T)(*((U*)ptr));
	};

	size_t i;

	std::shared_ptr<NDArray> parent;
	T (*castfunc)(void* ptr);
};

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

protected:
	virtual void* getAddr(size_t i) = 0;
	friend NDAccess;
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
	NDArrayStore(std::initializer_list<size_t> a_args);
	NDArrayStore(const std::vector<size_t>& dim);
	
	~NDArrayStore() { delete[] _m_data; };

	/* 
	 * get / set functions
	 */
	T& operator[](std::initializer_list<int64_t> index);
	T& operator[](int64_t pixel);
	
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

	protected:
	T* _m_data;
	size_t _m_dim[D];	// overall image dimension
	size_t _m_stride[D]; // steps between pixels

	void updateStrides();
	
	void* getAddr(size_t i) { return &_m_data[i]; };
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
