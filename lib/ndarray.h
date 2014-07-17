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

template <typename T> class NDAccess;

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
	
	virtual int64_t getLinIndex(std::initializer_list<int64_t> index) const = 0;
	virtual int64_t getLinIndex(const int64_t* index) const = 0;
	virtual int64_t getLinIndex(const std::vector<int64_t>& index) const = 0;

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

	virtual int64_t getLinIndex(std::initializer_list<int64_t> index) const;
	virtual int64_t getLinIndex(const int64_t* index) const;
	virtual int64_t getLinIndex(const std::vector<int64_t>& index) const;

	protected:
	T* _m_data;
	size_t _m_dim[D];	// overall image dimension
	size_t _m_stride[D]; // steps between pixels

	void updateStrides();
	
};

template<typename T>
class NDAccess
{
public:
	NDAccess(std::shared_ptr<NDArray> in) : parent(in)
	{
		switch(in->type()) {
			case UINT8:
				castget = castgetStatic<uint8_t>;
				castset = castsetStatic<uint8_t>;
				break;
			case INT8:
				castget = castgetStatic<int8_t>;
				castset = castsetStatic<int8_t>;
				break;
			case UINT16:
				castget = castgetStatic<uint16_t>;
				castset = castsetStatic<uint16_t>;
				break;
			case INT16:
				castget = castgetStatic<int16_t>;
				castset = castsetStatic<int16_t>;
				break;
			case UINT32:
				castget = castgetStatic<uint32_t>;
				castset = castsetStatic<uint32_t>;
				break;
			case INT32:
				castget = castgetStatic<int32_t>;
				castset = castsetStatic<int32_t>;
				break;
			case UINT64:
				castget = castgetStatic<uint64_t>;
				castset = castsetStatic<uint64_t>;
				break;
			case INT64:
				castget = castgetStatic<int64_t>;
				castset = castsetStatic<int64_t>;
				break;
			case FLOAT32:
				castget = castgetStatic<float>;
				castset = castsetStatic<float>;
				break;
			case FLOAT64:
				castget = castgetStatic<double>;
				castset = castsetStatic<double>;
				break;
			case FLOAT128:
				castget = castgetStatic<long double>;
				castset = castsetStatic<long double>;
				break;
			case COMPLEX64:
				castget = castgetStatic<cfloat_t>;
				castset = castsetStatic<cfloat_t>;
				break;
			case COMPLEX128:
				castget = castgetStatic<cdouble_t>;
				castset = castsetStatic<cdouble_t>;
				break;
			case COMPLEX256:
				castget = castgetStatic<cquad_t>;
				castset = castsetStatic<cquad_t>;
				break;
			case RGB24:
				castget = castgetStatic<rgb_t>;
				castset = castsetStatic<rgb_t>;
				break;
			case RGBA32:
				castget = castgetStatic<rgba_t>;
				castset = castsetStatic<rgba_t>;
				break;
			default:
			case UNKNOWN_TYPE:
				castget = castgetStatic<uint8_t>;
				castset = castsetStatic<uint8_t>;
				throw std::invalid_argument("Unknown type to NDAccess");
				break;
		}
	};
	
	/**
	 * @brief Gets value linear position in array, then casts to T
	 *
	 * @return value
	 */
	T operator[](int64_t index)
	{
		return castget(parent->__getAddr(index)); 
	};
	
	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @return value
	 */
	T get(const std::vector<int64_t>& index)
	{
		return castget(parent->__getAddr(index)); 
	};
	
	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @return value
	 */
	T operator[](const std::vector<int64_t>& index)
	{
		return castget(parent->__getAddr(index)); 
	};
	
	/**
	 * @brief Casts to the appropriate type then sets array at given index.
	 *
	 * @return current value
	 */
	void set(const std::vector<int64_t>& index, T v)
	{
		return castset(parent->__getAddr(index), v); 
	};
	
	/**
	 * @brief Casts to the appropriate type then sets array at given index.
	 *
	 * @return current value
	 */
	void set(int64_t index, T v)
	{
		return castset(parent->__getAddr(index), v); 
	};
	
private:
	

	/**
	 * @brief This is a wrapper function that will be called to safely cast 
	 * from the underlying type.
	 *
	 * @tparam U Underlying type of pixel, figured out in the constructor
	 * @param ptr Pointer to memory where the pixel is.
	 *
	 * @return Correctly cast value
	 */
	template <typename U>
	static T castgetStatic(void* ptr)
	{
		return (T)(*(static_cast<U*>(ptr)));
	};
	
	/**
	 * @brief This is a wrapper function that will be called to safely cast 
	 * to the underlying type.
	 *
	 * @tparam U Underlying type of pixel, figured out in the constructor
	 * @param ptr Pointer to memory where the pixel is.
	 * @param val new value to write
	 *
	 */
	template <typename U>
	static void castsetStatic(void* ptr, const T& val)
	{
		(*(static_cast<U*>(ptr))) = (U)val;
	};

	/**
	 * @brief Where to get the dat a from. Also the shared_ptr prevents dealloc
	 */
	std::shared_ptr<NDArray> parent;

	/**
	 * @brief Function pointer to the correct function for casting from the 
	 * underlying type
	 *
	 * @param ptr location in memory where the pixel is stored
	 */
	T (*castget)(void* ptr);

	/**
	 * @brief Function pointer to the correct function for casting to the 
	 * underlying type. This should be set during construction.
	 *
	 *
	 * @param ptr pointer to memory 
	 * @param v value to cast and write
	 */
	void (*castset)(void* ptr, const T& v);
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
