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
 * @file ndarray.h
 *
 *****************************************************************************/

/******************************************************************************
 * @file ndarray.h
 * @brief This file contains the definition for NDarray and its derived types.
 * The derived types are templated over dimensionality and pixel type.
 ******************************************************************************/

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

using std::shared_ptr;

/******************************************************************************
 * Define Types
 *****************************************************************************/
enum PixelT {UNKNOWN_TYPE=0, UINT8=2, INT16=4, INT32=8, FLOAT32=16,
	COMPLEX64=32, FLOAT64=64, RGB24=128, INT8=256, UINT16=512, UINT32=768,
	INT64=1024, UINT64=1280, FLOAT128=1536, COMPLEX128=1792, COMPLEX256=2048,
	RGBA32=2304 };

class NDArray;

/******************************************************************************
 * Basic Functions.
 ******************************************************************************/

/**
 * @brief Creates a new NDArray with dimensions set by ndim, and size set by
 * size. Output pixel type is decided by ptype variable.
 *
 * @param ndim number of image dimensions
 * @param size size of image, in each dimension
 * @param ptype Pixel type npl::PixelT
 *
 * @return New image, default orientation
 */
shared_ptr<NDArray> createNDArray(size_t ndim, const size_t* size, PixelT ptype);

/**
 * @brief Creates a new NDArray with dimensions set by ndim, and size set by
 * size. Output pixel type is decided by ptype variable.
 *
 * @param size size of image, in each dimension, number of dimensions decied by
 * length of size vector
 * @param ptype Pixel type npl::PixelT
 *
 * @return New image, default orientation
 */
shared_ptr<NDArray> createNDArray(const std::vector<size_t>& dim, PixelT ptype);

/******************************************************************************
 * Classes.
 ******************************************************************************/

/**
 * @brief Pure virtual interface to interact with an ND array
 */
class NDArray : public std::enable_shared_from_this<NDArray>
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
	
	shared_ptr<NDArray> getPtr()  {
		return shared_from_this();
	};
	
	shared_ptr<const NDArray> getConstPtr() const {
		return shared_from_this();
	};

	virtual void* data() = 0;
	virtual const void* data() const = 0;

	/**
	 * @brief Performs a deep copy of the entire array.
	 *
	 * @return Copied array.
	 */
	virtual shared_ptr<NDArray> copy() const = 0;

	/**
	 * @brief Create a new array that is a copy of the input, possibly with new
	 * dimensions and pixeltype. The new array will have all overlapping pixels
	 * copied from the old array.
	 *
	 * This function just calls the outside copyCast, the reason for this
	 * craziness is that making a template function nested in the already
	 * huge number of templates I have kills the compiler, so we call an
	 * outside function that calls templates that has all combinations of D,T.
	 *
	 * @param in Input array, anything that can be copied will be
	 * @param newdims Number of dimensions in output array
	 * @param newsize Size of output array
	 * @param newtype Type of pixels in output array
	 *
	 * @return Image with overlapping sections cast and copied from 'in'
	 */
	virtual shared_ptr<NDArray> copyCast(size_t newdims, const size_t* newsize,
			PixelT newtype) const = 0;

	/**
	 * @brief Create a new array that is a copy of the input, with same dimensions
	 * but pxiels cast to newtype. The new array will have all overlapping pixels
	 * copied from the old array.
	 *
	 * This function just calls the outside copyCast, the reason for this
	 * craziness is that making a template function nested in the already
	 * huge number of templates I have kills the compiler, so we call an
	 * outside function that calls templates that has all combinations of D,T.
	 *
	 * @param in Input array, anything that can be copied will be
	 * @param newtype Type of pixels in output array
	 *
	 * @return Image with overlapping sections cast and copied from 'in'
	 */
	virtual shared_ptr<NDArray> copyCast(PixelT newtype) const = 0;

	/**
	 * @brief Create a new array that is a copy of the input, possibly with new
	 * dimensions or size. The new array will have all overlapping pixels
	 * copied from the old array. The new array will have the same pixel type as
	 * the input array
	 *
	 * This function just calls the outside copyCast, the reason for this
	 * craziness is that making a template function nested in the already
	 * huge number of templates I have kills the compiler, so we call an
	 * outside function that calls templates that has all combinations of D,T.
	 *
	 * @param in Input array, anything that can be copied will be
	 * @param newdims Number of dimensions in output array
	 * @param newsize Size of output array
	 *
	 * @return Image with overlapping sections cast and copied from 'in'
	 */
	virtual shared_ptr<NDArray> copyCast(size_t newdims,
				const size_t* newsize) const = 0;

	
//	virtual int opself(const NDArray* right, double(*func)(double,double),
//			bool elevR) = 0;
//	virtual std::shared_ptr<NDArray> opnew(const NDArray* right,
//			double(*func)(double,double), bool elevR) = 0;

	virtual void* __getAddr(std::initializer_list<int64_t> index) const = 0;
	virtual void* __getAddr(size_t len, const int64_t* index) const = 0;
	virtual void* __getAddr(const std::vector<int64_t>& index) const = 0;
	virtual void* __getAddr(int64_t i) const = 0;
	virtual void* __getAddr(int64_t x, int64_t y, int64_t z, int64_t t) const = 0;
	
	virtual int64_t getLinIndex(std::initializer_list<int64_t> index) const = 0;
	virtual int64_t getLinIndex(size_t len, const int64_t* index) const = 0;
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
template <size_t D, typename T>
class NDArrayStore : public virtual NDArray
{
public:
	/**
	 * @brief Constructor with initializer list. Orientation will be default
	 * (direction = identity, spacing = 1, origin = 0).
	 *
	 * @param a_args dimensions of input, the length of this initializer list
	 * may not be fully used if a_args is longer than D. If it is shorter
	 * then D then additional dimensions are left as size 1.
	 */
	NDArrayStore(const std::initializer_list<size_t>& dim);

	/**
	 * @brief Constructor with vector. Orientation will be default
	 * (direction = identity, spacing = 1, origin = 0).
	 *
	 * @param a_args dimensions of input, the length of this initializer list
	 * may not be fully used if a_args is longer than D. If it is shorter
	 * then D then additional dimensions are left as size 1.
	 */
	NDArrayStore(const std::vector<size_t>& dim);
	
	/**
	 * @brief Constructor with array of length len, Orientation will be default
	 * (direction = identity, spacing = 1, origin = 0).
	 *
	 * @param len Length of array 'size'
	 * @param size dimensions of input, the length of this initializer list
	 * may not be fully used if a_args is longer than D. If it is shorter
	 * then D then additional dimensions are left as size 1.
	 */
	NDArrayStore(size_t len, const size_t* dim);
	
	/**
	 * @brief Constructor which uses a preexsting array, to graft into the
	 * array. No new allocation will be performed, however ownership of the
	 * array will be taken, meaning it could be deleted anytime after this
	 * constructor completes.
	 *
	 * @param len Length of array 'size'
	 * @param size dimensions of input, the length of this initializer list
	 * may not be fully used if a_args is longer than D. If it is shorter
	 * then D then additional dimensions are left as size 1.
	 * @param ptr Pointer to data array, should be allocated with new, and
	 * size should be exactly sizeof(T)*size[0]*size[1]*...*size[len-1]
	 */
	NDArrayStore(size_t len, const size_t* dim, T* ptr);
	
	~NDArrayStore() { delete[] _m_data; };

	/*
	 * get / set functions
	 */
	T& operator[](const std::vector<int64_t>& index);
	T& operator[](std::initializer_list<int64_t> index);
	T& operator[](int64_t pixel);
	
	const T& operator[](const std::vector<int64_t>& index) const;
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

	virtual void resize(const size_t dim[D]);

	// return the pixel type
	virtual PixelT type() const;
	
	/**
	 * @brief Returns a pointer to the data array. Be careful
	 *
	 * @return Pointer to data
	 */
	void* data() {return _m_data; };
	
	/**
	 * @brief Returns a pointer to the data array. Be careful
	 *
	 * @return Pointer to data
	 */
	const void* data() const { return _m_data; };

	/**
	 * @brief Grafts data of the given dimensions into the image, effectively
	 * changing the image size.
	 *
	 * @param dim[D] Dimensions of image
	 * @param ptr Pointer to data which we will take control of
	 */
	void graft(const size_t dim[D], T* ptr);
	
	/**************************************************************************
	 * Duplication Functions
	 *************************************************************************/
	
	/**
	 * @brief Performs a deep copy of the entire array
	 *
	 * @return Copied array.
	 */
	virtual shared_ptr<NDArray> copy() const;

	/**
	 * @brief Create a new array that is a copy of the input, possibly with new
	 * dimensions and pixeltype. The new array will have all overlapping pixels
	 * copied from the old array.
	 *
	 * This function just calls the outside copyCast, the reason for this
	 * craziness is that making a template function nested in the already
	 * huge number of templates I have kills the compiler, so we call an
	 * outside function that calls templates that has all combinations of D,T.
	 *
	 * @param in Input array, anything that can be copied will be
	 * @param newdims Number of dimensions in output array
	 * @param newsize Size of output array
	 * @param newtype Type of pixels in output array
	 *
	 * @return Image with overlapping sections cast and copied from 'in'
	 */
	virtual shared_ptr<NDArray> copyCast(size_t newdims, const size_t* newsize,
			PixelT newtype) const;

	/**
	 * @brief Create a new array that is a copy of the input, with same dimensions
	 * but pxiels cast to newtype. The new array will have all overlapping pixels
	 * copied from the old array.
	 *
	 * This function just calls the outside copyCast, the reason for this
	 * craziness is that making a template function nested in the already
	 * huge number of templates I have kills the compiler, so we call an
	 * outside function that calls templates that has all combinations of D,T.
	 *
	 * @param in Input array, anything that can be copied will be
	 * @param newtype Type of pixels in output array
	 *
	 * @return Image with overlapping sections cast and copied from 'in'
	 */
	virtual shared_ptr<NDArray> copyCast(PixelT newtype) const;

	/**
	 * @brief Create a new array that is a copy of the input, possibly with new
	 * dimensions or size. The new array will have all overlapping pixels
	 * copied from the old array. The new array will have the same pixel type as
	 * the input array
	 *
	 * This function just calls the outside copyCast, the reason for this
	 * craziness is that making a template function nested in the already
	 * huge number of templates I have kills the compiler, so we call an
	 * outside function that calls templates that has all combinations of D,T.
	 *
	 * @param in Input array, anything that can be copied will be
	 * @param newdims Number of dimensions in output array
	 * @param newsize Size of output array
	 *
	 * @return Image with overlapping sections cast and copied from 'in'
	 */
	virtual shared_ptr<NDArray> copyCast(size_t newdims, const size_t* newsize) const;

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
	
	inline virtual void* __getAddr(size_t len, const int64_t* index) const
	{
		return &_m_data[getLinIndex(len, index)];
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
	virtual int64_t getLinIndex(size_t len, const int64_t* index) const;
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
	
	T* _m_data;
	size_t _m_stride[D]; // steps between pixels
	size_t _m_dim[D];	// overall image dimension

	protected:

	void updateStrides();
	
};


#undef VIRTGETSET
#undef GETSET

} //npl

#endif
