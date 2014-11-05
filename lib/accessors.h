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
 * @file accessors.h Provides accessors to the NDArray data structure. Due to
 * the fact that dimensionality and pixel type are not carried around in the
 * container type, it is necessary to provide generic accessors that will cast
 * input data to the correct type and return it, or take input, cast it to the
 * underlying type and set the internal pixel. While this might seem
 * round-about, it allows you to write general purpose algorithms functions
 * without concerning with dimensionality or pixel type.
 *
 *****************************************************************************/

#ifndef ACCESSORS_H
#define ACCESSORS_H

#include <stdexcept>

#include "ndarray.h"
#include "mrimage.h"
#include "basic_functions.h"
#include "utility.h"
#include "iterators.h"

namespace npl {

/** \defgroup Accessors Accessors for NDArray/Image
 *
 * Accessors are used to get and set pixel data. Since
 * the pixel type is hidden in images and arrays, accessors perform the
 * necessary casting. All Accessors have names that end with View or ConstView
 * Thus
 *
 * \code{.cpp}
 * NDView<double> dacc(img);
 * double v = dacc[index];
 *
 * NDView<std::complex<double>> cacc(img);
 * std::complex<double> c = cacc[index];
 * \endcode
 *
 * Can both be used to access the same data. Of course upon writing or if
 * you need a particular precision you should cast the image to ensure
 * that you aren't losing anything. While this may seem convoluted it allowes
 * for general purpose coding of functions without having to maintain the
 * type in every single function declaration. This is what you would
 * effectively do in C if you had a void*.
 *
 * @{
 */

/**
 * @brief This is a basic accessor class, which allows for accessing
 * array data in the type specified by the template.
 *
 * @tparam T Value to return on access
 */
template<typename T>
class NDView
{
public:
	NDView(std::shared_ptr<NDArray> in) : parent(in)
	{
		setArray(in);
	};

	NDView() : parent(NULL) {} ;

	void setArray(ptr<NDArray> in)
	{
		parent = in;
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
				throw std::invalid_argument("Unknown type to NDView");
				break;
		}
	}

	/**
	 * @brief Gets value linear position in array, then casts to T
	 *
	 * @return value
	 */
	T operator[](int64_t index)
	{
		return castget(this->parent->__getAddr(index));
	};

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @param index n-d index to access
	 *
	 * @return value
	 */
	T get(const std::vector<int64_t>& index)
	{
		return castget(this->parent->__getAddr(index));
	};

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @param len length of index array
	 * @param index n-d index to access
	 *
	 * @return value
	 */
	T get(size_t len, int64_t* index)
	{
		return castget(this->parent->__getAddr(len, index));
	};

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @param index n-d index to access
	 *
	 * @return value
	 */
	T operator[](const std::vector<int64_t>& index)
	{
		return castget(this->parent->__getAddr(index));
	};

	/**
	 * @brief Casts to the appropriate type then sets array at given index.
	 *
	 * @param len length of index array
	 * @param index n-d index to access
	 * @param v value to set at index
	 *
	 * @return current value
	 */
	void set(size_t len, const int64_t* index, T v)
	{
		return castset(this->parent->__getAddr(len, index), v);
	};

	/**
	 * @brief Casts to the appropriate type then sets array at given index.
	 *
	 * @param v value to set at index
	 * @param index n-d index to access
	 *
	 * @return current value
	 */
	void set(const std::vector<int64_t>& index, T v)
	{
		return castset(this->parent->__getAddr(index), v);
	};

	/**
	 * @brief Casts to the appropriate type then sets array at given index.
	 *
	 * @param v value to set at index
	 * @param index n-d index to access
	 *
	 * @return current value
	 */
	void set(int64_t index, T v)
	{
		return castset(this->parent->__getAddr(index), v);
	};

	int64_t tlen() { return this->parent->tlen(); };

protected:


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
 * @brief This is a basic accessor class, which allows for accessing
 * array data in the type specified by the template.
 *
 * @tparam T Value to return on access
 */
template<typename T>
class NDConstView
{
public:
	NDConstView(std::shared_ptr<const NDArray> in) : parent(in)
	{
		setArray(in);
	}

	NDConstView() : parent(NULL) {} ;

	void setArray(ptr<const NDArray> in)
	{
		parent = in;
		switch(in->type()) {
			case UINT8:
				castget = castgetStatic<uint8_t>;
				break;
			case INT8:
				castget = castgetStatic<int8_t>;
				break;
			case UINT16:
				castget = castgetStatic<uint16_t>;
				break;
			case INT16:
				castget = castgetStatic<int16_t>;
				break;
			case UINT32:
				castget = castgetStatic<uint32_t>;
				break;
			case INT32:
				castget = castgetStatic<int32_t>;
				break;
			case UINT64:
				castget = castgetStatic<uint64_t>;
				break;
			case INT64:
				castget = castgetStatic<int64_t>;
				break;
			case FLOAT32:
				castget = castgetStatic<float>;
				break;
			case FLOAT64:
				castget = castgetStatic<double>;
				break;
			case FLOAT128:
				castget = castgetStatic<long double>;
				break;
			case COMPLEX64:
				castget = castgetStatic<cfloat_t>;
				break;
			case COMPLEX128:
				castget = castgetStatic<cdouble_t>;
				break;
			case COMPLEX256:
				castget = castgetStatic<cquad_t>;
				break;
			case RGB24:
				castget = castgetStatic<rgb_t>;
				break;
			case RGBA32:
				castget = castgetStatic<rgba_t>;
				break;
			default:
			case UNKNOWN_TYPE:
				castget = castgetStatic<uint8_t>;
				throw std::invalid_argument("Unknown type to NDConstView");
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
		return castget(this->parent->__getAddr(index));
	};

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @param index n-d index to access
	 *
	 * @return value
	 */
	T get(const std::vector<int64_t>& index)
	{
		return castget(this->parent->__getAddr(index));
	};

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @param len length of index array
	 * @param index n-d index to access
	 *
	 * @return value
	 */
	T get(size_t len, int64_t* index)
	{
		return castget(this->parent->__getAddr(len, index));
	};

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @param index n-d index to access
	 *
	 * @return value
	 */
	T operator[](const std::vector<int64_t>& index)
	{
		return castget(this->parent->__getAddr(index));
	};

	int64_t tlen() { return this->parent->tlen(); };

protected:


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
	 * @brief Where to get the dat a from. Also the shared_ptr prevents dealloc
	 */
	std::shared_ptr<const NDArray> parent;

	/**
	 * @brief Function pointer to the correct function for casting from the
	 * underlying type
	 *
	 * @param ptr location in memory where the pixel is stored
	 */
	T (*castget)(void* ptr);

};

/**
 * @brief The purpose of this class is to view an image as a 3D
 * image rather than a ND image. Therefore all dimensions above the third will
 * be ignored and index 0 will be used.
 *
 * @tparam T Type of value to cast and return
 */
template<typename T>
class Pixel3DView : public NDView<T>
{
public:
	Pixel3DView(std::shared_ptr<NDArray> in) : NDView<T>(in)
	{ };

	Pixel3DView();
	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @return value
	 */
	T operator()(int64_t x=0, int64_t y=0, int64_t z=0)
	{
		return this->castget(this->parent->__getAddr(x,y,z,0));
	};

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @return value
	 */
	T get(int64_t x, int64_t y, int64_t z)
	{
		return this->castget(this->parent->__getAddr(x,y,z,0));
	};

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @return value
	 */
	void set(int64_t x, int64_t y, int64_t z, T v)
	{
		this->castset(this->parent->__getAddr(x,y,z,0), v);
	};

protected:

	// Remove functions that aren't relevent from NDView
	T operator[](int64_t i) { (void)(i); return T(); };
	T get(const std::vector<int64_t>& i) {  (void)(i); return T(); };
	T operator[](const std::vector<int64_t>& i) { (void)(i); return T(); };
};

/**
 * @brief The purpose of this class is to view an image as a 3D+vector dimension
 * image rather than a 4+D image. Therefore all dimensions above the third are
 * cast as a vector. If there is demand I may create a matrixx verion as well
 *
 * @tparam T Type of value to cast and return
 */
template<typename T>
class Vector3DConstView : public NDConstView<T>
{
public:
	Vector3DConstView(std::shared_ptr<const NDArray> in) : NDConstView<T>(in)
	{ };

	Vector3DConstView() {};

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @return value
	 */
	virtual
		T operator()(int64_t x=0, int64_t y=0, int64_t z=0, int64_t t=0)
		{
			return this->castget(this->parent->__getAddr(x,y,z,t));
		};

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @return value
	 */
	virtual
		T get(int64_t x=0, int64_t y=0, int64_t z=0, int64_t t=0)
		{
			return this->castget(this->parent->__getAddr(x,y,z,t));
		};

private:
	//////////////////////////////////////////////////////
	// Hide Non-3D Functrions from NDConstView
	//////////////////////////////////////////////////////

	/**
	 * @brief Gets value linear position in array, then casts to T
	 *
	 * @return value
	 */
	T operator[](int64_t index)
	{
		return castget(this->parent->__getAddr(index));
	};

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @param index n-d index to access
	 *
	 * @return value
	 */
	T get(const std::vector<int64_t>& index)
	{
		return castget(this->parent->__getAddr(index));
	};

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @param len length of index array
	 * @param index n-d index to access
	 *
	 * @return value
	 */
	T get(size_t len, int64_t* index)
	{
		return castget(this->parent->__getAddr(len, index));
	};

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @param index n-d index to access
	 *
	 * @return value
	 */
	T operator[](const std::vector<int64_t>& index)
	{
		return castget(this->parent->__getAddr(index));
	};

};

/**
 * @brief The purpose of this class is to view an image as a 3D+vector dimension
 * image rather than a 4+D image. Therefore all dimensions above the third are
 * cast as a vector. If there is demand I may create a matrixx verion as well
 *
 * @tparam T Type of value to cast and return
 */
template<typename T>
class Vector3DView : public NDView<T>
{
public:
	Vector3DView(std::shared_ptr<NDArray> in) : NDView<T>(in)
	{ };

	Vector3DView() {};
	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @return value
	 */
	T operator()(int64_t x=0, int64_t y=0, int64_t z=0, int64_t t=0)
	{
		return this->castget(this->parent->__getAddr(x,y,z,t));
	};

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @return value
	 */
	T get(int64_t x=0, int64_t y=0, int64_t z=0, int64_t t=0)
	{
		return this->castget(this->parent->__getAddr(x,y,z,t));
	};

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @return value
	 */
	void set(int64_t x, int64_t y, int64_t z, int64_t t, T v)
	{
		this->castset(this->parent->__getAddr(x,y,z,t), v);
	};
private:
	//////////////////////////////////////////////////////
	// Hide Non-3D Functrions from NDConstView
	//////////////////////////////////////////////////////

	/**
	 * @brief Gets value linear position in array, then casts to T
	 *
	 * @return value
	 */
	T operator[](int64_t index)
	{
		return castget(this->parent->__getAddr(index));
	};

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @param index n-d index to access
	 *
	 * @return value
	 */
	T get(const std::vector<int64_t>& index)
	{
		return castget(this->parent->__getAddr(index));
	};

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @param len length of index array
	 * @param index n-d index to access
	 *
	 * @return value
	 */
	T get(size_t len, int64_t* index)
	{
		return castget(this->parent->__getAddr(len, index));
	};

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @param index n-d index to access
	 *
	 * @return value
	 */
	T operator[](const std::vector<int64_t>& index)
	{
		return castget(this->parent->__getAddr(index));
	};

	/**
	 * @brief Casts to the appropriate type then sets array at given index.
	 *
	 * @param len length of index array
	 * @param index n-d index to access
	 * @param v value to set at index
	 *
	 * @return current value
	 */
	void set(size_t len, const int64_t* index, T v)
	{
		return castset(this->parent->__getAddr(len, index), v);
	};

	/**
	 * @brief Casts to the appropriate type then sets array at given index.
	 *
	 * @param v value to set at index
	 * @param index n-d index to access
	 *
	 * @return current value
	 */
	void set(const std::vector<int64_t>& index, T v)
	{
		return castset(this->parent->__getAddr(index), v);
	};

	/**
	 * @brief Casts to the appropriate type then sets array at given index.
	 *
	 * @param v value to set at index
	 * @param index n-d index to access
	 *
	 * @return current value
	 */
	void set(int64_t index, T v)
	{
		return castset(this->parent->__getAddr(index), v);
	};

};

/****************************************
 * Interpolators
 ***************************************/

////////////////
//Linear
///////////////

/* Linear Kernel Sampling */
double linKern(double x)
{
	return fabs(1-fmin(1,fabs(x)));
}

/**
 * @brief The purpose of this class is to view an image as a continuous
 * ND image and to sample at a continuous ND-position within.
 *
 * @tparam T Type of value to cast and return
 */
template<typename T>
class LinInterpNDView : public NDConstView<T>
{
public:
	LinInterpNDView(std::shared_ptr<const NDArray> in,
			BoundaryConditionT bound = ZEROFLUX)
		: NDConstView<T>(in), m_boundmethod(bound), m_ras(false)
	{ };

	LinInterpNDView() : m_boundmethod(ZEROFLUX), m_ras(false) {} ;

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @param x	x-dimension
	 * @param y	y-dimension
	 * @param z	z-dimension
	 * @param t	4th dimension
	 * @param u	5th dimension
	 * @param v	6th dimension
	 * @param w	7th dimension
	 * @param q	8th dimension
	 *
	 * @return value Interpolated value at given position
	 */
	T operator()(double x=0, double y=0, double z=0, double t=0, double u = 0,
			double v = 0, double w = 0, double q = 0)
	{
		double tmp[8] = {x,y,z,t,u,v,w,q};
		return get(8, tmp);
	};


	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @param index  Position in ND-space to interpolate
	 *
	 * @return value Interpolated value at given position
	 */
	T operator()(const std::vector<float>& index)
	{
		assert(index.size() <= 8);
		double tmp[8];
		size_t ii=0;
		for(auto it = index.begin(); it != index.end() && ii<8; ++it, ++ii)
			tmp[ii] = *it;
		return get(std::min(8UL, index.size()), tmp);
	};

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @param index  Position in ND-space to interpolate
	 *
	 * @return value Interpolated value at given position
	 */
	T operator()(const std::vector<double>& index)
	{
		return get(index.size(), index.data());
	};

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @param index  Position in ND-space to interpolate
	 *
	 * @return value Interpolated value at given position
	 */
	T operator()(std::initializer_list<double> index)
	{
		assert(index.size() <= 8);
		double tmp[8];
		size_t ii=0;
		for(auto it = index.begin(); it != index.end() && ii<8; ++it, ++ii)
			tmp[ii] = *it;
		return get(std::min(8UL, index.size()), tmp);
	};

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @param index  Position in ND-space to interpolate
	 *
	 * @return value Interpolated value at given position
	 */
	T operator()(std::initializer_list<float> index)
	{
		assert(index.size() <= 8);
		double tmp[8];
		size_t ii=0;
		for(auto it = index.begin(); it != index.end() && ii<8; ++it, ++ii)
			tmp[ii] = *it;
		return get(std::min(8UL, index.size()), tmp);
	};

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @return value
	 */
	T get(const vector<double>& cindex)
	{
		return get(cindex.size(), cindex.data());
	}

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @param index n-d index to access
	 *
	 * @return value
	 */
	T get(const std::vector<int64_t>& index)
	{
		assert(index.size() <= 8);
		double tmp[8];
		size_t ii=0;
		for(auto it = index.begin(); it != index.end() && ii<8; ++it, ++ii)
			tmp[ii] = *it;
		return get(std::min(8UL, index.size()), tmp);
	};

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @param len length of index array
	 * @param index n-d index to access
	 *
	 * @return value
	 */
	T get(size_t len, int64_t* index)
	{
		assert(len <= 8);
		double tmp[8];
		for(size_t ii=0; ii<len && ii<8; ii++)
			tmp[ii] = index[ii];
		return get(std::min(8UL, len), tmp);
	};

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @param index n-d index to access
	 *
	 * @return value
	 */
	T operator[](const std::vector<int64_t>& index)
	{
		assert(index.size() <= 8);
		double tmp[8];
		size_t ii=0;
		for(auto it = index.begin(); it != index.end() && ii<8; ++it, ++ii)
			tmp[ii] = *it;
		return get(std::min(8UL, index.size()), tmp);
	};

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @return value
	 */
	T get(size_t len, const double* incindex)
	{
		// Initialize variables
		int ndim = this->parent->ndim();
		assert(ndim <= MAXDIM);
		const size_t* dim = this->parent->dim();
		int64_t index[MAXDIM];
		double cindex[MAXDIM];

		// convert RAS to index
		if(m_ras) {
			auto tmp = dPtrCast<const MRImage>(this->parent);
			tmp->pointToIndex(len, incindex, cindex);
		} else {
			for(size_t dd=0; dd<ndim; dd++) {
				if(dd < len)
					cindex[dd] = incindex[dd];
				else
					cindex[dd] = 0;
			}
		}

		Counter<> count(ndim);
		for(size_t dd=0; dd<ndim; dd++)
			count.sz[dd] = 2;

		// compute weighted pixval by iterating over neighbors
		T pixval = 0;
		do {
			double weight = 1;
			bool iioutside = false;

			//set index
			for(int dd = 0; dd < ndim; dd++) {
				index[dd] = floor(cindex[dd]) + count.pos[dd];
				weight *= linKern(index[dd] - cindex[dd]);
				iioutside = iioutside || index[dd] < 0 || index[dd] >= dim[dd];
			}

			// if the current point maps outside, then we need to deal with it
			if(iioutside) {
				if(m_boundmethod == ZEROFLUX) {
					// clamp
					for(size_t dd=0; dd<ndim; dd++)
						index[dd] = clamp<int64_t>(0, dim[dd]-1, index[dd]);
				} else if(m_boundmethod == WRAP) {
					// wrap
					for(size_t dd=0; dd<ndim; dd++)
						index[dd] = wrap<int64_t>(0, dim[dd]-1, index[dd]);
				} else {
					// set wieght to zero, then just clamp
					weight = 0;
					for(size_t dd=0; dd<ndim; dd++)
						index[dd] = clamp<int64_t>(0, dim[dd]-1, index[dd]);
				}
			}

			T v = this->castget(this->parent->__getAddr(ndim, index));
			pixval += weight*v;
		} while(count.advance());

		return pixval;
	}


	BoundaryConditionT m_boundmethod;

	/**
	 * @brief if true, then this assumes the inputs are RAS coordinates rather
	 * than indexes. Default is false
	 */
	bool m_ras;

protected:

	/**
	 * @brief Gets value at array index and then casts to T
	 * doesn't make sense for interpolation
	 *
	 * @return value
	 */
	T operator[](int64_t i) { (void)(i); return T(); };
};

/**
 * @brief The purpose of this class is to view an image as a continuous
 * 3D+vector dimension image rather than a 4+D image. Therefore all dimensions
 * above the third are cast as a vector and interpolation is only performed
 * between 3D points, with the 4th dimension assumed to be non-spatial. The
 * would be applicable if the upper dimensions are of a different type
 * than the first 3.
 *
 * @tparam T Type of value to cast and return
 */
template<typename T>
class LinInterp3DView : public Vector3DConstView<T>
{
public:
	LinInterp3DView(std::shared_ptr<const NDArray> in,
			BoundaryConditionT bound = ZEROFLUX)
		: Vector3DConstView<T>(in), m_boundmethod(bound), m_ras(false)
	{ };

	LinInterp3DView() : m_boundmethod(ZEROFLUX), m_ras(false) {} ;

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @return value
	 */
	T operator()(double x=0, double y=0, double z=0, int64_t t=0)
	{
		return get(x,y,z,t);
	};

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @return value
	 */
	T get(double x=0, double y=0, double z=0, int64_t t=0)
	{
		// figure out size of dimensions in parent
		size_t dim[4];
		dim[0] = this->parent->dim(0);
		dim[1] = this->parent->ndim() > 1 ? this->parent->dim(1) : 1;
		dim[2] = this->parent->ndim() > 2 ? this->parent->dim(2) : 1;
		dim[3] = this->parent->tlen();

		// deal with t being outside bounds
		if(t < 0 || t >= dim[3]) {
			if(m_boundmethod == ZEROFLUX) {
				// clamp
				t = clamp<int64_t>(0, dim[3]-1, t);
			} else if(m_boundmethod == WRAP) {
				// wrap
				t = wrap<int64_t>(0, dim[3]-1, t);
			} else {
				return 0;
			}
		}

		// initialize variables
		double cindex[4] = {x,y,z,(double)t};
		int64_t index[4] = {0,0,0,t};

		// convert RAS to cindex
		if(m_ras) {
			auto tmp = dPtrCast<const MRImage>(this->parent);
			tmp->pointToIndex(3, cindex, cindex);
		}

		Counter<> count(3);
		for(size_t dd=0; dd<3; dd++)
			count.sz[dd] = 2;

		T pixval = 0;
		bool iioutside = false;
		do {
			double weight = 1;

			//set index
			for(int dd = 0; dd < 3; dd++) {
				index[dd] = floor(cindex[dd]) + count.pos[dd];
				weight *= linKern(index[dd] - cindex[dd]);
				iioutside = iioutside || index[dd] < 0 || index[dd] >= dim[dd];
			}

			// if the current point maps outside, then we need to deal with it
			if(iioutside) {
				if(m_boundmethod == ZEROFLUX) {
					// clamp
					for(size_t dd=0; dd<3; dd++)
						index[dd] = clamp<int64_t>(0, dim[dd]-1, index[dd]);
				} else if(m_boundmethod == WRAP) {
					// wrap
					for(size_t dd=0; dd<3; dd++)
						index[dd] = wrap<int64_t>(0, dim[dd]-1, index[dd]);
				} else {
					// set wieght to zero, then just clamp
					weight = 0;
					for(size_t dd=0; dd<3; dd++)
						index[dd] = clamp<int64_t>(0, dim[dd]-1, index[dd]);
				}
			}

			T v = this->castget(this->parent->__getAddr(index[0], index[1],index[2],t));
			pixval += weight*v;
		} while(count.advance());

		return pixval;
	}

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @return value
	 */
	T operator()(int64_t x=0, int64_t y=0, int64_t z=0, int64_t t=0)
	{
		return get((double)x,(double)y,(double)z,t);
	};

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @return value
	 */
	T get(int64_t x=0, int64_t y=0, int64_t z=0, int64_t t=0)
	{
		return get((double)x,(double)y,(double)z,t);
	};

	BoundaryConditionT m_boundmethod;

	/**
	 * @brief if true, then this assumes the inputs are RAS coordinates rather
	 * than indexes. Default is false
	 */
	bool m_ras;

};

/**
 * @brief General purpose Nearest-Neighbor interpolator
 *
 * @tparam T Type of value to cast and return
 */
template<typename T>
class NNInterpNDView : public NDConstView<T>
{
public:
	NNInterpNDView(std::shared_ptr<const NDArray> in,
			BoundaryConditionT bound = ZEROFLUX)
		: NDConstView<T>(in), m_boundmethod(bound), m_ras(false)
	{ };

	NNInterpNDView() : m_boundmethod(ZEROFLUX), m_ras(false) {} ;

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @param x	x-dimension
	 * @param y	y-dimension
	 * @param z	z-dimension
	 * @param t	4th dimension
	 * @param u	5th dimension
	 * @param v	6th dimension
	 * @param w	7th dimension
	 * @param q	8th dimension
	 *
	 * @return value Interpolated value at given position
	 */
	T operator()(double x=0, double y=0, double z=0, double t=0, double u = 0,
			double v = 0, double w = 0, double q = 0)
	{
		double tmp[8] = {x,y,z,t,u,v,w,q};
		return get(8, tmp);
	};


	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @param index  Position in ND-space to interpolate
	 *
	 * @return value Interpolated value at given position
	 */
	T operator()(const std::vector<float>& index)
	{
		assert(index.size() <= 8);
		double tmp[8];
		size_t ii=0;
		for(auto it = index.begin(); it != index.end() && ii<8; ++it, ++ii)
			tmp[ii] = *it;
		return get(std::min(8UL, index.size()), tmp);
	};

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @param index n-d index to access
	 *
	 * @return value
	 */
	T get(const std::vector<int64_t>& index)
	{
		assert(index.size() <= 8);
		double tmp[8];
		size_t ii=0;
		for(auto it = index.begin(); it != index.end() && ii<8; ++it, ++ii)
			tmp[ii] = *it;
		return get(std::min(8UL, index.size()), tmp);
	};

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @param len length of index array
	 * @param index n-d index to access
	 *
	 * @return value
	 */
	T get(size_t len, int64_t* index)
	{
		assert(len <= 8);
		double tmp[8];
		for(size_t ii=0; ii < len && ii<8; ++ii)
			tmp[ii] = index[ii];
		return get(std::min(8UL, len), tmp);
	};

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @param index n-d index to access
	 *
	 * @return value
	 */
	T operator[](const std::vector<int64_t>& index)
	{
		assert(index.size() <= 8);
		double tmp[8];
		size_t ii=0;
		for(auto it = index.begin(); it != index.end() && ii<8; ++it, ++ii)
			tmp[ii] = *it;
		return get(std::min(8UL, index.size()), tmp);
	};

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @param index  Position in ND-space to interpolate
	 *
	 * @return value Interpolated value at given position
	 */
	T operator()(const std::vector<double>& index)
	{
		return get(index.size(), index.data());
	};

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @param index  Position in ND-space to interpolate
	 *
	 * @return value Interpolated value at given position
	 */
	T operator()(std::initializer_list<double> index)
	{
		assert(index.size() <= 8);
		double tmp[8];
		size_t ii=0;
		for(auto it = index.begin(); it != index.end() && ii<8; ++it, ++ii)
			tmp[ii] = *it;
		return get(std::min(8UL, index.size()), tmp);
	};

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @param index  Position in ND-space to interpolate
	 *
	 * @return value Interpolated value at given position
	 */
	T operator()(std::initializer_list<float> index)
	{
		assert(index.size() <= 8);
		double tmp[8];
		size_t ii=0;
		for(auto it = index.begin(); it != index.end() && ii<8; ++it, ++ii)
			tmp[ii] = *it;
		return get(std::min(8UL, index.size()), tmp);
	};

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @return value
	 */
	T get(size_t len, const double* incindex)
	{
		// convert RAS to index
		size_t ndim = this->parent->ndim();
		assert(ndim <= MAXDIM);
		double cindex[MAXDIM];
		int64_t index[MAXDIM];

		// convert RAS to index
		if(m_ras) {
			auto tmp = dPtrCast<const MRImage>(this->parent);
			tmp->pointToIndex(len, incindex, cindex);
		} else {
			for(size_t dd=0; dd<ndim; dd++) {
				if(dd < len)
					cindex[dd] = incindex[dd];
				else
					cindex[dd] = 0;
			}
		}


		// initialize variables
		const size_t* dim = this->parent->dim();

		// round values from cindex
		if(m_boundmethod == ZEROFLUX) {
			// clamp
			for(size_t dd=0; dd<ndim; dd++) {
				double C = dd < len ? cindex[dd] : 0;
				index[dd] = clamp<int64_t>(0, dim[dd]-1, round(C));
			}
		} else if(m_boundmethod == WRAP) {
			// wrap
			for(size_t dd=0; dd<ndim; dd++) {
				double C = dd < len ? cindex[dd] : 0;
				index[dd] = wrap<int64_t>(0, dim[dd]-1, round(C));
			}
		} else {
			for(size_t dd=0; dd<ndim; dd++) {
				double C = dd < len ? cindex[dd] : 0;
				index[dd] = round(C);
				if(index[dd] < 0 || index[dd] > dim[dd]-1)
					return 0;
			}
		}

		return this->castget(this->parent->__getAddr(ndim, index));
	}

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @return value
	 */
	T get(const vector<double>& cindex)
	{
		return get(cindex.size(), cindex.data());
	}

	BoundaryConditionT m_boundmethod;

	/**
	 * @brief if true, then this assumes the inputs are RAS coordinates rather
	 * than indexes. Default is false
	 */
	bool m_ras;

private:
	///////////////////////////////////////////////////////////
	// Hide Unused Functions From Parent
	///////////////////////////////////////////////////////////

	/**
	 * @brief Gets value linear position in array, then casts to T
	 *
	 * @return value
	 */
	T operator[](int64_t index)
	{
		return castget(this->parent->__getAddr(index));
	};
};

/**
 * @brief The purpose of this class is to view an image as a continuous
 * 3D+vector dimension image rather than a 4+D image. Therefore all dimensions
 * above the third are cast as a vector and interpolation is only performed
 * between 3D points, with the 4th dimension assumed to be non-spatial. The
 * would be applicable if the upper dimensions are of a different type
 * than the first 3.
 *
 * @tparam T Type of value to cast and return
 */
template<typename T>
class NNInterp3DView : public Vector3DConstView<T>
{
public:
	NNInterp3DView(std::shared_ptr<NDArray> in,
			BoundaryConditionT bound = ZEROFLUX)
		: Vector3DConstView<T>(in), m_boundmethod(bound), m_ras(false)
	{ };

	NNInterp3DView() : m_boundmethod(ZEROFLUX), m_ras(false) {} ;

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @return value
	 */
	T operator()(double x=0, double y=0, double z=0, int64_t t=0)
	{
		return get(x,y,z,t);
	};

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @return value
	 */
	T get(double x=0, double y=0, double z=0, int64_t t=0)
	{
		// convert RAS to cindex
		if(m_ras) {
			double cindex[3] = {x,y,z};
			auto tmp = dPtrCast<const MRImage>(this->parent);
			tmp->pointToIndex(3, cindex, cindex);
			x = cindex[0];
			y = cindex[1];
			z = cindex[2];
		}


		// interpolate
		int64_t i = round(x);
		int64_t j = round(y);
		int64_t k = round(z);
		size_t xdim = this->parent->dim(0);
		size_t ydim = this->parent->ndim() > 1 ? this->parent->dim(1) : 1;
		size_t zdim = this->parent->ndim() > 2 ? this->parent->dim(2) : 1;
		size_t tdim = this->parent->tlen();

		bool xout = (i < 0 || i >= xdim);
		bool yout = (j < 0 || j >= ydim);
		bool zout = (k < 0 || k >= zdim);
		bool tout = (t < 0 || t >= tdim);

		if(xout || yout || zout || tout) {
			//			outside = true;
			switch(m_boundmethod) {
				case ZEROFLUX:
					i = clamp<int64_t>(0, xdim-1, i);
					j = clamp<int64_t>(0, ydim-1, j);
					k = clamp<int64_t>(0, zdim-1, k);
					t = clamp<int64_t>(0, tdim-1, t);
					break;
				case WRAP:
					i = wrap<int64_t>(0, xdim-1, i);
					j = wrap<int64_t>(0, ydim-1, j);
					k = wrap<int64_t>(0, zdim-1, k);
					t = wrap<int64_t>(0, tdim-1, t);
					break;
				case CONSTZERO:
				default:
					return 0;
					break;
			}
		}

		return this->castget(this->parent->__getAddr(i,j,k,t));
	};

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @return value
	 */
	T operator()(int64_t x=0, int64_t y=0, int64_t z=0, int64_t t=0)
	{
		return get((double)x, (double)y, (double)z, t);
	};

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @return value
	 */
	T get(int64_t x=0, int64_t y=0, int64_t z=0, int64_t t=0)
	{
		return get((double)x, (double)y, (double)z, t);
	};


	BoundaryConditionT m_boundmethod;

	/**
	 * @brief if true, then this assumes the inputs are RAS coordinates rather
	 * than indexes. Default is false
	 */
	bool m_ras;

private:

};

////////////////////////
// Lanczos
////////////////////////

/**
 * @brief The purpose of this class is to view an image as a continuous
 * 3D+vector dimension image rather than a 4+D image. Therefore all dimensions
 * above the third are cast as a vector and interpolation is only performed
 * between 3D points, with the 4th dimension assumed to be non-spatial. The
 * would be applicable if the upper dimensions are of a different type
 * than the first 3.
 *
 * @tparam T Type of value to cast and return
 */
template<typename T>
class LanczosInterpNDView : public NDConstView<T>
{
public:
	LanczosInterpNDView(std::shared_ptr<const NDArray> in,
			BoundaryConditionT bound = ZEROFLUX)
		: NDConstView<T>(in), m_boundmethod(bound), m_ras(false),
		m_radius(2)
	{ };

	LanczosInterpNDView() : m_boundmethod(ZEROFLUX), m_ras(false), m_radius(2) {} ;

	void setRadius(size_t rad) { m_radius = rad; };
	size_t getRadius() { return m_radius; };

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @return value
	 */
	T operator()(double x=0, double y=0, double z=0, double t=0, double u=0,
			double v=0, double w=0)
	{
		double tmp[8] = {x,y,z,t,u,v,w};
		return get(8, tmp);
	};

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @param index n-d index to access
	 *
	 * @return value
	 */
	T get(const std::vector<int64_t>& index)
	{
		assert(index.size() <= 8);
		double tmp[8];
		size_t ii=0;
		for(auto it = index.begin(); it != index.end() && ii<8; ++it, ++ii)
			tmp[ii] = *it;
		return get(std::min(8UL, index.size()), tmp);
	};

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @param len length of index array
	 * @param index n-d index to access
	 *
	 * @return value
	 */
	T get(size_t len, int64_t* index)
	{
		assert(len <= 8);
		double tmp[8];
		for(size_t ii=0; ii<len && ii<8; ii++)
			tmp[ii] = index[ii];
		return get(std::min(8UL, len), tmp);
	};

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @param index n-d index to access
	 *
	 * @return value
	 */
	T operator[](const std::vector<int64_t>& index)
	{
		assert(index.size() <= 8);
		double tmp[8];
		size_t ii=0;
		for(auto it = index.begin(); it != index.end() && ii<8; ++it, ++ii)
			tmp[ii] = *it;
		return get(std::min(8UL, index.size()), tmp);
	};

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @return value
	 */
	T get(const std::vector<double>& incoord)
	{
		return get(incoord.size(), incoord.data());
	}

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @return value
	 */
	T get(size_t len, const double* incoord)
	{
		// figure out size of dimensions in parent
		const size_t* dim = this->parent->dim();
		size_t ndim = this->parent->ndim();
		assert(ndim < MAXDIM);
		int64_t index[MAXDIM];
		double cindex[MAXDIM];

		// convert RAS to index
		if(m_ras) {
			auto tmp = dPtrCast<const MRImage>(this->parent);
			tmp->pointToIndex(len, incoord, cindex);
		} else {
			for(size_t dd=0; dd<ndim; dd++) {
				if(dd < len)
					cindex[dd] = incoord[dd];
				else
					cindex[dd] = 0;
			}
		}

		const int KPOINTS = 1+m_radius*2;

		// 1D version of the weights and indices
		double karray[MAXDIM][KPOINTS];
		int64_t indarray[MAXDIM][KPOINTS];

		for(int dd = 0; dd < ndim; dd++) {
			for(int64_t ii=-m_radius; ii<=m_radius; ii++){
				int64_t i = round(cindex[dd])+ii;
				indarray[dd][ii+m_radius] = i;
				karray[dd][ii+m_radius] = lanczosKern(i-cindex[dd], m_radius);
			}
		}

		// initialize variables
		Counter<int, MAXDIM> count(ndim);
		for(size_t dd=0; dd<ndim; dd++)
			count.sz[dd] = 1+m_radius*2;

		// compute weighted pixval by iterating over neighbors, which are
		// combinations of KPOINTS
		T pixval = 0;
		do {
			double weight = 1;
			bool iioutside = false;

			//set index
			for(int dd = 0; dd < ndim; dd++) {
				index[dd] = indarray[dd][count.pos[dd]];
				weight *= karray[dd][count.pos[dd]];
				iioutside = iioutside || index[dd] < 0 || index[dd] >= dim[dd];
			}

			if(iioutside) {
				if(m_boundmethod == ZEROFLUX) {
					// clamp
					for(size_t dd=0; dd<ndim; dd++)
						index[dd] = clamp<int64_t>(0, dim[dd]-1, index[dd]);
				} else if(m_boundmethod == WRAP) {
					// wrap
					for(size_t dd=0; dd<ndim; dd++)
						index[dd] = wrap<int64_t>(0, dim[dd]-1, index[dd]);
				} else { // zero outside
					// set wieght to zero, then just clamp
					weight = 0;
					for(size_t dd=0; dd<ndim; dd++)
						index[dd] = clamp<int64_t>(0, dim[dd]-1, index[dd]);
				}
			}

			T v = this->castget(this->parent->__getAddr(ndim, index));
			pixval += weight*v;
		} while(count.advance());

		return pixval;
	}

	BoundaryConditionT m_boundmethod;

	/**
	 * @brief if true, then this assumes the inputs are RAS coordinates rather
	 * than indexes. Default is false
	 */
	bool m_ras;

protected:

	/**
	 * @brief Gets value linear position in array, then casts to T
	 *
	 * @return value
	 */
	T operator[](int64_t index)
	{
		return castget(this->parent->__getAddr(index));
	};

	int64_t m_radius;
};


/**
 * @brief The purpose of this class is to view an image as a continuous
 * 3D+vector dimension image rather than a 4+D image. Therefore all dimensions
 * above the third are cast as a vector and interpolation is only performed
 * between 3D points, with the 4th dimension assumed to be non-spatial. The
 * would be applicable if the upper dimensions are of a different type
 * than the first 3.
 *
 * @tparam T Type of value to cast and return
 */
template<typename T>
class LanczosInterp3DView : public Vector3DConstView<T>
{
public:
	LanczosInterp3DView(std::shared_ptr<const NDArray> in,
			BoundaryConditionT bound = ZEROFLUX)
		: Vector3DConstView<T>(in), m_boundmethod(bound), m_ras(false),
		m_radius(2)
	{ };

	LanczosInterp3DView() : m_boundmethod(ZEROFLUX), m_ras(false), m_radius(2) {} ;

	void setRadius(size_t rad) { m_radius = rad; };
	size_t getRadius() { return m_radius; };

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @return value
	 */
	T operator()(double x=0, double y=0, double z=0, int64_t t=0)
	{
		return get(x,y,z,t);
	};

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @return value
	 */
	T operator()(int64_t x=0, int64_t y=0, int64_t z=0, int64_t t=0)
	{
		return get((double)x,(double)y,(double)z,t);
	};

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @return value
	 */
	T get(int64_t x=0, int64_t y=0, int64_t z=0, int64_t t=0)
	{
		return get((double)x,(double)y,(double)z,t);
	};


	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @return value
	 */
	T get(double x=0, double y=0, double z=0, int64_t t=0)
	{
		// figure out size of dimensions in parent
		size_t dim[4];
		double cindex[3] = {x,y,z};
		int64_t index[3];
		const int ndim = 3;
		dim[0] = this->parent->dim(0);
		dim[1] = this->parent->ndim() > 1 ? this->parent->dim(1) : 1;
		dim[2] = this->parent->ndim() > 2 ? this->parent->dim(2) : 1;
		dim[3] = this->parent->tlen();

		// deal with t being outside bounds
		if(t < 0 || t >= dim[3]) {
			if(m_boundmethod == ZEROFLUX) {
				// clamp
				t = clamp<int64_t>(0, dim[3]-1, t);
			} else if(m_boundmethod == WRAP) {
				// wrap
				t = wrap<int64_t>(0, dim[3]-1, t);
			} else {
				return 0;
			}
		}

		// convert RAS to cindex
		if(m_ras) {
			auto tmp = dPtrCast<const MRImage>(this->parent);
			tmp->pointToIndex(3, cindex, cindex);
		}

		const int KPOINTS = 1+m_radius*2;
		const int DIM = 3;

		// 1D version of the weights and indices
		double karray[DIM][KPOINTS];
		int64_t indarray[DIM][KPOINTS];

		for(int dd = 0; dd < DIM; dd++) {
			for(int64_t ii=-m_radius; ii<=m_radius; ii++){
				int64_t i = round(cindex[dd])+ii;
				indarray[dd][ii+m_radius] = i;
				karray[dd][ii+m_radius] = lanczosKern(i-cindex[dd], m_radius);
			}
		}

		T pixval = 0;
		Counter<int, MAXDIM> count(ndim);
		for(size_t dd=0; dd<ndim; dd++)
			count.sz[dd] = 1+m_radius*2;

		// compute weighted pixval by iterating over neighbors, which are
		// combinations of KPOINTS
		do {
			double weight = 1;
			bool iioutside = false;

			//set index
			for(int dd = 0; dd < ndim; dd++) {
				index[dd] = indarray[dd][count.pos[dd]];
				weight *= karray[dd][count.pos[dd]];
				iioutside = iioutside || index[dd] < 0 || index[dd] >= dim[dd];
			}

			// if the current point maps outside, then we need to deal with it
			if(iioutside) {
				if(m_boundmethod == ZEROFLUX) {
					// clamp
					for(size_t dd=0; dd<ndim; dd++)
						index[dd] = clamp<int64_t>(0, dim[dd]-1, index[dd]);
				} else if(m_boundmethod == WRAP) {
					// wrap
					for(size_t dd=0; dd<ndim; dd++)
						index[dd] = wrap<int64_t>(0, dim[dd]-1, index[dd]);
				} else {
					// set wieght to zero, then just clamp
					weight = 0;
					for(size_t dd=0; dd<ndim; dd++)
						index[dd] = clamp<int64_t>(0, dim[dd]-1, index[dd]);
				}
			}

			T v = this->castget(this->parent->__getAddr(index[0], index[1],index[2],t));
			pixval += weight*v;
		} while(count.advance());

		return pixval;
	}

	BoundaryConditionT m_boundmethod;

	/**
	 * @brief if true, then this assumes the inputs are RAS coordinates rather
	 * than indexes. Default is false
	 */
	bool m_ras;

protected:

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @return value
	 */
	T operator[](int64_t i) { (void)(i); return T(); };
	T get(const std::vector<int64_t>& i) { (void)(i); return T(); };
	T operator[](const std::vector<int64_t>& i) { (void)(i); return T(); };
	int64_t m_radius;
};



/****************************************************************************
 * BSpline
 ***************************************************************************/

/**
 * @brief This is a specialized viewer for computing the value of a cubic
 * B-Spline interpolation from the parameters. Thus the input to the
 * constructor or setArray must be a parameter image.
 *
 * estimate(in) will generate the parameters that minimize the least squares
 * difference from the given image (in).
 *
 * createOverlay(in) will create a new parameter array in params that extends
 * for two knots outside the input image (in)
 *
 * @tparam T Value to interpolate as
 */
template<typename T>
class BSplineView : public NDView<T>
{
public:
	BSplineView(std::shared_ptr<MRImage> params,
			BoundaryConditionT bound = ZEROFLUX)
		: NDView<T>(params), m_boundmethod(bound), m_ras(false)
	{ };

	BSplineView() : m_boundmethod(ZEROFLUX), m_ras(false){} ;

	void createOverlay(ptr<const MRImage> overlay, double bspace)
	{
		size_t ndim = overlay->ndim();
		assert(ndim <= MAXDIM);
		VectorXd spacing(overlay->ndim());
		VectorXd origin(overlay->ndim());
		size_t osize[MAXDIM];

		// get spacing and size
		for(size_t dd=0; dd<ndim; ++dd) {
			osize[dd] = 4+ceil(overlay->dim(dd)*overlay->spacing(dd)/bspace);
			spacing[dd] = bspace;
		}

		auto params = dPtrCast<MRImage>(overlay->createAnother(ndim, osize,
					FLOAT64));
		this->parent = params;
		params->setDirection(overlay->getDirection(), false);
		params->setSpacing(spacing, false);

		// compute center of input
		VectorXd indc(ndim); // center index
		for(size_t dd=0; dd<ndim; dd++)
			indc[dd] = (overlay->dim(dd)-1.)/2.;
		VectorXd ptc(ndim); // point center
		overlay->indexToPoint(ndim, indc.array().data(), ptc.array().data());

		// compute origin from center index (x_c) and center of input (c):
		// o = c-R(sx_c)
		for(size_t dd=0; dd<ndim; dd++)
			indc[dd] = (osize[dd]-1.)/2.;
		origin = ptc - overlay->getDirection()*(spacing.asDiagonal()*indc);
		params->setOrigin(origin, false);

		this->setArray(params);
	};

	/**
	 * @brief Samples the BSpline function at the specified point.
	 *
	 * @param len Length of point vector
	 * @param point Point (if m_ras) or continuous index in parameter space
	 * @param dir Dimension to take derivative in, must be >= 0
	 * @param val Return value at specified point
	 * @param dval Return derivative of Bspline in dir direction
	 *
	 * @return true if the specified point used boundary conditions (outside
	 * values)
	 */
	bool get(size_t len, const double* point, int dir, double& val, double& dval)
	{
		assert(this->parent);
		// initialize variables
		int ndim = this->parent->ndim();
		assert(ndim <= MAXDIM);
		const size_t* dim = this->parent->dim();
		int64_t index[MAXDIM];
		double cindex[MAXDIM];

		// convert RAS to index, or just copy into local array
		if(m_ras) {
			auto tmp = dPtrCast<const MRImage>(this->parent);
			tmp->pointToIndex(len, point, cindex);
		} else {
			for(size_t dd=0; dd<ndim; dd++) {
				if(dd < len)
					cindex[dd] = point[dd];
				else
					cindex[dd] = 0;
			}
		}

		Counter<> count(ndim);
		for(size_t dd=0; dd<ndim; dd++)
			count.sz[dd] = 5;

		dval = 0;
		val = 0;
		bool border = false;
		do {
			double weight = 1;
			double dweight = 1;
			bool iioutside = false;

			//set index
			for(int dd = 0; dd < ndim; dd++) {
				index[dd] = round(cindex[dd]) + count.pos[dd] - 2l;
				weight *= B3kern(index[dd] - cindex[dd]);
				if(dd == dir)
					dweight *= -dB3kern(index[dd] - cindex[dd])/
						getParams()->spacing(dd);
				else
					dweight *= B3kern(index[dd] - cindex[dd]);
				iioutside = iioutside || index[dd] < 0 || index[dd] >= dim[dd];
			}

			// if the current point maps outside, then we need to deal with it
			if(iioutside) {
				if(m_boundmethod == ZEROFLUX) {
					// clamp
					for(size_t dd=0; dd<ndim; dd++)
						index[dd] = clamp<int64_t>(0, dim[dd]-1, index[dd]);
				} else if(m_boundmethod == WRAP) {
					// wrap
					for(size_t dd=0; dd<ndim; dd++)
						index[dd] = wrap<int64_t>(0, dim[dd]-1, index[dd]);
				} else {
					// set wieght to zero, then just clamp
					weight = 0;
					for(size_t dd=0; dd<ndim; dd++)
						index[dd] = clamp<int64_t>(0, dim[dd]-1, index[dd]);
				}
			}

			// Update border with outside
			border |= iioutside;

			// Compute Values
			T v = this->castget(this->parent->__getAddr(ndim, index));
			dval += dweight*v;
			val += weight*v;
		} while(count.advance());

		return border;
	};

	/**
	 * @brief Simple sample function, just retrieve value at point, does not
	 * perform derivative
	 *
	 * @param len Length of point array
	 * @param point Point (or if m_ras = false, then continuous index in
	 * parameter space)
	 *
	 * @return Value at point
	 */
	double get(size_t len, double* point)
	{
		assert(this->parent);
		// initialize variables
		int ndim = this->parent->ndim();
		assert(ndim <= MAXDIM);
		const size_t* dim = this->parent->dim();
		int64_t index[MAXDIM];
		double cindex[MAXDIM];

		// convert RAS to index, or just copy into local array
		if(m_ras) {
			auto tmp = dPtrCast<const MRImage>(this->parent);
			tmp->pointToIndex(len, point, cindex);
		} else {
			for(size_t dd=0; dd<ndim; dd++) {
				if(dd < len)
					cindex[dd] = point[dd];
				else
					cindex[dd] = 0;
			}
		}

		Counter<> count(ndim);
		for(size_t dd=0; dd<ndim; dd++)
			count.sz[dd] = 5;

		double val = 0;
		bool border = false;
		do {
			double weight = 1;
			bool iioutside = false;

			//set index
			for(int dd = 0; dd < ndim; dd++) {
				index[dd] = round(cindex[dd]) + count.pos[dd] - 2l;
				weight *= B3kern(index[dd] - cindex[dd]);
				iioutside = iioutside || index[dd] < 0 || index[dd] >= dim[dd];
			}

			// if the current point maps outside, then we need to deal with it
			if(iioutside) {
				if(m_boundmethod == ZEROFLUX) {
					// clamp
					for(size_t dd=0; dd<ndim; dd++)
						index[dd] = clamp<int64_t>(0, dim[dd]-1, index[dd]);
				} else if(m_boundmethod == WRAP) {
					// wrap
					for(size_t dd=0; dd<ndim; dd++)
						index[dd] = wrap<int64_t>(0, dim[dd]-1, index[dd]);
				} else {
					// set wieght to zero, then just clamp
					weight = 0;
					for(size_t dd=0; dd<ndim; dd++)
						index[dd] = clamp<int64_t>(0, dim[dd]-1, index[dd]);
				}
			}

			border |= iioutside;
			T v = this->castget(this->parent->__getAddr(ndim, index));
			val += weight*v;
		} while(count.advance());

		return val;
	};

	/**
	 * @brief Perform full-on reconstruction in the space of the input image
	 *
	 * @param input
	 *
	 * @return
	 */
	ptr<MRImage> reconstruct(ptr<const MRImage> input)
	{
		assert(this->parent);
		auto params = dPtrCast<MRImage>(this->parent);
		if(params->getDirection() != input->getDirection())
			return reconstructNotAligned(input);
		else
			return reconstructAligned(input);
	};

	ptr<MRImage> reconstructNotAligned(ptr<const MRImage> input)
	{
		assert(this->parent);
		(void)(input);
		throw INVALID_ARGUMENT("Not yet implemented");
		return NULL;
	};

	ptr<MRImage> reconstructAligned(ptr<const MRImage> input)
	{
		assert(this->parent);
		if(input->ndim() != this->parent->ndim()) {
			throw INVALID_ARGUMENT("Not sure how to deal with different "
					"dimensions for B-Spline right now");
		}
		auto params = getParams();
		auto out = dPtrCast<MRImage>(input->createAnother());

		// for each kernel, iterate over the points in the neighborhood
		size_t ndim = input->ndim();
		assert(ndim <= MAXDIM);
		double cind[MAXDIM];
		int64_t ind[MAXDIM];
		double scale[MAXDIM];
		int64_t center[MAXDIM];
		size_t winsize[MAXDIM];
		int64_t roistart[MAXDIM];

		for(size_t dd=0; dd<ndim; dd++) {
			winsize[dd] = ceil(5*params->spacing(dd)/out->spacing(dd));
			scale[dd] = out->spacing(dd)/params->spacing(dd);
		}

		// We go through each parameter, and compute the weight of the B-spline
		// parameter at each pixel within the range (2 indexes in parameter
		// space, 2*S_B/S_I indexs in pixel space)
		NDIter<double> oit(out);
		for(NDConstIter<double> bit(params); !bit.eof(); ++bit) {

			// get continuous index of pixel
			bit.index(ndim, cind);
			params->indexToPoint(ndim, cind, cind);
			out->pointToIndex(ndim, cind, cind);

			// construct weights / construct ROI
			for(size_t dd=0; dd<ndim; dd++) {
				center[dd] = round(cind[dd]);
				roistart[dd] = center[dd]-((int64_t)winsize[dd]/2);
			}

			oit.setROI(ndim, winsize, roistart);
			for(oit.goBegin(); !oit.eof(); ++oit) {
				double weight = 1;
				oit.index(ndim, ind);
				for(size_t dd=0; dd<ndim; dd++) {
					double dist = (ind[dd] - cind[dd])*scale[dd];
					weight *= B3kern(dist);
				}
				oit.set(oit.get()+(*bit)*weight);
			}
		}

		return out;
	};

	double thinPlateEnergy()
	{
		using std::pow;

		auto params = getParams();
		assert(params->ndim() == 3);

		double dphi2_dx2 = 0;
		double dphi2_dy2 = 0;
		double dphi2_dz2 = 0;
		double dphi2_dxz = 0;
		double dphi2_dxy = 0;
		double dphi2_dyz = 0;

		// We use NN interpolator because it is fast and handles boundary 
		// conditions
		NNInterpNDView<double> dvw(params);
		double ind1[3];
		double ind2[3];

		// Create a counter to iterate over [0,4] ie [-2,2] in 6 directions
		Counter<int, 3> counter(3);
		for(size_t dd=0; dd<3;dd++)
			counter.sz[dd] = 9;

		//integrate over all the knots
		Counter<int64_t, 3> it(3, (int64_t*)params->dim());
		do {
			double phi_abc = dvw.get(3, it.pos);
			do {
				for(size_t dd=0; dd<3; dd++)
					ind1[dd] = it.pos[dd] + counter.pos[dd] - 4;

				double xv = U200[counter.pos[0]][counter.pos[1]][counter.pos[2]];
				double yv = U200[counter.pos[1]][counter.pos[0]][counter.pos[2]];
				double zv = U200[counter.pos[2]][counter.pos[1]][counter.pos[0]];
				double xyv = U110[counter.pos[0]][counter.pos[1]][counter.pos[2]];
				double xzv = U110[counter.pos[0]][counter.pos[2]][counter.pos[1]];
				double yzv = U110[counter.pos[1]][counter.pos[2]][counter.pos[0]];
				double phi_lmn = dvw.get(3, ind1);
				dphi2_dx2 += phi_abc*phi_lmn*xv;
				dphi2_dy2 += phi_abc*phi_lmn*yv;
				dphi2_dz2 += phi_abc*phi_lmn*zv;
				dphi2_dxy += phi_abc*phi_lmn*xyv;
				dphi2_dyz += phi_abc*phi_lmn*yzv;
				dphi2_dxz += phi_abc*phi_lmn*xzv;
			} while(counter.advance());
		} while(it.advance());

		double v = dphi2_dx2/pow(params->spacing(0),4) +
			dphi2_dy2/pow(params->spacing(1),4) +
			dphi2_dz2/pow(params->spacing(2),4) +
			2*dphi2_dxy/(pow(params->spacing(0),2)*pow(params->spacing(1),2)) +
			2*dphi2_dxz/(pow(params->spacing(0),2)*pow(params->spacing(2),2)) +
			2*dphi2_dyz/(pow(params->spacing(1),2)*pow(params->spacing(2),2));

		return v;
	};

	double thinPlateEnergy(size_t len, double* grad)
	{
		using std::pow;

		auto params = getParams();
		if(len != getParams()->elements())
			throw INVALID_ARGUMENT("Incorrect length of grad array");

		assert(params->ndim() == 3);

		double dphi2_dx2 = 0;
		double dphi2_dy2 = 0;
		double dphi2_dz2 = 0;
		double dphi2_dxz = 0;
		double dphi2_dxy = 0;
		double dphi2_dyz = 0;
		double reg = 0;

		NNInterpNDView<double> dvw(params);
		double ind1[3];
		double ind2[3];

		// Create a counter to iterate over [0,4] ie [-2,2] in 6 directions
		Counter<int, 6> counter(6);
		for(size_t dd=0; dd<6;dd++)
			counter.sz[dd] = 5;

		//integrate over all the knots
		Counter<int64_t, 3> it(3, (int64_t*)params->dim());
		int ii = 0;
		do {
			double tmp_dphi2_dx2 = 0;
			double tmp_dphi2_dy2 = 0;
			double tmp_dphi2_dz2 = 0;
			double tmp_dphi2_dxz = 0;
			double tmp_dphi2_dxy = 0;
			double tmp_dphi2_dyz = 0;
			do {
				double phi = 0;
				double Vat   = vConv  [counter.pos[0]][counter.pos[3]];
				double Vbu   = vConv  [counter.pos[1]][counter.pos[4]];
				double Vcv   = vConv  [counter.pos[2]][counter.pos[5]];
				double dVat  = dvConv [counter.pos[0]][counter.pos[3]];
				double dVbu  = dvConv [counter.pos[1]][counter.pos[4]];
				double dVcv  = dvConv [counter.pos[2]][counter.pos[5]];
				double ddVat = ddvConv[counter.pos[0]][counter.pos[3]];
				double ddVbu = ddvConv[counter.pos[1]][counter.pos[4]];
				double ddVcv = ddvConv[counter.pos[2]][counter.pos[5]];

				// Variables to Compute the Value
				for(size_t dd=0; dd<3; dd++) {
					ind1[dd] = it.pos[dd] + counter.pos[dd] - 2;
					ind2[dd] = it.pos[dd] + counter.pos[dd+3] - 2;
				}
				phi = dvw.get(3, ind1)*dvw.get(3, ind2);

				dphi2_dx2 += phi*ddVat*Vbu*Vcv;
				dphi2_dy2 += phi*Vat*ddVbu*Vcv;
				dphi2_dz2 += phi*Vat*Vbu*ddVcv;
				dphi2_dxy += phi*dVat*dVbu*Vcv;
				dphi2_dyz += phi*Vat*dVbu*dVcv;
				dphi2_dxz += phi*dVat*Vbu*dVcv;

				// Variables to Compute the Derivative
				for(size_t dd=0; dd<3; dd++) {
					ind1[dd] = it.pos[dd] + counter.pos[dd] - counter.pos[dd+3];
					ind2[dd] = it.pos[dd] + counter.pos[dd+3] - counter.pos[dd];
				}
				phi = dvw.get(3, ind1)+dvw.get(3, ind2);

				tmp_dphi2_dx2 += phi*ddVat*Vbu*Vcv;
				tmp_dphi2_dy2 += phi*Vat*ddVbu*Vcv;
				tmp_dphi2_dz2 += phi*Vat*Vbu*ddVcv;
				tmp_dphi2_dxy += phi*dVat*dVbu*Vcv;
				tmp_dphi2_dyz += phi*Vat*dVbu*dVcv;
				tmp_dphi2_dxz += phi*dVat*Vbu*dVcv;

			} while(counter.advance());

			reg = tmp_dphi2_dx2/pow(params->spacing(0),4) + 
				tmp_dphi2_dy2/pow(params->spacing(1),4) + 
				tmp_dphi2_dz2/pow(params->spacing(2),4) + 
				2*tmp_dphi2_dxy/(pow(params->spacing(0),2)*pow(params->spacing(1),2)) +
				2*tmp_dphi2_dxz/(pow(params->spacing(0),2)*pow(params->spacing(2),2)) +
				2*tmp_dphi2_dyz/(pow(params->spacing(1),2)*pow(params->spacing(2),2));
			grad[ii++] = reg;
		} while(it.advance());

		double v = dphi2_dx2/pow(params->spacing(0),4) +
			dphi2_dy2/pow(params->spacing(1),4) +
			dphi2_dz2/pow(params->spacing(2),4) +
			2*dphi2_dxy/(pow(params->spacing(0),2)*pow(params->spacing(1),2)) +
			2*dphi2_dxz/(pow(params->spacing(0),2)*pow(params->spacing(2),2)) +
			2*dphi2_dyz/(pow(params->spacing(1),2)*pow(params->spacing(2),2));

		return v;
	};

	/**
	 * @brief Computes the regularization term by integrating over the
	 * entire space for each knot. Thankfully integrals can be pre-computed
	 * (vConv, dvConv, ddvConv). See equations 66-71 in 
	 * docs/bspline/fmri_dist_correct_2013-12-06.pdf
	 *
	 * @return regularization value
	 */
	double jacobianDet(int dir)
	{
		using std::pow;

		auto params = getParams();
		assert(params->ndim() == 3);

		// We use NN interpolator because it is fast and handles boundary 
		// conditions
		NNInterpNDView<double> dvw(params);
		double ind1[3];
		double ind2[3];
		double reg = 0;

		// Create a counter to iterate over [0,4] ie [-2,2] in 6 directions
		Counter<int, 6> counter(6);
		for(size_t dd=0; dd<6;dd++)
			counter.sz[dd] = 5;

		//integrate over all the knots
		Counter<int64_t, 3> it(3, (int64_t*)params->dim());
		do {
			do {
				for(size_t dd=0; dd<3; dd++) {
					ind1[dd] = it.pos[dd] + counter.pos[dd] - 2;
					ind2[dd] = it.pos[dd] + counter.pos[dd+3] - 2;
				}
				double phi = dvw.get(3, ind1)*dvw.get(3, ind2);

				switch(dir) {
				case 0:
					reg += phi*
						dvConv[counter.pos[0]][counter.pos[3]]*
						 vConv[counter.pos[1]][counter.pos[4]]*
						 vConv[counter.pos[2]][counter.pos[5]];
				break;
				case 1:
					reg += phi*
						 vConv[counter.pos[0]][counter.pos[3]]*
						dvConv[counter.pos[1]][counter.pos[4]]*
						 vConv[counter.pos[2]][counter.pos[5]];
				break;
				case 2:
					reg += phi*
						 vConv[counter.pos[0]][counter.pos[3]]*
						 vConv[counter.pos[1]][counter.pos[4]]*
						dvConv[counter.pos[2]][counter.pos[5]];
				break;
				}
			} while(counter.advance());
		} while(it.advance());

		return reg/pow(params->spacing(dir),2);
	};

	/**
	 * @brief Computes the gradient of regularization for each of the knots.
	 * Thankfully integrals can be pre-computed
	 * (uConv, duConv, dduConv). See equations 54-59 in 
	 * docs/bspline/fmri_dist_correct_2013-12-06.pdf
	 *
	 * @param output gradient of each parameter with respect to the deform 
	 * 			regularization
	 */
	double jacobianDet(int dir, size_t len, double* grad)
	{
		using std::pow;

		auto params = getParams();
		assert(params->ndim() == 3);
		if(len != getParams()->elements())
			throw INVALID_ARGUMENT("Incorrect length of grad array");

		// We use NN interpolator because it is fast and handles boundary 
		// conditions
		NNInterpNDView<double> dvw(params);
		double ind1[3];
		double ind2[3];

		// Create a counter to iterate over [0,4] ie [-2,2] in 6 directions
		Counter<int, 6> counter(6);
		for(size_t dd=0; dd<6;dd++)
			counter.sz[dd] = 5;

		//integrate over all the knots
		Counter<int64_t, 3> it(3, (int64_t*)params->dim());

		double reg = 0;
		size_t ii=0;
		do {
			double dreg = 0;
			do {
				double phi = 1;
				double dphi = 1;

				for(size_t dd=0; dd<3; dd++) {
					ind1[dd] = it.pos[dd] + counter.pos[dd+3] - counter.pos[dd];
					ind2[dd] = it.pos[dd] + counter.pos[dd] - counter.pos[dd+3];
				}
				phi = dvw.get(3, ind1)*dvw.get(3, ind2);
				dphi = dvw.get(3, ind1)+dvw.get(3, ind2);

				switch(dir) {
					case 0:
						reg += phi*
							dvConv[counter.pos[0]][counter.pos[3]]*
							 vConv[counter.pos[1]][counter.pos[4]]*
							 vConv[counter.pos[2]][counter.pos[5]];
						dreg += dphi*
							dvConv[counter.pos[0]][counter.pos[3]]*
							 vConv[counter.pos[1]][counter.pos[4]]*
							 vConv[counter.pos[2]][counter.pos[5]];
					break;
					case 1:
						reg += phi*
							 vConv[counter.pos[0]][counter.pos[3]]*
							dvConv[counter.pos[1]][counter.pos[4]]*
							 vConv[counter.pos[2]][counter.pos[5]];
						dreg += dphi*
							 vConv[counter.pos[0]][counter.pos[3]]*
							dvConv[counter.pos[1]][counter.pos[4]]*
							 vConv[counter.pos[2]][counter.pos[5]];
					break;
					case 2:
						reg += phi*
							 vConv[counter.pos[0]][counter.pos[3]]*
							 vConv[counter.pos[1]][counter.pos[4]]*
							dvConv[counter.pos[2]][counter.pos[5]];
						dreg += dphi*
							 vConv[counter.pos[0]][counter.pos[3]]*
							 vConv[counter.pos[1]][counter.pos[4]]*
							dvConv[counter.pos[2]][counter.pos[5]];
					break;
				}
			} while(counter.advance());

			grad[ii++] = dreg/pow(params->spacing(dir),2);
		} while(it.advance());

		return reg/pow(params->spacing(dir),2);
	};

	/**
	 * @brief Return the parameter image.
	 *
	 * @return
	 */
	ptr<MRImage> getParams() { return dPtrCast<MRImage>(this->parent); };

	/**
	 * @brief How to handle boundaries (ZEROFLUX for constant outside bounds,
	 * ZERO for outside to be 0 and WRAP to wrap values, this might also be
	 * called periodic)
	 */
	BoundaryConditionT m_boundmethod;

	/**
	 * @brief if true, then this assumes the inputs are RAS coordinates rather
	 * than indexes. Default is false
	 */
	bool m_ras;

	double vConv[5][5] = {
		{0.000031001984126984125,0.0010788690476190477,0.0013950892857142857,0.0000992063492063492,0.},
		{0.0010788690476190477,0.058643353174603174,0.11707589285714286,0.02101934523809524,0.0000992063492063492},
		{0.0013950892857142857,0.11707589285714286,0.362016369047619,0.11707589285714286,0.0013950892857142857},
		{0.0000992063492063492,0.02101934523809524,0.11707589285714286,0.058643353174603174,0.0010788690476190477},
		{0.,0.0000992063492063492,0.0013950892857142857,0.0010788690476190477,0.000031001984126984125}};
	double dvConv[5][5] = {
		{0.0015625,0.013541666666666667,-0.0109375,-0.004166666666666667,0.},
		{0.013541666666666667,0.24479166666666666,-0.07604166666666666,-0.178125,-0.004166666666666667},
		{-0.0109375,-0.07604166666666666,0.17395833333333333,-0.07604166666666666,-0.0109375},
		{-0.004166666666666667,-0.178125,-0.07604166666666666,0.24479166666666666,0.013541666666666667},
		{0.,-0.004166666666666667,-0.0109375,0.013541666666666667,0.0015625}};
	double ddvConv[5][5] = {
		{0.041666666666666664,0.,-0.125,0.08333333333333333,0.},
		{0.,0.4166666666666667,-0.75,0.25,0.08333333333333333},
		{-0.125,-0.75,1.75,-0.75,-0.125},
		{0.08333333333333333,0.25,-0.75,0.4166666666666667,0.},
		{0.,0.08333333333333333,-0.125,0.,0.041666666666666664}};

	double U200[9][9][9] = 
{{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
 {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
  {0.0,
   6.561266481901402e-09,
   7.873519778281682e-07,
   7.814468379944569e-06,
   1.5852019820273786e-05,
   7.814468379944569e-06,
   7.873519778281682e-07,
   6.561266481901402e-09,
   0.0},
  {0.0,
   7.873519778281682e-07,
   9.44822373393802e-05,
   0.0009377362055933485,
   0.0019022423784328549,
   0.0009377362055933485,
   9.44822373393802e-05,
   7.873519778281682e-07,
   0.0},
  {0.0,
   7.81446837994457e-06,
   0.0009377362055933487,
   0.00930703184051398,
   0.018879755605946087,
   0.00930703184051398,
   0.0009377362055933487,
   7.81446837994457e-06,
   0.0},
  {0.0,
   1.585201982027379e-05,
   0.0019022423784328547,
   0.018879755605946083,
   0.03829847988578147,
   0.018879755605946083,
   0.0019022423784328547,
   1.585201982027379e-05,
   0.0},
  {0.0,
   7.81446837994457e-06,
   0.0009377362055933487,
   0.00930703184051398,
   0.018879755605946087,
   0.00930703184051398,
   0.0009377362055933487,
   7.81446837994457e-06,
   0.0},
  {0.0,
   7.873519778281682e-07,
   9.44822373393802e-05,
   0.0009377362055933485,
   0.0019022423784328549,
   0.0009377362055933485,
   9.44822373393802e-05,
   7.873519778281682e-07,
   0.0},
  {0.0,
   6.561266481901402e-09,
   7.873519778281682e-07,
   7.814468379944569e-06,
   1.5852019820273786e-05,
   7.814468379944569e-06,
   7.873519778281682e-07,
   6.561266481901402e-09,
   0.0},
  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
 {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
  {0.0,
   0.0,
   8.933550615572699e-23,
   6.617444900424221e-22,
   -1.4097225589434974e-21,
   6.617444900424221e-22,
   8.933550615572699e-23,
   0.0,
   0.0},
  {0.0,
   -1.3896634290890865e-22,
   -2.948733447629033e-20,
   3.095640724418451e-19,
   -1.2023219095968517e-18,
   3.095640724418451e-19,
   -2.948733447629033e-20,
   -1.3896634290890865e-22,
   0.0},
  {0.0,
   -1.2176098616780567e-21,
   -3.814295240604521e-20,
   -2.910855193019005e-18,
   -5.991280757250155e-18,
   -2.910855193019005e-18,
   -3.814295240604521e-20,
   -1.2176098616780567e-21,
   0.0},
  {0.0,
   -4.5135110123955955e-21,
   -1.0007148332605274e-18,
   -7.31572264940066e-18,
   2.069796097619653e-17,
   -7.31572264940066e-18,
   -1.0007148332605274e-18,
   -4.5135110123955955e-21,
   0.0},
  {0.0,
   -1.2176098616780567e-21,
   -3.814295240604521e-20,
   -2.910855193019005e-18,
   -5.991280757250155e-18,
   -2.910855193019005e-18,
   -3.814295240604521e-20,
   -1.2176098616780567e-21,
   0.0},
  {0.0,
   -1.3896634290890865e-22,
   -2.948733447629033e-20,
   3.095640724418451e-19,
   -1.2023219095968517e-18,
   3.095640724418451e-19,
   -2.948733447629033e-20,
   -1.3896634290890865e-22,
   0.0},
  {0.0,
   0.0,
   8.933550615572699e-23,
   6.617444900424221e-22,
   -1.4097225589434974e-21,
   6.617444900424221e-22,
   8.933550615572699e-23,
   0.0,
   0.0},
  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
 {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
  {0.0,
   -5.9051398337112626e-08,
   -7.086167800453517e-06,
   -7.033021541950112e-05,
   -0.00014266817838246408,
   -7.033021541950112e-05,
   -7.086167800453517e-06,
   -5.9051398337112626e-08,
   0.0},
  {0.0,
   -7.086167800453517e-06,
   -0.0008503401360544218,
   -0.00843962585034014,
   -0.017120181405895694,
   -0.00843962585034014,
   -0.0008503401360544218,
   -7.086167800453517e-06,
   0.0},
  {0.0,
   -7.033021541950112e-05,
   -0.00843962585034014,
   -0.08376328656462587,
   -0.16991780045351476,
   -0.08376328656462587,
   -0.00843962585034014,
   -7.033021541950112e-05,
   0.0},
  {0.0,
   -0.00014266817838246406,
   -0.017120181405895694,
   -0.16991780045351473,
   -0.3446863189720333,
   -0.16991780045351473,
   -0.017120181405895694,
   -0.00014266817838246406,
   0.0},
  {0.0,
   -7.033021541950112e-05,
   -0.00843962585034014,
   -0.08376328656462587,
   -0.16991780045351476,
   -0.08376328656462587,
   -0.00843962585034014,
   -7.033021541950112e-05,
   0.0},
  {0.0,
   -7.086167800453517e-06,
   -0.0008503401360544218,
   -0.00843962585034014,
   -0.017120181405895694,
   -0.00843962585034014,
   -0.0008503401360544218,
   -7.086167800453517e-06,
   0.0},
  {0.0,
   -5.9051398337112626e-08,
   -7.086167800453517e-06,
   -7.033021541950112e-05,
   -0.00014266817838246408,
   -7.033021541950112e-05,
   -7.086167800453517e-06,
   -5.9051398337112626e-08,
   0.0},
  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
 {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
  {0.0,
   1.0498026371042242e-07,
   1.2597631645250694e-05,
   0.0001250314940791132,
   0.00025363231712438063,
   0.0001250314940791132,
   1.2597631645250694e-05,
   1.0498026371042242e-07,
   0.0},
  {0.0,
   1.259763164525069e-05,
   0.001511715797430084,
   0.015003779289493578,
   0.0304358780549257,
   0.015003779289493578,
   0.001511715797430084,
   1.259763164525069e-05,
   0.0},
  {0.0,
   0.0001250314940791132,
   0.015003779289493578,
   0.14891250944822357,
   0.3020760896951375,
   0.14891250944822357,
   0.015003779289493578,
   0.0001250314940791132,
   0.0},
  {0.0,
   0.0002536323171243806,
   0.0304358780549257,
   0.3020760896951375,
   0.612775678172503,
   0.3020760896951375,
   0.0304358780549257,
   0.0002536323171243806,
   0.0},
  {0.0,
   0.0001250314940791132,
   0.015003779289493578,
   0.14891250944822357,
   0.3020760896951375,
   0.14891250944822357,
   0.015003779289493578,
   0.0001250314940791132,
   0.0},
  {0.0,
   1.259763164525069e-05,
   0.001511715797430084,
   0.015003779289493578,
   0.0304358780549257,
   0.015003779289493578,
   0.001511715797430084,
   1.259763164525069e-05,
   0.0},
  {0.0,
   1.0498026371042242e-07,
   1.2597631645250694e-05,
   0.0001250314940791132,
   0.00025363231712438063,
   0.0001250314940791132,
   1.2597631645250694e-05,
   1.0498026371042242e-07,
   0.0},
  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
 {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
  {0.0,
   -5.9051398337112626e-08,
   -7.086167800453517e-06,
   -7.033021541950112e-05,
   -0.00014266817838246408,
   -7.033021541950112e-05,
   -7.086167800453517e-06,
   -5.9051398337112626e-08,
   0.0},
  {0.0,
   -7.086167800453517e-06,
   -0.0008503401360544218,
   -0.00843962585034014,
   -0.017120181405895694,
   -0.00843962585034014,
   -0.0008503401360544218,
   -7.086167800453517e-06,
   0.0},
  {0.0,
   -7.033021541950112e-05,
   -0.00843962585034014,
   -0.08376328656462587,
   -0.16991780045351476,
   -0.08376328656462587,
   -0.00843962585034014,
   -7.033021541950112e-05,
   0.0},
  {0.0,
   -0.00014266817838246406,
   -0.017120181405895694,
   -0.16991780045351473,
   -0.3446863189720333,
   -0.16991780045351473,
   -0.017120181405895694,
   -0.00014266817838246406,
   0.0},
  {0.0,
   -7.033021541950112e-05,
   -0.00843962585034014,
   -0.08376328656462587,
   -0.16991780045351476,
   -0.08376328656462587,
   -0.00843962585034014,
   -7.033021541950112e-05,
   0.0},
  {0.0,
   -7.086167800453517e-06,
   -0.0008503401360544218,
   -0.00843962585034014,
   -0.017120181405895694,
   -0.00843962585034014,
   -0.0008503401360544218,
   -7.086167800453517e-06,
   0.0},
  {0.0,
   -5.9051398337112626e-08,
   -7.086167800453517e-06,
   -7.033021541950112e-05,
   -0.00014266817838246408,
   -7.033021541950112e-05,
   -7.086167800453517e-06,
   -5.9051398337112626e-08,
   0.0},
  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
 {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
  {0.0,
   0.0,
   8.933550615572699e-23,
   6.617444900424221e-22,
   -1.4097225589434974e-21,
   6.617444900424221e-22,
   8.933550615572699e-23,
   0.0,
   0.0},
  {0.0,
   -1.3896634290890865e-22,
   -2.948733447629033e-20,
   3.095640724418451e-19,
   -1.2023219095968517e-18,
   3.095640724418451e-19,
   -2.948733447629033e-20,
   -1.3896634290890865e-22,
   0.0},
  {0.0,
   -1.2176098616780567e-21,
   -3.814295240604521e-20,
   -2.910855193019005e-18,
   -5.991280757250155e-18,
   -2.910855193019005e-18,
   -3.814295240604521e-20,
   -1.2176098616780567e-21,
   0.0},
  {0.0,
   -4.5135110123955955e-21,
   -1.0007148332605274e-18,
   -7.31572264940066e-18,
   2.069796097619653e-17,
   -7.31572264940066e-18,
   -1.0007148332605274e-18,
   -4.5135110123955955e-21,
   0.0},
  {0.0,
   -1.2176098616780567e-21,
   -3.814295240604521e-20,
   -2.910855193019005e-18,
   -5.991280757250155e-18,
   -2.910855193019005e-18,
   -3.814295240604521e-20,
   -1.2176098616780567e-21,
   0.0},
  {0.0,
   -1.3896634290890865e-22,
   -2.948733447629033e-20,
   3.095640724418451e-19,
   -1.2023219095968517e-18,
   3.095640724418451e-19,
   -2.948733447629033e-20,
   -1.3896634290890865e-22,
   0.0},
  {0.0,
   0.0,
   8.933550615572699e-23,
   6.617444900424221e-22,
   -1.4097225589434974e-21,
   6.617444900424221e-22,
   8.933550615572699e-23,
   0.0,
   0.0},
  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
 {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
  {0.0,
   6.561266481901402e-09,
   7.873519778281682e-07,
   7.814468379944569e-06,
   1.5852019820273786e-05,
   7.814468379944569e-06,
   7.873519778281682e-07,
   6.561266481901402e-09,
   0.0},
  {0.0,
   7.873519778281682e-07,
   9.44822373393802e-05,
   0.0009377362055933485,
   0.0019022423784328549,
   0.0009377362055933485,
   9.44822373393802e-05,
   7.873519778281682e-07,
   0.0},
  {0.0,
   7.81446837994457e-06,
   0.0009377362055933487,
   0.00930703184051398,
   0.018879755605946087,
   0.00930703184051398,
   0.0009377362055933487,
   7.81446837994457e-06,
   0.0},
  {0.0,
   1.585201982027379e-05,
   0.0019022423784328547,
   0.018879755605946083,
   0.03829847988578147,
   0.018879755605946083,
   0.0019022423784328547,
   1.585201982027379e-05,
   0.0},
  {0.0,
   7.81446837994457e-06,
   0.0009377362055933487,
   0.00930703184051398,
   0.018879755605946087,
   0.00930703184051398,
   0.0009377362055933487,
   7.81446837994457e-06,
   0.0},
  {0.0,
   7.873519778281682e-07,
   9.44822373393802e-05,
   0.0009377362055933485,
   0.0019022423784328549,
   0.0009377362055933485,
   9.44822373393802e-05,
   7.873519778281682e-07,
   0.0},
  {0.0,
   6.561266481901402e-09,
   7.873519778281682e-07,
   7.814468379944569e-06,
   1.5852019820273786e-05,
   7.814468379944569e-06,
   7.873519778281682e-07,
   6.561266481901402e-09,
   0.0},
  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
 {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}}};

	double U110[9][9][9] = 
{{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
	{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
	{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
	{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
	{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
	{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
	{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
	{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
	{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
	{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
	{0.0,
	1.3778659611992942e-08,
	1.6534391534391533e-06,
	1.64103835978836e-05,
	3.328924162257494e-05,
	1.64103835978836e-05,
	1.6534391534391533e-06,
	1.3778659611992942e-08,
	0.0},
	{0.0,
	3.3068783068783074e-07,
	3.9682539682539696e-05,
	0.0003938492063492064,
	0.0007989417989417992,
	0.0003938492063492064,
	3.9682539682539696e-05,
	3.3068783068783074e-07,
	0.0},
	{0.0,
	2.0667989417989414e-07,
	2.4801587301587298e-05,
	0.0002461557539682539,
	0.0004993386243386242,
	0.0002461557539682539,
	2.4801587301587298e-05,
	2.0667989417989414e-07,
	0.0},
	{0.0,
	-1.1022927689594355e-06,
	-0.0001322751322751323,
	-0.0013128306878306879,
	-0.0026631393298059956,
	-0.0013128306878306879,
	-0.0001322751322751323,
	-1.1022927689594355e-06,
	0.0},
	{0.0,
	2.0667989417989414e-07,
	2.4801587301587298e-05,
	0.0002461557539682539,
	0.0004993386243386242,
	0.0002461557539682539,
	2.4801587301587298e-05,
	2.0667989417989414e-07,
	0.0},
	{0.0,
	3.3068783068783074e-07,
	3.9682539682539696e-05,
	0.0003938492063492064,
	0.0007989417989417992,
	0.0003938492063492064,
	3.9682539682539696e-05,
	3.3068783068783074e-07,
	0.0},
	{0.0,
	1.3778659611992942e-08,
	1.6534391534391533e-06,
	1.64103835978836e-05,
	3.328924162257494e-05,
	1.64103835978836e-05,
	1.6534391534391533e-06,
	1.3778659611992942e-08,
	0.0},
	{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
	{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
	{0.0,
	3.306878306878308e-07,
	3.9682539682539696e-05,
	0.00039384920634920654,
	0.000798941798941799,
	0.00039384920634920654,
	3.9682539682539696e-05,
	3.306878306878308e-07,
	0.0},
	{0.0,
	7.936507936507936e-06,
	0.0009523809523809524,
	0.009452380952380952,
	0.019174603174603164,
	0.009452380952380952,
	0.0009523809523809524,
	7.936507936507936e-06,
	0.0},
	{0.0,
	4.960317460317461e-06,
	0.0005952380952380953,
	0.005907738095238096,
	0.011984126984126979,
	0.005907738095238096,
	0.0005952380952380953,
	4.960317460317461e-06,
	0.0},
	{0.0,
	-2.6455026455026446e-05,
	-0.003174603174603174,
	-0.03150793650793652,
	-0.06391534391534391,
	-0.03150793650793652,
	-0.003174603174603174,
	-2.6455026455026446e-05,
	0.0},
	{0.0,
	4.960317460317461e-06,
	0.0005952380952380953,
	0.005907738095238096,
	0.011984126984126979,
	0.005907738095238096,
	0.0005952380952380953,
	4.960317460317461e-06,
	0.0},
	{0.0,
	7.936507936507936e-06,
	0.0009523809523809524,
	0.009452380952380952,
	0.019174603174603164,
	0.009452380952380952,
	0.0009523809523809524,
	7.936507936507936e-06,
	0.0},
	{0.0,
	3.306878306878308e-07,
	3.9682539682539696e-05,
	0.00039384920634920654,
	0.000798941798941799,
	0.00039384920634920654,
	3.9682539682539696e-05,
	3.306878306878308e-07,
	0.0},
	{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
	{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
	{0.0,
	2.0667989417989414e-07,
	2.48015873015873e-05,
	0.00024615575396825384,
	0.0004993386243386241,
	0.00024615575396825384,
	2.48015873015873e-05,
	2.0667989417989414e-07,
	0.0},
	{0.0,
	4.960317460317462e-06,
	0.0005952380952380951,
	0.005907738095238095,
	0.011984126984126982,
	0.005907738095238095,
	0.0005952380952380951,
	4.960317460317462e-06,
	0.0},
	{0.0,
	3.100198412698412e-06,
	0.0003720238095238095,
	0.003692336309523805,
	0.007490079365079362,
	0.003692336309523805,
	0.0003720238095238095,
	3.100198412698412e-06,
	0.0},
	{0.0,
	-1.6534391534391525e-05,
	-0.001984126984126985,
	-0.019692460317460303,
	-0.039947089947089946,
	-0.019692460317460303,
	-0.001984126984126985,
	-1.6534391534391525e-05,
	0.0},
	{0.0,
	3.100198412698412e-06,
	0.0003720238095238095,
	0.003692336309523805,
	0.007490079365079362,
	0.003692336309523805,
	0.0003720238095238095,
	3.100198412698412e-06,
	0.0},
	{0.0,
	4.960317460317462e-06,
	0.0005952380952380951,
	0.005907738095238095,
	0.011984126984126982,
	0.005907738095238095,
	0.0005952380952380951,
	4.960317460317462e-06,
	0.0},
	{0.0,
	2.0667989417989414e-07,
	2.48015873015873e-05,
	0.00024615575396825384,
	0.0004993386243386241,
	0.00024615575396825384,
	2.48015873015873e-05,
	2.0667989417989414e-07,
	0.0},
	{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
	{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
	{0.0,
	-1.1022927689594353e-06,
	-0.00013227513227513228,
	-0.001312830687830688,
	-0.002663139329805994,
	-0.001312830687830688,
	-0.00013227513227513228,
	-1.1022927689594353e-06,
	0.0},
	{0.0,
	-2.6455026455026446e-05,
	-0.003174603174603174,
	-0.031507936507936485,
	-0.06391534391534393,
	-0.031507936507936485,
	-0.003174603174603174,
	-2.6455026455026446e-05,
	0.0},
	{0.0,
	-1.653439153439153e-05,
	-0.001984126984126985,
	-0.019692460317460303,
	-0.03994708994708995,
	-0.019692460317460303,
	-0.001984126984126985,
	-1.653439153439153e-05,
	0.0},
	{0.0,
	8.818342151675487e-05,
	0.010582010582010571,
	0.10502645502645507,
	0.2130511463844796,
	0.10502645502645507,
	0.010582010582010571,
	8.818342151675487e-05,
	0.0},
	{0.0,
	-1.653439153439153e-05,
	-0.001984126984126985,
	-0.019692460317460303,
	-0.03994708994708995,
	-0.019692460317460303,
	-0.001984126984126985,
	-1.653439153439153e-05,
	0.0},
	{0.0,
	-2.6455026455026446e-05,
	-0.003174603174603174,
	-0.031507936507936485,
	-0.06391534391534393,
	-0.031507936507936485,
	-0.003174603174603174,
	-2.6455026455026446e-05,
	0.0},
	{0.0,
	-1.1022927689594353e-06,
	-0.00013227513227513228,
	-0.001312830687830688,
	-0.002663139329805994,
	-0.001312830687830688,
	-0.00013227513227513228,
	-1.1022927689594353e-06,
	0.0},
	{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
	{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
	{0.0,
	2.0667989417989414e-07,
	2.48015873015873e-05,
	0.00024615575396825384,
	0.0004993386243386241,
	0.00024615575396825384,
	2.48015873015873e-05,
	2.0667989417989414e-07,
	0.0},
	{0.0,
	4.960317460317462e-06,
	0.0005952380952380951,
	0.005907738095238095,
	0.011984126984126982,
	0.005907738095238095,
	0.0005952380952380951,
	4.960317460317462e-06,
	0.0},
	{0.0,
	3.100198412698412e-06,
	0.0003720238095238095,
	0.003692336309523805,
	0.007490079365079362,
	0.003692336309523805,
	0.0003720238095238095,
	3.100198412698412e-06,
	0.0},
	{0.0,
	-1.6534391534391525e-05,
	-0.001984126984126985,
	-0.019692460317460303,
	-0.039947089947089946,
	-0.019692460317460303,
	-0.001984126984126985,
	-1.6534391534391525e-05,
	0.0},
	{0.0,
	3.100198412698412e-06,
	0.0003720238095238095,
	0.003692336309523805,
	0.007490079365079362,
	0.003692336309523805,
	0.0003720238095238095,
	3.100198412698412e-06,
	0.0},
	{0.0,
	4.960317460317462e-06,
	0.0005952380952380951,
	0.005907738095238095,
	0.011984126984126982,
	0.005907738095238095,
	0.0005952380952380951,
	4.960317460317462e-06,
	0.0},
	{0.0,
	2.0667989417989414e-07,
	2.48015873015873e-05,
	0.00024615575396825384,
	0.0004993386243386241,
	0.00024615575396825384,
	2.48015873015873e-05,
	2.0667989417989414e-07,
	0.0},
	{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
	{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
	{0.0,
	3.306878306878308e-07,
	3.9682539682539696e-05,
	0.00039384920634920654,
	0.000798941798941799,
	0.00039384920634920654,
	3.9682539682539696e-05,
	3.306878306878308e-07,
	0.0},
	{0.0,
	7.936507936507936e-06,
	0.0009523809523809524,
	0.009452380952380952,
	0.019174603174603164,
	0.009452380952380952,
	0.0009523809523809524,
	7.936507936507936e-06,
	0.0},
	{0.0,
	4.960317460317461e-06,
	0.0005952380952380953,
	0.005907738095238096,
	0.011984126984126979,
	0.005907738095238096,
	0.0005952380952380953,
	4.960317460317461e-06,
	0.0},
	{0.0,
	-2.6455026455026446e-05,
	-0.003174603174603174,
	-0.03150793650793652,
	-0.06391534391534391,
	-0.03150793650793652,
	-0.003174603174603174,
	-2.6455026455026446e-05,
	0.0},
	{0.0,
	4.960317460317461e-06,
	0.0005952380952380953,
	0.005907738095238096,
	0.011984126984126979,
	0.005907738095238096,
	0.0005952380952380953,
	4.960317460317461e-06,
	0.0},
	{0.0,
	7.936507936507936e-06,
	0.0009523809523809524,
	0.009452380952380952,
	0.019174603174603164,
	0.009452380952380952,
	0.0009523809523809524,
	7.936507936507936e-06,
	0.0},
	{0.0,
	3.306878306878308e-07,
	3.9682539682539696e-05,
	0.00039384920634920654,
	0.000798941798941799,
	0.00039384920634920654,
	3.9682539682539696e-05,
	3.306878306878308e-07,
	0.0},
	{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
	{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
	{0.0,
	1.3778659611992942e-08,
	1.6534391534391533e-06,
	1.64103835978836e-05,
	3.328924162257494e-05,
	1.64103835978836e-05,
	1.6534391534391533e-06,
	1.3778659611992942e-08,
	0.0},
	{0.0,
	3.3068783068783074e-07,
	3.9682539682539696e-05,
	0.0003938492063492064,
	0.0007989417989417992,
	0.0003938492063492064,
	3.9682539682539696e-05,
	3.3068783068783074e-07,
	0.0},
	{0.0,
	2.0667989417989414e-07,
	2.4801587301587298e-05,
	0.0002461557539682539,
	0.0004993386243386242,
	0.0002461557539682539,
	2.4801587301587298e-05,
	2.0667989417989414e-07,
	0.0},
	{0.0,
	-1.1022927689594355e-06,
	-0.0001322751322751323,
	-0.0013128306878306879,
	-0.0026631393298059956,
	-0.0013128306878306879,
	-0.0001322751322751323,
	-1.1022927689594355e-06,
	0.0},
	{0.0,
	2.0667989417989414e-07,
	2.4801587301587298e-05,
	0.0002461557539682539,
	0.0004993386243386242,
	0.0002461557539682539,
	2.4801587301587298e-05,
	2.0667989417989414e-07,
	0.0},
	{0.0,
	3.3068783068783074e-07,
	3.9682539682539696e-05,
	0.0003938492063492064,
	0.0007989417989417992,
	0.0003938492063492064,
	3.9682539682539696e-05,
	3.3068783068783074e-07,
	0.0},
	{0.0,
	1.3778659611992942e-08,
	1.6534391534391533e-06,
	1.64103835978836e-05,
	3.328924162257494e-05,
	1.64103835978836e-05,
	1.6534391534391533e-06,
	1.3778659611992942e-08,
	0.0},
	{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
	{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
	{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
	{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
	{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
	{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
	{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
	{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
	{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
	{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}}};


};

/**
 * @}
 */

} // namespace npl

#endif //ACCESSORS_H
