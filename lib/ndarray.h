/*******************************************************************************
This file is part of Neuro Programs and Libraries (NPL), 

Written and Copyrighted by by Micah C. Chambers (micahc.vt@gmail.com)

The Neuro Programs and Libraries is free software: you can redistribute it and/or
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

#ifndef NDARRAY_H
#define NDARRAY_H

#include "npltypes.h"
#include "slicer.h"

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
	virtual TYPE GETFUNC(std::initializer_list<size_t> index) const = 0; \
	virtual TYPE GETFUNC(size_t d, const size_t* index) const = 0; \
	virtual TYPE GETFUNC(size_t index) const = 0; \
	virtual void SETFUNC(std::initializer_list<size_t> index, TYPE) = 0; \
	virtual void SETFUNC(size_t d, const size_t* index, TYPE) = 0; \
	virtual void SETFUNC(size_t index, TYPE) = 0; \

#define GETSET(TYPE, GETFUNC, SETFUNC) \
	TYPE GETFUNC(std::initializer_list<size_t> index) const; \
	TYPE GETFUNC(size_t d, const size_t* index) const; \
	TYPE GETFUNC(size_t index) const; \
	void SETFUNC(std::initializer_list<size_t> index, TYPE); \
	void SETFUNC(size_t d, const size_t* index, TYPE); \
	void SETFUNC(size_t index, TYPE); \

// Iterator Functions
#define CITERFUNCS(TYPE, GETFUNC)\
	TYPE GETFUNC() const \
	{\
		assert(m_parent);\
		return m_parent->GETFUNC(Slicer::get());\
	};\
	TYPE GETFUNC(size_t d, int64_t* dindex, bool* outside=NULL) \
	{\
		assert(m_parent);\
		return m_parent->GETFUNC(Slicer::offset(d, dindex, outside));\
	};\
	TYPE GETFUNC(std::initializer_list<int64_t> dindex, bool* outside=NULL) \
	{\
		assert(m_parent);\
		return m_parent->GETFUNC(Slicer::offset(dindex, outside));\
	};\

#define ITERFUNCS(TYPE, GETFUNC, SETFUNC)\
	TYPE GETFUNC() const \
	{\
		assert(m_parent);\
		return m_parent->GETFUNC(Slicer::get());\
	};\
	void SETFUNC(TYPE v) const \
	{\
		assert(m_parent);\
		m_parent->SETFUNC(Slicer::get(), v);\
	};\
	TYPE GETFUNC(size_t d, int64_t* dindex, bool* outside = NULL) \
	{\
		assert(m_parent);\
		return m_parent->GETFUNC(Slicer::offset(d, dindex, outside));\
	};\
	TYPE GETFUNC(std::initializer_list<int64_t> dindex, bool* outside = NULL) \
	{\
		assert(m_parent);\
		return m_parent->GETFUNC(Slicer::offset(dindex, outside));\
	};\
	void SETFUNC(size_t d, int64_t* dindex, TYPE v, bool* outside=NULL) \
	{\
		assert(m_parent);\
		m_parent->SETFUNC(Slicer::offset(d, dindex, outside), v);\
	};\
	void SETFUNC(std::initializer_list<int64_t> dindex, TYPE v, \
			bool* outside=NULL) \
	{\
		assert(m_parent);\
		m_parent->SETFUNC(Slicer::offset(dindex, outside), v);\
	};


namespace npl {

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
	virtual size_t getAddr(std::initializer_list<size_t> index) const = 0;
	virtual size_t getAddr(const std::vector<size_t>& index) const = 0;
	virtual size_t getAddr(const size_t* index) const = 0;

	VIRTGETSET(double, get_dbl, set_dbl);
	VIRTGETSET(int64_t, get_int, set_int);
	VIRTGETSET(cdouble_t, get_cdbl, set_cdbl);
	VIRTGETSET(cfloat_t, get_cfloat, set_cfloat);
	VIRTGETSET(rgba_t, get_rgba, set_rgba);
	VIRTGETSET(long double, get_quad, set_quad);
	VIRTGETSET(cquad_t, get_cquad, set_cquad);

	virtual size_t ndim() const = 0;
	virtual size_t bytes() const = 0;
	virtual size_t elements() const = 0;
	virtual size_t dim(size_t dir) const = 0;
	virtual const size_t* dim() const = 0;

	virtual std::shared_ptr<NDArray> clone() const = 0;
	virtual int opself(const NDArray* right, double(*func)(double,double), 
			bool elevR) = 0;
	virtual std::shared_ptr<NDArray> opnew(const NDArray* right, 
			double(*func)(double,double), bool elevR) = 0;

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
	GETSET(double, get_dbl, set_dbl);
	GETSET(int64_t, get_int, set_int);
	GETSET(cdouble_t, get_cdbl, set_cdbl);
	GETSET(cfloat_t, get_cfloat, set_cfloat);
	GETSET(rgba_t, get_rgba, set_rgba);
	GETSET(long double, get_quad, set_quad);
	GETSET(cquad_t, get_cquad, set_cquad);

	// Get Address
	virtual size_t getAddr(std::initializer_list<size_t> index) const;
	virtual size_t getAddr(const std::vector<size_t>& index) const;
	virtual size_t getAddr(const size_t* index) const;
	
	T& operator[](std::initializer_list<size_t> index);
	T& operator[](const std::vector<size_t>& index);
	T& operator[](const size_t* index);
	T& operator[](size_t pixel);
	
	const T& operator[](std::initializer_list<size_t> index) const;
	const T& operator[](const std::vector<size_t>& index) const;
	const T& operator[](const size_t* index) const;
	const T& operator[](size_t pixel) const;

	/* 
	 * General Information 
	 */
	virtual size_t ndim() const;
	virtual size_t bytes() const;
	virtual size_t elements() const;
	virtual size_t dim(size_t dir) const;
	virtual const size_t* dim() const;

	virtual void resize(size_t dim[D]);

	/* 
	 * Higher Level Operations
	 */
	virtual std::shared_ptr<NDArray> clone() const;
	virtual int opself(const NDArray* right, double(*func)(double,double), 
			bool elevR);
	virtual std::shared_ptr<NDArray> opnew(const NDArray* right, 
			double(*func)(double,double), bool elevR);

	protected:
	T* _m_data;
	size_t _m_dim[D];	// overall image dimension
	size_t _m_stride[D]; // steps between pixels

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
#undef CITERFUNCS
#undef ITERFUNCS

} //npl

#endif
