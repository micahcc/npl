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
	TYPE GETFUNC(size_t d, int64_t* dindex, bool* outside) \
	{\
		assert(m_parent);\
		return m_parent->GETFUNC(Slicer::offset(d, dindex, outside));\
	};\
	TYPE GETFUNC(std::initializer_list<int64_t> dindex, bool* outside) \
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
	TYPE GETFUNC(size_t d, int64_t* dindex, bool* outside) \
	{\
		assert(m_parent);\
		return m_parent->GETFUNC(Slicer::offset(d, dindex, outside));\
	};\
	TYPE GETFUNC(std::initializer_list<int64_t> dindex, bool* outside) \
	{\
		assert(m_parent);\
		return m_parent->GETFUNC(Slicer::offset(dindex, outside));\
	};\
	void SETFUNC(size_t d, int64_t* dindex, TYPE v, bool* outside) \
	{\
		assert(m_parent);\
		m_parent->SETFUNC(Slicer::offset(d, dindex, outside), v);\
	};\
	void SETFUNC(std::initializer_list<int64_t> dindex, TYPE v, \
			bool* outside) \
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
	class iterator;
	class const_iterator;

	virtual iterator begin() {
		return iterator(this);
	};

	virtual iterator begin(const std::list<size_t>& order) {
		return iterator(this, order);
	};
	
	virtual const_iterator cbegin() const {
		return const_iterator(this);
	};

	virtual const_iterator cbegin(const std::list<size_t>& order) const {
		return const_iterator(this, order);
	};

	/*
	 * get / set functions
	 */
	// Get Address
	virtual size_t getAddr(std::initializer_list<size_t> index) const = 0;
	virtual size_t getAddr(const std::vector<size_t>& index) const = 0;

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

	/* 
	 * Iterator Declaration
	 */
	class iterator : public virtual Slicer {
	public:
		iterator(iterator&& other) : m_parent(other.m_parent) {} ;
		iterator(NDArray* parent, const std::list<size_t>& order) 
		{
			m_parent = parent;
			std::vector<size_t> dim(m_parent->ndim());
			for(size_t ii=0; ii<m_parent->ndim(); ii++)
				dim[ii] = m_parent->dim(ii);
			updateDim(dim);
			setOrder(order);
		};
		iterator(NDArray* parent) 
		{
			m_parent = parent;
			std::vector<size_t> dim(m_parent->ndim());
			std::list<size_t> order;
			for(size_t ii=0; ii<m_parent->ndim(); ii++)
				dim[ii] = m_parent->dim(ii);
			updateDim(dim);
			setOrder(order);
		};

		ITERFUNCS(double, get_dbl, set_dbl);
		ITERFUNCS(int64_t, get_int, set_int);
		ITERFUNCS(cdouble_t, get_cdbl, set_cdbl);
		ITERFUNCS(cfloat_t, get_cfloat, set_cfloat);
		ITERFUNCS(rgba_t, get_rgba, set_rgba);
		ITERFUNCS(long double, get_quad, set_quad);
		ITERFUNCS(cquad_t, get_cquad, set_cquad);

	protected:
		iterator() {} ;
		NDArray* m_parent;
	};

	class const_iterator : public virtual Slicer {
	public:
		const_iterator(const_iterator&& other) : m_parent(other.m_parent) {} ;
		const_iterator(const NDArray* parent, const std::list<size_t>& order) 
		{
			m_parent = parent;
			std::vector<size_t> dim(m_parent->ndim());
			for(size_t ii=0; ii<m_parent->ndim(); ii++)
				dim[ii] = m_parent->dim(ii);
			updateDim(dim);
			setOrder(order);
		};
		const_iterator(const NDArray* parent) 
		{
			m_parent = parent;
			std::vector<size_t> dim(m_parent->ndim());
			std::list<size_t> order;
			for(size_t ii=0; ii<m_parent->ndim(); ii++)
				dim[ii] = m_parent->dim(ii);
			updateDim(dim);
			setOrder(order);
		};
		
		CITERFUNCS(double, get_dbl);
		CITERFUNCS(int64_t, get_int);
		CITERFUNCS(cdouble_t, get_cdbl);
		CITERFUNCS(cfloat_t, get_cfloat);
		CITERFUNCS(rgba_t, get_rgba);
		CITERFUNCS(long double, get_quad);
		CITERFUNCS(cquad_t, get_cquad);
	
	protected:
		const_iterator() {} ;
		const NDArray* m_parent;
	};

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

	T* _m_data;
	size_t _m_dim[D];	// overall image dimension
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
