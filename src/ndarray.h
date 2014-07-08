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

// virtual get and set function macro, ie
// VIRTGETSET(double, dbl); 
// generates:
// double dbl(std::initializer_list<size_t> index) const;
// void dbl(std::initializer_list<size_t> index, double newval) const;
// double dbl(const std::vector<size_t>& index) const;
// void dbl(const std::vector<size_t>& index, double newval) const;
// double dbl(const size_t* index) const;
// void dbl(const size_t* index, double newval) const;
#define VIRTGETSET(TYPE, FNAME) \
	virtual TYPE FNAME(std::initializer_list<size_t> index) const = 0; \
	virtual TYPE FNAME(const std::vector<size_t>& index) const = 0; \
	virtual TYPE FNAME(const size_t* index) const = 0; \
	virtual TYPE FNAME(size_t index) const = 0; \
	virtual void FNAME(std::initializer_list<size_t> index, TYPE) = 0; \
	virtual void FNAME(const std::vector<size_t>& index, TYPE) = 0; \
	virtual void FNAME(const size_t* index, TYPE) = 0; \
	virtual void FNAME(size_t index, TYPE) = 0; \

#define GETSET(TYPE, FNAME) \
	TYPE FNAME(std::initializer_list<size_t> index) const; \
	TYPE FNAME(const std::vector<size_t>& index) const; \
	TYPE FNAME(const size_t* index) const; \
	TYPE FNAME(size_t index) const; \
	void FNAME(std::initializer_list<size_t> index, TYPE); \
	void FNAME(const std::vector<size_t>& index, TYPE); \
	void FNAME(const size_t* index, TYPE); \
	void FNAME(size_t index, TYPE); \

#define CITERFUNCS(TYPE, FUNCNAME)\
	TYPE FUNCNAME() const \
	{\
		assert(m_parent);\
		return m_parent->FUNCNAME(Slicer::get());\
	};\
	TYPE FUNCNAME(int64_t* dindex, bool* outside) \
	{\
		assert(m_parent);\
		return m_parent->FUNCNAME(Slicer::offset(dindex, outside));\
	};\
	TYPE FUNCNAME(std::initializer_list<int64_t> dindex, bool* outside) \
	{\
		assert(m_parent);\
		return m_parent->FUNCNAME(Slicer::offset(dindex, outside));\
	};\

#define ITERFUNCS(TYPE, FUNCNAME)\
	TYPE FUNCNAME() const \
	{\
		assert(m_parent);\
		return m_parent->FUNCNAME(Slicer::get());\
	};\
	void FUNCNAME(TYPE v) const \
	{\
		assert(m_parent);\
		m_parent->FUNCNAME(Slicer::get(), v);\
	};\
	TYPE FUNCNAME(int64_t* dindex, bool* outside) \
	{\
		assert(m_parent);\
		return m_parent->FUNCNAME(Slicer::offset(dindex, outside));\
	};\
	TYPE FUNCNAME(std::initializer_list<int64_t> dindex, bool* outside) \
	{\
		assert(m_parent);\
		return m_parent->FUNCNAME(Slicer::offset(dindex, outside));\
	};\
	void FUNCNAME(int64_t* dindex, TYPE v, bool* outside) \
	{\
		assert(m_parent);\
		m_parent->FUNCNAME(Slicer::offset(dindex, outside), v);\
	};\
	void FUNCNAME(std::initializer_list<int64_t> dindex, TYPE v, \
			bool* outside) \
	{\
		assert(m_parent);\
		m_parent->FUNCNAME(Slicer::offset(dindex, outside), v);\
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

	iterator begin() {
		return iterator(this);
	};

	iterator begin(const std::list<size_t>& order) {
		return iterator(this, order);
	};
	
	const_iterator cbegin() const {
		return const_iterator(this);
	};

	const_iterator cbegin(const std::list<size_t>& order) const {
		return const_iterator(this, order);
	};

	/*
	 * get / set functions
	 */
	// Get Address
	virtual size_t getAddr(std::initializer_list<size_t> index) const = 0;
	virtual size_t getAddr(const std::vector<size_t>& index) const = 0;
	virtual size_t getAddr(const size_t* index) const = 0;

	VIRTGETSET(double, dbl);
	VIRTGETSET(int64_t, int64);
	VIRTGETSET(cdouble_t, cdbl);
	VIRTGETSET(cfloat_t, cfloat);
	VIRTGETSET(rgba_t, rgba);
	VIRTGETSET(long double, quad);
	VIRTGETSET(cquad_t, cquad);

	virtual size_t ndim() const = 0;
	virtual size_t bytes() const = 0;
	virtual size_t dim(size_t dir) const = 0;
	virtual const size_t* dim() const = 0;

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

		ITERFUNCS(double, dbl);
		ITERFUNCS(int64_t, int64);
		ITERFUNCS(cdouble_t, cdbl);
		ITERFUNCS(cfloat_t, cfloat);
		ITERFUNCS(rgba_t, rgba);
		ITERFUNCS(long double, quad);
		ITERFUNCS(cquad_t, cquad);

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
		
		CITERFUNCS(double, dbl);
		CITERFUNCS(int64_t, int64);
		CITERFUNCS(cdouble_t, cdbl);
		CITERFUNCS(cfloat_t, cfloat);
		CITERFUNCS(rgba_t, rgba);
		CITERFUNCS(long double, quad);
		CITERFUNCS(cquad_t, cquad);
	
	protected:
		const_iterator() {} ;
		const NDArray* m_parent;
	};

};

#undef ITERFUNCS
#undef CITERFUNCS

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
	NDArrayStore(size_t dim[D]);
	NDArrayStore(const std::vector<size_t>& dim);
	
	~NDArrayStore() { delete[] _m_data; };

	/* 
	 * get / set functions
	 */
	GETSET(double, dbl);
	GETSET(int64_t, int64);
	GETSET(cdouble_t, cdbl);
	GETSET(cfloat_t, cfloat);
	GETSET(rgba_t, rgba);
	GETSET(long double, quad);
	GETSET(cquad_t, cquad);

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
	virtual size_t dim(size_t dir) const;
	virtual const size_t* dim() const;

	virtual void resize(size_t dim[D]);

	T* _m_data;
	size_t _m_dim[D];	// overall image dimension
};

#undef VIRTGETSET
#undef GETSET

} //npl

#endif
