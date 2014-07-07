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

#define ITER(TYPE, CALLFUNC, CNAME, FNAME)										\
	class CNAME : public Slicer													\
	{																			\
	public:																		\
		CNAME() : Slicer(), m_parent(NULL) {} ;									\
		CNAME(NDArray* parent, const std::list<size_t>& order) {				\
			m_parent = parent;													\
			std::vector<size_t> dim(m_parent->ndim());							\
			for(size_t ii=0; ii<m_parent->ndim(); ii++)							\
				dim[ii] = m_parent->dim(ii);									\
			updateDim(dim);														\
			setOrder(order);													\
		}																		\
		CNAME(NDArray* parent) {												\
			m_parent = parent;													\
			std::vector<size_t> dim(m_parent->ndim());							\
			for(size_t ii=0; ii<m_parent->ndim(); ii++)							\
				dim[ii] = m_parent->dim(ii);									\
			updateDim(dim);														\
		}																		\
		TYPE operator*() {														\
			assert(m_parent);													\
			return m_parent->CALLFUNC(Slicer::get());							\
		};																		\
		TYPE get() {															\
			assert(m_parent);													\
			return m_parent->CALLFUNC(Slicer::get());							\
		};																		\
		void set(TYPE v) {														\
			assert(m_parent);													\
			m_parent->CALLFUNC(Slicer::get(), v);								\
		};																		\
	private:																	\
		NDArray* m_parent;														\
	};																			\
	CNAME FNAME(const std::list<size_t>& order) { return CNAME(this, order); };

#define CONSTITER(TYPE, CALLFUNC, CNAME, FNAME)										\
	class CNAME : public Slicer													\
	{																			\
	public:																		\
		CNAME() : Slicer(), m_parent(NULL) {} ;									\
		CNAME(const NDArray* parent, const std::list<size_t>& order) : m_parent(parent) { \
			m_parent = parent;													\
			std::vector<size_t> dim(m_parent->ndim());							\
			for(size_t ii=0; ii<m_parent->ndim(); ii++)							\
				dim[ii] = m_parent->dim(ii);									\
			updateDim(dim);														\
			setOrder(order);													\
		}																		\
		CNAME(NDArray* parent) {												\
			m_parent = parent;													\
			std::vector<size_t> dim(m_parent->ndim());							\
			for(size_t ii=0; ii<m_parent->ndim(); ii++)							\
				dim[ii] = m_parent->dim(ii);									\
			updateDim(dim);														\
		}																		\
		TYPE operator*() {														\
			assert(m_parent);													\
			return m_parent->CALLFUNC(Slicer::get());							\
		};																		\
		TYPE get() {															\
			assert(m_parent);													\
			return m_parent->CALLFUNC(Slicer::get());							\
		};																		\
		TYPE offset(int64_t* dindex) {											\
			assert(m_parent);													\
			return m_parent->CALLFUNC(Slicer::offset(dindex));					\
		};																		\
		TYPE offset(std::initializer_list<int64_t> dindex) {					\
			assert(m_parent);													\
			return m_parent->CALLFUNC(Slicer::offset(dindex));					\
		};																		\
	private:																	\
		const NDArray* m_parent;												\
	};																			\
	CNAME FNAME(const std::list<size_t>& order) const { return CNAME(this, order); };



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

	VIRTGETSET(double, dbl);
	VIRTGETSET(int64_t, int64);
	VIRTGETSET(cdouble_t, cdbl);
	VIRTGETSET(cfloat_t, cfloat);
	VIRTGETSET(rgba_t, rgba);
	VIRTGETSET(long double, quad);
	VIRTGETSET(cquad_t, cquad);

	ITER(double, dbl, dbl_iter, begin_dbl);
	ITER(int64_t, int64, int64_iter, begin_int64);
	ITER(cdouble_t, cdbl, cdbl_iter, begin_cdbl);
	ITER(cfloat_t, cfloat, cfloat_iter, begin_cfloat);
	ITER(rgba_t, rgba, rgba_iter, begin_rgba);
	ITER(long double, quad, quad_iter, begin_quad);
	ITER(cquad_t, cquad, cquad_iter, begin_cquad);
	
	CONSTITER(double, dbl, dbl_citer, cbegin_dbl);
	CONSTITER(int64_t, int64, int64_citer, cbegin_int64);
	CONSTITER(cdouble_t, cdbl, cdbl_citer, cbegin_cdbl);
	CONSTITER(cfloat_t, cfloat, cfloat_citer, cbegin_cfloat);
	CONSTITER(rgba_t, rgba, rgba_citer, cbegin_rgba);
	CONSTITER(long double, quad, quad_citer, cbegin_quad);
	CONSTITER(cquad_t, cquad, cquad_citer, cbegin_cquad);

	virtual size_t ndim() const = 0;
	virtual size_t bytes() const = 0;
	virtual size_t dim(size_t dir) const = 0;
	virtual const size_t* dim() const = 0;

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
