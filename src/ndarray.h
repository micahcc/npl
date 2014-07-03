#ifndef NDARRAY_H
#define NDARRAY_H

#include <cstddef>
#include <cmath>
#include <initializer_list>
#include <vector>
#include <cstdint>

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
	VIRTGETSET(int32_t, int32);
	VIRTGETSET(int64_t, int64);

	virtual size_t ndim() const = 0;
	virtual size_t bytes() const = 0;
	virtual size_t dim(size_t dir) const = 0;
	virtual const size_t* dim() const = 0;

//	template <typename T = double>
//	class iterator : public Slicer
//	{
//	public:
//		iterator();
//		iterator(iterator&& other);
//		iterator(const iterator& other);
//	};
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
	GETSET(int32_t, int32);
	GETSET(int64_t, int64);

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

typedef NDArrayStore<2, double> Matrix;

#undef VIRTGETSET
#undef GETSET

#endif
