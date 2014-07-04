#ifndef NDARRAY_H
#define NDARRAY_H

#include <cstddef>
#include <cmath>
#include <initializer_list>
#include <vector>
#include <cstdint>
#include <complex>

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

struct c32_t;
struct rgba_t;

struct c64_t: public std::complex<double>
{
	explicit operator float();
	explicit operator double();
	explicit operator int64_t();
	explicit operator uint64_t();
	explicit operator int32_t();
	explicit operator uint32_t();
	explicit operator int16_t();
	explicit operator uint16_t();
	explicit operator int8_t();
	explicit operator uint8_t();
	explicit operator c32_t();
	explicit operator std::complex<float>();
//	explicit operator rgba_t();
	
	c64_t(double re = 0, double im = 0) : std::complex<double>(re, im) {} ;
//	explicit c64_t(float);
//	explicit c64_t(double);
//	explicit c64_t(int64_t);
	explicit c64_t(c32_t);
	explicit c64_t(std::complex<float>);
	explicit c64_t(rgba_t);
};

struct c32_t : public std::complex<float>
{
	explicit operator float();
	explicit operator double();
	explicit operator int64_t();
	explicit operator uint64_t();
	explicit operator int32_t();
	explicit operator uint32_t();
	explicit operator int16_t();
	explicit operator uint16_t();
	explicit operator int8_t();
	explicit operator uint8_t();
	explicit operator c64_t();
	explicit operator std::complex<double>();
//	explicit operator rgba_t();
	
	c32_t(double re = 0, double im = 0) : std::complex<float>(re, im) {} ;
//	explicit c32_t(float);
//	explicit c32_t(double);
//	explicit c32_t(int64_t);
	explicit c32_t(c64_t);
	explicit c32_t(std::complex<double>);
	explicit c32_t(rgba_t);
};

struct rgba_t
{
	char red;
	char green;
	char blue;
	char alpha;

	explicit operator float();
	explicit operator double();
	explicit operator int64_t();
	explicit operator uint64_t();
	explicit operator int32_t();
	explicit operator uint32_t();
	explicit operator int16_t();
	explicit operator uint16_t();
	explicit operator int8_t();
	explicit operator uint8_t();
//	explicit operator c64_t();
//	explicit operator c32_t();
	
	rgba_t(char r, char g, char b, char a = 0) : 
		red(r), green(g), blue(b), alpha(a) {} ;
	explicit rgba_t(float);
	explicit rgba_t(double);
	explicit rgba_t(int64_t);
	explicit rgba_t(uint64_t);
	explicit rgba_t(int32_t);
	explicit rgba_t(uint32_t);
	explicit rgba_t(int16_t);
	explicit rgba_t(uint16_t);
	explicit rgba_t(int8_t);
	explicit rgba_t(uint8_t);
	explicit rgba_t(c64_t);
	explicit rgba_t(c32_t);
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
	virtual size_t getAddr(std::initializer_list<size_t> index) const = 0;
	virtual size_t getAddr(const std::vector<size_t>& index) const = 0;
	virtual size_t getAddr(const size_t* index) const = 0;

	VIRTGETSET(double, dbl);
	VIRTGETSET(int64_t, int64);
	VIRTGETSET(c64_t, c64);
	VIRTGETSET(c32_t, c32);
	VIRTGETSET(rgba_t, rgba);

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
	GETSET(int64_t, int64);
	GETSET(c64_t, c64);
	GETSET(c32_t, c32);
	GETSET(rgba_t, rgba);

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
