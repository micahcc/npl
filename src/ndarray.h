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
// double dbl(std::initializer_list<size_t> index) const = 0;
// void dbl(std::initializer_list<size_t> index, double newval) const = 0;
// double dbl(const std::vector<size_t>& index) const = 0;
// void dbl(const std::vector<size_t>& index, double newval) const = 0;
// double dbl(const size_t* index) const = 0;
// void dbl(const size_t* index, double newval) const = 0;
#define VIRTGETSET(TYPE, FNAME) \
	virtual TYPE FNAME(std::initializer_list<size_t> index) const = 0; \
	virtual TYPE FNAME(const std::vector<size_t>& index) const = 0; \
	virtual TYPE FNAME(const size_t* index) const = 0; \
	virtual TYPE FNAME(size_t index) const = 0; \
	virtual void FNAME(std::initializer_list<size_t> index, TYPE) = 0; \
	virtual void FNAME(const std::vector<size_t>& index, TYPE) = 0; \
	virtual void FNAME(const size_t* index, TYPE) = 0; \
	virtual void FNAME(size_t index, TYPE) = 0; \


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

// macros to actually create the get/set functions, note that get and set are 
// the same name, just different arguments
#define GETSET(TYPE, FNAME, TEMPLTYPE) \
	virtual TYPE FNAME(std::initializer_list<size_t> index) const 		\
	{																	\
		return _m_data[getAddr(index)];									\
	}																	\
	virtual TYPE FNAME(const std::vector<size_t>& index) const 			\
	{																	\
		return _m_data[getAddr(index)];									\
	}																	\
	virtual TYPE FNAME(const size_t* index) const 						\
	{																	\
		return _m_data[getAddr(index)];									\
	}																	\
	virtual TYPE FNAME(size_t addr)	const								\
	{																	\
		return _m_data[addr];											\
	}																	\
	virtual void FNAME(std::initializer_list<size_t> index, TYPE newval)\
	{																	\
		_m_data[getAddr(index)] = (TEMPLTYPE)newval;					\
	}																	\
	virtual void FNAME(const std::vector<size_t>& index, TYPE newval)	\
	{																	\
		_m_data[getAddr(index)] = (TEMPLTYPE)newval;					\
	}																	\
	virtual void FNAME(const size_t* index, TYPE newval)				\
	{																	\
		_m_data[getAddr(index)] = (TEMPLTYPE)newval;					\
	}																	\
	virtual void FNAME(size_t addr, TYPE newval)						\
	{																	\
		_m_data[addr] = (TEMPLTYPE)newval;								\
	}																	\


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
	
	~NDArrayStore() { delete[] _m_data; };

	/* 
	 * get / set functions
	 */

	// Get Address
	virtual size_t getAddr(std::initializer_list<size_t> index) const;
	virtual size_t getAddr(const std::vector<size_t>& index) const;
	virtual size_t getAddr(const size_t* index) const;

	GETSET(double, dbl, T);
	GETSET(int32_t, int32, T);
	GETSET(int64_t, int64, T);
	
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

/**
 * @brief Initializes an array with a size and a chache size. The layout will
 * be cubes in each dimension to make the clusters.
 *
 * @tparam D
 * @tparam T
 * @param dim[D]
 * @param csize
 */
template <int D, typename T>
NDArrayStore<D,T>::NDArrayStore(size_t dim[D])
{
	size_t dsize = 1;
	for(size_t ii=0; ii<D; ii++) {
		_m_dim[ii] = dim[ii];
		dsize *= _m_dim[ii];
	}

	_m_data = new T[dsize];
}

template <int D, typename T>
NDArrayStore<D,T>::NDArrayStore(std::initializer_list<size_t> a_args)
{
	size_t dsize = 1;
	size_t ii;
	
	// set dimensions with matching size to the minimum length, ignoring
	// any extra parts of a_args
	auto it = a_args.begin();
	for(ii=0; ii < D && it != a_args.end(); ii++, ++it) {
		_m_dim[ii] = *it;
		dsize *= _m_dim[ii];
	}

	// make any remaining dimensions size 1
	for(ii=0; ii < D; ii++) {
		_m_dim[ii] = 1;
		dsize *= _m_dim[ii];
	}

	_m_data = new T[dsize];
}

template <int D, typename T>
void NDArrayStore<D,T>::resize(size_t dim[D])
{
	delete[] _m_data;

	size_t dsize = 1;
	for(size_t ii=0; ii<D; ii++) {
		_m_dim[ii] = dim[ii];
		dsize *= _m_dim[ii];
	}

	_m_data = new T[dsize];
}

template <int D, typename T>
size_t NDArrayStore<D,T>::bytes() const
{
	size_t out = 1;
	for(size_t ii=0; ii<D; ii++)
		out*= _m_dim[ii];
	return out*sizeof(T);
}

template <int D, typename T>
size_t NDArrayStore<D,T>::ndim() const
{
	return D;
}

template <int D, typename T>
size_t NDArrayStore<D,T>::dim(size_t dir) const
{
	return _m_dim[dir];
}

template <int D, typename T>
const size_t* NDArrayStore<D,T>::dim() const
{
	return _m_dim;
}
	
/* 
 * Get Address of a Particular Index 
 */
template <int D, typename T>
inline
size_t NDArrayStore<D,T>::getAddr(std::initializer_list<size_t> index) const
{
	size_t tmp[D];
	
	size_t ii = 0;
	for(auto it=index.begin(); ii<D && it != index.end(); ii++, ++it) 
		tmp[ii] = *it;
	for( ; ii<D ; ii++) 
		tmp[ii] = 0;

	return getAddr(tmp);
}

template <int D, typename T>
inline
size_t NDArrayStore<D,T>::getAddr(const size_t* index) const
{
	size_t loc = index[0];
	size_t jump = _m_dim[0];           // jump to global position
	for(size_t ii=1; ii<D; ii++) {
		loc += index[ii]*jump;
		jump *= _m_dim[ii];
	}
	return loc;
}

template <int D, typename T>
inline
size_t NDArrayStore<D,T>::getAddr(const std::vector<size_t>& index) const
{
	size_t tmp[D];
	
	size_t ii = 0;
	for(auto it=index.begin(); ii<D && it != index.end(); ii++, ++it) 
		tmp[ii] = *it;
	for( ; ii<D ; ii++) 
		tmp[ii] = 0;

	return getAddr(tmp);
}

template <int D, typename T>
const T& NDArrayStore<D,T>::operator[](std::initializer_list<size_t> index) const
{
	return _m_data[getAddr(index)];
}

template <int D, typename T>
const T& NDArrayStore<D,T>::operator[](const std::vector<size_t>& index) const
{
	return _m_data[getAddr(index)];
}

template <int D, typename T>
const T& NDArrayStore<D,T>::operator[](const size_t* index) const
{
	return _m_data[getAddr(index)];
}

template <int D, typename T>
const T& NDArrayStore<D,T>::operator[](size_t pixel) const
{
	return _m_data[pixel];
}

template <int D, typename T>
T& NDArrayStore<D,T>::operator[](std::initializer_list<size_t> index) 
{
	return _m_data[getAddr(index)];
}

template <int D, typename T>
T& NDArrayStore<D,T>::operator[](const std::vector<size_t>& index) 
{
	return _m_data[getAddr(index)];
}

template <int D, typename T>
T& NDArrayStore<D,T>::operator[](const size_t* index) 
{
	return _m_data[getAddr(index)];
}

template <int D, typename T>
T& NDArrayStore<D,T>::operator[](size_t pixel) 
{
	return _m_data[pixel];
}
#endif
