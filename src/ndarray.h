#ifndef NDARRAY_H
#define NDARRAY_H

#include <cstddef>
#include <cmath>
#include <initializer_list>

/**
 * @brief Pure virtual interface to interact with an ND array
 */
class NDArray
{
public:
	virtual double getD(std::initializer_list<size_t> index) = 0;
	
	virtual double operator()(std::initializer_list<size_t> index) = 0;

	virtual int getI(std::initializer_list<size_t> index) = 0;
	
	virtual void setD(double newval, std::initializer_list<size_t> index) = 0;

	virtual void setI(int newval, std::initializer_list<size_t> index) = 0;

	virtual size_t getBytes() = 0;
	virtual size_t getNDim() = 0;
	virtual size_t dim(size_t dir) = 0;
	virtual size_t* dim() = 0;
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
	
	~NDArrayStore() { delete[] _m_data; };

	virtual size_t getAddr(std::initializer_list<size_t> index);
	virtual size_t getAddr(size_t index[D]);

	virtual double getD(std::initializer_list<size_t> index);

	virtual int getI(std::initializer_list<size_t> index);
	
	virtual void setD(double newval, std::initializer_list<size_t> index);

	virtual void setI(int newval, std::initializer_list<size_t> index);

	virtual double operator()(std::initializer_list<size_t> index);

	virtual size_t getBytes();
	virtual size_t getNDim();
	virtual size_t dim(size_t dir);
	virtual size_t* dim();

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
inline
size_t NDArrayStore<D,T>::getAddr(std::initializer_list<size_t> index)
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
size_t NDArrayStore<D,T>::getAddr(size_t index[D])
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
double NDArrayStore<D,T>::getD(std::initializer_list<size_t> index)
{
	size_t ii = getAddr(index);
	return (double)_m_data[ii];
}

template <int D, typename T>
int NDArrayStore<D,T>::getI(std::initializer_list<size_t> index)
{
	return (int)_m_data[getAddr(index)];
}

template <int D, typename T>
void NDArrayStore<D,T>::setD(double newval, std::initializer_list<size_t> index)
{
	_m_data[getAddr(index)] = (T)newval;
}

template <int D, typename T>
void NDArrayStore<D,T>::setI(int newval, std::initializer_list<size_t> index)
{
	_m_data[getAddr(index)] = (T)newval;
}

template <int D, typename T>
double NDArrayStore<D,T>::operator()(std::initializer_list<size_t> index)
{
	return (double)_m_data[getAddr(index)];
}

template <int D, typename T>
size_t NDArrayStore<D,T>::getBytes()
{
	size_t out = 1;
	for(size_t ii=0; ii<D; ii++)
		out*= _m_dim[ii];
	return out*sizeof(T);
}

template <int D, typename T>
size_t NDArrayStore<D,T>::getNDim() 
{
	return D;
}

template <int D, typename T>
size_t NDArrayStore<D,T>::dim(size_t dir)
{
	return _m_dim[dir];
}

template <int D, typename T>
size_t* NDArrayStore<D,T>::dim()
{
	return _m_dim;
}

#endif
