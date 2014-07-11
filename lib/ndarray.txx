
#include "npltypes.h"
#include <iostream>

namespace npl {

//rmacros to actually create the get/set functions, note that get and set are 
// the same name, just different arguments
#define GETSETIMP(TYPE, FNAME) \
	template <int D, typename T>												\
	TYPE NDArrayStore<D,T>::FNAME(std::initializer_list<size_t> index) const 	\
	{																			\
		return (TYPE)_m_data[getAddr(index)];									\
	}																			\
	template <int D, typename T>												\
	TYPE NDArrayStore<D,T>::FNAME(const std::vector<size_t>& index) const 		\
	{																			\
		return (TYPE)_m_data[getAddr(index)];									\
	}																			\
	template <int D, typename T>												\
	TYPE NDArrayStore<D,T>::FNAME(const size_t* index) const 					\
	{																			\
		return (TYPE)_m_data[getAddr(index)];									\
	}																			\
	template <int D, typename T>												\
	TYPE NDArrayStore<D,T>::FNAME(size_t addr)	const							\
	{																			\
		return (TYPE)_m_data[addr];												\
	}																			\
	template <int D, typename T>												\
	void NDArrayStore<D,T>::FNAME(std::initializer_list<size_t> index, TYPE val)\
	{																			\
		_m_data[getAddr(index)] = (T)val;										\
	}																			\
	template <int D, typename T>												\
	void NDArrayStore<D,T>::FNAME(const std::vector<size_t>& index, TYPE val)	\
	{																			\
		_m_data[getAddr(index)] = (T)val;										\
	}																			\
	template <int D, typename T>												\
	void NDArrayStore<D,T>::FNAME(const size_t* index, TYPE val)				\
	{																			\
		_m_data[getAddr(index)] = (T)val;										\
	}																			\
	template <int D, typename T>												\
	void NDArrayStore<D,T>::FNAME(size_t addr, TYPE val)						\
	{																			\
		_m_data[addr] = (T)val;													\
	}																			\


GETSETIMP(double, dbl);
GETSETIMP(int64_t, int64);
GETSETIMP(cdouble_t, cdbl);
GETSETIMP(cfloat_t, cfloat);
GETSETIMP(rgba_t, rgba);
GETSETIMP(long double, quad);
GETSETIMP(cquad_t, cquad);

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
	for(; ii < D; ii++) {
		_m_dim[ii] = 1;
		dsize *= _m_dim[ii];
	}

	_m_data = new T[dsize];
}

template <int D, typename T>
NDArrayStore<D,T>::NDArrayStore(const std::vector<size_t>& a_args)
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
	for(; ii < D; ii++) {
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
size_t NDArrayStore<D,T>::elements() const
{
	size_t out = 1;
	for(size_t ii=0; ii<D; ii++)
		out*= _m_dim[ii];
	return out;
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

template <int D, typename T>
int NDArrayStore<D,T>::opself(const NDArray* right, 
		double(*func)(double,double), bool elevR)
{
	bool canElev = false;
	bool comp = comparable(this, right, NULL, &canElev);
	if(comp) {
		auto lit = begin();
		auto rit = right->cbegin();
		for(; !lit.isEnd() && !rit.isEnd(); ++lit, ++rit) {
			double result = func(lit.dbl(), rit.dbl());
			lit.dbl(result);
		}
	} else if(canElev && elevR) {
		// match dimensions, to make this work, we need to iterate through the 
		// common dimensions fastest, the unique dimensions slowest, the way
		// iterators work, if you specify an order, you specify the fastest
		std::list<size_t> commondim;
		for(size_t ii=0; ii < ndim() && ii < right->ndim(); ii++) {
			if(right->dim(ii) != 1)
				commondim.push_front(ii);
		}

		for(auto lit = begin(commondim); !lit.isEnd() ; ) {
			// iterate together until right hits the end, then restart
			auto rit = right->cbegin(commondim);
			for( ; !lit.isEnd() && !rit.isEnd(); ++lit, ++rit) {
				double result = func(lit.dbl(), rit.dbl());
				lit.dbl(result);
			}
		}
	} else {
		std::cerr << "Input Images are not conformable, failing" << endl;
		return -1;
	}

	return 0;
}

template <int D, typename T>
std::shared_ptr<NDArray> NDArrayStore<D,T>::opnew(const NDArray* right, 
		double(*func)(double,double), bool elevR)
{
	auto out = clone();
	if(out->opself(right, func, elevR) != 0)
		return NULL;
	return out;
}

template <int D, typename T>
std::shared_ptr<NDArray> NDArrayStore<D,T>::clone() const
{
	std::vector<size_t> newdims(_m_dim, _m_dim+D);
	auto out = std::make_shared<NDArrayStore<D,T>>(newdims);

	size_t total = 1;
	for(size_t ii=0; ii<D; ii++)
		total *= _m_dim[ii];

	std::copy(_m_data, _m_data+total, out->_m_data);
	std::copy(_m_dim, _m_dim+D, out->_m_dim);

	return out;
}

} //npl
#undef GETSETIMPL
