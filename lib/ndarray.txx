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

#include "npltypes.h"
#include <iostream>

namespace npl {

template <int D, typename T>
NDArrayStore<D,T>::NDArrayStore(const std::vector<size_t>& a_args) : _m_data(NULL)
{
	size_t tmp[D];
	
	// set dimensions with matching size to the minimum length, ignoring
	// any extra parts of a_args
	auto it = a_args.begin();
	for(size_t ii=0; ii < D && it != a_args.end(); ii++, ++it) {
		tmp[ii] = *it;
	}

	resize(tmp);
}

template <int D, typename T>
void NDArrayStore<D,T>::updateStrides()
{
	_m_stride[D-1] = 1;
	for(int64_t ii=D-2; ii>=0; ii--) 
		_m_stride[ii] = _m_stride[ii+1]*_m_dim[ii+1];
}

template <int D, typename T>
void NDArrayStore<D,T>::resize(size_t dim[D])
{
	if(_m_data)
		delete[] _m_data;

	size_t dsize = 1;
	for(size_t ii=0; ii<D; ii++) {
		_m_dim[ii] = dim[ii];
		dsize *= _m_dim[ii];
	}

	_m_data = new T[dsize];

	updateStrides();
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
 * TODO add *outside
 */
template <int D, typename T>
int64_t NDArrayStore<D,T>::getLinIndex(std::initializer_list<int64_t> index) const
{
	int64_t out = 0;

	// copy the dimensions 
	int64_t ii=0;
	for(auto it=index.begin(); it != index.end() && ii<D; ii++, ++it) {
		assert(*it >= 0);
		assert(*it < _m_dim[ii]);
		// set position
		out += _m_stride[ii]*(*it);
	}
	
	assert(out < elements());
	return out;
}

template <int D, typename T>
inline
int64_t NDArrayStore<D,T>::getLinIndex(const int64_t* index) const
{
	int64_t out = 0;

	// copy the dimensions 
	for(size_t ii = 0; ii<D; ii++) {
		assert(index[ii] >= 0);
		assert(index[ii] < _m_dim[ii]);

		// set position
		out += _m_stride[ii]*index[ii];
	}

	assert(out < elements());
	return out;
}

template <int D, typename T>
inline
int64_t NDArrayStore<D,T>::getLinIndex(const std::vector<int64_t>& index) const
{
	size_t out = 0;

	// copy the dimensions 
	size_t ii=0;
	for(auto it=index.begin(); it != index.end() && ii<D; ++it, ii++) {
		assert(index[ii] >= 0);
		assert(index[ii] < _m_dim[ii]);

		// set position
		out += _m_stride[ii]*(*it);
	}
	
	assert(out < elements());
	return out;
}
	
/**
 * @brief Used instead of the normal mapper, we want all the upper dimensions 
 * to be treated as flat
 *
 * @param x
 * @param y
 * @param z
 * @param t
 *
 * @return 
 */
template <int D, typename T>
inline
int64_t NDArrayStore<D,T>::getLinIndex(int64_t x, int64_t y, int64_t z, 
			int64_t t) const
{
	int64_t out = 0;
	int64_t tmp[4] = {x,y,z,t};
	for(size_t ii=0; ii<3; ii++) {
		assert(tmp[ii] >= 0 && tmp[ii] < dim(ii));
		if(ii<D)
			out += tmp[ii]*_m_stride[ii];
	}

	// assert(z < _m_stride[2]);
	out += t;
	return out;
};
	
template <int D, typename T>
const T& NDArrayStore<D,T>::operator[](std::initializer_list<int64_t> index) const
{
	return _m_data[getLinIndex(index)];
}

template <int D, typename T>
const T& NDArrayStore<D,T>::operator[](const std::vector<int64_t>& index) const
{
	return _m_data[getLinIndex(index)];
}

template <int D, typename T>
const T& NDArrayStore<D,T>::operator[](const int64_t* index) const
{
	return _m_data[getLinIndex(index)];
}

template <int D, typename T>
const T& NDArrayStore<D,T>::operator[](int64_t pixel) const
{
	return _m_data[pixel];
}

template <int D, typename T>
T& NDArrayStore<D,T>::operator[](std::initializer_list<int64_t> index) 
{
	return _m_data[getLinIndex(index)];
}

template <int D, typename T>
T& NDArrayStore<D,T>::operator[](const std::vector<int64_t>& index) 
{
	return _m_data[getLinIndex(index)];
}

template <int D, typename T>
T& NDArrayStore<D,T>::operator[](const int64_t* index) 
{
	return _m_data[getLinIndex(index)];
}

template <int D, typename T>
T& NDArrayStore<D,T>::operator[](int64_t pixel) 
{
	return _m_data[pixel];
}

//template <int D, typename T>
//int NDArrayStore<D,T>::opself(const NDArray* right, 
//		double(*func)(double,double), bool elevR)
//{
//	bool canElev = false;
//	bool comp = comparable(this, right, NULL, &canElev);
//	if(comp) {
//		for(size_t ii=0; ii<elements(); ii++) {
//			double result = func(get_dbl(ii), right->get_dbl(ii));
//			set_dbl(ii, result);
//		}
//	} else if(canElev && elevR) {
//		// match dimensions, to make this work, we need to iterate through the 
//		// common dimensions fastest, the unique dimensions slowest, the way
//		// iterators work, if you specify an order, you specify the fastest
//		std::list<size_t> commondim;
//		for(size_t ii=0; ii < ndim() && ii < right->ndim(); ii++) {
//			if(right->dim(ii) != 1)
//				commondim.push_front(ii);
//		}
//		for(Slicer lit(ndim(), dim(), commondim); !lit.isEnd() ; ) {
//			// iterate together until right hits the end, then restart
//			for(Slicer rit(ndim(), dim(), commondim); 
//						!lit.isEnd() && !rit.isEnd(); ++lit, ++rit) {
//				double result = func(get_dbl(*rit), get_dbl(*rit));
//				set_dbl(*lit, result);
//			}
//		}
//	} else {
//		std::cerr << "Input Images are not conformable, failing" << endl;
//		return -1;
//	}
//
//	return 0;
//}
//
//template <int D, typename T>
//std::shared_ptr<NDArray> NDArrayStore<D,T>::opnew(const NDArray* right, 
//		double(*func)(double,double), bool elevR)
//{
//	auto out = clone();
//	if(out->opself(right, func, elevR) != 0)
//		return NULL;
//	return out;
//}
//
//template <int D, typename T>
//std::shared_ptr<NDArray> NDArrayStore<D,T>::clone() const
//{
//	std::vector<size_t> newdims(_m_dim, _m_dim+D);
//	auto out = std::make_shared<NDArrayStore<D,T>>(newdims);
//
//	size_t total = 1;
//	for(size_t ii=0; ii<D; ii++)
//		total *= _m_dim[ii];
//
//	std::copy(_m_data, _m_data+total, out->_m_data);
//	std::copy(_m_dim, _m_dim+D, out->_m_dim);
//	std::copy(_m_stride, _m_stride+D, out->_m_stride);
//
//	return out;
//}

/* 
 * type() Function specialized for all available types
 */
template <int D, typename T>
PixelT NDArrayStore<D,T>::type() const 
{
	if(typeid(T) == typeid(uint8_t)) 
		return UINT8;
	else if(typeid(T) == typeid(int8_t)) 
		return INT8;
	else if(typeid(T) == typeid(uint16_t)) 
		return UINT16;
	else if(typeid(T) == typeid(int16_t)) 
		return INT16;
	else if(typeid(T) == typeid(uint32_t)) 
		return UINT32;
	else if(typeid(T) == typeid(int32_t)) 
		return INT32;
	else if(typeid(T) == typeid(uint64_t)) 
		return UINT64;
	else if(typeid(T) == typeid(int64_t)) 
		return INT64;
	else if(typeid(T) == typeid(float)) 
		return FLOAT32;
	else if(typeid(T) == typeid(double)) 
		return FLOAT64;
	else if(typeid(T) == typeid(long double)) 
		return FLOAT128;
	else if(typeid(T) == typeid(cfloat_t)) 
		return COMPLEX64;
	else if(typeid(T) == typeid(cdouble_t)) 
		return COMPLEX128;
	else if(typeid(T) == typeid(cquad_t)) 
		return COMPLEX256;
	else if(typeid(T) == typeid(rgb_t)) 
		return RGB24;
	else if(typeid(T) == typeid(rgba_t)) 
		return RGBA32;
	return UNKNOWN_TYPE;
}


} //npl
#undef GETSETIMPL
