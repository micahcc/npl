/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * 	http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file ndarray.txx
 *
 *****************************************************************************/

#include "npltypes.h"
#include "slicer.h"
#include "version.h"

#include <iostream>

namespace npl {

template <size_t D, typename T>
NDArrayStore<D,T>::NDArrayStore(const std::initializer_list<size_t>& a_args) :
			_m_data(NULL)
{
	size_t tmp[D];
	
	// set dimensions with matching size to the minimum length, ignoring
	// any extra parts of a_args
	auto it = a_args.begin();
	for(size_t ii=0; ii < D; ii++) {
		if(it != a_args.end()) {
			tmp[ii] = *it;
			++it;
		} else {
			tmp[ii] = 1;
		}
	}

	resize(tmp);
}

template <size_t D, typename T>
NDArrayStore<D,T>::NDArrayStore(const std::vector<size_t>& a_args) : _m_data(NULL)
{
	size_t tmp[D];
	
	// set dimensions with matching size to the minimum length, ignoring
	// any extra parts of a_args
	auto it = a_args.begin();
	for(size_t ii=0; ii < D; ii++) {
		if(it != a_args.end()) {
			tmp[ii] = *it;
			++it;
		} else {
			tmp[ii] = 1;
		}
	}

	resize(tmp);
}

template <size_t D, typename T>
NDArrayStore<D,T>::NDArrayStore(size_t len, const size_t* a_args) : _m_data(NULL)
{
	size_t tmp[D];
	
	// set dimensions with matching size to the minimum length, ignoring
	// any extra parts of a_args
	for(size_t ii=0; ii<D; ii++) {
		if(ii < len)
			tmp[ii] = a_args[ii];
		else
			tmp[ii] = 1;
	}

	resize(tmp);
}

template <size_t D, typename T>
NDArrayStore<D,T>::NDArrayStore(size_t len, const size_t* dim, T* ptr,
        const std::function<void(void*)>& deleter) : _m_data(NULL)
{
	if(len != D) {
		std::cerr << "Length of input size (" << len << ") array to " << D
			<< "D array is not allowed." << endl;
	}
	graft(dim, ptr, deleter);
}

template <size_t D, typename T>
void NDArrayStore<D,T>::updateStrides()
{
	_m_stride[D-1] = 1;
	for(int64_t ii=D-2; ii>=0; ii--)
		_m_stride[ii] = _m_stride[ii+1]*_m_dim[ii+1];
}

/**
 * @brief Graft an ND dataset into the NDArray. Any old data is deleted and
 * the dimensions are set to those passed. *
 *
 * @tparam D Array dimensionality
 * @tparam T Type of data stored
 * @param dim[D] dimension of graft data, and new dimensions after this
 * @param ptr Pointer to new data
 */
template <size_t D, typename T>
void NDArrayStore<D,T>::graft(const size_t dim[D], T* ptr, 
        const std::function<void(void*)>& deleter)
{
	if(_m_data)
		m_freefunc(_m_data);

	for(size_t ii=0; ii<D; ii++)
		_m_dim[ii] = dim[ii];

	_m_data = ptr;
    m_freefunc = deleter;

	updateStrides();
}

/**
 * @brief If the size is different from the current size, then it allocates
 * a new chunk of memory, fills it with zeros and then copies the values from
 * the original image into the new image. If the size is the same, it does
 * nothing.
 *
 * @tparam D
 * @tparam T
 * @param dim[D]
 */
template <size_t D, typename T>
void NDArrayStore<D,T>::resize(const size_t dim[D])
{
	if(_m_data) {
		///////////////////////////////////////////////////////
		// Do nothing if the input dimensions match the current
		///////////////////////////////////////////////////////
		bool samedim = true;
		for(size_t ii=0; ii<D; ii++) {
			if(_m_dim[ii] != dim[ii]) {
				samedim = false;
				break;
			}
		}

		if(samedim)
			return;
	
		////////////////////////////////
		// Create New Data and zero fill
		////////////////////////////////
		size_t dsize = 1;
		for(size_t ii=0; ii<D; ii++)
			dsize *= dim[ii];
		T* newdata = new T[dsize];
		std::fill(newdata, newdata+dsize, (T)0);
		
		// copy the old array to the new, by creating slicers with regions of
		//interest which have the minimum size of the original or new
		int64_t roi_lower[D];
		int64_t roi_upper[D];
		for(size_t dd=0; dd < D; dd++) {
			roi_lower[dd] = 0;
			roi_upper[dd] = std::min<int64_t>(dim[dd], _m_dim[dd]);
		}

		Slicer oldit(D, _m_dim);
		Slicer newit(D, dim);
		oldit.setROI(D, roi_lower, roi_upper);
		newit.setROI(D, roi_lower, roi_upper);

		// copy the data
		for(oldit.goBegin(), newit.goBegin(); !newit.eof(); ++newit, ++oldit) {
			newdata[*newit] = _m_data[*oldit];
		}
		assert(newit.eof() && oldit.eof());

		// now copy the dimension
		for(size_t ii=0; ii<D; ii++) {
			_m_dim[ii] = dim[ii];
		}

		// set up data pointer
        m_freefunc(_m_data);
		_m_data = newdata;
        m_freefunc = [](void* p) {delete[] (T*)p;};
	} else {
		// just create the data
		size_t dsize = 1;
		for(size_t ii=0; ii<D; ii++) {
			_m_dim[ii] = dim[ii];
			dsize *= _m_dim[ii];
		}
		
		// allocate
		_m_data = new T[dsize];
        m_freefunc = [](void* p) {delete[] (T*)p;};

		// zero fill
		std::fill(_m_data, _m_data+dsize, (T)0);
	}

	updateStrides();
}

template <size_t D, typename T>
size_t NDArrayStore<D,T>::bytes() const
{
	size_t out = 1;
	for(size_t ii=0; ii<D; ii++)
		out*= _m_dim[ii];
	return out*sizeof(T);
}

template <size_t D, typename T>
size_t NDArrayStore<D,T>::elements() const
{
	size_t out = 1;
	for(size_t ii=0; ii<D; ii++)
		out*= _m_dim[ii];
	return out;
}

template <size_t D, typename T>
size_t NDArrayStore<D,T>::ndim() const
{
	return D;
}

template <size_t D, typename T>
size_t NDArrayStore<D,T>::dim(size_t dir) const
{
	return _m_dim[dir];
}

template <size_t D, typename T>
const size_t* NDArrayStore<D,T>::dim() const
{
	return _m_dim;
}

/*
 * Get Address of a Particular Index
 * TODO add *outside
 */
template <size_t D, typename T>
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

//template <size_t D, typename T>
//inline
//int64_t NDArrayStore<D,T>::getLinIndex(const int64_t* index) const
//{
//	int64_t out = 0;
//
//	// copy the dimensions
//	for(size_t ii = 0; ii<D; ii++) {
//		assert(index[ii] >= 0);
//		assert(index[ii] < _m_dim[ii]);
//
//		// set position
//		out += _m_stride[ii]*index[ii];
//	}
//
//	assert(out < elements());
//	return out;
//}

template <size_t D, typename T>
inline
int64_t NDArrayStore<D,T>::getLinIndex(size_t len, const int64_t* index) const
{
	size_t out = 0;

	// copy the dimensions
	for(size_t ii = 0; ii < len && ii<D; ii++) {
		assert(index[ii] >= 0);
		assert(index[ii] < _m_dim[ii]);

		// set position
		out += _m_stride[ii]*index[ii];
	}
	
	assert(out < elements());
	return out;
}

template <size_t D, typename T>
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
template <size_t D, typename T>
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

template <size_t D, typename T>
int NDArrayStore<D,T>::writeJSONArray(gzFile file) const
{
    ostringstream oss;
    oss << "{\n\"version\" : \"" << __version__<< "\",\n\"comment\" : \"supported "
        "type variables: uint8, int16, int32, float, cfloat, double, RGB, "
        "int8, uint16, uint32, int64, uint64, quad, cdouble, cquad, RGBA\",\n";
    oss << "\"type\": " << '"' << pixelTtoString(type()) << "\",\n";
    oss << "\"size\": [";
    for(size_t ii=0; ii<D; ii++) {
        if(ii) oss << ", ";
        oss << dim(ii);
    }
    oss << "],\n";

    int64_t index[D]; 
    oss << "\"values\" : ";
    for(NDConstIter<T> it(getConstPtr()); !it.eof(); ++it) {
        it.index(D, index);
        if(index[D-1] == 0)
            oss << "\n";
        for(int64_t dd=D-1; dd>=0; dd--) {
            if(index[dd] == 0) {
                oss << "[";
            } else {
                break;
            }
        }
        oss << *it;;
        for(int64_t dd=D-1; dd>=0; dd--) {
            if(index[dd] == dim(dd)-1) {
                oss << "]";
            } else {
                oss << ", ";
                break;
            }
        }
    }
    oss << "\n}\n";

    if(gzwrite(file, oss.str().c_str(), oss.str().length()) > 0)
        return 0;
	return -1;
}
	
template <size_t D, typename T>
const T& NDArrayStore<D,T>::operator[](std::initializer_list<int64_t> index) const
{
	return _m_data[getLinIndex(index)];
}

template <size_t D, typename T>
const T& NDArrayStore<D,T>::operator[](const std::vector<int64_t>& index) const
{
	return _m_data[getLinIndex(index)];
}
//
//template <size_t D, typename T>
//const T& NDArrayStore<D,T>::operator[](const int64_t* index) const
//{
//	return _m_data[getLinIndex(index)];
//}

template <size_t D, typename T>
const T& NDArrayStore<D,T>::operator[](int64_t pixel) const
{
	return _m_data[pixel];
}

template <size_t D, typename T>
T& NDArrayStore<D,T>::operator[](std::initializer_list<int64_t> index)
{
	return _m_data[getLinIndex(index)];
}

template <size_t D, typename T>
T& NDArrayStore<D,T>::operator[](const std::vector<int64_t>& index)
{
	return _m_data[getLinIndex(index)];
}

//template <size_t D, typename T>
//T& NDArrayStore<D,T>::operator[](const int64_t* index)
//{
//	return _m_data[getLinIndex(index)];
//}
//
template <size_t D, typename T>
T& NDArrayStore<D,T>::operator[](int64_t pixel)
{
	return _m_data[pixel];
}

//template <size_t D, typename T>
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
//template <size_t D, typename T>
//ptr<NDArray> NDArrayStore<D,T>::opnew(const NDArray* right,
//		double(*func)(double,double), bool elevR)
//{
//	auto out = clone();
//	if(out->opself(right, func, elevR) != 0)
//		return NULL;
//	return out;
//}
//
//template <size_t D, typename T>
//ptr<NDArray> NDArrayStore<D,T>::clone() const
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

/**
 * @brief Performs a deep copy of the entire array and all metadata.
 *
 * @return Copied array.
 */
template <size_t D, typename T>
ptr<NDArray> NDArrayStore<D,T>::copy() const
{
	ptr<NDArrayStore> out(new NDArrayStore<D,T>(D, this->_m_dim));
	for(size_t ii=0; ii<elements(); ii++)
		out->_m_data[ii] = this->_m_data[ii];

	return out;
}

/**
 * @brief Creates an identical array, but does not initialize pixel values.
 *
 * @return New array.
 */
template <size_t D, typename T>
ptr<NDArray> NDArrayStore<D,T>::createAnother() const
{
    return createNDArray(D, this->dim(), type());
}
	
/**
 * @brief Create a new array that is a copy of the input, possibly with new
 * dimensions and pixeltype. The new array will have all overlapping pixels
 * copied from the old array.
 *
 * This function just calls the outside copyCast, the reason for this
 * craziness is that making a template function nested in the already
 * huge number of templates I have kills the compiler, so we call an
 * outside function that calls templates that has all combinations of D,T.
 *
 * @param in Input array, anything that can be copied will be
 * @param newdims Number of dimensions in output array
 * @param newsize Size of output array
 * @param newtype Type of pixels in output array
 *
 * @return Image with overlapping sections cast and copied from 'in'
 */
template <size_t D, typename T>
ptr<NDArray> NDArrayStore<D,T>::copyCast(size_t newdims,
		const size_t* newsize, PixelT newtype) const
{
	return _copyCast(getConstPtr(), newdims, newsize, newtype);
}

/**
 * @brief Create a new array that is a copy of the input, with same dimensions
 * but pxiels cast to newtype. The new array will have all overlapping pixels
 * copied from the old array.
 *
 * This function just calls the outside copyCast, the reason for this
 * craziness is that making a template function nested in the already
 * huge number of templates I have kills the compiler, so we call an
 * outside function that calls templates that has all combinations of D,T.
 *
 * @param in Input array, anything that can be copied will be
 * @param newtype Type of pixels in output array
 *
 * @return Image with overlapping sections cast and copied from 'in'
 */
template <size_t D, typename T>
ptr<NDArray> NDArrayStore<D,T>::copyCast(PixelT newtype) const
{
	return _copyCast(getConstPtr(), newtype);
}

/**
 * @brief Create a new array that is a copy of the input, possibly with new
 * dimensions or size. The new array will have all overlapping pixels
 * copied from the old array. The new array will have the same pixel type as
 * the input array
 *
 * This function just calls the outside copyCast, the reason for this
 * craziness is that making a template function nested in the already
 * huge number of templates I have kills the compiler, so we call an
 * outside function that calls templates that has all combinations of D,T.
 *
 * @param in Input array, anything that can be copied will be
 * @param newdims Number of dimensions in output array
 * @param newsize Size of output array
 *
 * @return Image with overlapping sections cast and copied from 'in'
 */
template <size_t D, typename T>
ptr<NDArray> NDArrayStore<D,T>::copyCast(size_t newdims,
		const size_t* newsize) const
{
	return _copyCast(getConstPtr(), newdims, newsize);
}

/**
 * @brief Create a new array that is a copy of the input, possibly with new
 * dimensions or size. The new array will have all overlapping pixels
 * copied from the old array. The new array will have the same pixel type as
 * the input array. If len > ndim(), then the output may have more dimensions 
 * then the input, and in fact the extra dimensions may be larger than the
 * input image. If this happens, then data will still be extracted, but only
 * the overlapping segments of the new and old image will be copied. Also note
 * that index[] will not be accessed above ndim()
 *
 * @param len     Length of index/newsize arrays
 * @param index   ROI Index to start copying from.
 * @param size    ROI size. Note length 0 dimensions will be removed, while
 * length 1 dimensions will be left. 
 *
 * @return Image with overlapping sections cast and copied from 'in'
 */
template <size_t D, typename T>
ptr<NDArray> NDArrayStore<D,T>::extractCast(size_t len, const int64_t* index,
        const size_t* size) const
{
    return extractCast(len, index, size, type());
}

/**
 * @brief Create a new array that is a copy of the input, possibly with new
 * dimensions or size. The new array will have all overlapping pixels
 * copied from the old array. The new array will have the same pixel type as
 * the input array. If len > ndim(), then the output may have more dimensions 
 * then the input, and in fact the extra dimensions may be larger than the
 * input image. If this happens, then data will still be extracted, but only
 * the overlapping segments of the new and old image will be copied.
 *
 * @param len     Length of index/size arrays
 * @param index   Index to start copying from.
 * @param size Size of output image. Note length 0 dimensions will be
 * removed, while length 1 dimensions will be left. 
 *
 * @return Image with overlapping sections cast and copied from 'in'
 */
template <size_t D, typename T>
ptr<NDArray> NDArrayStore<D,T>::extractCast(size_t len, 
        const size_t* size) const
{
    return extractCast(len, NULL, size, type());
}

/**
 * @brief Create a new array that is a copy of the input, possibly with new
 * dimensions or size. The new array will have all overlapping pixels
 * copied from the old array. The new array will have the same pixel type as
 * the input array
 *
 * @param len     Length of index/size arrays
 * @param index   Index to start copying from.
 * @param size Size of output image. Note length 0 dimensions will be
 * removed, while length 1 dimensions will be left. 
 * @param newtype Pixel type of output image.
 *
 * @return Image with overlapping sections cast and copied from 'in'
 */
template <size_t D, typename T>
ptr<NDArray> NDArrayStore<D,T>::extractCast(size_t len, const int64_t* index,
        const size_t* size, PixelT newtype) const
{
    assert(size);
    assert(len < 10);
    
    int64_t ilower[D];
    int64_t iupper[D];
    
    size_t newdim = 0;
    size_t newsize[10];
    int64_t olower[10];
    int64_t oupper[10];

    // determine output size
    for(size_t dd=0; dd<len; dd++) {
        if(size[dd] > 0) {
            newsize[newdim] = size[dd];
            olower[newdim] = 0;
            oupper[newdim] = size[dd]-1;
            newdim++;
        }
    }
    
    // create ROI in input image
    for(size_t dd=0; dd<D; dd++) {
        if(dd < len) {
            if(index)
                ilower[dd] = index[dd];
            else
                ilower[dd] = 0;

            if(size[dd] > 0) 
                iupper[dd] = ilower[dd]+size[dd]-1;
            else
                iupper[dd] = ilower[dd];
        } else {
            ilower[dd] = 0;
            iupper[dd] = 0;
        }

        if(iupper[dd] >= dim(dd)) {
            throw INVALID_ARGUMENT("Extracted Region is outside the input "
                    "image FOV");
        }
    }
    
    // create output
    auto out = createNDArray(newdim, newsize, newtype);
    copyROI(getConstPtr(), ilower, iupper, out, olower, oupper, newtype);

    return out;
}

/**
 * @brief Create a new array that is a copy of the input, possibly with new
 * dimensions or size. The new array will have all overlapping pixels
 * copied from the old array. The new array will have the same pixel type as
 * the input array. Index assumed to be [0,0,...], so the output image will 
 * start at the origin of this image.
 *
 * @param len     Length of index/size arrays
 * @param size Size of output image. Note length 0 dimensions will be
 * removed, while length 1 dimensions will be left. 
 * @param newtype Pixel type of output image.
 *
 * @return Image with overlapping sections cast and copied from 'in'
 */
template <size_t D, typename T>
ptr<NDArray> NDArrayStore<D,T>::extractCast(size_t len, 
        const size_t* size, PixelT newtype) const
{
    return extractCast(len, NULL, size, newtype);
}

/*
 * type() Function specialized for all available types
 */
template <size_t D, typename T>
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
