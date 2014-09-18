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
 * @file mrimage.cpp
 *
 *****************************************************************************/

#include "mrimage.h"
#include "iterators.h"
#include <iostream>

#include "ndarray.h"
#include "nifti.h"
#include "byteswap.h"
#include "slicer.h"
#include "macros.h"

#include "zlib.h"

#include <cstring>

using std::make_shared;

namespace npl {

/******************************************************************************
 * Helper Functions. We put these here because they are internally by MRImage
 * and MRImageStore; we don't want to call across libraries if we can avoid
 * it.
 ******************************************************************************/

/**
 * @brief Template helper for creating new images.
 *
 * @tparam T Type of voxels
 * @param len Length of input dimension array
 * @param dim Size of new image
 *
 * @return New MRImage with defaults set
 */
template <typename T>
ptr<MRImage> createMRImageHelp(size_t len, const size_t* dim)
{
	switch(len) {
		case 1:
			return make_shared<MRImageStore<1, T>>(len, dim);
		case 2:
			return make_shared<MRImageStore<2, T>>(len, dim);
		case 3:
			return make_shared<MRImageStore<3, T>>(len, dim);
		case 4:
			return make_shared<MRImageStore<4, T>>(len, dim);
		case 5:
			return make_shared<MRImageStore<5, T>>(len, dim);
		case 6:
			return make_shared<MRImageStore<6, T>>(len, dim);
		case 7:
			return make_shared<MRImageStore<7, T>>(len, dim);
		case 8:
			return make_shared<MRImageStore<8, T>>(len, dim);
		default:
			std::cerr << "Unsupported len, dimension: " << len << std::endl;
			return NULL;
	}

	return NULL;
}

/**
 * @brief Creates a new MRImage with dimensions set by ndim, and size set by
 * size. Output pixel type is decided by type variable.
 *
 * @param ndim number of image dimensions
 * @param size size of image, in each dimension
 * @param type Pixel type npl::PixelT
 *
 * @return New image, default orientation
 */
ptr<MRImage> createMRImage(size_t ndim, const size_t* size, PixelT type)
{
	switch(type) {
         case UINT8:
			return createMRImageHelp<uint8_t>(ndim, size);
        break;
         case INT16:
			return createMRImageHelp<int16_t>(ndim, size);
        break;
         case INT32:
			return createMRImageHelp<int32_t>(ndim, size);
        break;
         case FLOAT32:
			return createMRImageHelp<float>(ndim, size);
        break;
         case COMPLEX64:
			return createMRImageHelp<cfloat_t>(ndim, size);
        break;
         case FLOAT64:
			return createMRImageHelp<double>(ndim, size);
        break;
         case RGB24:
			return createMRImageHelp<rgb_t>(ndim, size);
        break;
         case INT8:
			return createMRImageHelp<int8_t>(ndim, size);
        break;
         case UINT16:
			return createMRImageHelp<uint16_t>(ndim, size);
        break;
         case UINT32:
			return createMRImageHelp<uint32_t>(ndim, size);
        break;
         case INT64:
			return createMRImageHelp<int64_t>(ndim, size);
        break;
         case UINT64:
			return createMRImageHelp<uint64_t>(ndim, size);
        break;
         case FLOAT128:
			return createMRImageHelp<long double>(ndim, size);
        break;
         case COMPLEX128:
			return createMRImageHelp<cdouble_t>(ndim, size);
        break;
         case COMPLEX256:
			return createMRImageHelp<cquad_t>(ndim, size);
        break;
         case RGBA32:
			return createMRImageHelp<rgba_t>(ndim, size);
        break;
		 default:
		return NULL;
	}
	return NULL;
}

/**
 * @brief Creates a new MRImage with dimensions set by ndim, and size set by
 * size. Output pixel type is decided by type variable.
 *
 * @param dim size of image, in each dimension, number of dimensions decied by
 * length of size vector
 * @param type Pixel type npl::PixelT
 *
 * @return New image, default orientation
 */
ptr<MRImage> createMRImage(const std::vector<size_t>& dim, PixelT type)
{
	return createMRImage(dim.size(), dim.data(), type);
}

/**
 * @brief Template helper for creating new images.
 *
 * @tparam T Type of voxels
 * @param len Length of input dimension array
 * @param dim Size of new image
 * @param ptr data to use (instead of allocating new data)
 * @param deleter function that can destroy ptr properly
 *
 * @return New MRImage with defaults set
 */
template <typename T>
ptr<MRImage> createMRImageHelp(size_t len, const size_t* dim,
        void* ptr, std::function<void(void*)> deleter)
{
	switch(len) {
		case 1:
			return make_shared<MRImageStore<1, T>>(len, dim, (T*)ptr, deleter);
		case 2:
			return make_shared<MRImageStore<2, T>>(len, dim, (T*)ptr, deleter);
		case 3:
			return make_shared<MRImageStore<3, T>>(len, dim, (T*)ptr, deleter);
		case 4:
			return make_shared<MRImageStore<4, T>>(len, dim, (T*)ptr, deleter);
		case 5:
			return make_shared<MRImageStore<5, T>>(len, dim, (T*)ptr, deleter);
		case 6:
			return make_shared<MRImageStore<6, T>>(len, dim, (T*)ptr, deleter);
		case 7:
			return make_shared<MRImageStore<7, T>>(len, dim, (T*)ptr, deleter);
		case 8:
			return make_shared<MRImageStore<8, T>>(len, dim, (T*)ptr, deleter);
		default:
			std::cerr << "Unsupported len, dimension: " << len << std::endl;
			return NULL;
	}

	return NULL;
}

/**
 * @brief Creates a new MRImage with dimensions set by ndim, and size set by
 * size. Output pixel type is decided by type variable.
 *
 * @param ndim number of image dimensions
 * @param size size of image, in each dimension
 * @param type Pixel type npl::PixelT
 * @param ptr Pointer to data block
 * @param deleter function to delete data block
 *
 * @return New image, default orientation
 */
ptr<MRImage> createMRImage(size_t ndim, const size_t* size, PixelT type,
        void* ptr, std::function<void(void*)> deleter)
{
	switch(type) {
         case UINT8:
			return createMRImageHelp<uint8_t>(ndim, size, ptr, deleter);
        break;
         case INT16:
			return createMRImageHelp<int16_t>(ndim, size, ptr, deleter);
        break;
         case INT32:
			return createMRImageHelp<int32_t>(ndim, size, ptr, deleter);
        break;
         case FLOAT32:
			return createMRImageHelp<float>(ndim, size, ptr, deleter);
        break;
         case COMPLEX64:
			return createMRImageHelp<cfloat_t>(ndim, size, ptr, deleter);
        break;
         case FLOAT64:
			return createMRImageHelp<double>(ndim, size, ptr, deleter);
        break;
         case RGB24:
			return createMRImageHelp<rgb_t>(ndim, size, ptr, deleter);
        break;
         case INT8:
			return createMRImageHelp<int8_t>(ndim, size, ptr, deleter);
        break;
         case UINT16:
			return createMRImageHelp<uint16_t>(ndim, size, ptr, deleter);
        break;
         case UINT32:
			return createMRImageHelp<uint32_t>(ndim, size, ptr, deleter);
        break;
         case INT64:
			return createMRImageHelp<int64_t>(ndim, size, ptr, deleter);
        break;
         case UINT64:
			return createMRImageHelp<uint64_t>(ndim, size, ptr, deleter);
        break;
         case FLOAT128:
			return createMRImageHelp<long double>(ndim, size, ptr, deleter);
        break;
         case COMPLEX128:
			return createMRImageHelp<cdouble_t>(ndim, size, ptr, deleter);
        break;
         case COMPLEX256:
			return createMRImageHelp<cquad_t>(ndim, size, ptr, deleter);
        break;
         case RGBA32:
			return createMRImageHelp<rgba_t>(ndim, size, ptr, deleter);
        break;
		 default:
		return NULL;
	}
	return NULL;
};

/**
 * @brief Creates a new MRImage with dimensions set by ndim, and size set by
 * size. Output pixel type is decided by type variable.
 *
 * @param dim size of image, in each dimension, number of dimensions decied by
 * length of size vector
 * @param type Pixel type npl::PixelT
 * @param ptr Pointer to data block
 * @param deleter function to delete data block
 *
 * @return New image, default orientation
 */
ptr<MRImage> createMRImage(const std::vector<size_t>& dim, PixelT type,
        void* ptr, std::function<void(void*)> deleter)
{
    return createMRImage(dim.size(), dim.data(), type, ptr, deleter);
}

/**
 * @brief Sets slice timing from duration, start, and end order variables.
 *
 * @param duration Duration of data collection for each slice (same unit as TR)
 * @param start First slice collected
 * @param end Last slice collected
 * @param order Order of slice collection. Defined by NIFTI format
 */
void MRImage::updateSliceTiming(double duration, int start, int end, SliceOrderT order)
{
	m_slice_duration = duration;
	m_slice_start = start;
	m_slice_end = end;

	switch(order) {
		case NIFTI_SLICE_SEQ_INC:
			m_slice_order = SEQ;
			for(int ii=m_slice_start; ii<=m_slice_end; ii++)
				m_slice_timing[ii] = ii*m_slice_duration;
		break;
		case NIFTI_SLICE_SEQ_DEC:
			m_slice_order = RSEQ;
			for(int ii=m_slice_end; ii>=m_slice_start; ii--)
				m_slice_timing[ii] = ii*m_slice_duration;
		break;
		case NIFTI_SLICE_ALT_INC:
			m_slice_order = ALT;
			for(int ii=m_slice_start; ii<=m_slice_end; ii+=2)
				m_slice_timing[ii] = ii*m_slice_duration;
			for(int ii=m_slice_start+1; ii<=m_slice_end; ii+=2)
				m_slice_timing[ii] = ii*m_slice_duration;
		break;
		case NIFTI_SLICE_ALT_DEC:
			m_slice_order = RALT;
			for(int ii=m_slice_end; ii>=m_slice_start; ii-=2)
				m_slice_timing[ii] = ii*m_slice_duration;
			for(int ii=m_slice_end-1; ii>=m_slice_start; ii-=2)
				m_slice_timing[ii] = ii*m_slice_duration;
		break;
		case NIFTI_SLICE_ALT_INC2:
			m_slice_order = ALT_SHFT;
			for(int ii=m_slice_start+1; ii<=m_slice_end; ii+=2)
				m_slice_timing[ii] = ii*m_slice_duration;
			for(int ii=m_slice_start; ii<=m_slice_end; ii+=2)
				m_slice_timing[ii] = ii*m_slice_duration;
		break;
		case NIFTI_SLICE_ALT_DEC2:
			m_slice_order = RALT_SHFT;
			for(int ii=m_slice_end-1; ii>=m_slice_start; ii-=2)
				m_slice_timing[ii] = ii*m_slice_duration;
			for(int ii=m_slice_end; ii>=m_slice_start; ii-=2)
				m_slice_timing[ii] = ii*m_slice_duration;
		break;
		default:
		case UNKNOWN_SLICE:
			m_slice_order = UNKNOWN_SLICE;
			m_slice_timing.clear();
		break;
	}
}
/**
 * @brief Helper function that casts all the elements as the given type then uses
 * the same type to set all the elements of the output image. Only overlapping
 * sections of the images are copied.
 *
 * @tparam T Type to cast to
 * @param in Input image to copy
 * @param out Output image to write to
 */
template <typename T>
void _copyCast_help(ptr<const MRImage> in, ptr<MRImage> out)
{

	// Set up slicers to iterate through the input and output images. Only
	// common dimensions are iterated over, and only the minimum of the two
	// sizes are used for ROI. so a 10x10x10 image cast to a 20x5 image will
	// iterator copy ROI 10x5x1
	OrderConstIter<T> iit(in);
	OrderIter<T> oit(out);
	
	std::vector<std::pair<int64_t,int64_t>> roi(max(out->ndim(), in->ndim()));
	for(size_t ii=0; ii<roi.size(); ii++) {
		if(ii < min(out->ndim(), in->ndim())) {
			roi[ii].first = 0;
			roi[ii].second = min(out->dim(ii), in->dim(ii))-1;
		} else {
			roi[ii].first = 0;
			roi[ii].second = 0;
		}
	}

	iit.setROI(roi);
	oit.setROI(roi);

	// use the larger of the two order vectors as the reference
	if(in->ndim() < out->ndim()) {
		iit.setOrder(oit.getOrder());
	} else {
		oit.setOrder(iit.getOrder());
	}

	// perform copy/cast
	for(iit.goBegin(), oit.goBegin(); !oit.eof() && !iit.eof(); ++oit, ++iit)
		oit.set(*iit);

}

/**
 * @brief Create a new image that is a copy of the input, possibly with new
 * dimensions and pixeltype. The new image will have all overlapping pixels
 * copied from the old image.
 *
 * @param in Input image, anything that can be copied will be
 * @param newdims Number of dimensions in output image
 * @param newsize Size of output image
 * @param newtype Type of pixels in output image
 *
 * @return Image with overlapping sections cast and copied from 'in'
 */
ptr<MRImage> _copyCast(ptr<const MRImage> in, size_t newdims,
		const size_t* newsize, PixelT newtype)
{
	auto out = createMRImage(newdims, newsize, newtype);
	
	// copy image metadata
	out->m_freqdim = in->m_freqdim;
	out->m_slicedim = in->m_slicedim;
	out->m_phasedim = in->m_phasedim;
	out->m_slice_duration = in->m_slice_duration;
	out->m_slice_start = in->m_slice_start;
	out->m_slice_end = in->m_slice_end;
	out->m_slice_timing = in->m_slice_timing;
	out->m_slice_order = in->m_slice_order;
	out->setOrient(in->getOrigin(), in->getSpacing(), in->getDirection(), true);

	switch(newtype) {
		case UINT8:
			_copyCast_help<uint8_t>(in, out);
			break;
		case INT16:
			_copyCast_help<int16_t>(in, out);
			break;
		case INT32:
			_copyCast_help<int32_t>(in, out);
			break;
		case FLOAT32:
			_copyCast_help<float>(in, out);
			break;
		case COMPLEX64:
			_copyCast_help<cfloat_t>(in, out);
			break;
		case FLOAT64:
			_copyCast_help<double>(in, out);
			break;
		case RGB24:
			_copyCast_help<rgb_t>(in, out);
			break;
		case INT8:
			_copyCast_help<int8_t>(in, out);
			break;
		case UINT16:
			_copyCast_help<uint16_t>(in, out);
			break;
		case UINT32:
			_copyCast_help<uint32_t>(in, out);
			break;
		case INT64:
			_copyCast_help<int64_t>(in, out);
			break;
		case UINT64:
			_copyCast_help<uint64_t>(in, out);
			break;
		case FLOAT128:
			_copyCast_help<long double>(in, out);
			break;
		case COMPLEX128:
			_copyCast_help<cdouble_t>(in, out);
			break;
		case COMPLEX256:
			_copyCast_help<cquad_t>(in, out);
			break;
		case RGBA32:
			_copyCast_help<rgba_t>(in, out);
			break;
		default:
			return NULL;
	}
	return out;
}

/**
 * @brief Create a new image that is a copy of the input, with same dimensions
 * but pxiels cast to newtype. The new image will have all overlapping pixels
 * copied from the old image.
 *
 * @param in Input image, anything that can be copied will be
 * @param newtype Type of pixels in output image
 *
 * @return Image with overlapping sections cast and copied from 'in'
 */
ptr<MRImage> _copyCast(ptr<const MRImage> in, PixelT newtype)
{
	return _copyCast(in, in->ndim(), in->dim(), newtype);
}

/**
 * @brief Create a new image that is a copy of the input, possibly with new
 * dimensions or size. The new image will have all overlapping pixels
 * copied from the old image. The new image will have the same pixel type as
 * the input image
 *
 * @param in Input image, anything that can be copied will be
 * @param newdims Number of dimensions in output image
 * @param newsize Size of output image
 *
 * @return Image with overlapping sections cast and copied from 'in'
 */
ptr<MRImage> _copyCast(ptr<const MRImage> in, size_t newdims,
		const size_t* newsize)
{
	return _copyCast(in, newdims, newsize, in->type());
}


/****************************************************************************
 * MRImage (Base Class) Function Implementations
 ****************************************************************************/

/**
 * @brief Copies metadata from another image. This includes slice timing,
 * anything read from nifti files, spacing, orientation etc, but NOT 
 * pixel data, size, and dimensionality. 
 *
 * @param in Other image to copy from
 */
void MRImage::copyMetadata(ptr<const MRImage> in)
{
	// copy image metadata
	this->m_freqdim = in->m_freqdim;
	this->m_slicedim = in->m_slicedim;
	this->m_phasedim = in->m_phasedim;
	this->m_slice_duration = in->m_slice_duration;
	this->m_slice_start = in->m_slice_start;
	this->m_slice_end = in->m_slice_end;
	this->m_slice_timing = in->m_slice_timing;
	this->m_slice_order = in->m_slice_order;
	this->setOrient(in->getOrigin(), in->getSpacing(), in->getDirection(), 1);
};

/**
 * @brief Returns true of the other image has matching orientation as this.
 * If checksize = true, then it will also check the size of the two images
 * and return true if both orientation and size match, and false if they
 * don't. 
 *
 * @param other MRimage to compare.
 * @param checksize Whether to enforce identical size as well as orientation
 *
 * @return True if the two images have matching orientation information.
 */
bool MRImage::matchingOrient(ptr<const MRImage> other, bool checksize) const
{
    if(ndim() != other->ndim())
        return false;
    
    double err = 0;
    double THRESH = 0.000001;

    // check spacing
    err = 0;
    for(size_t dd=0; dd<ndim(); dd++) 
        err += pow(spacing(dd)-other->spacing(dd),2);
    if(err > THRESH)
        return false;

    // Check Origin
    err = 0;
    for(size_t dd=0; dd<ndim(); dd++) 
        err += pow(origin(dd)-other->origin(dd),2);
    if(err > THRESH)
        return false;
    
    // check direction
    err = 0;
    for(size_t dd=0; dd<ndim(); dd++) {
        for(size_t ee=0; ee<ndim(); ee++) {
            err += pow(direction(dd,ee)-other->direction(dd,ee),2);
        }
    }
    if(err > THRESH)
        return false;
  
    if(checksize) {
        for(size_t dd=0; dd<ndim(); dd++) {
            if(dim(dd) != other->dim(dd))
                return false;
        }
    }

    return true;
};

/******************************************************************************
 * Pre-Compile Image Types
 ******************************************************************************/
template class MRImageStore<1, double>;
template class MRImageStore<1, long double>;
template class MRImageStore<1, cdouble_t>;
template class MRImageStore<1, cquad_t>;
template class MRImageStore<1, float>;
template class MRImageStore<1, cfloat_t>;
template class MRImageStore<1, int64_t>;
template class MRImageStore<1, uint64_t>;
template class MRImageStore<1, int32_t>;
template class MRImageStore<1, uint32_t>;
template class MRImageStore<1, int16_t>;
template class MRImageStore<1, uint16_t>;
template class MRImageStore<1, int8_t>;
template class MRImageStore<1, uint8_t>;
template class MRImageStore<1, rgba_t>;
template class MRImageStore<1, rgb_t>;

template class MRImageStore<2, double>;
template class MRImageStore<2, long double>;
template class MRImageStore<2, cdouble_t>;
template class MRImageStore<2, cquad_t>;
template class MRImageStore<2, float>;
template class MRImageStore<2, cfloat_t>;
template class MRImageStore<2, int64_t>;
template class MRImageStore<2, uint64_t>;
template class MRImageStore<2, int32_t>;
template class MRImageStore<2, uint32_t>;
template class MRImageStore<2, int16_t>;
template class MRImageStore<2, uint16_t>;
template class MRImageStore<2, int8_t>;
template class MRImageStore<2, uint8_t>;
template class MRImageStore<2, rgba_t>;
template class MRImageStore<2, rgb_t>;

template class MRImageStore<3, double>;
template class MRImageStore<3, long double>;
template class MRImageStore<3, cdouble_t>;
template class MRImageStore<3, cquad_t>;
template class MRImageStore<3, float>;
template class MRImageStore<3, cfloat_t>;
template class MRImageStore<3, int64_t>;
template class MRImageStore<3, uint64_t>;
template class MRImageStore<3, int32_t>;
template class MRImageStore<3, uint32_t>;
template class MRImageStore<3, int16_t>;
template class MRImageStore<3, uint16_t>;
template class MRImageStore<3, int8_t>;
template class MRImageStore<3, uint8_t>;
template class MRImageStore<3, rgba_t>;
template class MRImageStore<3, rgb_t>;

template class MRImageStore<4, double>;
template class MRImageStore<4, long double>;
template class MRImageStore<4, cdouble_t>;
template class MRImageStore<4, cquad_t>;
template class MRImageStore<4, float>;
template class MRImageStore<4, cfloat_t>;
template class MRImageStore<4, int64_t>;
template class MRImageStore<4, uint64_t>;
template class MRImageStore<4, int32_t>;
template class MRImageStore<4, uint32_t>;
template class MRImageStore<4, int16_t>;
template class MRImageStore<4, uint16_t>;
template class MRImageStore<4, int8_t>;
template class MRImageStore<4, uint8_t>;
template class MRImageStore<4, rgba_t>;
template class MRImageStore<4, rgb_t>;

template class MRImageStore<5, double>;
template class MRImageStore<5, long double>;
template class MRImageStore<5, cdouble_t>;
template class MRImageStore<5, cquad_t>;
template class MRImageStore<5, float>;
template class MRImageStore<5, cfloat_t>;
template class MRImageStore<5, int64_t>;
template class MRImageStore<5, uint64_t>;
template class MRImageStore<5, int32_t>;
template class MRImageStore<5, uint32_t>;
template class MRImageStore<5, int16_t>;
template class MRImageStore<5, uint16_t>;
template class MRImageStore<5, int8_t>;
template class MRImageStore<5, uint8_t>;
template class MRImageStore<5, rgba_t>;
template class MRImageStore<5, rgb_t>;

template class MRImageStore<6, double>;
template class MRImageStore<6, long double>;
template class MRImageStore<6, cdouble_t>;
template class MRImageStore<6, cquad_t>;
template class MRImageStore<6, float>;
template class MRImageStore<6, cfloat_t>;
template class MRImageStore<6, int64_t>;
template class MRImageStore<6, uint64_t>;
template class MRImageStore<6, int32_t>;
template class MRImageStore<6, uint32_t>;
template class MRImageStore<6, int16_t>;
template class MRImageStore<6, uint16_t>;
template class MRImageStore<6, int8_t>;
template class MRImageStore<6, uint8_t>;
template class MRImageStore<6, rgba_t>;
template class MRImageStore<6, rgb_t>;

template class MRImageStore<7, double>;
template class MRImageStore<7, long double>;
template class MRImageStore<7, cdouble_t>;
template class MRImageStore<7, cquad_t>;
template class MRImageStore<7, float>;
template class MRImageStore<7, cfloat_t>;
template class MRImageStore<7, int64_t>;
template class MRImageStore<7, uint64_t>;
template class MRImageStore<7, int32_t>;
template class MRImageStore<7, uint32_t>;
template class MRImageStore<7, int16_t>;
template class MRImageStore<7, uint16_t>;
template class MRImageStore<7, int8_t>;
template class MRImageStore<7, uint8_t>;
template class MRImageStore<7, rgba_t>;
template class MRImageStore<7, rgb_t>;

template class MRImageStore<8, double>;
template class MRImageStore<8, long double>;
template class MRImageStore<8, cdouble_t>;
template class MRImageStore<8, cquad_t>;
template class MRImageStore<8, float>;
template class MRImageStore<8, cfloat_t>;
template class MRImageStore<8, int64_t>;
template class MRImageStore<8, uint64_t>;
template class MRImageStore<8, int32_t>;
template class MRImageStore<8, uint32_t>;
template class MRImageStore<8, int16_t>;
template class MRImageStore<8, uint16_t>;
template class MRImageStore<8, int8_t>;
template class MRImageStore<8, uint8_t>;
template class MRImageStore<8, rgba_t>;
template class MRImageStore<8, rgb_t>;


} // npl

#include "mrimage.txx"


