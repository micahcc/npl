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
shared_ptr<MRImage> createMRImageHelp(size_t len, const size_t* dim)
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
shared_ptr<MRImage> createMRImage(size_t ndim, const size_t* size, PixelT type)
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
shared_ptr<MRImage> createMRImage(const std::vector<size_t>& dim, PixelT type)
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
shared_ptr<MRImage> createMRImageHelp(size_t len, const size_t* dim,
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
shared_ptr<MRImage> createMRImage(size_t ndim, const size_t* size, PixelT type,
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
shared_ptr<MRImage> createMRImage(const std::vector<size_t>& dim, PixelT type,
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
void _copyCast_help(shared_ptr<const MRImage> in, shared_ptr<MRImage> out)
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
			roi[ii].second = min(out->dim(ii), in->dim(ii));
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
shared_ptr<MRImage> _copyCast(shared_ptr<const MRImage> in, size_t newdims,
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
	out->setOrient(in->origin(), in->spacing(), in->direction(), 1);

	
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
shared_ptr<MRImage> _copyCast(shared_ptr<const MRImage> in, PixelT newtype)
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
shared_ptr<MRImage> _copyCast(shared_ptr<const MRImage> in, size_t newdims,
		const size_t* newsize)
{
	return _copyCast(in, newdims, newsize, in->type());
}

/****************************************************************************
 * Input Functions
 ****************************************************************************/
/**
 * @brief Helper function for readNiftiImage. End users should use readMRImage
 *
 * @tparam T Type of pixels to read
 * @param file Already opened gzFile
 * @param vox_offset Offset to start reading at
 * @param dim Dimensions of input image
 * @param pixsize Size, in bytes, of each pixel
 * @param doswap Whether to perform byte swapping on the pixels
 *
 * @return New MRImage with loaded pixels
 */
template <typename T>
shared_ptr<MRImage> readPixels(gzFile file, size_t vox_offset,
		const std::vector<size_t>& dim, size_t pixsize, bool doswap)
{
	// jump to voxel offset
	gzseek(file, vox_offset, SEEK_SET);

	/*
	 * Create Slicer Object to iterate through image slices
	 */

	// dim 0 is the fastest in nifti images, so go in that order
	Slicer slicer(dim.size(), dim.data());
	slicer.setOrder({}, true);

	T tmp(0);
	shared_ptr<MRImage> out;

	// someday this all might be simplify by using MRImage* and the
	// dbl or int64 functions, as long as we trust that the type is
	// going to be good enough to caputre the underlying pixle type
	switch(dim.size()) {
		case 1: {
			auto typed = std::make_shared<MRImageStore<1, T>>(dim);
			for(slicer.goBegin(); !slicer.isEnd(); ++slicer) {
				gzread(file, &tmp, pixsize);
				if(doswap) swap<T>(&tmp);
				(*typed)[*slicer] = tmp;
			}
			out = typed;
			} break;
		case 2:{
			auto typed = std::make_shared<MRImageStore<2, T>>(dim);
			for(slicer.goBegin(); !slicer.isEnd(); ++slicer) {
				gzread(file, &tmp, pixsize);
				if(doswap) swap<T>(&tmp);
				(*typed)[*slicer] = tmp;
			}
			out = typed;
			} break;
		case 3:{
			auto typed = std::make_shared<MRImageStore<3, T>>(dim);
			for(slicer.goBegin(); !slicer.isEnd(); ++slicer) {
				gzread(file, &tmp, pixsize);
				if(doswap) swap<T>(&tmp);
				(*typed)[*slicer] = tmp;
			}
			out = typed;
			} break;
		case 4:{
			auto typed = std::make_shared<MRImageStore<4, T>>(dim);
			for(slicer.goBegin(); !slicer.isEnd(); ++slicer) {
				gzread(file, &tmp, pixsize);
				if(doswap) swap<T>(&tmp);
				(*typed)[*slicer] = tmp;
			}
			out = typed;
			} break;
		case 5:{
			auto typed = std::make_shared<MRImageStore<5, T>>(dim);
			for(slicer.goBegin(); !slicer.isEnd(); ++slicer) {
				gzread(file, &tmp, pixsize);
				if(doswap) swap<T>(&tmp);
				(*typed)[*slicer] = tmp;
			}
			out = typed;
			} break;
		case 6:{
			auto typed = std::make_shared<MRImageStore<6, T>>(dim);
			for(slicer.goBegin(); !slicer.isEnd(); ++slicer) {
				gzread(file, &tmp, pixsize);
				if(doswap) swap<T>(&tmp);
				(*typed)[*slicer] = tmp;
			}
			out = typed;
			} break;
		case 7:{
			auto typed = std::make_shared<MRImageStore<7, T>>(dim);
			for(slicer.goBegin(); !slicer.isEnd(); ++slicer) {
				gzread(file, &tmp, pixsize);
				if(doswap) swap<T>(&tmp);
				(*typed)[*slicer] = tmp;
			}
			out = typed;
			} break;
		case 8:{
			auto typed = std::make_shared<MRImageStore<8, T>>(dim);
			for(slicer.goBegin(); !slicer.isEnd(); ++slicer) {
				gzread(file, &tmp, pixsize);
				if(doswap) swap<T>(&tmp);
				(*typed)[*slicer] = tmp;
			}
			out = typed;
			} break;
	};

	return out;
}

/**
 * @brief Function to parse nifti1header. End users should use readMRimage.
 *
 * @param file already open gzFile, although it will seek to begin
 * @param header Header to fill in
 * @param doswap Whether to byteswap header elements
 * @param verbose Whether to print out header information
 *
 * @return 0 if successful
 */
int readNifti1Header(gzFile file, nifti1_header* header, bool* doswap,
		bool verbose)
{
	// seek to 0
	gzseek(file, 0, SEEK_SET);

	static_assert(sizeof(nifti1_header) == 348, "Error, nifti header packing failed");

	// read header
	gzread(file, header, sizeof(nifti1_header));
	if(strncmp(header->magic, "n+1", 3)) {
		gzclearerr(file);
		gzrewind(file);
		return 1;
	}

	// byte swap
	int64_t npixel = 1;
	if(header->sizeof_hdr != 348) {
		*doswap = true;
		swap(&header->sizeof_hdr);
		if(header->sizeof_hdr != 348) {
			swap(&header->sizeof_hdr);
			return -1;
		}
		swap(&header->ndim);
		for(size_t ii=0; ii<7; ii++)
			swap(&header->dim[ii]);
		swap(&header->intent_p1);
		swap(&header->intent_p2);
		swap(&header->intent_p3);
		swap(&header->intent_code);
		swap(&header->datatype);
		swap(&header->bitpix);
		swap(&header->slice_start);
		swap(&header->qfac);
		for(size_t ii=0; ii<7; ii++)
			swap(&header->pixdim[ii]);
		swap(&header->vox_offset);
		swap(&header->scl_slope);
		swap(&header->scl_inter);
		swap(&header->slice_end);
		swap(&header->cal_max);
		swap(&header->cal_min);
		swap(&header->slice_duration);
		swap(&header->toffset);
		swap(&header->glmax);
		swap(&header->glmin);
		swap(&header->qform_code);
		swap(&header->sform_code);
		
		for(size_t ii=0; ii<3; ii++)
			swap(&header->quatern[ii]);
		for(size_t ii=0; ii<3; ii++)
			swap(&header->qoffset[ii]);
		for(size_t ii=0; ii<12; ii++)
			swap(&header->saffine[ii]);

		for(int32_t ii=0; ii<header->ndim; ii++)
			npixel *= header->dim[ii];
	}
	
	if(verbose) {
		std::cerr << "sizeof_hdr=" << header->sizeof_hdr << std::endl;
		std::cerr << "data_type=" << header->data_type << std::endl;
		std::cerr << "db_name=" << header->db_name << std::endl;
		std::cerr << "extents=" << header->extents  << std::endl;
		std::cerr << "session_error=" << header->session_error << std::endl;
		std::cerr << "regular=" << header->regular << std::endl;

		std::cerr << "magic =" << header->magic  << std::endl;
		std::cerr << "datatype=" << header->datatype << std::endl;
		std::cerr << "bitpix=" << header->bitpix << std::endl;
		std::cerr << "ndim=" << header->ndim << std::endl;
		for(size_t ii=0; ii < 7; ii++)
			std::cerr << "dim["<<ii<<"]=" << header->dim[ii] << std::endl;
		std::cerr << "intent_p1 =" << header->intent_p1  << std::endl;
		std::cerr << "intent_p2 =" << header->intent_p2  << std::endl;
		std::cerr << "intent_p3 =" << header->intent_p3  << std::endl;
		std::cerr << "qfac=" << header->qfac << std::endl;
		for(size_t ii=0; ii < 7; ii++)
			std::cerr << "pixdim["<<ii<<"]=" << header->pixdim[ii] << std::endl;
		std::cerr << "vox_offset=" << header->vox_offset << std::endl;
		std::cerr << "scl_slope =" << header->scl_slope  << std::endl;
		std::cerr << "scl_inter =" << header->scl_inter  << std::endl;
		std::cerr << "cal_max=" << header->cal_max << std::endl;
		std::cerr << "cal_min=" << header->cal_min << std::endl;
		std::cerr << "slice_duration=" << header->slice_duration << std::endl;
		std::cerr << "toffset=" << header->toffset << std::endl;
		std::cerr << "glmax=" << header->glmax  << std::endl;
		std::cerr << "glmin=" << header->glmin  << std::endl;
		std::cerr << "slice_start=" << header->slice_start << std::endl;
		std::cerr << "slice_end=" << header->slice_end << std::endl;
		std::cerr << "descrip=" << header->descrip << std::endl;
		std::cerr << "aux_file=" << header->aux_file << std::endl;
		std::cerr << "qform_code =" << header->qform_code  << std::endl;
		std::cerr << "sform_code =" << header->sform_code  << std::endl;
		for(size_t ii=0; ii < 3; ii++){
			std::cerr << "quatern["<<ii<<"]="
				<< header->quatern[ii] << std::endl;
		}
		for(size_t ii=0; ii < 3; ii++){
			std::cerr << "qoffset["<<ii<<"]="
				<< header->qoffset[ii] << std::endl;
		}
		for(size_t ii=0; ii < 3; ii++) {
			for(size_t jj=0; jj < 4; jj++) {
				std::cerr << "saffine["<<ii<<"*4+"<<jj<<"]="
					<< header->saffine[ii*4+jj] << std::endl;
			}
		}
		std::cerr << "slice_code=" << (int)header->slice_code << std::endl;
		std::cerr << "xyzt_units=" << header->xyzt_units << std::endl;
		std::cerr << "intent_code =" << header->intent_code  << std::endl;
		std::cerr << "intent_name=" << header->intent_name << std::endl;
		std::cerr << "dim_info.bits.freqdim=" << header->dim_info.bits.freqdim << std::endl;
		std::cerr << "dim_info.bits.phasedim=" << header->dim_info.bits.phasedim << std::endl;
		std::cerr << "dim_info.bits.slicedim=" << header->dim_info.bits.slicedim << std::endl;
	}
	
	return 0;
}

/**
 * @brief Reads a nifti image, given an already open gzFile.
 *
 * @param file gzFile to read from
 * @param verbose whether to print out information during header parsing
 *
 * @return New MRImage with values from header and pixels set
 */
shared_ptr<MRImage> readNiftiImage(gzFile file, bool verbose)
{
	bool doswap = false;
	int16_t datatype = 0;
	size_t start;
	std::vector<size_t> dim;
	size_t psize;
	int qform_code = 0;
	std::vector<double> pixdim;
	std::vector<double> offset;
	std::vector<double> quatern(3,0);
	double qfac;
	double slice_duration = 0;
	int slice_code = 0;
	int slice_start = 0;
	int slice_end = 0;
	int freqdim = 0;
	int phasedim = 0;
	int slicedim = 0;

	int ret = 0;
	nifti1_header header1;
	nifti2_header header2;
	if((ret = readNifti1Header(file, &header1, &doswap, verbose)) == 0) {
		start = header1.vox_offset;
		dim.resize(header1.ndim, 0);
		for(int64_t ii=0; ii<header1.ndim && ii < 7; ii++) {
			dim[ii] = header1.dim[ii];
		}
		psize = (header1.bitpix >> 3);
		qform_code = header1.qform_code;
		datatype = header1.datatype;

		slice_code = header1.slice_code;
		slice_duration = header1.slice_duration;
		slice_start = header1.slice_start;
		slice_end = header1.slice_end;
		freqdim = (int)(header1.dim_info.bits.freqdim)-1;
		phasedim = (int)(header1.dim_info.bits.phasedim)-1;
		slicedim = (int)(header1.dim_info.bits.slicedim)-1;

		// pixdim
		pixdim.resize(header1.ndim, 0);
		for(int64_t ii=0; ii<header1.ndim && ii < 7; ii++)
			pixdim[ii] = header1.pixdim[ii];

		// offset
		offset.resize(4, 0);
		for(int64_t ii=0; ii<header1.ndim && ii < 3; ii++)
			offset[ii] = header1.qoffset[ii];
		if(header1.ndim > 3)
			offset[3] = header1.toffset;

		// quaternion
		for(int64_t ii=0; ii<3 && ii<header1.ndim; ii++)
			quatern[ii] = header1.quatern[ii];
		qfac = header1.qfac;
	}

	if(ret!=0 && (ret = readNifti2Header(file, &header2, &doswap, verbose)) == 0) {
		start = header2.vox_offset;
		dim.resize(header2.ndim, 0);
		for(int64_t ii=0; ii<header2.ndim && ii < 7; ii++) {
			dim[ii] = header2.dim[ii];
		}
		psize = (header2.bitpix >> 3);
		qform_code = header2.qform_code;
		datatype = header2.datatype;
		
		slice_code = header2.slice_code;
		slice_duration = header2.slice_duration;
		slice_start = header2.slice_start;
		slice_end = header2.slice_end;
		freqdim = (int)(header2.dim_info.bits.freqdim)-1;
		phasedim = (int)(header2.dim_info.bits.phasedim)-1;
		slicedim = (int)(header2.dim_info.bits.slicedim)-1;

		// pixdim
		pixdim.resize(header2.ndim, 0);
		for(int64_t ii=0; ii<header2.ndim && ii < 7; ii++)
			pixdim[ii] = header2.pixdim[ii];
		
		// offset
		offset.resize(4, 0);
		for(int64_t ii=0; ii<header2.ndim && ii < 3; ii++)
			offset[ii] = header2.qoffset[ii];
		if(header2.ndim > 3)
			offset[3] = header2.toffset;
		
		// quaternion
		for(int64_t ii=0; ii<3 && ii<header2.ndim; ii++)
			quatern[ii] = header2.quatern[ii];
		qfac = header2.qfac;
	}

	shared_ptr<MRImage> out;

	// create image
	switch(datatype) {
		// 8 bit
		case NIFTI_TYPE_INT8:
			out = readPixels<int8_t>(file, start, dim, psize, doswap);
		break;
		case NIFTI_TYPE_UINT8:
			out = readPixels<uint8_t>(file, start, dim, psize, doswap);
		break;
		// 16  bit
		case NIFTI_TYPE_INT16:
			out = readPixels<int16_t>(file, start, dim, psize, doswap);
		break;
		case NIFTI_TYPE_UINT16:
			out = readPixels<uint16_t>(file, start, dim, psize, doswap);
		break;
		// 32 bit
		case NIFTI_TYPE_INT32:
			out = readPixels<int32_t>(file, start, dim, psize, doswap);
		break;
		case NIFTI_TYPE_UINT32:
			out = readPixels<uint32_t>(file, start, dim, psize, doswap);
		break;
		// 64 bit int
		case NIFTI_TYPE_INT64:
			out = readPixels<int64_t>(file, start, dim, psize, doswap);
		break;
		case NIFTI_TYPE_UINT64:
			out = readPixels<uint64_t>(file, start, dim, psize, doswap);
		break;
		// floats
		case NIFTI_TYPE_FLOAT32:
			out = readPixels<float>(file, start, dim, psize, doswap);
		break;
		case NIFTI_TYPE_FLOAT64:
			out = readPixels<double>(file, start, dim, psize, doswap);
		break;
		case NIFTI_TYPE_FLOAT128:
			out = readPixels<long double>(file, start, dim, psize, doswap);
		break;
		// RGB
		case NIFTI_TYPE_RGB24:
			out = readPixels<rgb_t>(file, start, dim, psize, doswap);
		break;
		case NIFTI_TYPE_RGBA32:
			out = readPixels<rgba_t>(file, start, dim, psize, doswap);
		break;
		case NIFTI_TYPE_COMPLEX256:
			out = readPixels<cquad_t>(file, start, dim, psize, doswap);
		break;
		case NIFTI_TYPE_COMPLEX128:
			out = readPixels<cdouble_t>(file, start, dim, psize, doswap);
		break;
		case NIFTI_TYPE_COMPLEX64:
			out = readPixels<cfloat_t>(file, start, dim, psize, doswap);
		break;
	}

	if(!out)
		return NULL;

	/*
	 * Now that we have an Image*, we can fill in the remaining values from
	 * the header
	 */

	// figure out orientation
	if(qform_code > 0) {
		/*
		 * set spacing
		 */
		for(size_t ii=0; ii<out->ndim(); ii++)
			out->spacing()[ii] = pixdim[ii];
		
		/*
		 * set origin
		 */
		// x,y,z
		for(size_t ii=0; ii<out->ndim(); ii++) {
			out->origin()[ii] = offset[ii];
		}
		
		// calculate a, copy others
		double b = quatern[0];
		double c = quatern[1];
		double d = quatern[2];
		double a = sqrt(1.0-(b*b+c*c+d*d));

		// calculate R, (was already identity)
		out->direction()(0, 0) = a*a+b*b-c*c-d*d;

		if(out->ndim() > 1) {
			out->direction()(0,1) = 2*b*c-2*a*d;
			out->direction()(1,0) = 2*b*c+2*a*d;
			out->direction()(1,1) = a*a+c*c-b*b-d*d;
		}

		if(qfac != -1)
			qfac = 1;
		
		if(out->ndim() > 2) {
			out->direction()(0,2) = qfac*(2*b*d+2*a*c);
			out->direction()(1,2) = qfac*(2*c*d-2*a*b);
			out->direction()(2,2) = qfac*(a*a+d*d-c*c-b*b);
			out->direction()(2,1) = 2*c*d+2*a*b;
			out->direction()(2,0) = 2*b*d-2*a*c;
		}
		
		if(verbose) {
			std::cerr << "Direction:" << std::endl;
			std::cerr << out->direction() << endl;;
		}

		// finally update affine, but scale pixdim[z] by qfac temporarily
		out->updateAffine();
		if(verbose) {
			std::cerr << "Affine:" << std::endl;
			std::cerr << out->affine() << endl;;
		}
//	} else if(header.sform_code > 0) {
//		/* use the sform, since no qform exists */
//
//		// origin, last column
//		double di = 0, dj = 0, dk = 0;
//		for(size_t ii=0; ii<3 && ii<out->ndim(); ii++) {
//			di += pow(header.saffine[4*ii+0],2); //column 0
//			dj += pow(header.saffine[4*jj+1],2); //column 1
//			dk += pow(header.saffine[4*kk+2],2); //column 2
//			out->origin()[ii] = header.saffine[4*ii+3]; //column 3
//		}
//		
//		// set direction and spacing
//		out->m_spacing[0] = sqrt(di);
//		out->m_dir[0*out->ndim()+0] = header.saffine[4*0+0]/di;
//
//		if(out->ndim() > 1) {
//			out->m_spacing[1] = sqrt(dj);
//			out->m_dir[0*out->ndim()+1] = header.saffine[4*0+1]/dj;
//			out->m_dir[1*out->ndim()+1] = header.saffine[4*1+1]/dj;
//			out->m_dir[1*out->ndim()+0] = header.saffine[4*1+0]/di;
//		}
//		if(out->ndim() > 2) {
//			out->m_spacing[2] = sqrt(dk);
//			out->m_dir[0*out->ndim()+2] = header.saffine[4*0+2]/dk;
//			out->m_dir[1*out->ndim()+2] = header.saffine[4*1+2]/dk;
//			out->m_dir[2*out->ndim()+2] = header.saffine[4*2+2]/dk;
//			out->m_dir[2*out->ndim()+1] = header.saffine[4*2+1]/dj;
//			out->m_dir[2*out->ndim()+0] = header.saffine[4*2+0]/di;
//		}
//
//		// affine matrix
//		updateAffine();
	} else {
		// only spacing changes
		for(size_t ii=0; ii<dim.size(); ii++)
			out->spacing()[ii] = pixdim[ii];
		out->updateAffine();
	}

	/**************************************************************************
	 * Medical Imaging Varaibles Variables
	 **************************************************************************/
	
	// direct copies
	out->m_freqdim = freqdim;
	out->m_phasedim = phasedim;
	out->m_slicedim = slicedim;
	
	// slice timing
	out->updateSliceTiming(slice_duration,  slice_start, slice_end,
			(SliceOrderT)slice_code);

	return out;
}

/**
 * @brief Reads a nifti2 header from an already-open gzFile. End users should
 * use readMRImage instead.
 *
 * @param file Already opened gzFile, will seek to 0
 * @param header Header to put data into
 * @param doswap whether to swap header fields
 * @param verbose Whether to print information about header
 *
 * @return 0 if successful
 */
int readNifti2Header(gzFile file, nifti2_header* header, bool* doswap,
		bool verbose)
{
	// seek to 0
	gzseek(file, 0, SEEK_SET);

	static_assert(sizeof(nifti2_header) == 540, "Error, nifti header packing failed");

	// read header
	gzread(file, header, sizeof(nifti2_header));
	if(strncmp(header->magic, "n+2", 3)) {
		gzclearerr(file);
		gzrewind(file);
		return -1;
	}

	// byte swap
	int64_t npixel = 1;
	if(header->sizeof_hdr != 540) {
		*doswap = true;
		swap(&header->sizeof_hdr);
		if(header->sizeof_hdr != 540) {
			swap(&header->sizeof_hdr);
			return -1;
		}
		swap(&header->datatype);
		swap(&header->bitpix);
		swap(&header->ndim);
		for(size_t ii=0; ii<7; ii++)
			swap(&header->dim[ii]);
		swap(&header->intent_p1);
		swap(&header->intent_p2);
		swap(&header->intent_p3);
		swap(&header->qfac);
		for(size_t ii=0; ii<7; ii++)
			swap(&header->pixdim[ii]);
		swap(&header->vox_offset);
		swap(&header->scl_slope);
		swap(&header->scl_inter);
		swap(&header->cal_max);
		swap(&header->cal_min);
		swap(&header->slice_duration);
		swap(&header->toffset);
		swap(&header->slice_start);
		swap(&header->slice_end);
//		swap(&header->glmax);
//		swap(&header->glmin);
		swap(&header->qform_code);
		swap(&header->sform_code);
		
		for(size_t ii=0; ii<3; ii++)
			swap(&header->quatern[ii]);
		for(size_t ii=0; ii<3; ii++)
			swap(&header->qoffset[ii]);
		for(size_t ii=0; ii<12; ii++)
			swap(&header->saffine[ii]);
		
		swap(&header->slice_code);
		swap(&header->xyzt_units);
		swap(&header->intent_code);

		for(int32_t ii=0; ii<header->ndim; ii++)
			npixel *= header->dim[ii];
	}
	
	if(verbose) {
		std::cerr << "sizeof_hdr=" << header->sizeof_hdr << std::endl;
		std::cerr << "magic =" << header->magic  << std::endl;
		std::cerr << "datatype=" << header->datatype << std::endl;
		std::cerr << "bitpix=" << header->bitpix << std::endl;
		std::cerr << "ndim=" << header->ndim << std::endl;
		for(size_t ii=0; ii < 7; ii++)
			std::cerr << "dim["<<ii<<"]=" << header->dim[ii] << std::endl;
		std::cerr << "intent_p1 =" << header->intent_p1  << std::endl;
		std::cerr << "intent_p2 =" << header->intent_p2  << std::endl;
		std::cerr << "intent_p3 =" << header->intent_p3  << std::endl;
		std::cerr << "qfac=" << header->qfac << std::endl;
		for(size_t ii=0; ii < 7; ii++)
			std::cerr << "pixdim["<<ii<<"]=" << header->pixdim[ii] << std::endl;
		std::cerr << "vox_offset=" << header->vox_offset << std::endl;
		std::cerr << "scl_slope =" << header->scl_slope  << std::endl;
		std::cerr << "scl_inter =" << header->scl_inter  << std::endl;
		std::cerr << "cal_max=" << header->cal_max << std::endl;
		std::cerr << "cal_min=" << header->cal_min << std::endl;
		std::cerr << "slice_duration=" << header->slice_duration << std::endl;
		std::cerr << "toffset=" << header->toffset << std::endl;
		std::cerr << "slice_start=" << header->slice_start << std::endl;
		std::cerr << "slice_end=" << header->slice_end << std::endl;
		std::cerr << "descrip=" << header->descrip << std::endl;
		std::cerr << "aux_file=" << header->aux_file << std::endl;
		std::cerr << "qform_code =" << header->qform_code  << std::endl;
		std::cerr << "sform_code =" << header->sform_code  << std::endl;
		for(size_t ii=0; ii < 3; ii++){
			std::cerr << "quatern["<<ii<<"]="
				<< header->quatern[ii] << std::endl;
		}
		for(size_t ii=0; ii < 3; ii++){
			std::cerr << "qoffset["<<ii<<"]="
				<< header->qoffset[ii] << std::endl;
		}
		for(size_t ii=0; ii < 3; ii++) {
			for(size_t jj=0; jj < 4; jj++) {
				std::cerr << "saffine["<<ii<<"*4+"<<jj<<"]="
					<< header->saffine[ii*4+jj] << std::endl;
			}
		}
		std::cerr << "slice_code=" << (int)header->slice_code << std::endl;
		std::cerr << "xyzt_units=" << header->xyzt_units << std::endl;
		std::cerr << "intent_code =" << header->intent_code  << std::endl;
		std::cerr << "intent_name=" << header->intent_name << std::endl;
		std::cerr << "dim_info.bits.freqdim=" << header->dim_info.bits.freqdim << std::endl;
		std::cerr << "dim_info.bits.phasedim=" << header->dim_info.bits.phasedim << std::endl;
		std::cerr << "dim_info.bits.slicedim=" << header->dim_info.bits.slicedim << std::endl;
		std::cerr << "unused_str=" << header->unused_str << std::endl;
	}
	
	return 0;
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
void MRImage::copyMetadata(shared_ptr<const MRImage> in)
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
	this->setOrient(in->origin(), in->spacing(), in->direction(), 1);
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


