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
 * @file mrimage_utils.cpp
 *
 *****************************************************************************/

#include "mrimage.h"
#include "iterators.h"
#include "accessors.h"
#include "ndarray_utils.h"
#include "mrimage_utils.h"
#include "byteswap.h"

#include <string>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <memory>
#include <cstring>

namespace npl {

using std::vector;
using std::shared_ptr;

double gaussKern(double x)
{
	const double PI = acos(-1);
	const double den = 1./sqrt(2*PI);
	return den*exp(-x*x/(2));
}

/**
 * @brief Smooths an image in 1 dimension
 *
 * @param in Input/output image to smooth
 * @param dim dimensions to smooth in. If you are smoothing individual volumes
 * of an fMRI you would provide dim={0,1,2}
 * @param stddev standard deviation in physical units index*spacing
 *
 */
void gaussianSmooth1D(shared_ptr<MRImage> inout, size_t dim,
		double stddev)
{
	//TODO figure out how to scale this properly, including with stddev and
	//spacing
	if(dim >= inout->ndim()) {
		throw std::out_of_range("Invalid dimension specified for 1D gaussian "
				"smoothing");
	}

	std::vector<int64_t> index(dim, 0);
	stddev /= inout->spacing()[dim];
	std::vector<double> buff(inout->dim(dim));

	// for reading have the kernel iterator
	KernelIter<double> kit(inout);
	std::vector<size_t> radius(inout->ndim(), 0);
	for(size_t dd=0; dd<inout->ndim(); dd++) {
		if(dd == dim)
			radius[dd] = round(2*stddev);
	}
	kit.setRadius(radius);
	kit.goBegin();

	// calculate normalization factor
	double normalize = 0;
	int64_t rad = radius[dim];
	for(int64_t ii=-rad; ii<=rad; ii++)
		normalize += gaussKern(ii/stddev);

	// for writing, have the regular iterator
	OrderIter<double> it(inout);
	it.setOrder(kit.getOrder());
	it.goBegin();
	while(!it.eof()) {

		// perform kernel math, writing to buffer
		for(size_t ii=0; ii<inout->dim(dim); ii++, ++kit) {
			double tmp = 0;
			for(size_t kk=0; kk<kit.ksize(); kk++) {
				double dist = kit.from_center(kk, dim);
				double nval = kit[kk];
				double stddist = dist/stddev;
				double weighted = gaussKern(stddist)*nval/normalize;
				tmp += weighted;
			}
			buff[ii] = tmp;
		}
		
		// write back out
		for(size_t ii=0; ii<inout->dim(dim); ii++, ++it)
			it.set(buff[ii]);

	}
}

/**
 * @brief Reads an MRI image. Right now only nift images are supported. later
 * on, it will try to load image using different reader functions until one
 * suceeds.
 *
 * @param filename Name of input file to read
 * @param verbose Whether to print out information as the file is read
 *
 * @return Loaded image
 */
shared_ptr<MRImage> readMRImage(std::string filename, bool verbose)
{
	const size_t BSIZE = 1024*1024; //1M
	auto gz = gzopen(filename.c_str(), "rb");

	if(!gz) {
		throw std::ios_base::failure("Could not open " + filename + " for readin");
		return NULL;
	}
	gzbuffer(gz, BSIZE);
	
	shared_ptr<MRImage> out;

	if((out = readNiftiImage(gz, verbose))) {
		gzclose(gz);
		return out;
	}

	throw std::ios_base::failure("Error reading " + filename );
	return NULL;
}

/**
 * @brief Writes out an MRImage to the file fn. Bool indicates whether to use
 * nifti2 (rather than nifti1) format.
 *
 * @param img Image to write.
 * @param fn Filename
 * @param nifti2 Whether the use nifti2 format
 *
 * @return 0 if successful
 */
int writeMRImage(MRImage* img, std::string fn, bool nifti2)
{
	if(!img)
		return -1;
	double version = 1;
	if(nifti2)
		version = 2;
	return img->write(fn, version);
}

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

/**
 * @brief Writes out information about an MRImage
 *
 * @param out Output ostream
 * @param img Image to write information about
 *
 * @return More ostream
 */
ostream& operator<<(ostream &out, const MRImage& img)
{
	out << "---------------------------" << endl;
	out << img.ndim() << "D Image" << endl;
	for(int64_t ii=0; ii<(int64_t)img.ndim(); ii++) {
		out << "dim[" << ii << "]=" << img.dim(ii);
		if(img.m_freqdim == ii)
			out << " (frequency-encode)";
		if(img.m_phasedim == ii)
			out << " (phase-encode)";
		if(img.m_slicedim == ii)
			out << " (slice-encode)";
		out << endl;
	}

	out << "Direction: " << endl;
	for(size_t ii=0; ii<img.ndim(); ii++) {
		cerr << "[ ";
		for(size_t jj=0; jj<img.ndim(); jj++) {
			cerr << std::setw(10) << std::setprecision(3) << img.direction()(ii,jj);
		}
		cerr << "] " << endl;
	}
	
	out << "Spacing: " << endl;
	for(size_t ii=0; ii<img.ndim(); ii++) {
		out << "[ " << std::setw(10) << std::setprecision(3)
			<< img.spacing()[ii] << "] ";
	}
	out << endl;

	out << "Origin: " << endl;
	for(size_t ii=0; ii<img.ndim(); ii++) {
		out << "[ " << std::setw(10) << std::setprecision(3)
			<< img.origin()[ii] << "] ";
	}
	out << endl;
	
	out << "Affine: " << endl;
	for(size_t ii=0; ii<img.ndim()+1; ii++) {
		cerr << "[ ";
		for(size_t jj=0; jj<img.ndim()+1; jj++) {
			cerr << std::setw(10) << std::setprecision(3) << img.affine()(ii,jj);
		}
		cerr << "] " << endl;
	}

	switch(img.type()) {
		case UNKNOWN_TYPE:
		case UINT8:
			out << "UINT8" << endl;
			break;
		case INT16:
			out << "INT16" << endl;
			break;
		case INT32:
			out << "INT32" << endl;
			break;
		case FLOAT32:
			out << "FLOAT32" << endl;
			break;
		case COMPLEX64:
			out << "COMPLEX64" << endl;
			break;
		case FLOAT64:
			out << "FLOAT64" << endl;
			break;
		case RGB24:
			out << "RGB24" << endl;
			break;
		case INT8:
			out << "INT8" << endl;
			break;
		case UINT16:
			out << "UINT16" << endl;
			break;
		case UINT32:
			out << "UINT32" << endl;
			break;
		case INT64:
			out << "INT64" << endl;
			break;
		case UINT64:
			out << "UINT64" << endl;
			break;
		case FLOAT128:
			out << "FLOAT128" << endl;
			break;
		case COMPLEX128:
			out << "COMPLEX128" << endl;
			break;
		case COMPLEX256:
			out << "COMPLEX256" << endl;
			break;
		case RGBA32:
			out << "RGBA32" << endl;
			break;
		default:
			out << "UNKNOWN" << endl;
			break;
	}

	out << "Slice Duration: " << img.m_slice_duration << endl;
	out << "Slice Start: " << img.m_slice_start << endl;
	out << "Slice End: " << img.m_slice_end << endl;
	switch(img.m_slice_order) {
		case SEQ:
			out << "Slice Order: Increasing Sequential" << endl;
			break;
		case RSEQ:
			out << "Slice Order: Decreasing Sequential" << endl;
			break;
		case ALT:
			out << "Slice Order: Increasing Alternating" << endl;
			break;
		case RALT:
			out << "Slice Order: Decreasing Alternating" << endl;
			break;
		case ALT_SHFT:
			out << "Slice Order: Alternating Starting at "
				<< img.m_slice_start+1 << " (not "
				<< img.m_slice_start << ")" << endl;
			break;
		case RALT_SHFT:
			out << "Slice Order: Decreasing Alternating Starting at "
				<< img.m_slice_end-1 << " not ( " << img.m_slice_end << endl;
			break;
		case UNKNOWN_SLICE:
		default:
			out << "Slice Order: Not Set" << endl;
			break;
	}
	out << "Slice Timing: " << endl;
	for(auto it=img.m_slice_timing.begin(); it != img.m_slice_timing.end(); ++it) {
		out << std::setw(10) << it->first << std::setw(10) << std::setprecision(3)
			<< it->second << ",";
	}
	out << endl;
	out << "---------------------------" << endl;
	return out;
}

/**
 * @brief Perform fourier transform on the dimensions specified. Those
 * dimensions will be padded out. The output of this will be a double.
 * If len = 0 or dim == NULL, then ALL dimensions will be transformed.
 *
 * @param in Input image to inverse fourier trnasform
 * @param len length of input dimension array
 * @param dim dimensions to transform
 *
 * @return Image with specified dimensions in the real domain. Image will
 * differ in size from input.
 */
shared_ptr<MRImage> ifft_c2r(shared_ptr<const MRImage> in)
{
	auto out = dynamic_pointer_cast<MRImage>(ifft_c2r(
				dynamic_pointer_cast<const NDArray>(in)));
	for(size_t dd = 0; dd<in->ndim(); dd++)
		out->spacing()[dd] = 1./(in->spacing()[dd]*out->dim(dd));
	return out;
}

/**
 * @brief Perform fourier transform on the dimensions specified. Those
 * dimensions will be padded out. The output of this will be a complex double.
 * If len = 0 or dim == NULL, then ALL dimensions will be transformed.
 *
 * @param in Input image to fourier transform
 *
 * @return Complex image, which is the result of inverse fourier transforming
 * the (Real) input image. Note that the last dimension only contains the real
 * frequencies, but all other dimensions contain both
 */
shared_ptr<MRImage> fft_r2c(shared_ptr<const MRImage> in)
{
	auto out = dynamic_pointer_cast<MRImage>(fft_r2c(
				dynamic_pointer_cast<const NDArray>(in)));
	for(size_t dd = 0; dd<in->ndim(); dd++)
		out->spacing()[dd] = 1./(in->spacing()[dd]*out->dim(dd));
	return out;
}


} // npl


