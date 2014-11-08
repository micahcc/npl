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
 * @file nplio.cpp Readers and Writers for npl::MRImage and npl::NDarray
 *
 *****************************************************************************/

#include "mrimage.h"
#include "mrimage_utils.h"
#include "ndarray.h"
#include "ndarray_utils.h"
#include "macros.h"
#include "iterators.h"
#include "byteswap.h"

#include "zlib.h"

#include <string>
#include <iostream>
using std::string;
using std::to_string;
using std::cerr;
using std::endl;

namespace npl 
{

/*****************************************************************
 * Helper Functoins
 ***************************************************************/

/**
 * @brief Helper function for readNiftiImage. End users should use readMRImage
 *
 * @tparam T Type of pixels to read
 * @param arr NDArray or Image to write to
 * @param file Already opened gzFile
 * @param vox_offset Offset to start reading at
 * @param pixsize Size, in bytes, of each pixel
 * @param doswap Whether to perform byte swapping on the pixels
 *
 */
template <typename T>
void readPixels(ptr<NDArray> arr, gzFile file, size_t vox_offset, 
        size_t pixsize, bool doswap)
{
    if(pixsize != sizeof(T)) {
        throw INVALID_ARGUMENT("Pixel size in file ("+to_string(pixsize)
                +")does not match actual size of "+typeid(T).name());
    }

	// jump to voxel offset
	gzseek(file, vox_offset, SEEK_SET);

    // need to reverse order from nifti
    NDIter<T> it(arr);
    it.setOrder({}, true); 
    for(it.goBegin(); !it.eof(); ++it) {
        T tmp; 
        if(gzread(file, &tmp, sizeof(T)) <= 0)
            throw RUNTIME_ERROR("Input file ended abruptly");
        if(doswap) swap<T>(&tmp);
        it.set(tmp);
    }
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
 * @brief Reads a nifti image, given an already open gzFile.
 *
 * @param file gzFile to read from
 * @param verbose whether to print out information during header parsing
 * @param makearray Rather than making an image, make and NDArray
 * @param nopixeldata Don't actually read the pixel data, but still create
 * the NDArray
 *
 * @return New MRImage with values from header and pixels set
 */
ptr<NDArray> readNiftiImage(gzFile file, bool verbose, bool makearray,
        bool nopixeldata = false)
{
	bool doswap = false;
	PixelT datatype = UNKNOWN_TYPE;
	size_t start = SIZE_MAX;
	std::vector<size_t> dim;
	size_t psize = 0;
	int qform_code = 0;
	int sform_code = 0;
	std::vector<double> pixdim;
	std::vector<double> offset;
	std::vector<double> quatern(3,0);
	std::vector<double> saffine(12,0);
	double qfac = -1;
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
		sform_code = header1.sform_code;
		datatype = (PixelT)header1.datatype;

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
        
        // saffine
        for(size_t ii=0; ii<12; ii++)
            saffine[ii] = header1.saffine[ii];
	}

	if(ret!=0 && (ret = readNifti2Header(file, &header2, &doswap, verbose)) == 0) {
		start = header2.vox_offset;
		dim.resize(header2.ndim, 0);
		for(int64_t ii=0; ii<header2.ndim && ii < 7; ii++) {
			dim[ii] = header2.dim[ii];
		}
		psize = (header2.bitpix >> 3);
		qform_code = header2.qform_code;
		sform_code = header2.sform_code;
		datatype = (PixelT)header2.datatype;
		
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

        // saffine
        for(size_t ii=0; ii<12; ii++)
            saffine[ii] = header2.saffine[ii];
	}

    ptr<NDArray> out;
    if(makearray) {
        // just create an array, not an image
        out = createNDArray(dim.size(), dim.data(), datatype);

        if(verbose) 
            std::cerr << (*out) << std::endl;

    } else {
        // create an image, get orientation
        out = createMRImage(dim.size(), dim.data(), datatype);
        auto oimage = dPtrCast<MRImage>(out);
	
        // figure out orientation
        if(qform_code > 0) {
            /*
             * set spacing
             */
            for(size_t ii=0; ii<oimage->ndim(); ii++)
                oimage->spacing(ii) = pixdim[ii];

            /*
             * set origin
             */
            // x,y,z
            for(size_t ii=0; ii<oimage->ndim(); ii++) {
                oimage->origin(ii) = offset[ii];
            }

            // calculate a, copy others
            double b = quatern[0];
            double c = quatern[1];
            double d = quatern[2];
			double a = 1.0-(b*b+c*c+d*d);
			
			// if a is extremeley small (or negative), renormalize, make a=0
			if(a < 1e-7) {
				a = 1./sqrt(b*b+c*c+d*d);
				b *= a;
				c *= a;
				d *= a;
				a = 0;
			} else {
				a = sqrt(a);
			}

            // calculate R, (was already identity)
            MatrixXd tmpdirection = oimage->getDirection();
            tmpdirection(0,0) = a*a+b*b-c*c-d*d;

            if(oimage->ndim() > 1) {
                tmpdirection(0,1) = 2*b*c-2*a*d;
                tmpdirection(1,0) = 2*b*c+2*a*d;
                tmpdirection(1,1) = a*a+c*c-b*b-d*d;
            }

            if(qfac != -1)
                qfac = 1;

            if(oimage->ndim() > 2) {
                tmpdirection(0,2) = qfac*(2*b*d+2*a*c);
                tmpdirection(1,2) = qfac*(2*c*d-2*a*b);
                tmpdirection(2,2) = qfac*(a*a+d*d-c*c-b*b);
                tmpdirection(2,1) = 2*c*d+2*a*b;
                tmpdirection(2,0) = 2*b*d-2*a*c;
            }

            oimage->setDirection(tmpdirection, true);
        } else if(sform_code > 0) {
            /* use the sform, since no qform exists */

            // origin, last column
            double di = 0, dj = 0, dk = 0;
            for(size_t ii=0; ii<3 && ii<oimage->ndim(); ii++) {
                di += pow(saffine[4*ii+0],2); //column 0
                dj += pow(saffine[4*ii+1],2); //column 1
                dk += pow(saffine[4*ii+2],2); //column 2
                oimage->origin(ii) = saffine[4*ii+3]; //column 3
            }
            di = sqrt(di);
            dj = sqrt(dj);
            dk = sqrt(dk);

            // set direction and spacing
            oimage->spacing(0) = sqrt(di);
            oimage->direction(0,0) = saffine[4*0+0]/di;

            if(oimage->ndim() > 1) {
                oimage->spacing(1) = sqrt(dj);
                oimage->direction(0,1) = saffine[4*0+1]/dj;
                oimage->direction(1,1) = saffine[4*1+1]/dj;
                oimage->direction(1,0) = saffine[4*1+0]/di;
            }
            if(oimage->ndim() > 2) {
                oimage->spacing(2) = sqrt(dk);
                oimage->direction(0,2) = saffine[4*0+2]/dk;
                oimage->direction(1,2) = saffine[4*1+2]/dk;
                oimage->direction(2,2) = saffine[4*2+2]/dk;
                oimage->direction(2,1) = saffine[4*2+1]/dj;
                oimage->direction(2,0) = saffine[4*2+0]/di;
            }

        } else {
            // only spacing changes
            for(size_t ii=0; ii<dim.size(); ii++)
                oimage->spacing(ii) = pixdim[ii];
        }

        /**************************************************************************
         * Medical Imaging Varaibles Variables
         **************************************************************************/

        // direct copies
        oimage->m_freqdim = freqdim;
        oimage->m_phasedim = phasedim;
        oimage->m_slicedim = slicedim;

        // slice timing
        oimage->updateSliceTiming(slice_duration,  slice_start, slice_end,
                (SliceOrderT)slice_code);


        if(verbose) 
            std::cerr << *oimage  << std::endl;

    }

    if(!nopixeldata) {
        // copy pixels
        switch(datatype) {
            // 8 bit
            case INT8:
                readPixels<int8_t>(out, file, start, psize, doswap);
                break;
            case UINT8:
                readPixels<uint8_t>(out, file, start, psize, doswap);
                break;
                // 16  bit
            case INT16:
                readPixels<int16_t>(out, file, start, psize, doswap);
                break;
            case UINT16:
                readPixels<uint16_t>(out, file, start, psize, doswap);
                break;
                // 32 bit
            case INT32:
                readPixels<int32_t>(out, file, start, psize, doswap);
                break;
            case UINT32:
                readPixels<uint32_t>(out, file, start, psize, doswap);
                break;
                // 64 bit int
            case INT64:
                readPixels<int64_t>(out, file, start, psize, doswap);
                break;
            case UINT64:
                readPixels<uint64_t>(out, file, start, psize, doswap);
                break;
                // floats
            case FLOAT32:
                readPixels<float>(out, file, start, psize, doswap);
                break;
            case FLOAT64:
                readPixels<double>(out, file, start, psize, doswap);
                break;
            case FLOAT128:
                readPixels<long double>(out, file, start, psize, doswap);
                break;
                // RGB
            case RGB24:
                readPixels<rgb_t>(out, file, start, psize, doswap);
                break;
            case RGBA32:
                readPixels<rgba_t>(out, file, start, psize, doswap);
                break;
            case COMPLEX256:
                readPixels<cquad_t>(out, file, start, psize, doswap);
                break;
            case COMPLEX128:
                readPixels<cdouble_t>(out, file, start, psize, doswap);
                break;
            case COMPLEX64:
                readPixels<cfloat_t>(out, file, start, psize, doswap);
                break;
            default:
            case UNKNOWN_TYPE:
                throw RUNTIME_ERROR("Unknown Pixel Type in input Nifti Image");
        }
    }
	return out;
}


/****************************************************************************
 * Read JSON Image and helper functions
 ****************************************************************************/

/**
 * @brief Runs until stop character is found, end of file, error, or
 * if a character does return true from stop/ignore/keep.
 *
 * @param file Input file
 * @param oss stream to write to
 * @param keeplast whether to keep the character that returns true for stop
 * @param stop stops if this returns true
 * @param ignore neither fails nor writes to oss if this returns true
 * @param keep writes character to oss if this returns true
 *
 * @return -2: error, -1 unexpected character, 1: eof, 0: OK
 */
int read(gzFile file, stringstream& oss, bool keeplast,
        std::function<bool(char)> stop,
        std::function<bool(char)> ignore,
        std::function<bool(char)> keep)
{
    int c;
    while((c = gzgetc(file)) >= 0) {
        if(stop(c)) {
            if(keeplast)
                oss << (char)c;
            return 0;
        } else if(ignore(c)) {
            continue;
        } else if(keep(c)) {
            oss << (char)c;
        } else {
            return -1;
        }
    }

    if(gzeof(file))
        return 1;
    else
        return -2;
}

/**
 * @brief Reads a string from a json file
 *
 * @param file
 * @param out
 *
 * @return 
 */
int readstring(gzFile file, std::string& out)
{
    // read key, find "
    stringstream oss;
    int ret = read(file, oss, false, 
            [&](char c){return c=='"';},
            [&](char c){return (c==' '||c=='\r'||c=='\n'||c=='\t');},
            [&](char c){(void)c; return false;});

    if(ret != 0) {
        return -1;
    }

    assert(oss.str() == "");
    // find closing "
    bool backslash = false;
    ret = read(file, oss, false, 
            [&](char c){return !backslash && c=='"';},
            [&](char c){return (c==' '||c=='\r'||c=='\n'||c=='\t');},
            [&](char c){backslash = (c=='\\'); return true;});

    if(ret != 0) {
        return -1;
    }

    out = oss.str();

    return 0;
}

/**
 * @brief Reads a "blah" : OR } 
 *
 * @param file File to read from
 * @param key either a key or ""
 *
 * @return 0 if key found, -1 if error occurred
 */
int readKey(gzFile file, string& key)
{
    // read key, find "
    stringstream oss;
    int ret = read(file, oss, false, 
            [&](char c){return c=='"';},
            [&](char c){return (c==' '||c=='\r'||c=='\n'||c=='\t');},
            [&](char c){(void)c; return false;});

    if(ret != 0) {
        return -1;
    }

    // find closing "
    bool backslash = false;
    ret = read(file, oss, false, 
            [&](char c){return !backslash && c=='"';},
            [&](char c){return (c==' '||c=='\r'||c=='\n'||c=='\t');},
            [&](char c){backslash = (c=='\\'); return true;});

    if(ret != 0) {
        return -1;
    }
    key = oss.str();
        
    // find colon
    ret = read(file, oss, false, 
            [&](char c){return c==':';},
            [&](char c){return (c==' '||c=='\r'||c=='\n'||c=='\t');},
            [&](char c){(void)c; return false;});
    if(ret != 0) {
        cerr << "Could not find : for key: " << oss.str() << endl;
        return -1;
    }

    return 0;
}

bool isspace(char c)
{
    return (c==' '||c=='\r'||c=='\n'||c=='\v'||c=='\f'||c=='\t');
}

bool isnumeric(char c)
{
    return isdigit(c) || c=='.' || c=='-' || c=='e' || c=='E' || c==',';
}

/**
 * @brief Reads an array of numbers from a json file
 *
 * @tparam T
 * @param file
 * @param oarray
 *
 * @return 
 */
template <typename T>
int readNumArray(gzFile file, vector<T>& oarray)
{
    stringstream ss;
    // find [
    int ret = read(file, ss, false, 
            [&](char c){return c=='[';},
            [&](char c){return (c==' '||c=='\r'||c=='\n'||c=='\t');},
            [&](char c){(void)c; return false;});

    if(ret != 0) {
        return -1;
    }

    // iterate through and try to find the end bracket that matches the
    // initial. We also ignore any middle [] and save all the spaces/numbers
    assert(ss.str() == "");
    int stack = 1;
    // find closing ]
    ret = read(file, ss, false, 
            [&](char c)
            {
                stack += (c=='['); 
                stack -= (c==']'); 
                return stack==0;
            }, 
            [](char c){return c=='[' || c==']';},
            [](char c){return isnumeric(c) || isspace(c);});

    if(ret != 0) 
        return -1;

#ifdef DEBUG
    cerr << "Array String:\n" << ss.str() << endl;
#endif 

    // now that we have the character, break them up
    string token;
    istringstream iss;
    T val;
    while(ss.good()) {
        std::getline(ss, token, ',');
        iss.clear();
        iss.str(token);
        iss >> val;

        if(iss.bad()) {
            cerr << "Invalid type foudn in string before: "<< ss.str() <<endl;
            return -1;
        }
        oarray.push_back(val);       
    }

    if(ss.bad()) {
        cerr << "Array ending not found!" << endl;
        return -1;
    }
    return 0;
}

/**
 * @brief Reads an MRI image. Right now only nift images are supported. later
 * on, it will try to load image using different reader functions until one
 * suceeds.
 *
 * @param file File to read from
 * @param verbose Whether to print out information as the file is read
 * @param makearray Whether to make an array, as opposed to an MRImage
 *
 * @return Loaded image
 */
shared_ptr<NDArray> readJSONImage(gzFile file, bool verbose, bool makearray) 
{
    // read to opening brace
    stringstream oss;
    int ret = read(file, oss, false, 
            [](char c){return c=='{';},
            [](char c){return (c==' '||c=='\r'||c=='\n'||c=='\t');},
            [](char c){(void)c; return false;});

    if(ret != 0) {
        cerr << "Expected Opening { but did not find one" << endl;
    }

    PixelT type = UNKNOWN_TYPE;
    vector<double> values;
    vector<double> spacing;
    vector<double> origin;
    vector<double> direction;
    vector<size_t> size;

    while(true) {
        string key;
        ret = readKey(file, key);

        if(ret < 0) {
            cerr << "Looking for key, but couldn't find one!" << endl;
            return NULL;
        }

        if(verbose)
            cerr << "Parsing key:" << key << endl;
        if(key == "type") {
            // read a string
            string value;
            ret = readstring(file, value);
            if(ret != 0) {
                cerr << "Expected string for key: " << key << 
                    " but could not parse" << endl;
                return NULL;
            }

            type = stringToPixelT(value);
            if(type == UNKNOWN_TYPE)
                return NULL;

        } else if(key == "size") { 
            ret = readNumArray<size_t>(file, size);
            if(ret != 0)  {
                cerr << "Expected array of non-negative integers for size!" <<
                    endl;
                return NULL;
            }
        } else if(key == "values") {
            ret = readNumArray(file, values);
            if(ret != 0)  {
                cerr << "Expected array of floats for values!" <<
                    endl;
                return NULL;
            }
        } else if(key == "spacing") {
            ret = readNumArray(file, spacing);
            if(ret != 0)  {
                cerr << "Expected array of floats for spacing!" <<
                    endl;
                return NULL;
            }
        } else if(key == "direction") {
            ret = readNumArray(file, direction);
            if(ret != 0)  {
                cerr << "Expected array of floats for direction!" <<
                    endl;
                return NULL;
            }
        } else if(key == "origin") {
            ret = readNumArray(file, origin);
            if(ret != 0)  {
                cerr << "Expected array of floats for origin!" <<
                    endl;
                return NULL;
            }
        } else if(key == "version" || key == "comment") {
            string value;
            ret = readstring(file, value);
        } else {
            cerr << "Error, Unknown key:" << key << endl;
            return NULL;
        }

        // should find a comma or closing brace
        oss.str("");
        ret = read(file, oss, true, 
                [](char c){return c==',' || c=='}';},
                [](char c){return isspace(c);},
                [](char c){(void)c; return false;});

        if(ret != 0) {
            cerr << "After a Key:Value Pair there should be either a } or ," 
                << endl;
            return NULL;
        }
        if(oss.str()[0] == '}')
            break;
    }

    size_t ndim = size.size();
    if(ndim == 0) {
        cerr << "No \"size\" tag found!" << endl; 
        return NULL;
    }
    if(type == UNKNOWN_TYPE) {
        cerr << "No type, or unknown type specified!" << endl; 
        return NULL;
    }

    ptr<NDArray> out;
    if(makearray) {
        out = createNDArray(size.size(), size.data(), type);
    } else {
        out = createMRImage(size.size(), size.data(), type);
        auto oimg= dPtrCast<MRImage>(out);

        // copy spacing
        if(spacing.size() > 0) {
            if(spacing.size() != ndim) {
                cerr << "Incorrect number of spacing values (" << spacing.size() 
                    << " vs " << ndim << ") given" << endl;
                return NULL;
            } else {
                for(size_t ii=0; ii<ndim; ii++)
                    oimg->spacing(ii) = spacing[ii];
            }
        }

        // copy origin  
        if(origin.size() > 0) {
            if(origin.size() != ndim) {
                cerr << "Incorrect number of origin values (" << origin.size() 
                    << " vs " << ndim << ") given" << endl;
                return NULL;
            } else {
                for(size_t ii=0; ii<ndim; ii++)
                    oimg->origin(ii) = origin[ii];
            }
        }

        // copy direction
        if(direction.size() > 0) {
            if(direction.size() != ndim*ndim) {
                cerr << "Incorrect number of origin values (" << direction.size() 
                    << " vs " << ndim << ") given" << endl;
                return NULL;
            }
            for(size_t ii=0; ii<ndim; ii++) {
                for(size_t jj=0; jj<ndim; jj++) {
                    oimg->direction(ii,jj) = direction[ii*ndim+jj];
                }
            }
        }
    }

    // copy values
    if(values.size() != out->elements()) {
        throw RUNTIME_ERROR("Incorrect number of values ("+
                to_string(values.size())+" vs "+to_string(ndim)+") given");
    }
    size_t ii=0;
    for(NDIter<float> it(out); !it.eof(); ++it, ++ii) 
        it.set(values[ii]);
    
    return out;
}


/*********************************************************
 * High Level Functions
 *********************************************************/

/**
 * @brief Reads an MRI image. Right now only nift images are supported. later
 * on, it will try to load image using different reader functions until one
 * suceeds.
 *
 * @param fn Name of input file to read
 * @param verbose Whether to print out information as the file is read
 * @param nopixeldata Don't actually read the pixel data, just the header and
 * create the image. So if you want to copy an image's orientation and
 * structure, this would be the way to do it without wasting time actually
 * reading.
 *
 * @return Loaded image
 */
ptr<MRImage> readMRImage(std::string fn, bool verbose, bool nopixeldata)
{
#if ZLIB_VERNUM >= 0x1280
	const size_t BSIZE = 1024*1024; //1M
#endif
	auto gz = gzopen(fn.c_str(), "rb");

	if(!gz) {
		throw std::ios_base::failure("Could not open " + fn + " for reading");
		return NULL;
	}
#if ZLIB_VERNUM >= 0x1280
	gzbuffer(gz, BSIZE);
#endif
	
	ptr<NDArray> out;
	
    // remove .gz to find the "real" format,
	if(fn.size()> 3 && fn.substr(fn.size()-3, 3) == ".gz") {
		fn = fn.substr(0, fn.size()-3);
	}
	
	if(fn.size() > 4 && fn.substr(fn.size()-4, 4) == ".nii") {
        //////////////////////////
        // Read Nifti Data
        //////////////////////////
        if((out = readNiftiImage(gz, verbose, false, nopixeldata))) {
            gzclose(gz);
            return dPtrCast<MRImage>(out);
        }
    } else if(fn.size() > 5 && fn.substr(fn.size()-5, 5) == ".json") {
        //////////////////////////
        // Read JSON data
        //////////////////////////
        if((out = readJSONImage(gz, verbose, false))) {
            gzclose(gz);
            return dPtrCast<MRImage>(out);
        }
	} else {
		std::cerr << "Unknown filetype: " << fn << std::endl;
		gzclose(gz);
        throw std::ios_base::failure("Error reading " + fn );
	}

    throw std::ios_base::failure("Error reading " + fn );
	return NULL;
}

/**
 * @brief Reads an array. Can read nifti's but orientation won't be read.
 *
 * @param fn Name of input file to read
 * @param verbose Whether to print out information as the file is read
 * @param nopixeldata Don't actually read the pixel data, just the header and
 * create the image. So if you want to copy an image's orientation and
 * structure, this would be the way to do it without wasting time actually
 * reading.
 *
 * @return Loaded image
 */
ptr<NDArray> readNDArray(std::string fn, bool verbose, bool nopixeldata)
{
#if ZLIB_VERNUM >= 0x1280
	const size_t BSIZE = 1024*1024; //1M
#endif
	auto gz = gzopen(fn.c_str(), "rb");

	if(!gz) {
		throw std::ios_base::failure("Could not open " + fn + " for readin");
		return NULL;
	}
#if ZLIB_VERNUM >= 0x1280
	gzbuffer(gz, BSIZE);
#endif
	
	ptr<NDArray> out;
	
    // remove .gz to find the "real" format,
	if(fn.size() > 3 && fn.substr(fn.size()-3, 3) == ".gz") {
		fn = fn.substr(0, fn.size()-3);
	}
	
	if(fn.size() > 4 && fn.substr(fn.size()-4, 4) == ".nii") {
        if((out = readNiftiImage(gz, verbose, true, nopixeldata))) {
            gzclose(gz);
            return out;
        }
    } else if(fn.size() > 5 && fn.substr(fn.size()-5, 5) == ".json") {
        //////////////////////////
        // Read JSON data
        //////////////////////////
        if((out = readJSONImage(gz, verbose, true))) {
            gzclose(gz);
            return out;
        }
	} else {
		std::cerr << "Unknown filetype: " << fn << std::endl;
		gzclose(gz);
        throw std::ios_base::failure("Error reading " + fn );
	}

    throw std::ios_base::failure("Error reading " + fn);
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
int writeMRImage(ptr<const MRImage> img, std::string fn, bool nifti2)
{
	if(!img)
		return -1;
	double version = 1;
	if(nifti2)
		version = 2;
	return img->write(fn, version);
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
int writeNDArray(ptr<const NDArray> img, std::string fn, bool nifti2)
{
	if(!img)
		return -1;
	double version = 1;
	if(nifti2)
		version = 2;
	return img->write(fn, version);
}

} // NPL

