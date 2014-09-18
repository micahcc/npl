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
 * @file mrimage_utils.cpp Utilities for operating on MR Images. These are
 * functions which are sensitive to MR variables, such as spacing, orientation,
 * slice timing etc. ndarray_utils contains more general purpose tools.
 *
 *****************************************************************************/

#include "mrimage.h"
#include "iterators.h"
#include "accessors.h"
#include "ndarray_utils.h"
#include "mrimage_utils.h"
#include "byteswap.h"
#include "macros.h"

#include "fftw3.h"

#include <string>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <memory>
#include <cstring>

namespace npl {

using std::vector;

#define VERYDEBUG

#ifdef VERYDEBUG
#define DEBUGWRITE(FOO) FOO 
#else
#define DEBUGWRITE(FOO) 
#endif

/**
 * @brief Writes a pair of images, one real, one imaginary or if absPhase is
 * set to true then an absolute image and a phase image. 
 *
 * @param basename Base filename _abs.nii.gz and _phase.nii.gz or _re.nii.gz
 * and _im.nii.gz will be appended, depending on absPhase
 * @param in Input image
 * @param absPhase Whether the break up into absolute and phase rather than
 * re/imaginary
 */
void writeComplex(std::string basename, ptr<const MRImage> in, 
        bool absPhase)
{
    auto img1 = dPtrCast<MRImage>(in->copyCast(FLOAT64));
    auto img2 = dPtrCast<MRImage>(in->copyCast(FLOAT64));

    OrderIter<double> it1(img1);
    OrderIter<double> it2(img2);
    OrderConstIter<cdouble_t> init(in);
    for(; !init.eof(); ++init, ++it1, ++it2) {
        if(absPhase) {
            it1.set(abs(*init));
            it2.set(arg(*init));
        } else {
            it1.set((*init).real());
            it2.set((*init).imag());
        }
    }

    if(absPhase) {
        img1->write(basename + "_abs.nii.gz");
        img2->write(basename + "_ang.nii.gz");
    } else {
        img1->write(basename + "_re.nii.gz");
        img2->write(basename + "_im.nii.gz");
    }
}

/**
 * @brief Performs forward FFT transform in N dimensions.
 *
 * @param in Input image
 * @param in_osize Size of output image (will be padded up to this prior to
 * FFT)
 *
 * @return Frequency domain of input. Note the output will be
 * COMPLEX128/CDOUBLE type
 */
ptr<MRImage> fft_forward(ptr<const MRImage> in, 
        const std::vector<size_t>& in_osize)
{

    // make sure osize matches input dimensions
    vector<size_t> osize(in_osize);
    osize.resize(in->ndim(), 1);
    size_t ndim = osize.size();

    // create padded NDArray, allocated with fftw
    size_t opixels = 1;
    vector<int> osize32(ndim);
    for(size_t ii=0; ii<ndim; ii++) {
        opixels *= osize[ii];
        osize32[ii] = osize[ii];

        if(osize[ii] < in->dim(ii)) 
            throw std::invalid_argument("Input image larger than output size!"
                    " In\n" + __FUNCTION_STR__);
    }

    auto outbuff = fftw_alloc_complex(opixels);
    auto output = createMRImage(osize.size(), osize.data(), CDOUBLE,
            outbuff, [](void* ptr) {fftw_free(ptr);});
    output->copyMetadata(in);
    
    // create ND FFTW Plan
    auto fwd = fftw_plan_dft((int)ndim, osize32.data(), outbuff, outbuff, 
            FFTW_FORWARD, FFTW_MEASURE);
    for(size_t ii=0; ii<opixels; ii++) {
        outbuff[ii][0] = 0;
        outbuff[ii][1] = 0;
    }
    
    // fill padded from input
    OrderConstIter<cdouble_t> iit(in);
    OrderIter<cdouble_t> pit(output);
    pit.setROI(ndim, in->dim());
    pit.setOrder(iit.getOrder());
    for(iit.goBegin(), pit.goBegin(); !iit.eof() && !pit.eof(); ++pit, ++iit) 
        pit.set(*iit);
    assert(iit.eof() && pit.eof());

    DEBUGWRITE(writeComplex("forward_prefft", output));

    // fourier transform
    fftw_execute(fwd);

#ifndef NDEBUG
    OrderIter<cdouble_t> it(output);;
    for(size_t ii=0; !it.eof(); ii++, ++it) {
        cdouble_t tmp(*it);
        assert(tmp.real() == outbuff[ii][0]);
        assert(tmp.imag() == outbuff[ii][1]);
    }
#endif 

    // normalize
    double normf = 1./opixels;
    for(size_t ii=0; ii<opixels; ii++) {
        outbuff[ii][0] = normf*outbuff[ii][0];
        outbuff[ii][1] = normf*outbuff[ii][1];
    }

    DEBUGWRITE(writeComplex("forward_postfft", output));

    return output;
}

/**
 * @brief Performs inverse FFT transform in N dimensions.
 *
 * @param in Input image
 * @param in_osize Size of output image. If this is smaller than the input then
 * the frequency domain will be trunkated, if it is larger then the fourier
 * domain will be padded ( output upsampled )
 *
 * @return Frequency domain of input. Note the output will be
 * COMPLEX128/CDOUBLE type
 */
ptr<MRImage> fft_backward(ptr<const MRImage> in,
        const std::vector<size_t>& in_osize)
{

    // make sure osize matches input dimensions
    vector<size_t> osize(in_osize);
    osize.resize(in->ndim(), 1);
    size_t ndim = osize.size();

    // create output NDArray, allocated with fftw
    size_t opixels = 1;
    vector<int> osize32(ndim);
    for(size_t ii=0; ii<ndim; ii++) {
        opixels *= osize[ii];
        osize32[ii] = osize[ii];
    }

    auto outbuff = fftw_alloc_complex(opixels);
    auto output = createMRImage(osize.size(), osize.data(), CDOUBLE,
            outbuff, [](void* ptr) {fftw_free(ptr);});
    output->copyMetadata(in);
    
    // create ND FFTW Plan
    auto plan = fftw_plan_dft((int)ndim, osize32.data(), outbuff, outbuff, 
            FFTW_BACKWARD, FFTW_MEASURE);
    for(size_t ii=0; ii<opixels; ii++) {
        outbuff[ii][0] = 0;
        outbuff[ii][1] = 0;
    }

    // fill padded from input
    NDConstAccess<cdouble_t> iacc(in);
    OrderIter<cdouble_t> it(output);
    vector<int64_t> iindex(ndim);
    vector<int64_t> oindex(ndim);
    for(it.goBegin(); !it.eof(); ++it) {
        it.index(oindex);

        // if the curent oindex doesn't exist in the input (due to output size
        // being larger than input size), then leave as 0
        bool skip = false;

        // compute input index, handling frequency unrwrapping
        int64_t ilen, olen;
        for(size_t dd=0; dd<ndim; dd++) {
            ilen = in->dim(dd);
            olen = output->dim(dd);

            if(oindex[dd] < olen/2) {
                iindex[dd] = oindex[dd];
                if(iindex[dd] >= ilen/2) {
                    skip = true;
                    break;
                }
            } else  {
                // negative frequencies
                iindex[dd] = ilen-(olen-oindex[dd]); 
                if(iindex[dd] < ilen/2) {
                    skip = true;
                    break;
                }
            }

        }

        if(skip)
            continue;

        it.set(iacc[iindex]);
    }
    
    // fourier transform
    fftw_execute(plan);

    return output;
}

/**
 * @brief Performs smoothing in each dimension, then downsamples so that pixel
 * spacing is roughly equal to FWHM.
 *
 * /todo less memory allocation, reuse fftw_alloc data
 *
 * @param in    Input image
 * @param sigma Standard deviation for smoothing
 *
 * @return  Smoothed and downsampled image
 */
ptr<MRImage> smoothDownsample(ptr<const MRImage> in, double sigma)
{

    size_t ndim = in->ndim();

    // convert mm to indices
    DBG3(cerr << "StdDev: " << sigma << "\n, Smoothing in Index Space:\n");
    vector<double> sd(ndim, sigma);
    for(size_t ii=0; ii<ndim; ii++) {
        sd[ii] /= in->spacing(ii);
        DBG3(cerr << ii << ": FWHM: " << sd_to_fwhm(sd[ii]) << " SD: " 
            << sd[ii] << endl);
    }


    // create downsampled image
    vector<size_t> isize(in->dim(), in->dim()+ndim);
    vector<size_t> psize(ndim);
    vector<size_t> dsize(ndim);
    vector<size_t> osize(ndim);
    size_t linelen = 0;
    for(size_t dd=0; dd<ndim; dd++) {
        // compute ratio
        double ratio;
        if(sd_to_fwhm(sd[dd]) < 2)
            ratio = 1;
        else
            ratio = 2/sd_to_fwhm(sd[dd]);

        psize[dd] = round2(isize[dd]*2);
        dsize[dd] = round2(psize[dd]*ratio);
        double pratio = ((double)psize[dd])/((double)isize[dd]);
        osize[dd] = round(dsize[dd]/pratio);
        
        if(psize[dd] > linelen)
            linelen = psize[dd];

        assert(psize[dd] >= dsize[dd]);
        assert(osize[dd] <= isize[dd]);
    }

    vector<size_t> roi(in->dim(), in->dim()+ndim);
    auto working = dPtrCast<MRImage>(in->copy());
    auto buffer1 = fftw_alloc_complex(linelen*2);
    auto buffer2 = &buffer1[linelen];
    for(size_t dd=0; dd<ndim; dd++) {
        auto fwd = fftw_plan_dft_1d((int)psize[dd], buffer1, buffer2,
                FFTW_FORWARD, FFTW_MEASURE);
        auto bwd = fftw_plan_dft_1d((int)dsize[dd], buffer1, buffer2,
                FFTW_BACKWARD, FFTW_MEASURE);

        // extract line
        ChunkIter<cdouble_t> it(working);
        it.setROI(roi.size(), roi.data());
        it.setLineChunk(dd);
        for(it.goBegin(); !it.eof(); it.nextChunk()) {
            int64_t ii=0;
            for(it.goChunkBegin(), ii=0; !it.eoc(); ++it, ++ii) {
                buffer1[ii][0] = (*it).real();
                buffer1[ii][1] = (*it).imag();
            }
            for(; ii<psize[dd]; ii++){
                buffer1[ii][0] = 0;
                buffer1[ii][1] = 0;
            }

            // fourier tansform line
            fftw_execute(fwd);

            double normf = 1./sqrt(psize[dd]*dsize[dd]);
            // positive frequencies
            for(ii=0; ii<dsize[dd]/2; ii++) {
                double ff = ii/(double)psize[dd];
                double w = exp(-M_PI*M_PI*ff*ff*2*sd[dd]*sd[dd]);
                buffer1[ii][0] = buffer2[ii][0]*w*normf;
                buffer1[ii][1] = buffer2[ii][1]*w*normf;
            }

            // negative frequencies
            for(ii=dsize[dd]/2; ii<dsize[dd]; ii++) {
                int64_t jj = psize[dd]-(dsize[dd]-ii);
                double ff = -(dsize[dd]-ii)/(double)psize[dd];
                double w = exp(-M_PI*M_PI*ff*ff*2*sd[dd]*sd[dd]);
                buffer1[ii][0] = buffer2[jj][0]*w*normf;
                buffer1[ii][1] = buffer2[jj][1]*w*normf;
            }

            // inverse fourier tansform
            fftw_execute(bwd);

            // write out (and zero extra area)
            for(it.goChunkBegin(), ii=0; ii<dsize[dd]; ++it, ++ii) {
                cdouble_t tmp(buffer2[ii][0], buffer2[ii][1]);
                it.set(tmp);
            }
            for(; !it.eoc(); ++it){
                cdouble_t tmp(0, 0);
                it.set(tmp);
            }
        }

        // update ROI
        roi[dd] = osize[dd];
        DBG3(cerr << isize[dd] << "->" << osize[dd] << endl);
    }

    // copy roi into output
    auto out = dPtrCast<MRImage>(working->copyCast(osize.size(), osize.data()));

    // set spacing
    for(size_t dd=0; dd<in->ndim(); dd++) 
        out->spacing(dd) *= ((double)psize[dd])/((double)dsize[dd]);

    fftw_free(buffer1);
    return out;
}

/**
 * @brief Smooths an image in 1 dimension
 *
 * @param inout Input/output image to smooth
 * @param dim dimensions to smooth in. If you are smoothing individual volumes
 * of an fMRI you would provide dim={0,1,2}
 * @param stddev standard deviation in physical units index*spacing
 *
 */
void gaussianSmooth1D(ptr<MRImage> inout, size_t dim,
		double stddev)
{
    const auto gaussKern = [](double x) 
    {
        const double PI = acos(-1);
        const double den = 1./sqrt(2*PI);
        return den*exp(-x*x/(2));
    };

	if(dim >= inout->ndim()) {
		throw std::out_of_range("Invalid dimension specified for 1D gaussian "
				"smoothing");
	}

	std::vector<int64_t> index(dim, 0);
	stddev /= inout->spacing(dim);
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

/*******************************************************************
 * Input/Output Functions
 ******************************************************************/

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
ptr<MRImage> readMRImage(std::string filename, bool verbose)
{
	const size_t BSIZE = 1024*1024; //1M
	auto gz = gzopen(filename.c_str(), "rb");

	if(!gz) {
		throw std::ios_base::failure("Could not open " + filename + " for readin");
		return NULL;
	}
	gzbuffer(gz, BSIZE);
	
	ptr<MRImage> out;
	
    // remove .gz to find the "real" format,
	if(filename.substr(filename.size()-3, 3) == ".gz") {
		filename = filename.substr(0, filename.size()-3);
	}
	
	if(filename.substr(filename.size()-4, 4) == ".nii") {
        if((out = readNiftiImage(gz, verbose))) {
            gzclose(gz);
            return out;
        }
    } else if(filename.substr(filename.size()-5, 5) == ".json") {
        if((out = readJSONImage(gz))) {
            gzclose(gz);
            return out;
        }
	} else {
		std::cerr << "Unknown filetype: " << filename.substr(filename.rfind('.'))
			<< std::endl;
		gzclose(gz);
        throw std::ios_base::failure("Error reading " + filename );
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
			cerr << std::setw(10) << std::setprecision(3) << img.direction(ii,jj);
		}
		cerr << "] " << endl;
	}
	
	out << "Spacing: " << endl;
	for(size_t ii=0; ii<img.ndim(); ii++) {
		out << "[ " << std::setw(10) << std::setprecision(3)
			<< img.spacing(ii) << "] ";
	}
	out << endl;

	out << "Origin: " << endl;
	for(size_t ii=0; ii<img.ndim(); ii++) {
		out << "[ " << std::setw(10) << std::setprecision(3)
			<< img.origin(ii) << "] ";
	}
	out << endl;

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

//
///**
// * @brief Uses fourier shift theorem to shift an image
// *
// * @param in Input image to shift
// * @param len length of dx array
// * @param dx movement in physical coordinates
// *
// * @return shifted image
// */
//ptr<MRImage> shiftImageFFT(ptr<MRImage> in, size_t len, double* dx)
//{
//
//	auto out = dPtrCast<MRImage>(in->copy());
//	std::vector<double> shift(len);
//	in->disOrientVector(len, dx, shift.data());
//
//	// for each dimension
//	for(size_t ii=0; ii<len && ii<in->ndim(); ii++) {
//		shiftImageFFT(out, ii, dx[ii]);
//	}
//
//	return out;
//}


} // npl


