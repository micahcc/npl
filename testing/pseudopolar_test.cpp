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
 * @file fft_rotate_test.cpp A test of automatica computation of the rotation 
 * axis, based on the pseudo-polar fourier transform
 *
 *****************************************************************************/

#include <version.h>
#include <string>
#include <stdexcept>

#include <Eigen/Geometry> 

#define DEBUG 1

#include "mrimage.h"
#include "ndarray_utils.h"
#include "iterators.h"
#include "accessors.h"
#include "basic_functions.h"
#include "basic_plot.h"
#include "chirpz.h"

#include "fftw3.h"

using namespace npl;
using namespace std;
using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::AngleAxisd;

void writeComplexAA(string basename, shared_ptr<const MRImage> in)
{
	auto absimg = dynamic_pointer_cast<MRImage>(in->copyCast(FLOAT64));
	auto angimg = dynamic_pointer_cast<MRImage>(in->copyCast(FLOAT64));

	OrderIter<double> rit(absimg);
	OrderIter<double> iit(angimg);
	OrderConstIter<cdouble_t> init(in);
	while(!init.eof()) {
		rit.set(abs(*init));
		iit.set(arg(*init));
		++init;
		++rit;
		++iit;
	}

	absimg->write(basename + "_abs.nii.gz");
	angimg->write(basename + "_ang.nii.gz");
}
void writeComplex(string basename, shared_ptr<const MRImage> in)
{
	auto re = dynamic_pointer_cast<MRImage>(in->copyCast(FLOAT64));
	auto im = dynamic_pointer_cast<MRImage>(in->copyCast(FLOAT64));

	OrderIter<double> rit(re);
	OrderIter<double> iit(im);
	OrderConstIter<cdouble_t> init(in);
	while(!init.eof()) {
		iit.set((*init).imag());
		rit.set((*init).real());
		++init;
		++rit;
		++iit;
	}

	re->write(basename + "_re.nii.gz");
	im->write(basename + "_im.nii.gz");
}

/**
 * @brief Performs a rotation of the image first by rotating around z, then
 * around y, then around x.
 *
 * @param rx Rotation around x, radians
 * @param ry Rotation around y, radians
 * @param rz Rotation around z, radians
 * @param in Input image
 *
 * @return 
 */
shared_ptr<MRImage> bruteForceRotate(Vector3d axis, double theta,
		shared_ptr<const MRImage> in)
{
	Matrix3d m;
	// negate because we are starting from the destination and mapping from
	// the source
	m = AngleAxisd(-theta, axis);
	LinInterp3DView<double> lin(in);
	auto out = dynamic_pointer_cast<MRImage>(in->copy());
	Vector3d ind;
	Vector3d cind;
	Vector3d center;
	for(size_t ii=0; ii<3 && ii<in->ndim(); ii++) {
		center[ii] = (in->dim(ii)-1)/2.;
	}

	for(Vector3DIter<double> it(out); !it.isEnd(); ++it) {
		it.index(3, ind.array().data());
		cind = m*(ind-center)+center;

		// set for each t
		for(size_t tt = 0; tt<in->tlen(); tt++) 
			it.set(tt, lin(cind[0], cind[1], cind[2], tt));
	}

	return out;
}

int closeCompare(shared_ptr<const MRImage> a, shared_ptr<const MRImage> b)
{
	if(a->ndim() != b->ndim()) {
		cerr << "Error image dimensionality differs" << endl;
		return -1;
	}
	
	for(size_t dd=0; dd<a->ndim(); dd++) {
		if(a->dim(dd) != b->dim(dd)) {
			cerr << "Image size in the " << dd << " direction differs" << endl;
			return -1;
		}
	}

	double corr = 0;
	double sum1 = 0;
	double sum2 = 0;
	double sumsq1 = 0;
	double sumsq2 = 0;
	size_t count = 0;
	OrderConstIter<double> ita(a);
	OrderConstIter<double> itb(b);
	itb.setOrder(ita.getOrder());
	for(ita.goBegin(), itb.goBegin(); !ita.eof() && !itb.eof(); ++ita, ++itb) {
		corr += abs(*ita)*abs(*itb);
		sum1 += abs(*ita);
		sum2 += abs(*itb);
		sumsq1 += abs(*ita)*abs(*ita);
		sumsq2 += abs(*itb)*abs(*itb);
		count++;
	}

	corr = sample_corr(count, sum1, sum2, sumsq1, sumsq2, corr);
	if(corr < .98) {
		return 1;
	}

	return 0;
}

// upsample in anglular directions
shared_ptr<MRImage> padFFT(shared_ptr<const MRImage> in, double* upsamp)
{
	if(in->ndim() != 3) {
		throw std::invalid_argument("Error, input image should be 3D!");
	}

	std::vector<size_t> osize(3, 0);
	for(size_t ii=0; ii<3; ii++) {
		osize[ii] = round2(in->dim(ii)*upsamp[ii]);
	}

#ifdef DEBUG
	writeComplex("prepadded", in);
#endif //DEBUG
	auto oimg = dynamic_pointer_cast<MRImage>(in->copyCast(osize.size(),
			osize.data(), COMPLEX128));
	
	// copy data
	std::vector<int64_t> index(in->ndim());
#ifdef DEBUG
	writeComplex("padded", oimg);
#endif //DEBUG

	// fourier transform
	for(size_t dd = 0; dd < 3; dd++) {
		auto buffer = fftw_alloc_complex((int)osize[dd]);
		fftw_plan fwd = fftw_plan_dft_1d((int)osize[dd], buffer, buffer, 
				FFTW_FORWARD, FFTW_MEASURE);
		
		ChunkIter<cdouble_t> it(oimg);
		it.setLineChunk(dd);
		for(it.goBegin(); !it.eof() ; it.nextChunk()) {
			it.index(index.size(), index.data());

			// fill from line
			for(size_t tt=0; !it.eoc(); ++it, tt++) {
				buffer[tt][0] = (*it).real();
				buffer[tt][1] = (*it).imag();
			}

			// fourier transform
			fftw_execute(fwd);
			
			// copy/shift
			// F += N/2 (even), for N = 4:
			// 0 -> 2 (f =  0)
			// 1 -> 3 (f = +1)
			// 2 -> 0 (f = -2)
			// 3 -> 1 (f = -1)
			it.goChunkBegin();
			double norm = 1./sqrt(osize[dd]);
			for(size_t tt=osize[dd]/2; !it.isChunkEnd(); ++it) {
				cdouble_t tmp(buffer[tt][0]*norm, buffer[tt][1]*norm);
				it.set(tmp);
				tt=(tt+1)%osize[dd];
			}
		}

		fftw_destroy_plan(fwd);
		fftw_free(buffer);
	}

#ifdef DEBUG
	writeComplexAA("padded_fft", oimg);
#endif //DEBUG
	
	return oimg;
}

/**
 * @brief 
 *
 * @param in		Input Image (in normal space)
 * @param prdim	Pseudo-Radius dimension
 *
 * @return 			Pseudo-polar space image
 */
shared_ptr<MRImage> pseudoPolarBrute(shared_ptr<MRImage> in, size_t prdim)
{
	if(in->ndim() != 3) 
		return NULL;

	double uscale = 2;
	double upsample[3] = {uscale,uscale,uscale};
	upsample[prdim] = 1;

	shared_ptr<MRImage> out = padFFT(in, upsample);
	shared_ptr<MRImage> tmp = padFFT(in, upsample);

	// interpolate along lines
	std::vector<double> index(3); // index space version of output
	double radius; 
	double angles[2]; 
	std::vector<double> index2(3); // 
	LinInterp3DView<cdouble_t> interp(tmp);
	for(OrderIter<cdouble_t> oit(out); !oit.eof(); ++oit) {
		oit.index(index.size(), index.data());
//		cerr << "[" << index[0] << ", " << index[1] << ", " << index[2] << "]" <<  "->" ;
		
		// make index into slope, then back to a flat index
		size_t jj=0;
		radius = (double)index[prdim]-((double)out->dim(prdim))/2.;
		for(size_t ii=0; ii<3; ii++) {
			if(ii != prdim) {
				double middle = out->dim(ii)/2.;
				double slope = uscale*(index[ii]-middle)/middle;
				angles[jj++] = slope;
			}
		}
//		cerr << "R: " << radius << ", 1: " << angles[0] << ", 2: " 
//				<< angles[1] << " -> ";

		// centered radius
		jj = 0;
		for(size_t ii=0; ii<3; ii++) {
			if(ii != prdim) 
				index2[ii] = angles[jj++]*radius+out->dim(ii)/2.;
			else
				index2[ii] = radius+out->dim(ii)/2.;
		}
//		cerr << "[" << index2[0] << ", " << index2[1] << ", " << index2[2] << "]" << endl;

		oit.set(interp(index2[0], index2[1], index2[2]));
	}

	return out;
}

shared_ptr<MRImage> createTestImageFreq(size_t sz1)
{
	// create an image
	int64_t index[3];
	size_t sz[] = {sz1, sz1, sz1};
	auto in = createMRImage(sizeof(sz)/sizeof(size_t), sz, COMPLEX128);

	// fill with a shape
	OrderIter<double> sit(in);
	double sum = 0;
	while(!sit.eof()) {
		sit.index(3, index);
		double v;
	
		// ball
//		if(pow(sz1/2.-index[0],2)+pow(sz1/2.-index[1],2)+pow(sz1/2.-index[2],2) < 20)
//			v = 1;
//		else 
//			v = 0;
		
		// lines in x direction, this actually won't work, because there
		// will be aliasing during down-sampling, 
		if((index[2]+index[1])%2 == 0)
			v = 1;
		else
			v = 0;
//
//		// lines in x direction
//		if(sin(index[2]) > 0 && sin(index[1]) > 0)
//			v = 1;
//		else
//			v = 0;
//
//		// gaussian
//		v = 1;
//		for(size_t ii=0; ii<3; ii++)
//			v *= std::exp(-pow(index[ii]-sz1/2.,2)/16);
		sit.set(v);
		sum += v;
		++sit;
	}
	
	for(sit.goBegin(); !sit.eof(); ++sit) 
		sit.set(sit.get()/sum);

	// perform inverse fourier transform
	auto buffer = fftw_alloc_complex((int)sz1);
	fftw_plan fwd = fftw_plan_dft_1d((int)sz1, buffer, buffer, 
			FFTW_BACKWARD, FFTW_MEASURE);
	for(size_t dd = 0; dd < 3; dd++) {
		ChunkIter<cdouble_t> it(in);
		it.setLineChunk(dd);
		for(it.goBegin(); !it.isEnd() ; it.nextChunk()) {
			it.index(3, index);

			// fill from line, shifting
			for(size_t tt=sz1/2; !it.isChunkEnd(); ++it) {
				buffer[tt][0] = (*it).real();
				buffer[tt][1] = (*it).imag();
				tt=(tt+1)%sz1;
			}

			// fourier transform
			fftw_execute(fwd);

			// copy
			it.goChunkBegin();
			for(size_t tt=0; !it.eoc(); ++it, tt++) {
				cdouble_t tmp(buffer[tt][0], buffer[tt][1]);
				it.set(tmp);
			}
		}
	}
	fftw_free(buffer);
	fftw_destroy_plan(fwd);
	cerr << "Finished filling"<<endl;

	return in;
}

shared_ptr<MRImage> createTestImageSpace(size_t sz1)
{
	// create an image
	int64_t index[3];
	size_t sz[] = {sz1, sz1, sz1};
	auto in = createMRImage(sizeof(sz)/sizeof(size_t), sz, COMPLEX128);

	// fill with a shape
	OrderIter<double> sit(in);
	double sum = 0;
	while(!sit.eof()) {
		sit.index(3, index);
		double v = index[0]>(sz[0]/2. - 2) && index[0]<(sz[0]/2. + 2) 
					&& index[1]>(sz[1]/2. - 2) && index[1]<(sz[1]/2. + 2) 
					&& index[2]>(sz[2]/2. - 2) && index[2]<(sz[2]/2. + 2);
		sit.set(v);
		sum += v;
		++sit;
	}
	
	for(sit.goBegin(); !sit.eof(); ++sit) 
		sit.set(sit.get()/sum);
	cerr << "Finished filling"<<endl;

	return in;
}

int testPseudoPolar()
{
//	auto in = createTestImageSpace(64);
	auto in = createTestImageFreq(64);
	writeComplex("input", in);
	
	// test the pseudopolar transforms
	for(size_t dd=0; dd<3; dd++) {
		cerr << "Computing With PseudoRadius = " << dd << endl;
		auto pp1_fft = dynamic_pointer_cast<MRImage>(pseudoPolar(in, dd));
		auto pp1_zoom = dynamic_pointer_cast<MRImage>(pseudoPolarZoom(in, dd));
		auto pp1_brute = pseudoPolarBrute(in, dd);

		writeComplexAA("fft_pp"+to_string(dd), pp1_fft);
		writeComplexAA("zoom_pp"+to_string(dd), pp1_zoom);
		writeComplexAA("brute_pp"+to_string(dd), pp1_brute);
		if(closeCompare(pp1_fft, pp1_brute) != 0) {
			cerr << "FFT and BruteForce pseudopolar differ" << endl;
			return -1;
		}
		if(closeCompare(pp1_zoom, pp1_brute) != 0) {
			cerr << "Zoom and BruteForce pseudopolar differ" << endl;
			return -1;
		}
	}
	
	return 0;
}

int main()
{
	// test the 'Power' Fourier Transform
	if(testPseudoPolar() != 0)
		return -1;

//	if(testPseudoPolar() != 0) 
//		return -1;
	
	return 0;
}




