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
 * @file mrimage_utils.h
 * @brief This file contains common functions which are useful for image
 * processing. Note that ndarray_utils.h has utilities which are more general
 * whereas this file contains functions which are specifically for image
 * processing.
 ******************************************************************************/

#ifndef MRIMAGE_UTILS_H
#define MRIMAGE_UTILS_H

#include "ndarray.h"
#include "mrimage.h"
#include "basic_functions.h"
#include "npltypes.h"

#include <string>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <memory>

namespace npl {

using std::vector;

/**
 * \defgroup MRImageUtilities MRImage Functions
 * @{
 */

/**
 * @brief Writes out information about an MRImage
 *
 * @param out Output ostream
 * @param img Image to write information about
 *
 * @return More ostream
 */
std::ostream& operator<<(std::ostream &out, const MRImage& img);

/*****************************************************************************
 * Kernel Functions
 ****************************************************************************/

/**
 * @brief Gaussian smooths an image in 1 direction. Questionable whether it
 * works. Seems to shift image.
 *
 * @param inout Input/Output image
 * @param dim Direction to smooth in
 * @param stddev in real space, for example millimeters.
 */
//void gaussianSmooth1D(ptr<MRImage> inout, size_t dim, double stddev);


/******************************************************
 * Resample Image Functions
 ******************************************************/

/**
 * @brief Performs smoothing in each dimension, then downsamples so that pixel
 * spacing is roughly equal to FWHM.
 *
 * @param in    Input image
 * @param sigma Standard deviation for smoothing
 * @param spacing Ouptut image spacing (isotropic). If this is <= 0, then sigma
 * will be used, which is a very conservative downsampling.
 *
 * @return  Smoothed and downsampled image
 */
ptr<MRImage> smoothDownsample(ptr<const MRImage> in,
			double sigma, double spacing = -1);

/**
 * @brief Performs fourier resampling using fourier transform and the provided window function.
 *
 * @param in Input image
 * @param spacing Desired output spacing
 * @param window Window function  to reduce ringing
 *
 * @return  Smoothed and downsampled image
 */
ptr<MRImage> resample(ptr<const MRImage> in, double* spacing,
		double(*window)(double, double) = hannWindow);

/******************************************************
 * FFT Tools
 *****************************************************/
/**
 * @brief Uses fourier shift theorem to shift an image
 *
 * @param in Input image to shift
 * @param len length of dx array
 * @param vect movement in physical coordinates, will be rotated using image
 * orientation prior to shifting
 *
 * @return shifted image
 */
ptr<MRImage> shiftImage(ptr<MRImage> in, size_t len, double* vect);

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
        bool absPhase = false);

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
        const std::vector<size_t>& in_osize);

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
        const std::vector<size_t>& in_osize);

/**
 * @brief Performs a shear on the image where the sheared dimension (dim) will
 * be shifted depending on the index in other dimensions (dist). 
 * (in units of pixels). Uses Lanczos interpolation.
 *
 * @param inout Input/output image
 * @param dim Dimension to shift/shear
 * @param len Length of dist array
 * @param dist Distance terms to travel. Shift[dim] = x0*dist[0]+x1*dist[1] ...
 * @param kern 1D interpolation kernel
 */
void shearImageKern(ptr<MRImage> inout, size_t dim, size_t len, 
        double* dist, double(*kern)(double,double) = npl::lanczosKern);

/**
 * @brief Performs a shear on the image where the sheared dimension (dim) will
 * be shifted depending on the index in other dimensions (dist). 
 * (in units of pixels), using FFT.
 *
 * @param inout Input/output image
 * @param dim Dimension to shift/shear
 * @param len Length of dist array
 * @param dist Distance terms to travel. Shift[dim] = x0*dist[0]+x1*dist[1] ...
 * @param window Windowing function of fourier domain (default sinc)
 */
void shearImageFFT(ptr<MRImage> inout, size_t dim, size_t len, double* dist,
		double(*window)(double,double) = npl::sincWindow);


/** @}  MRImageUtilities */
} // npl
#endif  //MRIMAGE_UTILS_H
