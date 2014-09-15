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
#include "npltypes.h"

#include <string>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <memory>

namespace npl {

using std::vector;
using std::shared_ptr;

/**
 * \defgroup MRImageUtilities MRImage Functions
 * @{
 */

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
shared_ptr<MRImage> readMRImage(std::string filename, bool verbose = false);


/**
 * @brief Writes out a MRImage to the given file. If specified it will be a
 * nifti version 2 file, rather than version 1.
 *
 * @param img MRImage to write to disk
 * @param fn Filename to write to
 * @param nifti2 whether to write version 2 of the nifti standard
 *
 * @return 0 if successful
 */
int writeMRImage(MRImage* img, std::string fn, bool nifti2 = false);

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
 * @brief Gaussian smooths an image in 1 direction.
 *
 * @param inout Input/Output image
 * @param dim Direction to smooth in
 * @param stddev in real space, for example millimeters.
 */
void gaussianSmooth1D(shared_ptr<MRImage> inout, size_t dim, double stddev);


/******************************************************
 * Resample Image Functions
 ******************************************************/

/**
 * @brief Performs smoothing in each dimension, then downsamples so that pixel
 * spacing is roughly equal to FWHM.
 *
 * @param in    Input image
 * @param sigma Standard deviation for smoothing
 *
 * @return  Smoothed and downsampled image
 */
shared_ptr<MRImage> smoothDownsample(shared_ptr<const MRImage> in, 
        double sigma);

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
shared_ptr<MRImage> shiftImage(shared_ptr<MRImage> in, size_t len, double* vect);

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
void writeComplex(std::string basename, shared_ptr<const MRImage> in, 
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
shared_ptr<MRImage> fft_forward(shared_ptr<const MRImage> in, 
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
shared_ptr<MRImage> fft_backward(shared_ptr<const MRImage> in,
        const std::vector<size_t>& in_osize);

/** @}  MRImageUtilities */
} // npl
#endif  //MRIMAGE_UTILS_H
