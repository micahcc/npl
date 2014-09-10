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
#include "matrix.h"
#include "nifti.h"

#include <string>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <memory>

namespace npl {

using std::vector;
using std::shared_ptr;

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
shared_ptr<MRImage> copyCast(shared_ptr<MRImage> in, size_t newdims,
		const size_t* newsize, PixelT newtype);

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
shared_ptr<MRImage> copyCast(shared_ptr<MRImage> in, size_t newdims,
		const size_t* newsize);

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
shared_ptr<MRImage> copyCast(shared_ptr<MRImage> in, PixelT newtype);

/**
 * @brief Reads a nifti image, given an already open gzFile.
 *
 * @param file gzFile to read from
 * @param verbose whether to print out information during header parsing
 *
 * @return New MRImage with values from header and pixels set
 */
shared_ptr<MRImage> readNiftiImage(gzFile file, bool verbose);

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
int readNifti2Header(gzFile file, nifti2_header* header, bool* doswap, bool verbose);

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
int readNifti1Header(gzFile file, nifti1_header* header, bool* doswap, bool verbose);


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
 * @brief Computes the derivative of the image in the specified direction. 
 * This is identical to the NDArray version, but it scales by the spacing.
 *
 * @param in    Input image/NDarray 
 * @param dir   Specify the dimension
 *
 * @return      Image storing the directional derivative of in
 */
shared_ptr<MRImage> derivative(shared_ptr<const MRImage> in, size_t dir);

/**
 * @brief Computes the derivative of the image. Computes all
 * directional derivatives of the input image and the output
 * image will have 1 higher dimension with derivative of 0 in the first volume
 * 1 in the second and so on.
 *
 * Thus a 2D image will produce a [X,Y,2] image and a 3D image will produce a 
 * [X,Y,Z,3] sized image.
 *
 * @param in    Input image/NDarray 
 *
 * @return 
 */
shared_ptr<MRImage> derivative(shared_ptr<const MRImage> in);


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

} // npl
#endif  //MRIMAGE_UTILS_H
