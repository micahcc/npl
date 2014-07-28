/*******************************************************************************
This file is part of Neuro Programs and Libraries (NPL), 

Written and Copyrighted by by Micah C. Chambers (micahc.vt@gmail.com)

The Neuro Programs and Libraries are free software: you can redistribute it
and/or modify it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

The Neural Programs and Libraries are distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along with
the Neural Programs Library.  If not, see <http://www.gnu.org/licenses/>.
*******************************************************************************/

#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

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
 * @brief Creates a new MRImage with dimensions set by ndim, and size set by
 * size. Output pixel type is decided by ptype variable.
 *
 * @param ndim number of image dimensions
 * @param size size of image, in each dimension
 * @param ptype Pixel type npl::PixelT
 *
 * @return New image, default orientation
 */
shared_ptr<MRImage> createMRImage(size_t ndim, const size_t* size, PixelT ptype);

/**
 * @brief Creates a new MRImage with dimensions set by ndim, and size set by
 * size. Output pixel type is decided by ptype variable.
 *
 * @param size size of image, in each dimension, number of dimensions decied by
 * length of size vector
 * @param ptype Pixel type npl::PixelT
 *
 * @return New image, default orientation
 */
shared_ptr<MRImage> createMRImage(const std::vector<size_t>& size, PixelT);

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


/**
 * @brief Gaussian smooths an image in 1 direction.
 *
 * @param inout Input/Output image
 * @param dim Direction to smooth in
 * @param stddev in real space, for example millimeters.
 */
void gaussianSmooth1D(shared_ptr<MRImage> inout, size_t dim, double stddev);

/**
 * @brief Smooths an image in 1 dimension, masked version. Only updates pixels
 * within masked region.
 *
 * @param in Input/output image to smooth
 * @param dim dimensions to smooth in. If you are smoothing individual volumes
 * of an fMRI you would provide dim={0,1,2}
 * @param stddev standard deviation in physical units index*spacing
 * @param mask Only smooth (alter) point within the mask, inverted by 'invert'
 * @param invert only smooth points outside the mask
 */
//void gaussianSmooth1D(shared_ptr<MRImage> inout, size_t dim, 
//		double stddev, shared_ptr<MRImage> mask, bool invert);

} // npl
#endif  //IMAGE_PROCESSING_H
