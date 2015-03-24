/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file nplio.cpp Readers and Writers for npl::MRImage and npl::NDarray
 *
 *****************************************************************************/
#ifndef NPLIO_H
#define NPLIO_H

#include "mrimage.h"

namespace npl
{

/******************************************************************************
 * /addtogroup MRImageUtilities
 * @{
 *****************************************************************************/

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
int writeMRImage(ptr<const MRImage> img, std::string fn, bool nifti2 = false);

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
int writeNDArray(ptr<const NDArray> img, std::string fn, bool nifti2 = false);

/**
 * @brief Reads an array. Can read nifti's but orientation won't be read.
 *
 * @param filename Name of input file to read
 * @param verbose Whether to print out information as the file is read
 * @param nopixeldata Don't actually read the pixel data, just the header and
 * create the image. So if you want to copy an image's orientation and
 * structure, this would be the way to do it without wasting time actually
 * reading.
 *
 * @return Loaded image
 */
ptr<NDArray> readNDArray(std::string filename, bool verbose = false,
		bool nopixeldata = false);

/**
 * @brief Reads an MRI image. Right now only nift images are supported. later
 * on, it will try to load image using different reader functions until one
 * suceeds.
 *
 * @param filename Name of input file to read
 * @param verbose Whether to print out information as the file is read
 * @param nopixeldata Don't actually read the pixel data, just the header and
 * create the image. So if you want to copy an image's orientation and
 * structure, this would be the way to do it without wasting time actually
 * reading.
 *
 * @return Loaded image
 */
ptr<MRImage> readMRImage(std::string filename, bool verbose = false,
		bool nopixeldata = false);

/** @} */

} // npl

#endif //NPLIO_H

