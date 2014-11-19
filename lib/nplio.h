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
