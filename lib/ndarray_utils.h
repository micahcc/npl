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

#ifndef ND_ALGOS_H
#define ND_ALGOS_H

#include "ndarray.h"
#include "npltypes.h"

#include <memory>

namespace npl {

using std::vector;
using std::shared_ptr;


/**
 * @brief Computes the N-Dimensional fourier transform on the input image.
 *
 * @param in Input image
 *
 * @return Output image (complex double)
 */
shared_ptr<NDArray> fft(shared_ptr<NDArray> in);

/**
 * @brief Computes the N-Dimensional inverse fourier transform on the input image.
 *
 * @param in Input image (should be complex)
 *
 * @return Output image (complex double)
 */
shared_ptr<NDArray> ifft(shared_ptr<NDArray> in);

/**
 * @brief Returns whether two NDArrays have the same dimensions, and therefore
 * can be element-by-element compared/operated on. elL is set to true if left
 * is elevatable to right (ie all dimensions match or are missing or are unary).
 * elR is the same but for the right. 
 *
 * Strictly R is elevatable if all dimensions that don't match are missing or 1
 * Strictly L is elevatable if all dimensions that don't match are missing or 1
 *
 * Examples of *elR = true (return false):
 *
 * left = [10, 20, 1]
 * right = [10, 20, 39]
 *
 * left = [10]
 * right = [10, 20, 39]
 *
 * Examples where neither elR or elL (returns true):
 *
 * left = [10, 20, 39]
 * right = [10, 20, 39]
 *
 * Examples where neither elR or elL (returns false):
 *
 * left = [10, 20, 9]
 * right = [10, 20, 39]
 *
 * left = [10, 1, 9]
 * right = [10, 20, 1]
 *
 * @param left	NDArray input
 * @param right NDArray input
 * @param elL Whether left is elevatable to right (see description of function)
 * @param elR Whether right is elevatable to left (see description of function)
 *
 * @return 
 */
bool comparable(const NDArray* left, const NDArray* right, 
		bool* elL = NULL, bool* elR = NULL);

/**
 * @brief Dilate an binary array repeatedly
 *
 * @param in Input to dilate
 * @param reps Number of radius-1 kernel dilations to perform
 *
 * @return Dilated Image
 */
shared_ptr<NDArray> dilate(shared_ptr<NDArray> in, size_t reps);

/**
 * @brief Erode an binary array repeatedly
 *
 * @param in Input to erode
 * @param reps Number of radius-1 kernel erosions to perform
 *
 * @return Eroded Image
 */
shared_ptr<NDArray> erode(shared_ptr<NDArray> in, size_t reps);

} // npl
#endif  //ND_ALGOS_H

