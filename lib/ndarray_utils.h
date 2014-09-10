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
 * @file ndarray_utils.h
 * @brief This file contains common functions which are useful for processing
 * of N-dimensional arrays and their derived counterparts (MRImage for
 * example). All of these functions return pointers to NDArray types, however
 * if an image is passed in, then the output will also be an image, you just
 * need to cast the output using std::dynamic_pointer_cast<MRImage>(out).
 * mrimage_utils.h is for more specific image-processing algorithm, this if for
 * generally data of any dimension, without regard to orientation.
 ******************************************************************************/

#ifndef NDARRAY_UTILS_H
#define NDARRAY_UTILS_H

#include "ndarray.h"
#include "npltypes.h"
#include "basic_functions.h"

#include <Eigen/Dense>
#include <memory>
#include <list>

namespace npl {

using std::vector;
using std::shared_ptr;


/**
 * @brief Perform fourier transform on the dimensions specified. Those
 * dimensions will be padded out. The output of this will be a double.
 * If len = 0 or dim == NULL, then ALL dimensions will be transformed.
 *
 * @param in Input image to inverse fourier trnasform
 *
 * @return Image with specified dimensions in the real domain. Image will
 * differ in size from input, and the last dimension will only contain the
 * positive frequencies
 */
shared_ptr<NDArray> ifft_c2r(shared_ptr<const NDArray> in);

/**
 * @brief Perform fourier transform on the dimensions specified. Those
 * dimensions will be padded out. The output of this will be a complex double.
 * If len = 0 or dim == NULL, then ALL dimensions will be transformed.
 *
 * @param in Input image to fourier trnasform
 *
 * @return Real image, which is the result of inverse fourier transforming
 * the (complex) input image.
 */
shared_ptr<NDArray> fft_r2c(shared_ptr<const NDArray> in);

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

/*************************
 * Basic Kernel Functions
 *************************/

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
shared_ptr<NDArray> derivative(shared_ptr<const NDArray> in);

/**
 * @brief Computes the derivative of the image in the specified direction. 
 *
 * @param in    Input image/NDarray 
 * @param dir   Specify the dimension
 *
 * @return      Image storing the directional derivative of in
 */
shared_ptr<NDArray> derivative(shared_ptr<const NDArray> in, size_t dir);

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


/**
 * @brief Smooths an image in 1 dimension
 *
 * @param inout Input/output image to smooth
 * @param dim dimensions to smooth in. If you are smoothing individual volumes
 * of an fMRI you would provide dim={0,1,2}
 * @param stddev standard deviation in physical units index*spacing
 *
 */
void gaussianSmooth1D(shared_ptr<NDArray> inout, size_t dim, double stddev);

/********************
 * Image Shifting 
 ********************/

/**
 * @brief Performs unidirectional shift in the direction of +dd, of distance 
 * (in units of pixels). Uses Lanczos interpolation.
 *
 * @param inout Input/output image
 * @param dd Dimension to shift, will be positive 
 * @param dist
 */
void shiftImageKern(shared_ptr<NDArray> inout, size_t dd, double dist);

/**
 * @brief Performs unidirectional shift in the direction of +dd, of distance 
 * (in units of pixels), using FFT.
 *
 * @param inout Input/output image
 * @param dim Dimension to shift, will be positive 
 * @param dist
 * @param window Windowing function to apply in fourier domain
 */
void shiftImageFFT(shared_ptr<NDArray> inout, size_t dim, double dist, 
		double(*window)(double,double) = npl::sincWindow);


/********************
 * Image Shearing 
 ********************/

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
void shearImageKern(shared_ptr<NDArray> inout, size_t dim, size_t len, 
        double* dist, double(*kern)(double,double) = npl::lanczosKernel);

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
void shearImageFFT(shared_ptr<NDArray> inout, size_t dim, size_t len, double* dist,
		double(*window)(double,double) = npl::sincWindow);

/**
 * @brief Decomposes a euler angle rotation using the rotation matrix made up 
 * of R = Rx*Ry*Rz. Note that this would be multiplying the input vector by Rz
 * then Ry, then Rx. This does not support angles > PI/4. To do that, you
 * should first do bulk rotation using 90 degree rotations (which requires not
 * interpolation).
 *
 * @param bestshears    List of the best fitting shears, should be applied in
 *                      forward order
 * @param Rx            Rotation about X axis
 * @param Ry            Rotation about Y axis
 * @param Rz            Rotation about Z axis
 *
 * @return              Success if 0
 */
int shearDecompose(std::list<Eigen::Matrix3d>& bestshears, 
        double Rx, double Ry, double Rz);

/**
 * @brief Tests shear results. If there is a solution (error is not NAN), then 
 * these should result small errors, so this checks that errors are small when
 * possible. Note that the rotation matrix would be given by R = Rx*Ry*Rz
 *
 * @param Rx rotation about X axis (last)
 * @param Ry rotation about Y axis (middle)
 * @param Rz rotation about Z axis (first)
 *
 * @return 
 */
int shearTest(double Rx, double Ry, double Rz);


/********************
 * Image Rotating
 ********************/

/**
 * @brief Performs a rotation using 3D intperolation kernel (lanczos)
 *
 * @param inout Input/output image
 * @param rx Rotation about x axis
 * @param ry Rotation about y axis
 * @param rz Rotation about z axis
 */
void rotateImageKern(shared_ptr<NDArray> inout, double rx, double ry, double rz);

/**
 * @brief Performs a rotation using fourier shift and shears, using FFT for 
 * unidirectional shifts, using FFT. Rotation matrix R = Rx*Ry*Rz
 *
 * @param inout Input/output image
 * @param rx Rotation about x axis
 * @param ry Rotation about y axis
 * @param rz Rotation about z axis
 * @param kern Kernel to perform 1D interpolation with. 
 */
int rotateImageShearKern(shared_ptr<NDArray> inout, double rx, double ry, double rz,
		double(*kern)(double,double) = npl::lanczosKernel);

/**
 * @brief Performs a rotation using fourier shift and shears, using FFT for 
 * unidirectional shifts, using FFT. Rotation matrix R = Rx*Ry*Rz
 *
 * @param inout Input/output image
 * @param rx Rotation about x axis
 * @param ry Rotation about y axis
 * @param rz Rotation about z axis
 * @param window Window function to apply in fourier domain
 */
int rotateImageShearFFT(shared_ptr<NDArray> inout, double rx, double ry, double rz,
		double(*window)(double,double) = npl::sincWindow);

/****************************
 * Radial Fourier Transforms
 ****************************/

/**
 * @brief Computes the pseudopolar-gridded fourier transform on the input
 * image, with prdim as the pseudo-radius direction. To sample the whole space
 * you would need to call this once for each of the dimensions, or use the
 * other function which does not take this argument, and returns a vector.
 *
 * @param in	Input image to compute pseudo-polar fourier transform on
 * @param prdim	Dimension to be the pseudo-radius in output
 *
 * @return 		Pseudo-polar sample fourier transform
 */
shared_ptr<NDArray> pseudoPolar(shared_ptr<const NDArray> in, size_t prdim);

/**
 * @brief Computes the pseudopolar-gridded fourier transform on the input
 * image returns a vector of pseudo-polar sampled image, one for each dimension
 * as the pseudo-radius.
 *
 * @param in	Input image to compute pseudo-polar fourier transform on
 *
 * @return 		Vector of Pseudo-polar sample fourier transforms, one for each
 * dimension
 */
std::vector<std::shared_ptr<NDArray>> pseudoPolar(shared_ptr<const NDArray> in);

/**
 * @brief Computes the pseudopolar-gridded fourier transform on the input
 * image, with prdim as the pseudo-radius direction. To sample the whole space
 * you would need to call this once for each of the dimensions, or use the
 * other function which does not take this argument, and returns a vector.
 * This function skips the chirpz transform by interpolation-zooming.
 *
 * @param inimg	Input image to compute pseudo-polar fourier transform on
 * @param prdim	Dimension to be the pseudo-radius in output
 *
 * @return 		Pseudo-polar sample fourier transform
 */
shared_ptr<NDArray> pseudoPolarZoom(shared_ptr<const NDArray> inimg, 
        size_t prdim);

} // npl
#endif  //NDARRAY_UTILS_H

