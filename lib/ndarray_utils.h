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
 * need to cast the output using dPtrCast<MRImage>(out).
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

/**
 * \defgroup NDarrayUtilities NDarray and Image Functions
 * @{
 */

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
ptr<NDArray> derivative(ptr<const NDArray> in);

/**
 * @brief Computes the derivative of the image in the specified direction. 
 *
 * @param in    Input image/NDarray 
 * @param dir   Specify the dimension
 *
 * @return      Image storing the directional derivative of in
 */
ptr<NDArray> derivative(ptr<const NDArray> in, size_t dir);

/**
 * @brief Computes the derivative of the image. Computes all  
 * directional derivatives of the input image and the output
 * image will have 1 higher dimension with derivative of 0 in the first volume
 * 1 in the second and so on.
 *
 * Thus a 2D image will produce a [X,Y,2] image and a 3D image will produce a 
 * [X,Y,Z,3] sized image.
 *
 * @param in    Input image/NDArray 
 * @param out	Derivative of input
 *
 * @return 0 if successful
 */
int derivative(ptr<const NDArray> in, ptr<NDArray> out);

/**
 * @brief Dilate an binary array repeatedly
 *
 * @param in Input to dilate
 * @param reps Number of radius-1 kernel dilations to perform
 *
 * @return Dilated Image
 */
ptr<NDArray> dilate(ptr<NDArray> in, size_t reps);

/**
 * @brief Erode an binary array repeatedly
 *
 * @param in Input to erode
 * @param reps Number of radius-1 kernel erosions to perform
 *
 * @return Eroded Image
 */
ptr<NDArray> erode(ptr<NDArray> in, size_t reps);


/**
 * @brief Smooths an image in 1 dimension
 *
 * @param inout Input/output image to smooth
 * @param dim dimensions to smooth in. If you are smoothing individual volumes
 * of an fMRI you would provide dim={0,1,2}
 * @param stddev standard deviation in physical units index*spacing
 *
 */
void gaussianSmooth1D(ptr<NDArray> inout, size_t dim, double stddev);

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
 * @param kern kernel to use for sampling
 */
void shiftImageKern(ptr<NDArray> inout, size_t dd, double dist,
		double(*kern)(double,double) = npl::lanczosKern);

/**
 * @brief Performs unidirectional shift in the direction of +dd, of distance 
 * (in units of pixels), using FFT.
 *
 * @param inout Input/output image
 * @param dim Dimension to shift, will be positive 
 * @param dist
 * @param window Windowing function to apply in fourier domain
 */
void shiftImageFFT(ptr<NDArray> inout, size_t dim, double dist, 
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
void shearImageKern(ptr<NDArray> inout, size_t dim, size_t len, 
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
void shearImageFFT(ptr<NDArray> inout, size_t dim, size_t len, double* dist,
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

/**
 * @brief Performs a rotation of the image first by rotating around z, then
 * around y, then around x. (Rx*Ry*Rz)
 *
 * @param rx Rotation around x, radians
 * @param ry Rotation around y, radians
 * @param rz Rotation around z, radians
 * @param in Input image
 *
 * @return Rotated image.
 */
ptr<NDArray> linearRotate(double rx, double ry, double rz, 
		ptr<const NDArray> in);

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
void rotateImageKern(ptr<NDArray> inout, double rx, double ry, double rz);

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
int rotateImageShearKern(ptr<NDArray> inout, double rx, double ry, double rz,
		double(*kern)(double,double) = npl::lanczosKern);

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
int rotateImageShearFFT(ptr<NDArray> inout, double rx, double ry, double rz,
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
ptr<NDArray> pseudoPolar(ptr<const NDArray> in, size_t prdim);

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
std::vector<ptr<NDArray>> pseudoPolar(ptr<const NDArray> in);

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
ptr<NDArray> pseudoPolarZoom(ptr<const NDArray> inimg, 
        size_t prdim);

/**
 * @brief Sets the middle of the image += radius (in index space) to 1,
 * everything else to 0
 *
 * @param inout Input/output image.
 * @param radius Radius (distance from center) to set to 1
 * @param alpha is the what the distance is raised to in each dimension (2 is
 * euclidian distance)
 */
void fillCircle(ptr<NDArray> inout, double radius, double alpha);

/**
 * @brief Fills image with the linear index at each pixel
 *
 * @param inout input/output image, will be filled with linear index
 *
 */
void fillLinear(ptr<NDArray> inout);

/**
 * @brief Fills image with the linear index at each pixel
 *
 * @param inout input/output image, will be filled with gaussian white noise
 *
 */
void fillGaussian(ptr<NDArray> inout);

/**
 * @brief Concatinates image in the direction specified by dir. So if dir
 * is 0, and two images, sized [32, 32, 34] and [12, 32, 34] were passed
 * in the input vector, then the output would be [44, 32, 34].
 *
 * @param images Input images, will be placed in order of input vector
 * @param dir Direction to concatinate, all dimesnions other than dir
 * must match in size
 *
 * @return New image that has had the images pasted together
 */
ptr<NDArray> concat(const vector<ptr<NDArray>>& images, size_t dir);

/**
 * @brief Concatinates images/arrays. 1 Extra dimension will be added, all the
 * lower dimensions of the images must match. An example with lastdim = false
 * would be 3 [32,32,32] images, which would result in 1 [32,32,32,3] image
 * with the orienation matching from the first image.
 *
 * @param images Array of images to concatinate
 *
 * @return New image with 1 extra dimension
 */
ptr<NDArray> concatElevate(const vector<ptr<NDArray>>& images);

/**
 * @brief Increases the number of dimensions by 1 then places the edges
 * in each dimension at indexes matching the direction of edge detection.
 * So an input 3D image will produce a 4D image with volume 0 the x edges,
 * volume 1 the y edges and volume 2 the z edges.
 *
 * @param img Input image ND
 *
 * @return Output image N+1D
 */
ptr<NDArray> sobelEdge(ptr<const NDArray> img);

/**
 * @brief Creates a new image with the specified dimension collapsed and the
 * values in each output point set to the sum of the values in the collapsed
 * dimension
 *
 * @param img Input image
 * @param dim Dimension to collapse
 * @param doabs Take the absolute value before summing up
 *
 * @return Image with 1 fewer dimensions, and dim sequezed out.
 */
ptr<NDArray> collapseSum(ptr<const NDArray> img, size_t dim, bool doabs=false);


/**
 * @brief Performs relabeling based on connected component using the two pass
 * algorithm.
 *
 * @param input Input labelmap image
 *
 * @return Relabeled image with connected components labeled together, and
 * non-connected components labeled separately
 */
ptr<NDArray> relabelConnected(ptr<NDArray> input);

/**
 * @brief Computes a threshold based on OTSU.
 *
 * @param in Input image.
 *
 * @return Threshold 
 */
double otsuThresh(ptr<const NDArray> in);

/**
 * @brief Computes a histogram, then spaces out intensities so that each
 * intensity has equal volume/area in the image.
 *
 * @param in Input image.
 *
 * @return Image that has been histogram equalized
 */
ptr<NDArray> histEqualize(ptr<const NDArray> in);

/** @}  NDArrayUtilities */

} // npl
#endif  //NDARRAY_UTILS_H

