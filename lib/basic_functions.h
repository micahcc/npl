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
 * @file basic_functions.h These are simple in-lineable functions for various
 * purposes, including bounding indexes, and window functions 
 *
 *****************************************************************************/
#ifndef BASIC_FUNCTIONS_H
#define BASIC_FUNCTIONS_H

#include <cmath>

namespace npl {

/**********************************************************
 * Functions for dealing with boundaries (wrap and clamp)
 **********************************************************/

/**
 * @brief Clamps value to range of [inf, sup]. Values outside will be pulled
 * to either sup or inf, whichever is closer.
 *
 * @tparam T Value of inf/sup/v/output
 * @param inf Lower bound (infinum)
 * @param sup Upper bound (supremum)
 * @param v Value to clamp
 *
 * @return
 */
template <typename T>
inline T clamp(T inf, T sup, T v)
{
	return std::min(sup, std::max(inf, v));
}

/**
 * @brief Wraps and index based on the range [inf, sup] (when v is outside that)
 * range. Thus inf = 1, sup = 5, v = 0 will wrap around to 4, and v = 6 would wrap
 * around to 1.
 *
 * @tparam T Value of inf/sup/v
 * @param inf lower bound (inclusive)
 * @param sup upper bound (inclusive)
 * @param v valueto wrap. Output will be the position wrapped.
 *
 * @return Wrapped value
 */
template <typename T>
inline T wrap(T inf, T sup, T v)
{
	T len = sup-inf+1;
	T vtmp = v-inf;
	T out = vtmp < 0 ? sup-((-vtmp-1)%len) : inf+(vtmp%len);
	return out;
}

/**********************************************************
 * Basic Statistical Functions
 **********************************************************/

/**
 * @brief Takes a count, sum and sumsqr and returns the sample variance.
 * This is slightly different than the variance definition and I can
 * never remember the exact formulation.
 *
 * @param count Number of samples
 * @param sum sum of samples
 * @param sumsqr sum of square of the samples
 *
 * @return sample variance
 */
inline
double sample_var(int count, double sum, double sumsqr)
{
	return (sumsqr-sum*sum/count)/(count-1);
}

/**********************************************************
 * Windows/Kernels
 **********************************************************/

/**
 * @brief Rectangle function centered at 0, with radius a, range should be = 2a
 *
 * @param x distance from center
 * @param a radius
 *
 * @return weight
 */
inline
double rectWindow(double x, double a)
{
	if(fabs(x) < a)
		return 1;
	else
		return 0;
}

/**
 * @brief Sinc function centered at 0, with radius a, range should be = 2a
 *
 * @param x distance from center
 * @param a radius
 *
 * @return weight
 */
inline
double sincWindow(double x, double a)
{
	const double PI = acos(-1);

	if(x == 0)
		return 1;
	else if(fabs(x) < a)
		return sin(PI*x/a)/(PI*x/a);
	else
		return 0;
}

/**
 * @brief Lanczos kernel function
 *
 * @param x distance from center
 * @param a radius of kernel
 *
 * @return weight
 */
inline
double lanczosKernel(double x, double a)
{
	const double PI = acos(-1);

	if(x == 0)
		return 1;
	else if(fabs(x) < a)
		return a*sin(PI*x)*sin(PI*x/a)/(PI*PI*x*x);
	else
		return 0;
}

/**
 * @brief Cotangent function
 *
 * @param v angle in radians
 *
 * @return Cotangent of angle
 */
inline
double cot(double v)
{
	return 1./tan(v);
}

/**
 * @brief Cosecant function
 *
 * @param v angle in radians
 *
 * @return Cosecant of angle
 */
inline
double csc(double v)
{
	return 1./sin(v);
}

/**
 * @brief Secand function
 *
 * @param v angle in radians
 *
 * @return Secand of angle
 */
inline
double sec(double v)
{
	return 1./cos(v);
}




} // npl

#endif //BASIC_FUNCTIONS_H
