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
#include <list>

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


/**
 * @brief Highest order bit
 *
 * @param num Input, output will be this with the highest order bit set
 *
 * @return 
 */
inline
int hob(int num)
{
	if (!num)
		return 0;

	int ret = 1;

	while(num >>= 1)
		ret <<= 1;

	return ret;
}


/**
 * @brief Round up to the nearest power of 2.
 *
 * @param in Number to round
 *
 * @return Rounded up umber
 */
inline
int64_t round2(int64_t in)
{
	int64_t just_hob = hob(in);
	if(just_hob == in)
		return in;
	else
		return (in<<1);
}

/**
 * @brief Provides a list of the prime-fractors of the input number
 *
 * @param f input number
 *
 * @return list of factors
 */
std::list<int64_t> factor(int64_t f)
{
	std::list<int64_t> factors;
	for(int64_t ii = 2; ii<=f; ii++) {
		while(f % ii == 0) {
			f = f/ii;
			factors.push_back(ii);
		}
	}

	return factors;
}

/**
 * @brief Rounds a number up to the nearest number that can be broken down into
 * 3,5,7
 *
 * @param in Input number
 *
 * @return Next number up that matches the requirement
 */
int64_t round357(int64_t in)
{
	// make it odd
	if(in %2 == 0)
		in++;

	bool acceptable = false;
	while(!acceptable) {
		acceptable = true;
		in += 2;

		// check the factors
		auto factors = factor(in);
		for(auto f : factors) {
			if(f != 3 && f != 5 && f != 7) {
				acceptable = false;
				break;
			}
		}
	}

	return in;
}



} // npl

#endif //BASIC_FUNCTIONS_H
