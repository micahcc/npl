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

#include <cstdlib>
#include <cmath>
#include <cassert>
#include <list>

#include "macros.h"

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

/**
 * @brief Computes the sample correlation.
 *
 * @param count 	Number of samples
 * @param sum1 		Sum of group 1
 * @param sum2		Sum of group 2
 * @param sumsq1	Sum of the sqaure of the elements of group1
 * @param sumsq2	Sum of the sqaure of the elements of group2
 * @param s1s2		Elements of group1*elements of group 2
 *
 * @return Sample Correlation
 */
inline
double sample_corr(int count, double sum1, double sum2, 
		double sumsq1, double sumsq2, double s1s2)
{
	return (count*s1s2-sum1*sum2)/
			sqrt((count*sumsq1-sum1*sum1)*(count*sumsq2-sum2*sum2));
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
	if(x == 0)
		return 1;
	else if(fabs(x) < a)
		return sin(M_PI*x/a)/(M_PI*x/a);
	else
		return 0;
}

/**
 * @brief Sinc function centered at 0, with radius a, range should be = 2a.
 * Zero < -a, > a
 *
 * @param x distance from center
 * @param a radius
 *
 * @return weight
 */
inline
double hannWindow(double x, double a)
{
	const double alpha = .54;
	const double beta = 1-alpha;
	if(fabs(x) < a)
		return alpha-beta*cos(M_PI*x/a+M_PI);
	else
		return 0;
}

/**
 * @brief Derivative of lanczos kernel with respect to x
 *
 * @param x Distance from center (0)
 * @param a Radius
 *
 * @return Weight
 */
double lanczosKern(double x, double a)
{
       if(x == 0)
               return 1;
       // a*Sin[Pi*x]*Sin[Pi*x/a]/(Pi*Pi*x*x)
       double v = a*sin(M_PI*x)*sin(M_PI*x/a)/(M_PI*M_PI*x*x);
       assert(v <= 1.001 && v >= -1.0001);
       return v;
}

/**
 * @brief Derivative of lanczos kernel with respect to x
 *
 * @param x Distance from center (0)
 * @param a Radius
 *
 * @return Weight
 */
double dLanczosKern(double x, double a)
{
       if(x == 0)
               return 0;
       double v = (M_PI*x*((cos((M_PI*x)/a)*sin(M_PI*x)) +
                   (a*sin((M_PI*x)/a)*cos(M_PI*x))) -
               (2*a*sin((M_PI*x)/a)*sin(M_PI*x)))/(M_PI*M_PI*x*x*x);
       assert(v >= -2 && v <= 10);
       return v;

}

/* Linear Kernel Sampling */
double linKern(double x, double a)
{
	return fabs(1-fmin(1,fabs(x/a)))/a;
}

/* Linear Kernel Sampling */
double dLinKern(double x, double a)
{
	if(x < -a || x > a)
		return 0;
	if(x < 0)
		return 1/a;
	else
		return -1/a; 
}

/****************************************************** 
 * Third Order BSpline kernel. X is distance from 0
 ****************************************************/

/**
 * @brief 3rd order B-Spline, radius 2 [-2,2]
 *
 * @param x Distance from center
 *
 * @return Weight
 */
inline
double B3kern(double x)
{
	switch((int)floor(x)) {
		case -2:
			return 4./3. + 2.*x + x*x + x*x*x/6.;
		break;
		case -1:
			return 2./3. - x*x - x*x*x/2.;
		break;
		case 0:
			return 2./3. - x*x + x*x*x/2.;
		break;
		case 1:
			return 4./3. - 2*x + x*x - x*x*x/6.;
		break;
		default:
			return 0;
		break;
	}
}

/**
 * @brief 3rd order B-Spline, variable radius (w)
 *
 * @param x Distance from center
 * @param r Radius
 *
 * @return Weight
 */
inline
double B3kern(double x, double r)
{
    return B3kern(x*2/r)*2/r;
}

/**
 * @brief 3rd order B-Spline derivative, radius 2 [-2,2]
 *
 * @param x Distance from center
 *
 * @return Weight
 */
inline
double dB3kern(double x)
{
	switch((int)floor(x)) {
		case -2:
			return (4+4*x+x*x)/2.;
		case -1:
			return -(4*x + 3*x*x)/2.;
		case 0:
			return (-4*x + 3*x*x)/2.;
		case 1:
			return -(x*x - 4*x + 4)/2.;
		default:
			return 0;
	}
	return 0;
}

/**
 * @brief 3rd order B-Spline, variable radius (w)
 *
 * @param x Distance from center
 * @param r Radius
 *
 * @return Weight
 */
inline
double dB3kern(double x, double r)
{
    return dB3kern(x*2/r)*4/(r*r);
}

/**
 * @brief 3rd order B-Spline, 2nd derivative, radius 2 [-2,2]
 *
 * @param x Distance from center
 *
 * @return Weight
 */
inline
double ddB3kern(double x)
{
	switch((int)floor(x)) {
		case -2:
			return 2 + x;
		case -1:
            return -2 - 3*x;
		case 0:
			return -2 + 3*x;
		case 1:
			return 2 - x;
		default:
			return 0;
	}
	return 0;
}

/**
 * @brief 3rd order B-Spline, variable radius (w)
 *
 * @param x Distance from center
 * @param r Radius
 *
 * @return Weight
 */
inline
double ddB3kern(double x, double r)
{
    return ddB3kern(x*2/r)*8/(r*r*r);
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
		return (just_hob<<1);
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

/**
 * @brief Very basic counter that iterates over an ND region.
 *
 * Example usage:
 *
 * size_t ret = 0;
 * size_t width = 5;
 * Counter c;
 * c.ndim = ndim;
 * for(size_t ii=0; ii<ndim; ii++)
 *   c.sz[ii] = width;
 * 
 * do {
 *   double weight = 1;
 *   for(int dd = 0; dd < ndim; dd++) 
 *     weight *= c.pos[dd];
 * } while(c.advance());
 *
 *
 * @tparam T Type of size/position
 * @tparam MAXDIM Maximum supported dimension, static array will be this size
 */
template <typename T = int, int MAXDIM=10>
struct Counter
{
	T sz[MAXDIM];
	T pos[MAXDIM];
	T ndim;
	
	/**
	 * @brief Default constructor. Just sizes pos to 0
	 */
	Counter() 
	{ 
		for(size_t dd=0; dd<MAXDIM; dd++) 
			pos[dd] = 0;
	};

	/**
	 * @brief Initialize counter with the specified dimension and stop point
	 *
	 * @param dim Number of dimensions
	 * @param stop
	 */
	Counter(T dim, T* stop)
	{
		if(dim > MAXDIM)
			throw INVALID_ARGUMENT("Dimension "+std::to_string(dim)+">="+
					std::to_string(MAXDIM));

		for(size_t dd=0; dd<dim; dd++) {
			sz[dd] = stop[dd];
			pos[dd] = 0;
		}
		ndim = dim;
	};

	/**
	 * @brief Advance through ND-counter. 0,0,0 to 0,0,1 and so on. If roder is
	 * true then this will start at 0,0,0 and then go to 1,0,0 then 2,0,0, etc.
	 *
	 * @param rorder Whether to reverse the order and do low dimension fastest
	 *
	 * @return 
	 */
	bool advance(bool rorder = false) {
		if(rorder) {
			for(int dd=0; dd<(int)ndim; dd++) {
				pos[dd]++;
				if(pos[dd] == sz[dd])
					pos[dd] = 0;
				else  
					return true;
			}
		} else {
			for(int dd=ndim-1; dd>= 0; dd--) {
				pos[dd]++;
				if(pos[dd] == sz[dd])
					pos[dd] = 0;
				else  
					return true;
			}
		}
		
		return false;
	};
};

} // npl

#endif //BASIC_FUNCTIONS_H
