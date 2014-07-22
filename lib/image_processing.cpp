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

#include "mrimage.h"
#include "iterators.h"
#include "accessors.h"


#include <string>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <memory>

namespace npl {

using std::vector;
using std::shared_ptr;

double gaussKern(double sigma, double x)
{
	const double PI = acos(-1);
	const double den = 1./sqrt(2*PI);
	return den*exp(-x*x/(2*sigma*sigma))/sigma;
}

/**
 * @brief Converts a point in RAS coordinate system to index.
 * If len < dimensions, additional dimensions are assumed to be 0. If len >
 * dimensions then additional values are ignored, and only the first DIM 
 * values will be transformed and written to ras.
 *
 * @param in Input image to smooth
 * @param stddev standard deviation in physical units index*spacing
 * @param dim dimensions to smooth in. If you are smoothing individual volumes
 * of an fMRI you would provide dim={0,1,2}
 *
 * @return Smoothed image
 */
shared_ptr<MRImage> gaussianSmooth(shared_ptr<MRImage> in, double stddev, 
		std::vector<size_t> dim = {})
{
	//TODO figure out how to scale this properly, including with stddev and 
	//spacing

	std::vector<double> window(in->ndim());
	for(size_t dd=0; dd<dim.size(); dd++) {
		if(dd >= in->ndim())
			throw std::out_of_range("Invalid dimensions passed to "
					"gaussian smoother");

		// convert stddev from spatial units to pixels
		double width = stddev/in->spacing()[dd];
		int64_t iwidth = round(width);

		// construct weights
		std::vector<double> weights(iwidth*2+2);
		for(int64_t ii=-iwidth; ii<= iwidth; ii++) {
			weights[ii+iwidth] = gaussKern(iwidth, width);
		}

		// construct window
		for(size_t ii=0; ii<in->ndim(); ii++) {
			if(dd == ii)
				window[ii] = std::make_pair<int64_t,int64_t>(-iwidth, iwidth);
			else
				window.push_back({0,0});
		}

		// create kernel iterator
		KernelIter kit(
	}
}


} // npl
#endif  //IMAGE_PROCESSING_H

