/*******************************************************************************
This file is part of Neuro Programs and Libraries (NPL), 

Written and Copyrighted by by Micah C. Chambers (micahc.vt@gmail.com)

The Neuro Programs and Libraries is free software: you can redistribute it and/or
modify it under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your option)
any later version.

The Neural Programs and Libraries are distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along with
the Neural Programs Library.  If not, see <http://www.gnu.org/licenses/>.
*******************************************************************************/

/******************************************************************************
 * @file fft_test.cpp
 * @brief This file is specifically to test forward, reverse of fft image
 * procesing functions.
 ******************************************************************************/

#include <version.h>
#include <string>
#include <stdexcept>

#include "fftw3.h"

using namespace npl;
using namespace std;

void fractionalFFT(const std::vector<double>& input, double alpha,
		std::vector<double>& real, std::vector<double>& imag)
{
	fftw_plan fwd_plan = fftw_plan_dft_1d((int)input.size(), idata, idata,
				FFTW_FORWARD, FFTW_MEASURE);
	fftw_plan rev_plan = fftw_plan_dft_1d((int)input.size(), idata, idata,
				FFTW_BACKWARD, FFTW_MEASURE);

	auto idata = fftw_alloc_complex(input.size());
	auto chirp = fftw_alloc_complex(input.size());
	for(size_t ii=0; ii<input.size(); ii++){
		idata[ii][0] = input[ii];
		idata[ii][1] = 0;
	}

	fftw_execute(fwd_plan);

	

}

int main()
{


	fftw_execute(plan);
	fftw_free(odata);
	return 0;
}




