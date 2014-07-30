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

#include "mrimage.h"
#include "mrimage_utils.h"
#include "iterators.h"
#include "accessors.h"

using namespace npl;
using namespace std;

int closeCompare(shared_ptr<const MRImage> a, shared_ptr<const MRImage> b)
{
	if(a->ndim() != b->ndim()) {
		cerr << "Error image dimensionality differs" << endl;
		return -1;
	}
	
	for(size_t dd=0; dd<a->ndim(); dd++) {
		if(a->dim(dd) != b->dim(dd)) {
			cerr << "Image size in the " << dd << " direction differs" << endl;
			return -1;
		}
	}

	OrderConstIter<double> ita(a);
	OrderConstIter<double> itb(b);
	itb.setOrder(ita.getOrder());
	for(ita.goBegin(), itb.goBegin(); !ita.eof() && !itb.eof(); ++ita, ++itb) {
		double diff = fabs(*ita - *itb);
		if(diff > 1E-10) {
			cerr << "Images differ!" << endl;
			return -1;
		}
	}

	return 0;
}

int main()
{
	// create an image
	int64_t index[3];
	size_t sz[] = {128, 128, 128};
	cerr << sizeof(sz) << endl;
	auto in = createMRImage(sizeof(sz)/sizeof(size_t), sz, FLOAT64);

	// fill with sphere
	OrderIter<double> sit(in);
	while(!sit.eof()) {
		sit.index(3, index);
		double dist = 0;
		for(size_t ii=0; ii<3 ; ii++) {
			dist += (index[ii]-sz[ii]/2.)*(index[ii]-sz[ii]/2.);
		}
		if(sqrt(dist) < 5)
			sit.set(1);
		else 
			sit.set(0);

		++sit;
	}
	
	// fourier transform image in xyz direction
	auto fft = fft_r2c(in);
	
	// perform fourier shift, +a
	// strictly the frequency for component k (where k = k-N/2,N/2]
	// double T = fft->dim(d)*in->spacing()[d];
	// double f = k/T; // where T is the total sampling period
	double shift[3] = {1, 5, 10};

	OrderIter<cdouble_t> fit(fft);
	std::complex<double> i(0, 1);
	const double PI = acos(-1);
	for(fit.goBegin(); !fit.isEnd(); ++fit) {
		fit.index(3, index);
		cdouble_t orig = *fit;;
		cdouble_t term = 0;
		for(size_t dd=0; dd<fft->ndim(); dd++)
			term += -2.0*PI*i*shift[dd]*(double)index[dd]/(double)fft->dim(dd);
		orig = orig*std::exp(term);
		fit.set(orig);
	}

	auto ifft = dynamic_pointer_cast<MRImage>(ifft_c2r(fft));

	// manual shift
	auto mshift = dynamic_pointer_cast<MRImage>(in->copy());
	NDAccess<double> acc(in);
	for(OrderIter<double> it(mshift); !it.eof(); ++it) {
		it.index(3, index);
		
		for(size_t dd = 0 ; dd < in->ndim(); dd++) 
			index[dd] = clamp<int64_t>(0, in->dim(dd)-1, index[dd]+shift[dd]);

		it.set(acc.get(3, index));

	}
	mshift->write("manual_shift.nii.gz");
	ifft->write("fourier_shift.nii.gz");

	if(closeCompare(ifft, mshift) != 0) 
		return -1;
	

	return 0;
}




