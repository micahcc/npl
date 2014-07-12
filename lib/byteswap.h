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

#ifndef BYTESWAP_H
#define BYTESWAP_H

#include "npltypes.h"

namespace npl {

template <typename T>
union Bytes 
{
	T iv;
	unsigned char bytes[sizeof(T)];
} __attribute__((packed));

template <typename T>
void swap(T* val)
{
	Bytes<T> tmp;
	tmp.iv = *val;
	for(size_t ii=0; ii<sizeof(T)/2; ii++)
		std::swap(tmp.bytes[sizeof(T)-ii-1], tmp.bytes[ii]);
	*val = tmp.iv;
}

// swap the real and imaginary parts individualy
template <>
void swap(cquad_t* val)
{
	Bytes<long double> real;
	Bytes<long double> imag;

	real.iv = val->real();
	imag.iv = val->imag();
	
	for(size_t ii=0; ii<sizeof(long double)/2; ii++) {
		std::swap(real.bytes[sizeof(long double)-ii-1], real.bytes[ii]);
		std::swap(imag.bytes[sizeof(long double)-ii-1], imag.bytes[ii]);
	}

	val->real(real.iv);
	val->imag(imag.iv);
}

// swap the real and imaginary parts individualy
template <>
void swap(cdouble_t* val)
{
	Bytes<double> real;
	Bytes<double> imag;

	real.iv = val->real();
	imag.iv = val->imag();
	
	for(size_t ii=0; ii<sizeof(double)/2; ii++) {
		std::swap(real.bytes[sizeof(double)-ii-1], real.bytes[ii]);
		std::swap(imag.bytes[sizeof(double)-ii-1], imag.bytes[ii]);
	}

	val->real(real.iv);
	val->imag(imag.iv);
}

// swap the real and imaginary parts individualy
template <>
void swap(cfloat_t* val)
{
	Bytes<float> real;
	Bytes<float> imag;

	real.iv = val->real();
	imag.iv = val->imag();
	
	for(size_t ii=0; ii<sizeof(float)/2; ii++) {
		std::swap(real.bytes[sizeof(float)-ii-1], real.bytes[ii]);
		std::swap(imag.bytes[sizeof(float)-ii-1], imag.bytes[ii]);
	}

	val->real(real.iv);
	val->imag(imag.iv);
}

// do no byte swapping
template <>
void swap(rgba_t* val)
{
	(void)(val);
}

template <>
void swap(rgb_t* val)
{
	(void)(val);
}

} // npl

#endif //BYTESWAP_H
