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
 * @file byteswap.h
 *
 *****************************************************************************/

#ifndef BYTESWAP_H
#define BYTESWAP_H

#include <stdexcept>
#include "npltypes.h"

namespace npl {

template <typename T>
union Bytes
{
	T iv;
	unsigned char bytes[sizeof(T)];
} __attribute__((packed));

bool little_endian()
{
	Bytes<int32_t> tmp;
	tmp.iv = 1;

	if(tmp.bytes[0] == 1)
		return true;
	else
		return false;
}

bool big_endian()
{
	Bytes<int32_t> tmp;
	tmp.iv = 1;

	if(tmp.bytes[3] == 1)
		return true;
	else
		return false;
}

template <typename T>
void swap(T* val)
{
	Bytes<T> tmp;
	tmp.iv = *val;
	for(size_t ii=0; ii<sizeof(T)/2; ii++)
		std::swap(tmp.bytes[sizeof(T)-ii-1], tmp.bytes[ii]);
	*val = tmp.iv;
}

template <>
inline
void swap<signed char>(signed char* val)
{
	(void)val;
}


template <>
inline
void swap<char>(char* val)
{
	(void)val;
}

template <>
inline
void swap<uint8_t>(uint8_t* val)
{
	(void)val;
}

// swap the real and imaginary parts individualy
template <>
inline
void swap<cquad_t>(cquad_t* val)
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
inline
void swap<cdouble_t>(cdouble_t* val)
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
inline
void swap<cfloat_t>(cfloat_t* val)
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
inline
void swap<rgba_t>(rgba_t* val)
{
	(void)val;
}

template <>
inline
void swap<rgb_t>(rgb_t* val)
{
	(void)val;
}

} // npl

#endif //BYTESWAP_H
