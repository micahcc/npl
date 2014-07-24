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

#ifndef NPLTYPES_H
#define NPLTYPES_H

#include <complex>

namespace npl {


struct cfloat_t;
struct cdouble_t;
struct cquad_t;
struct rgba_t;
struct rgb_t;

struct cdouble_t: public std::complex<double>
{
	explicit operator float() const;
	explicit operator double() const;
	explicit operator long double() const;
	explicit operator int64_t() const;
	explicit operator uint64_t() const;
	explicit operator int32_t() const;
	explicit operator uint32_t() const;
	explicit operator int16_t() const;
	explicit operator uint16_t() const;
	explicit operator int8_t() const;
	explicit operator uint8_t() const;
//	explicit operator cfloat_t() const;
//	explicit operator cquad_t() const;
	explicit operator std::complex<float>();
	explicit operator std::complex<long double>();

	cdouble_t(double re = 0, double im = 0) : std::complex<double>(re, im) {} ;
	explicit cdouble_t(cfloat_t);
	explicit cdouble_t(cquad_t);
	explicit cdouble_t(std::complex<float>);
	explicit cdouble_t(std::complex<long double>);
	explicit cdouble_t(rgba_t);
	explicit cdouble_t(rgb_t);
};

struct cfloat_t : public std::complex<float>
{
	explicit operator float() const;
	explicit operator double() const;
	explicit operator long double() const;
	explicit operator int64_t() const;
	explicit operator uint64_t() const;
	explicit operator int32_t() const;
	explicit operator uint32_t() const;
	explicit operator int16_t() const;
	explicit operator uint16_t() const;
	explicit operator int8_t() const;
	explicit operator uint8_t() const;
	explicit operator std::complex<double>();
	explicit operator std::complex<long double>();

	cfloat_t(double re = 0, double im = 0) : std::complex<float>(re, im) {} ;
	explicit cfloat_t(cdouble_t);
	explicit cfloat_t(cquad_t);
	explicit cfloat_t(std::complex<double>);
	explicit cfloat_t(std::complex<long double>);
	explicit cfloat_t(rgba_t);
	explicit cfloat_t(rgb_t);
};

struct cquad_t : public std::complex<long double>
{
	explicit operator float() const;
	explicit operator double() const;
	explicit operator long double() const;
	explicit operator int64_t() const;
	explicit operator uint64_t() const;
	explicit operator int32_t() const;
	explicit operator uint32_t() const;
	explicit operator int16_t() const;
	explicit operator uint16_t() const;
	explicit operator int8_t() const;
	explicit operator uint8_t() const;
	explicit operator std::complex<float>();
	explicit operator std::complex<double>();
	
	cquad_t(double re = 0, double im = 0) : std::complex<long double>(re, im) {} ;
	explicit cquad_t(cdouble_t);
	explicit cquad_t(cfloat_t);
	explicit cquad_t(std::complex<float>);
	explicit cquad_t(std::complex<double>);
	explicit cquad_t(rgba_t);
	explicit cquad_t(rgb_t);
};

struct rgb_t
{
	char red;
	char green;
	char blue;

	explicit operator float() const;
	explicit operator double() const;
	explicit operator long double() const;
	explicit operator int64_t() const;
	explicit operator uint64_t() const;
	explicit operator int32_t() const;
	explicit operator uint32_t() const;
	explicit operator int16_t() const;
	explicit operator uint16_t() const;
	explicit operator int8_t() const;
	explicit operator uint8_t() const;

	// math operators
	rgb_t& operator=(float v);
	rgb_t& operator=(double v);
	rgb_t& operator=(long double v);
	rgb_t& operator=(int64_t v);
	rgb_t& operator=(uint64_t v);
	rgb_t& operator=(int32_t v);
	rgb_t& operator=(uint32_t v);
	rgb_t& operator=(int16_t v);
	rgb_t& operator=(uint16_t v);
	rgb_t& operator=(int8_t v);
	rgb_t& operator=(uint8_t v);
	
	rgb_t() : red(0), green(0), blue(0) {} ;
	rgb_t(char r, char g, char b) : 
		red(r), green(g), blue(b) {} ;
	explicit rgb_t(float);
	explicit rgb_t(double);
	explicit rgb_t(long double);
	explicit rgb_t(int64_t);
	explicit rgb_t(uint64_t);
	explicit rgb_t(int32_t);
	explicit rgb_t(uint32_t);
	explicit rgb_t(int16_t);
	explicit rgb_t(uint16_t);
	explicit rgb_t(int8_t);
	explicit rgb_t(uint8_t);
	explicit rgb_t(cquad_t);
	explicit rgb_t(cdouble_t);
	explicit rgb_t(cfloat_t);
	explicit rgb_t(rgba_t);
};

struct rgba_t
{
	char red;
	char green;
	char blue;
	char alpha;

	explicit operator float() const;
	explicit operator double() const;
	explicit operator long double() const;
	explicit operator int64_t() const;
	explicit operator uint64_t() const;
	explicit operator int32_t() const;
	explicit operator uint32_t() const;
	explicit operator int16_t() const;
	explicit operator uint16_t() const;
	explicit operator int8_t() const;
	explicit operator uint8_t() const;
	
	rgba_t& operator=(float v);
	rgba_t& operator=(double v);
	rgba_t& operator=(long double v);
	rgba_t& operator=(int64_t v);
	rgba_t& operator=(uint64_t v);
	rgba_t& operator=(int32_t v);
	rgba_t& operator=(uint32_t v);
	rgba_t& operator=(int16_t v);
	rgba_t& operator=(uint16_t v);
	rgba_t& operator=(int8_t v);
	rgba_t& operator=(uint8_t v);
	
	rgba_t() : red(0), green(0), blue(0), alpha(0) {} ;
	rgba_t(char r, char g, char b, char a = 0) : 
		red(r), green(g), blue(b), alpha(a) {} ;
	explicit rgba_t(float);
	explicit rgba_t(double);
	explicit rgba_t(long double);
	explicit rgba_t(int64_t);
	explicit rgba_t(uint64_t);
	explicit rgba_t(int32_t);
	explicit rgba_t(uint32_t);
	explicit rgba_t(int16_t);
	explicit rgba_t(uint16_t);
	explicit rgba_t(int8_t);
	explicit rgba_t(uint8_t);
	explicit rgba_t(cquad_t);
	explicit rgba_t(cdouble_t);
	explicit rgba_t(cfloat_t);
	explicit rgba_t(rgb_t);
};

std::ostream& operator<< (std::ostream& stream, const rgb_t& v)
{
	stream << "(" <<v.red << "," << v.green << "," << v.blue << ")";
	return stream;
}

std::ostream& operator<< (std::ostream& stream, const rgba_t& v)
{
	stream << "(" <<v.red << "," << v.green << "," << v.blue << "," 
		<< v.alpha << ")";
	return stream;
}


// math

// cdouble - double
cdouble_t operator+(const cdouble_t& lhs, const double& rhs);
cdouble_t operator+(const double& lhs, const cdouble_t& rhs);
cdouble_t operator-(const cdouble_t& lhs, const double& rhs);
cdouble_t operator-(const double& lhs, const cdouble_t& rhs);
cdouble_t operator*(const cdouble_t& lhs, const double& rhs);
cdouble_t operator*(const double& lhs, const cdouble_t& rhs);
cdouble_t operator/(const cdouble_t& lhs, const double& rhs);
cdouble_t operator/(const double& lhs, const cdouble_t& rhs);

// cfloat - double
cfloat_t operator+(const cfloat_t& lhs, const double& rhs);
cfloat_t operator+(const double& lhs, const cfloat_t& rhs);
cfloat_t operator-(const cfloat_t& lhs, const double& rhs);
cfloat_t operator-(const double& lhs, const cfloat_t& rhs);
cfloat_t operator*(const cfloat_t& lhs, const double& rhs);
cfloat_t operator*(const double& lhs, const cfloat_t& rhs);
cfloat_t operator/(const cfloat_t& lhs, const double& rhs);
cfloat_t operator/(const double& lhs, const cfloat_t& rhs);

// rgb - double
rgb_t operator+(const rgb_t& lhs, const double& rhs);
rgb_t operator+(const double& lhs, const rgb_t& rhs);
rgb_t operator-(const rgb_t& lhs, const double& rhs);
rgb_t operator-(const double& lhs, const rgb_t& rhs);
rgb_t operator*(const rgb_t& lhs, const double& rhs);
rgb_t operator*(const double& lhs, const rgb_t& rhs);
rgb_t operator/(const rgb_t& lhs, const double& rhs);
rgb_t operator/(const double& lhs, const rgb_t& rhs);

// rgba - double
rgba_t operator+(const rgba_t& lhs, const double& rhs);
rgba_t operator+(const double& lhs, const rgba_t& rhs);
rgba_t operator-(const rgba_t& lhs, const double& rhs);
rgba_t operator-(const double& lhs, const rgba_t& rhs);
rgba_t operator*(const rgba_t& lhs, const double& rhs);
rgba_t operator*(const double& lhs, const rgba_t& rhs);
rgba_t operator/(const rgba_t& lhs, const double& rhs);
rgba_t operator/(const double& lhs, const rgba_t& rhs);

//cdouble_t operator*(const cfloat_t& lhs, const double& rhs)
//{
//	return cdouble_t(lhs.real()*rhs, lhs.imag()*rhs);
//}
//
//cdouble_t operator*(const cdouble_t& lhs, const double& rhs)
//{
//	return cdouble_t(lhs.real()*rhs, lhs.imag()*rhs);
//}
//
//cdouble_t operator*(const cdouble_t& lhs, const rgba_t& rhs)
//{
//	double v = (double)rhs;
//	return cdouble_t(lhs.real()*v, lhs.imag()*v);
//}

} //npl

#endif //NPLTYPES_H
