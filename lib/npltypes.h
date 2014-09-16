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
 * @file npltypes.h
 *
 *****************************************************************************/

#ifndef NPLTYPES_H
#define NPLTYPES_H

#include <Eigen/Dense>
#include <complex>
#include <memory>

namespace npl {

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Matrix;

/**
 * @brief Make the shared_ptr name shorter...
 *
 * @tparam T Type we are pointing at
 */
template <typename T>
using ptr = std::shared_ptr<T>;

/**
 * @brief Shorter name for dynamic_pointer_cast 
 *
 * @tparam F Cast from this type
 * @tparam T Cast to this type
 * @param in Input pointer (type F*)
 *
 * @return Output pointer (type T*)
 */
template <typename T, typename F>
ptr<T> dptrcast(const ptr<F>& in)
{
    return dptrcast<T>(in);
}

struct rgba_t;
struct rgb_t;

struct cdouble_t;
struct cfloat_t ;
struct cquad_t ;

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
//	explicit operator std::complex<float>();
//	explicit operator std::complex<long double>();
//
	cdouble_t(double re = 0, double im = 0) : std::complex<double>(re, im) {} ;
	explicit cdouble_t(const cfloat_t&);
	explicit cdouble_t(const cquad_t&);
	explicit cdouble_t(const std::complex<float>& );
	explicit cdouble_t(const std::complex<long double>&);
	cdouble_t(const std::complex<double>&);
//	explicit cdouble_t(rgba_t);
//	explicit cdouble_t(rgb_t);
//	explicit cdouble_t(const unsigned char& c) : std::complex<double>((double)c) {};
//	explicit cdouble_t(const char& c) : std::complex<double>((double)c) {};
	
	cdouble_t& operator=(const std::complex<double>& v);
	cdouble_t& operator=(const std::complex<float>& v);
	cdouble_t& operator=(const std::complex<long double>& v);
	cdouble_t& operator=(const cfloat_t& v);
	cdouble_t& operator=(const cquad_t& v);
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
//	explicit operator std::complex<double>();
//	explicit operator std::complex<long double>();
//
	cfloat_t(double re = 0, double im = 0) : std::complex<float>(re, im) {} ;
	explicit cfloat_t(const cdouble_t&);
	explicit cfloat_t(const cquad_t&);
	explicit cfloat_t(const std::complex<double>&);
	explicit cfloat_t(const std::complex<long double>&);
	cfloat_t(const std::complex<float>&);
//	explicit cfloat_t(rgba_t);
//	explicit cfloat_t(rgb_t);
//	explicit cfloat_t(const unsigned char& c) : std::complex<float>((float)c) {};
//	explicit cfloat_t(const char& c) : std::complex<float>((float)c) {};

	cfloat_t& operator=(const std::complex<double>& v);
	cfloat_t& operator=(const std::complex<float>& v);
	cfloat_t& operator=(const std::complex<long double>& v);
	cfloat_t& operator=(const cdouble_t& v);
	cfloat_t& operator=(const cquad_t& v);

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
//	explicit operator std::complex<float>();
//	explicit operator std::complex<double>();
//	
	cquad_t(double re = 0, double im = 0) : std::complex<long double>(re, im) {} ;
	explicit cquad_t(const cdouble_t&);
	explicit cquad_t(const cfloat_t&);
	explicit cquad_t(const std::complex<float>&);
	explicit cquad_t(const std::complex<double>&);
	cquad_t(const std::complex<long double>&);
//	explicit cquad_t(rgba_t);
//	explicit cquad_t(rgb_t);
//	
//	explicit cquad_t(const unsigned char& c) : std::complex<long double>((long double)c) {};
//	explicit cquad_t(const char& c) : std::complex<long double>((long double)c) {};
	
	cquad_t& operator=(const std::complex<double>& v);
	cquad_t& operator=(const std::complex<float>& v);
	cquad_t& operator=(const std::complex<long double>& v);
	cquad_t& operator=(const cdouble_t& v);
	cquad_t& operator=(const cfloat_t& v);
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
	explicit operator cquad_t() const;
	explicit operator cdouble_t() const;
	explicit operator cfloat_t() const;

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
	explicit operator cquad_t() const;
	explicit operator cdouble_t() const;
	explicit operator cfloat_t() const;
	
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


std::ostream& operator<< (std::ostream& stream, const rgb_t& v);
std::ostream& operator<< (std::ostream& stream, const rgba_t& v);

// math

//// cdouble - double
//cdouble_t operator+(const cdouble_t& lhs, const double& rhs);
//cdouble_t operator+(const double& lhs, const cdouble_t& rhs);
//cdouble_t operator-(const cdouble_t& lhs, const double& rhs);
//cdouble_t operator-(const double& lhs, const cdouble_t& rhs);
//cdouble_t operator*(const cdouble_t& lhs, const double& rhs);
//cdouble_t operator*(const double& lhs, const cdouble_t& rhs);
//cdouble_t operator/(const cdouble_t& lhs, const double& rhs);
//cdouble_t operator/(const double& lhs, const cdouble_t& rhs);
//
//// cfloat - double
//cfloat_t operator+(const cfloat_t& lhs, const double& rhs);
//cfloat_t operator+(const double& lhs, const cfloat_t& rhs);
//cfloat_t operator-(const cfloat_t& lhs, const double& rhs);
//cfloat_t operator-(const double& lhs, const cfloat_t& rhs);
//cfloat_t operator*(const cfloat_t& lhs, const double& rhs);
//cfloat_t operator*(const double& lhs, const cfloat_t& rhs);
//cfloat_t operator/(const cfloat_t& lhs, const double& rhs);
//cfloat_t operator/(const double& lhs, const cfloat_t& rhs);

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
