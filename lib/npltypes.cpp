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

#include "npltypes.h"

namespace npl {

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
T wrap(T inf, T sup, T v)
{
	T len = sup-inf+1;
	T vtmp = v-inf;
	T out = vtmp < 0 ? sup-((-vtmp-1)%len) : inf+(vtmp%len); 
	return out;
}



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


//////////////////////////////////////////////////////
// cdouble_t
//////////////////////////////////////////////////////
cdouble_t::operator float() const {
	return norm(*this);
};

cdouble_t::operator double() const {
	return norm(*this);
};

cdouble_t::operator long double() const {
	return norm(*this);
};

cdouble_t::operator int64_t() const {
	return (int64_t)norm(*this);
};

cdouble_t::operator uint64_t() const {
	return (uint64_t)norm(*this);
};

cdouble_t::operator int32_t() const {
	return (int32_t)norm(*this);
};

cdouble_t::operator uint32_t() const {
	return (uint32_t)norm(*this);
};

cdouble_t::operator int16_t() const {
	return (int16_t)norm(*this);
};

cdouble_t::operator uint16_t() const {
	return (uint16_t)norm(*this);
};

cdouble_t::operator int8_t() const {
	return (int8_t)norm(*this);
};

cdouble_t::operator uint8_t() const {
	return (uint8_t)norm(*this);
};

cdouble_t::operator std::complex<float>() {
	return std::complex<float>(real(), imag());
};

//cdouble_t::cdouble_t(double v) : std::complex<double>(v) {} 
//cdouble_t::cdouble_t(float v) : std::complex<double>(v) {} 
//cdouble_t::cdouble_t(int64_t v) : std::complex<double>(v) {} 
cdouble_t::cdouble_t(cfloat_t v): std::complex<double>(v) {} 
cdouble_t::cdouble_t(cquad_t v): std::complex<double>(v) {} 
cdouble_t::cdouble_t(std::complex<float> v): std::complex<double>(v) {} 
cdouble_t::cdouble_t(std::complex<long double> v): std::complex<double>(v) {} 
cdouble_t::cdouble_t(rgba_t v): std::complex<double>((double)v) {} 
cdouble_t::cdouble_t(rgb_t v): std::complex<double>((double)v) {} 
	
//////////////////////////////////////////////////////
// cfloat_t
//////////////////////////////////////////////////////
cfloat_t::operator float() const {
	return norm(*this);
};

cfloat_t::operator double() const {
	return norm(*this);
};

cfloat_t::operator long double() const {
	return norm(*this);
};

cfloat_t::operator int64_t() const {
	return (int64_t)norm(*this);
};

cfloat_t::operator uint64_t() const {
	return (uint64_t)norm(*this);
};

cfloat_t::operator int32_t() const {
	return (int32_t)norm(*this);
};

cfloat_t::operator uint32_t() const {
	return (uint32_t)norm(*this);
};

cfloat_t::operator int16_t() const {
	return (int16_t)norm(*this);
};

cfloat_t::operator uint16_t() const {
	return (uint16_t)norm(*this);
};

cfloat_t::operator int8_t() const {
	return (int8_t)norm(*this);
};

cfloat_t::operator uint8_t() const {
	return (uint8_t)norm(*this);
};

cfloat_t::operator std::complex<double>() {
	return std::complex<double>(real(), imag());
};

cfloat_t::cfloat_t(cdouble_t v): std::complex<float>(v) {} 
cfloat_t::cfloat_t(cquad_t v): std::complex<float>(v) {} 
cfloat_t::cfloat_t(std::complex<double> v): std::complex<float>(v) {} 
cfloat_t::cfloat_t(std::complex<long double> v): std::complex<float>(v) {} 
cfloat_t::cfloat_t(rgba_t v) : std::complex<float>((float)v) {} 
cfloat_t::cfloat_t(rgb_t v) : std::complex<float>((float)v) {} 

//////////////////////////////////////////////////////
// cquad_t
//////////////////////////////////////////////////////
cquad_t::operator float() const {
	return norm(*this);
};

cquad_t::operator double() const {
	return norm(*this);
};

cquad_t::operator long double() const {
	return norm(*this);
};

cquad_t::operator int64_t() const {
	return (int64_t)norm(*this);
};

cquad_t::operator uint64_t() const {
	return (uint64_t)norm(*this);
};

cquad_t::operator int32_t() const {
	return (int32_t)norm(*this);
};

cquad_t::operator uint32_t() const {
	return (uint32_t)norm(*this);
};

cquad_t::operator int16_t() const {
	return (int16_t)norm(*this);
};

cquad_t::operator uint16_t() const {
	return (uint16_t)norm(*this);
};

cquad_t::operator int8_t() const {
	return (int8_t)norm(*this);
};

cquad_t::operator uint8_t() const {
	return (uint8_t)norm(*this);
};

cquad_t::operator std::complex<double>() {
	return std::complex<double>(real(), imag());
};

//cquad_t::cquad_t(double v) : std::complex<float>(v) {} 
//cquad_t::cquad_t(float v) : std::complex<float>(v) {} 
//cquad_t::cquad_t(int64_t v) : std::complex<float>(v) {} 
cquad_t::cquad_t(cdouble_t v): std::complex<long double>(v) {} 
cquad_t::cquad_t(cfloat_t v): std::complex<long double>(v) {} 
cquad_t::cquad_t(std::complex<double> v): std::complex<long double>(v) {} 
cquad_t::cquad_t(std::complex<float> v): std::complex<long double>(v) {} 
cquad_t::cquad_t(rgba_t v) : std::complex<long double>((long double)v) {} 
cquad_t::cquad_t(rgb_t v) : std::complex<long double>((long double)v) {} 

//////////////////////////////////////////////////////
/// rgb_t
//////////////////////////////////////////////////////
rgb_t::operator float() const {
	return ((fabs(red)+fabs(green)+fabs(blue))); 
};

rgb_t::operator double() const {
	return ((fabs(red)+fabs(green)+fabs(blue))); 
};

rgb_t::operator long double() const {
	return ((fabs(red)+fabs(green)+fabs(blue))); 
};

rgb_t::operator int64_t() const {
	return (int64_t)((fabs(red)+fabs(green)+fabs(blue))); 
};

rgb_t::operator uint64_t() const {
	return (uint64_t)((fabs(red)+fabs(green)+fabs(blue))); 
};

rgb_t::operator int32_t() const {
	return (int32_t)((fabs(red)+fabs(green)+fabs(blue))); 
};

rgb_t::operator uint32_t() const {
	return (uint32_t)((fabs(red)+fabs(green)+fabs(blue))); 
};

rgb_t::operator int16_t() const {
	return (int16_t)((fabs(red)+fabs(green)+fabs(blue))); 
};

rgb_t::operator uint16_t() const {
	return (uint16_t)((fabs(red)+fabs(green)+fabs(blue))); 
};

rgb_t::operator int8_t() const {
	return (int8_t)((fabs(red)+fabs(green)+fabs(blue))); 
};

rgb_t::operator uint8_t() const {
	return (uint8_t)((fabs(red)+fabs(green)+fabs(blue))); 
};

//rgb_t::operator cdouble_t() const {
//	return (fabs(alpha)*(fabs(red)+fabs(green)+fabs(blue))); 
//};
//
//rgb_t::operator cfloat_t() const {
//	return (cfloat_t)(fabs(alpha)*(fabs(red)+fabs(green)+fabs(blue))); 
//};

rgb_t::rgb_t(long double v) : 
	red(clamp<double>(0,255, v)), 
	green(clamp<double>(0,255, v)), 
	blue(clamp<double>(0,255, v))
{
}

rgb_t::rgb_t(double v) : 
	red(clamp<double>(0,255, v)), 
	green(clamp<double>(0,255, v)), 
	blue(clamp<double>(0,255, v))
{
}
	
rgb_t::rgb_t(float v) : 
	red(clamp<float>(0,255, v)), 
	green(clamp<float>(0,255, v)), 
	blue(clamp<float>(0,255, v))
{
}

rgb_t::rgb_t(int64_t v) : 
	red(clamp<int64_t>(0,255, v)), 
	green(clamp<int64_t>(0,255, v)), 
	blue(clamp<int64_t>(0,255, v))
{
}

rgb_t::rgb_t(uint64_t v) : 
	red(clamp<int64_t>(0,255, v)), 
	green(clamp<int64_t>(0,255, v)), 
	blue(clamp<int64_t>(0,255, v))
{
}

rgb_t::rgb_t(int32_t v) : 
	red(clamp<int32_t>(0,255, v)), 
	green(clamp<int32_t>(0,255, v)), 
	blue(clamp<int32_t>(0,255, v))
{
}

rgb_t::rgb_t(uint32_t v) : 
	red(clamp<int32_t>(0,255, v)), 
	green(clamp<int32_t>(0,255, v)), 
	blue(clamp<int32_t>(0,255, v))
{
}

rgb_t::rgb_t(int16_t v) : 
	red(clamp<int16_t>(0,255, v)), 
	green(clamp<int16_t>(0,255, v)), 
	blue(clamp<int16_t>(0,255, v))
{
}

rgb_t::rgb_t(uint16_t v) : 
	red(clamp<int16_t>(0,255, v)), 
	green(clamp<int16_t>(0,255, v)), 
	blue(clamp<int16_t>(0,255, v))
{
}

rgb_t::rgb_t(int8_t v) : 
	red(clamp<int8_t>(0,255, v)), 
	green(clamp<int8_t>(0,255, v)), 
	blue(clamp<int8_t>(0,255, v))
{
}

rgb_t::rgb_t(uint8_t v) : 
	red(clamp<int8_t>(0,255, v)), 
	green(clamp<int8_t>(0,255, v)), 
	blue(clamp<int8_t>(0,255, v))
{
}

rgb_t::rgb_t(cfloat_t v) : 
	red(clamp<double>(0,255, (double)v)), 
	green(clamp<double>(0,255, (double)v)), 
	blue(clamp<double>(0,255, (double)v))
{
}

rgb_t::rgb_t(cdouble_t v) : 
	red(clamp<double>(0,255, (double)v)), 
	green(clamp<double>(0,255, (double)v)),
	blue(clamp<double>(0,255, (double)v))
{
}

rgb_t::rgb_t(cquad_t v) : 
	red(clamp<double>(0,255, (double)v)), 
	green(clamp<double>(0,255, (double)v)),
	blue(clamp<double>(0,255, (double)v))
{
}

rgb_t::rgb_t(rgba_t v) : 
	red(v.red), green(v.green), blue(v.blue)
{
}

rgb_t& rgb_t::operator=(float v)
{
	*this = (rgb_t)v;
	return *this;
}

rgb_t& rgb_t::operator=(double v)
{
	*this = (rgb_t)v;
	return *this;
}

rgb_t& rgb_t::operator=(long double v)
{
	*this = (rgb_t)v;
	return *this;
}

rgb_t& rgb_t::operator=(int64_t v)
{
	*this = (rgb_t)v;
	return *this;
}

rgb_t& rgb_t::operator=(uint64_t v)
{
	*this = (rgb_t)v;
	return *this;
}

rgb_t& rgb_t::operator=(int32_t v)
{
	*this = (rgb_t)v;
	return *this;
}

rgb_t& rgb_t::operator=(uint32_t v)
{
	*this = (rgb_t)v;
	return *this;
}

rgb_t& rgb_t::operator=(int16_t v)
{
	*this = (rgb_t)v;
	return *this;
}

rgb_t& rgb_t::operator=(uint16_t v)
{
	*this = (rgb_t)v;
	return *this;
}

rgb_t& rgb_t::operator=(int8_t v)
{
	*this = (rgb_t)v;
	return *this;
}

rgb_t& rgb_t::operator=(uint8_t v)
{
	*this = (rgb_t)v;
	return *this;
}


//////////////////////////////////////////////////////
/// rgba_t
//////////////////////////////////////////////////////
rgba_t::operator float() const {
	return fabs(alpha)*((fabs(red)+fabs(green)+fabs(blue))); 
};

rgba_t::operator double() const {
	return fabs(alpha)*((fabs(red)+fabs(green)+fabs(blue))); 
};

rgba_t::operator long double() const {
	return fabs(alpha)*((fabs(red)+fabs(green)+fabs(blue))); 
};

rgba_t::operator int64_t() const {
	return (int64_t)(fabs(alpha)*(fabs(red)+fabs(green)+fabs(blue))); 
};

rgba_t::operator uint64_t() const {
	return (uint64_t)(fabs(alpha)*(fabs(red)+fabs(green)+fabs(blue))); 
};

rgba_t::operator int32_t() const {
	return (int32_t)(fabs(alpha)*(fabs(red)+fabs(green)+fabs(blue))); 
};

rgba_t::operator uint32_t() const {
	return (uint32_t)(fabs(alpha)*(fabs(red)+fabs(green)+fabs(blue))); 
};

rgba_t::operator int16_t() const {
	return (int16_t)(fabs(alpha)*(fabs(red)+fabs(green)+fabs(blue))); 
};

rgba_t::operator uint16_t() const {
	return (uint16_t)(fabs(alpha)*(fabs(red)+fabs(green)+fabs(blue))); 
};

rgba_t::operator int8_t() const {
	return (int8_t)(fabs(alpha)*(fabs(red)+fabs(green)+fabs(blue))); 
};

rgba_t::operator uint8_t() const {
	return (uint8_t)(fabs(alpha)*(fabs(red)+fabs(green)+fabs(blue))); 
};

//rgba_t::operator cdouble_t() const {
//	return (fabs(alpha)*(fabs(red)+fabs(green)+fabs(blue))); 
//};
//
//rgba_t::operator cfloat_t() const {
//	return (cfloat_t)(fabs(alpha)*(fabs(red)+fabs(green)+fabs(blue))); 
//};

rgba_t::rgba_t(long double v) : 
	red(clamp<double>(0,255, v)), 
	green(clamp<double>(0,255, v)), 
	blue(clamp<double>(0,255, v)), 
	alpha(clamp<double>(0,255, v)) 
{
}

rgba_t::rgba_t(double v) : 
	red(clamp<double>(0,255, v)), 
	green(clamp<double>(0,255, v)), 
	blue(clamp<double>(0,255, v)), 
	alpha(clamp<double>(0,255, v)) 
{
}
	
rgba_t::rgba_t(float v) : 
	red(clamp<float>(0,255, v)), 
	green(clamp<float>(0,255, v)), 
	blue(clamp<float>(0,255, v)), 
	alpha(clamp<float>(0,255, v)) 
{
}

rgba_t::rgba_t(int64_t v) : 
	red(clamp<int64_t>(0,255, v)), 
	green(clamp<int64_t>(0,255, v)), 
	blue(clamp<int64_t>(0,255, v)), 
	alpha(clamp<int64_t>(0,255, v)) 
{
}

rgba_t::rgba_t(uint64_t v) : 
	red(clamp<int64_t>(0,255, v)), 
	green(clamp<int64_t>(0,255, v)), 
	blue(clamp<int64_t>(0,255, v)), 
	alpha(clamp<int64_t>(0,255, v)) 
{
}

rgba_t::rgba_t(int32_t v) : 
	red(clamp<int32_t>(0,255, v)), 
	green(clamp<int32_t>(0,255, v)), 
	blue(clamp<int32_t>(0,255, v)), 
	alpha(clamp<int32_t>(0,255, v)) 
{
}

rgba_t::rgba_t(uint32_t v) : 
	red(clamp<int32_t>(0,255, v)), 
	green(clamp<int32_t>(0,255, v)), 
	blue(clamp<int32_t>(0,255, v)), 
	alpha(clamp<int32_t>(0,255, v)) 
{
}

rgba_t::rgba_t(int16_t v) : 
	red(clamp<int16_t>(0,255, v)), 
	green(clamp<int16_t>(0,255, v)), 
	blue(clamp<int16_t>(0,255, v)), 
	alpha(clamp<int16_t>(0,255, v)) 
{
}

rgba_t::rgba_t(uint16_t v) : 
	red(clamp<int16_t>(0,255, v)), 
	green(clamp<int16_t>(0,255, v)), 
	blue(clamp<int16_t>(0,255, v)), 
	alpha(clamp<int16_t>(0,255, v)) 
{
}

rgba_t::rgba_t(int8_t v) : 
	red(clamp<int8_t>(0,255, v)), 
	green(clamp<int8_t>(0,255, v)), 
	blue(clamp<int8_t>(0,255, v)), 
	alpha(clamp<int8_t>(0,255, v)) 
{
}

rgba_t::rgba_t(uint8_t v) : 
	red(clamp<int8_t>(0,255, v)), 
	green(clamp<int8_t>(0,255, v)), 
	blue(clamp<int8_t>(0,255, v)), 
	alpha(clamp<int8_t>(0,255, v)) 
{
}

rgba_t::rgba_t(cfloat_t v) : 
	red(clamp<double>(0,255, (double)v)), 
	green(clamp<double>(0,255, (double)v)), 
	blue(clamp<double>(0,255, (double)v)), 
	alpha(clamp<double>(0,255, (double)v)) 
{
}

rgba_t::rgba_t(cdouble_t v) : 
	red(clamp<double>(0,255, (double)v)), 
	green(clamp<double>(0,255, (double)v)),
	blue(clamp<double>(0,255, (double)v)), 
	alpha(clamp<double>(0,255, (double)v)) 
{
}

rgba_t::rgba_t(cquad_t v) : 
	red(clamp<double>(0,255, (double)v)), 
	green(clamp<double>(0,255, (double)v)),
	blue(clamp<double>(0,255, (double)v)), 
	alpha(clamp<double>(0,255, (double)v)) 
{
}

rgba_t::rgba_t(rgb_t v) : 
	red(v.red), green(v.green), blue(v.blue), alpha(255)
{
}

rgba_t& rgba_t::operator=(float v)
{
	*this = (rgba_t)v;
	return *this;
}

rgba_t& rgba_t::operator=(double v)
{
	*this = (rgba_t)v;
	return *this;
}

rgba_t& rgba_t::operator=(long double v)
{
	*this = (rgba_t)v;
	return *this;
}

rgba_t& rgba_t::operator=(int64_t v)
{
	*this = (rgba_t)v;
	return *this;
}

rgba_t& rgba_t::operator=(uint64_t v)
{
	*this = (rgba_t)v;
	return *this;
}

rgba_t& rgba_t::operator=(int32_t v)
{
	*this = (rgba_t)v;
	return *this;
}

rgba_t& rgba_t::operator=(uint32_t v)
{
	*this = (rgba_t)v;
	return *this;
}

rgba_t& rgba_t::operator=(int16_t v)
{
	*this = (rgba_t)v;
	return *this;
}

rgba_t& rgba_t::operator=(uint16_t v)
{
	*this = (rgba_t)v;
	return *this;
}

rgba_t& rgba_t::operator=(int8_t v)
{
	*this = (rgba_t)v;
	return *this;
}

rgba_t& rgba_t::operator=(uint8_t v)
{
	*this = (rgba_t)v;
	return *this;
}

} //npl
