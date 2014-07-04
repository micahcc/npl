#include "ndarray.h"
#include "ndarray.txx"

template <typename T>
inline T clamp(T inf, T sup, T v) 
{
	return std::min(sup, std::max(inf, v));
}

// c64_t

c64_t::operator float() {
	return norm(*this);
};

c64_t::operator double() {
	return norm(*this);
};

c64_t::operator int64_t() {
	return (int64_t)norm(*this);
};

c64_t::operator uint64_t() {
	return (uint64_t)norm(*this);
};

c64_t::operator int32_t() {
	return (int32_t)norm(*this);
};

c64_t::operator uint32_t() {
	return (uint32_t)norm(*this);
};

c64_t::operator int16_t() {
	return (int16_t)norm(*this);
};

c64_t::operator uint16_t() {
	return (uint16_t)norm(*this);
};

c64_t::operator int8_t() {
	return (int8_t)norm(*this);
};

c64_t::operator uint8_t() {
	return (uint8_t)norm(*this);
};

c64_t::operator c32_t() {
	return c32_t(real(), imag());
};

c64_t::operator std::complex<float>() {
	return std::complex<float>(real(), imag());
};

//c64_t::c64_t(double v) : std::complex<double>(v) {} 
//c64_t::c64_t(float v) : std::complex<double>(v) {} 
//c64_t::c64_t(int64_t v) : std::complex<double>(v) {} 
c64_t::c64_t(c32_t v): std::complex<double>(v) {} 
c64_t::c64_t(std::complex<float> v): std::complex<double>(v) {} 
c64_t::c64_t(rgba_t v): std::complex<double>((double)v) {} 
	
// c32_t
c32_t::operator float() {
	return norm(*this);
};

c32_t::operator double() {
	return norm(*this);
};

c32_t::operator int64_t() {
	return (int64_t)norm(*this);
};

c32_t::operator uint64_t() {
	return (uint64_t)norm(*this);
};

c32_t::operator int32_t() {
	return (int32_t)norm(*this);
};

c32_t::operator uint32_t() {
	return (uint32_t)norm(*this);
};

c32_t::operator int16_t() {
	return (int16_t)norm(*this);
};

c32_t::operator uint16_t() {
	return (uint16_t)norm(*this);
};

c32_t::operator int8_t() {
	return (int8_t)norm(*this);
};

c32_t::operator uint8_t() {
	return (uint8_t)norm(*this);
};

c32_t::operator c64_t() {
	return c64_t(real(), imag());
};

c32_t::operator std::complex<double>() {
	return std::complex<double>(real(), imag());
};

//c32_t::c32_t(double v) : std::complex<float>(v) {} 
//c32_t::c32_t(float v) : std::complex<float>(v) {} 
//c32_t::c32_t(int64_t v) : std::complex<float>(v) {} 
c32_t::c32_t(c64_t v): std::complex<float>(v) {} 
c32_t::c32_t(std::complex<double> v): std::complex<float>(v) {} 
c32_t::c32_t(rgba_t v) : std::complex<float>((float)v) {} 

// rgba_t
rgba_t::operator float() {
	return fabs(alpha)*((fabs(red)+fabs(green)+fabs(blue))); 
};

rgba_t::operator double() {
	return fabs(alpha)*((fabs(red)+fabs(green)+fabs(blue))); 
};

rgba_t::operator int64_t() {
	return (int64_t)(fabs(alpha)*(fabs(red)+fabs(green)+fabs(blue))); 
};

rgba_t::operator uint64_t() {
	return (uint64_t)(fabs(alpha)*(fabs(red)+fabs(green)+fabs(blue))); 
};

rgba_t::operator int32_t() {
	return (int32_t)(fabs(alpha)*(fabs(red)+fabs(green)+fabs(blue))); 
};

rgba_t::operator uint32_t() {
	return (uint32_t)(fabs(alpha)*(fabs(red)+fabs(green)+fabs(blue))); 
};

rgba_t::operator int16_t() {
	return (int16_t)(fabs(alpha)*(fabs(red)+fabs(green)+fabs(blue))); 
};

rgba_t::operator uint16_t() {
	return (uint16_t)(fabs(alpha)*(fabs(red)+fabs(green)+fabs(blue))); 
};

rgba_t::operator int8_t() {
	return (int8_t)(fabs(alpha)*(fabs(red)+fabs(green)+fabs(blue))); 
};

rgba_t::operator uint8_t() {
	return (uint8_t)(fabs(alpha)*(fabs(red)+fabs(green)+fabs(blue))); 
};

//rgba_t::operator c64_t() {
//	return (fabs(alpha)*(fabs(red)+fabs(green)+fabs(blue))); 
//};
//
//rgba_t::operator c32_t() {
//	return (c32_t)(fabs(alpha)*(fabs(red)+fabs(green)+fabs(blue))); 
//};

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

rgba_t::rgba_t(c32_t v) : 
	red(clamp<double>(0,255, (double)v)), 
	green(clamp<double>(0,255, (double)v)), 
	blue(clamp<double>(0,255, (double)v)), 
	alpha(clamp<double>(0,255, (double)v)) 
{
}

rgba_t::rgba_t(c64_t v) : 
	red(clamp<double>(0,255, (double)v)), 
	green(clamp<double>(0,255, (double)v)),
	blue(clamp<double>(0,255, (double)v)), 
	alpha(clamp<double>(0,255, (double)v)) 
{
}
	
template class NDArrayStore<1, double>;
template class NDArrayStore<1, c64_t>;
template class NDArrayStore<1, float>;
template class NDArrayStore<1, c32_t>;
template class NDArrayStore<1, int64_t>;
template class NDArrayStore<1, uint64_t>;
template class NDArrayStore<1, int32_t>;
template class NDArrayStore<1, uint32_t>;
template class NDArrayStore<1, int16_t>;
template class NDArrayStore<1, uint16_t>;
template class NDArrayStore<1, int8_t>;
template class NDArrayStore<1, uint8_t>;
template class NDArrayStore<1, rgba_t>;

template class NDArrayStore<2, double>;
template class NDArrayStore<2, c64_t>;
template class NDArrayStore<2, float>;
template class NDArrayStore<2, c32_t>;
template class NDArrayStore<2, int64_t>;
template class NDArrayStore<2, uint64_t>;
template class NDArrayStore<2, int32_t>;
template class NDArrayStore<2, uint32_t>;
template class NDArrayStore<2, int16_t>;
template class NDArrayStore<2, uint16_t>;
template class NDArrayStore<2, int8_t>;
template class NDArrayStore<2, uint8_t>;
template class NDArrayStore<2, rgba_t>;

template class NDArrayStore<3, double>;
template class NDArrayStore<3, c64_t>;
template class NDArrayStore<3, float>;
template class NDArrayStore<3, c32_t>;
template class NDArrayStore<3, int64_t>;
template class NDArrayStore<3, uint64_t>;
template class NDArrayStore<3, int32_t>;
template class NDArrayStore<3, uint32_t>;
template class NDArrayStore<3, int16_t>;
template class NDArrayStore<3, uint16_t>;
template class NDArrayStore<3, int8_t>;
template class NDArrayStore<3, uint8_t>;
template class NDArrayStore<3, rgba_t>;

template class NDArrayStore<4, double>;
template class NDArrayStore<4, c64_t>;
template class NDArrayStore<4, float>;
template class NDArrayStore<4, c32_t>;
template class NDArrayStore<4, int64_t>;
template class NDArrayStore<4, uint64_t>;
template class NDArrayStore<4, int32_t>;
template class NDArrayStore<4, uint32_t>;
template class NDArrayStore<4, int16_t>;
template class NDArrayStore<4, uint16_t>;
template class NDArrayStore<4, int8_t>;
template class NDArrayStore<4, uint8_t>;
template class NDArrayStore<4, rgba_t>;

template class NDArrayStore<5, double>;
template class NDArrayStore<5, c64_t>;
template class NDArrayStore<5, float>;
template class NDArrayStore<5, c32_t>;
template class NDArrayStore<5, int64_t>;
template class NDArrayStore<5, uint64_t>;
template class NDArrayStore<5, int32_t>;
template class NDArrayStore<5, uint32_t>;
template class NDArrayStore<5, int16_t>;
template class NDArrayStore<5, uint16_t>;
template class NDArrayStore<5, int8_t>;
template class NDArrayStore<5, uint8_t>;
template class NDArrayStore<5, rgba_t>;

template class NDArrayStore<6, double>;
template class NDArrayStore<6, c64_t>;
template class NDArrayStore<6, float>;
template class NDArrayStore<6, c32_t>;
template class NDArrayStore<6, int64_t>;
template class NDArrayStore<6, uint64_t>;
template class NDArrayStore<6, int32_t>;
template class NDArrayStore<6, uint32_t>;
template class NDArrayStore<6, int16_t>;
template class NDArrayStore<6, uint16_t>;
template class NDArrayStore<6, int8_t>;
template class NDArrayStore<6, uint8_t>;
template class NDArrayStore<6, rgba_t>;

template class NDArrayStore<7, double>;
template class NDArrayStore<7, c64_t>;
template class NDArrayStore<7, float>;
template class NDArrayStore<7, c32_t>;
template class NDArrayStore<7, int64_t>;
template class NDArrayStore<7, uint64_t>;
template class NDArrayStore<7, int32_t>;
template class NDArrayStore<7, uint32_t>;
template class NDArrayStore<7, int16_t>;
template class NDArrayStore<7, uint16_t>;
template class NDArrayStore<7, int8_t>;
template class NDArrayStore<7, uint8_t>;
template class NDArrayStore<7, rgba_t>;

template class NDArrayStore<8, double>;
template class NDArrayStore<8, c64_t>;
template class NDArrayStore<8, float>;
template class NDArrayStore<8, c32_t>;
template class NDArrayStore<8, int64_t>;
template class NDArrayStore<8, uint64_t>;
template class NDArrayStore<8, int32_t>;
template class NDArrayStore<8, uint32_t>;
template class NDArrayStore<8, int16_t>;
template class NDArrayStore<8, uint16_t>;
template class NDArrayStore<8, int8_t>;
template class NDArrayStore<8, uint8_t>;
template class NDArrayStore<8, rgba_t>;

