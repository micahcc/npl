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
	explicit operator float();
	explicit operator double();
	explicit operator long double();
	explicit operator int64_t();
	explicit operator uint64_t();
	explicit operator int32_t();
	explicit operator uint32_t();
	explicit operator int16_t();
	explicit operator uint16_t();
	explicit operator int8_t();
	explicit operator uint8_t();
//	explicit operator cfloat_t();
//	explicit operator cquad_t();
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
	explicit operator float();
	explicit operator double();
	explicit operator long double();
	explicit operator int64_t();
	explicit operator uint64_t();
	explicit operator int32_t();
	explicit operator uint32_t();
	explicit operator int16_t();
	explicit operator uint16_t();
	explicit operator int8_t();
	explicit operator uint8_t();
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
	explicit operator float();
	explicit operator double();
	explicit operator long double();
	explicit operator int64_t();
	explicit operator uint64_t();
	explicit operator int32_t();
	explicit operator uint32_t();
	explicit operator int16_t();
	explicit operator uint16_t();
	explicit operator int8_t();
	explicit operator uint8_t();
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

	explicit operator float();
	explicit operator double();
	explicit operator long double();
	explicit operator int64_t();
	explicit operator uint64_t();
	explicit operator int32_t();
	explicit operator uint32_t();
	explicit operator int16_t();
	explicit operator uint16_t();
	explicit operator int8_t();
	explicit operator uint8_t();
	
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

	explicit operator float();
	explicit operator double();
	explicit operator long double();
	explicit operator int64_t();
	explicit operator uint64_t();
	explicit operator int32_t();
	explicit operator uint32_t();
	explicit operator int16_t();
	explicit operator uint16_t();
	explicit operator int8_t();
	explicit operator uint8_t();
	
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

} //npl

#endif //NPLTYPES_H
