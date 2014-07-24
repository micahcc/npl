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

#ifndef ACCESSORS_H
#define ACCESSORS_H

#include "ndarray.h"
#include "mrimage.h"
#include "basic_functions.h"

namespace npl {

/**
 * @brief This is a basic accessor class, which allows for accessing
 * array data in the type specified by the template.
 *
 * @tparam T Value to return on access
 */
template<typename T>
class NDAccess
{
public:
	NDAccess(std::shared_ptr<NDArray> in) : parent(in)
	{
		switch(in->type()) {
			case UINT8:
				castget = castgetStatic<uint8_t>;
				castset = castsetStatic<uint8_t>;
				break;
			case INT8:
				castget = castgetStatic<int8_t>;
				castset = castsetStatic<int8_t>;
				break;
			case UINT16:
				castget = castgetStatic<uint16_t>;
				castset = castsetStatic<uint16_t>;
				break;
			case INT16:
				castget = castgetStatic<int16_t>;
				castset = castsetStatic<int16_t>;
				break;
			case UINT32:
				castget = castgetStatic<uint32_t>;
				castset = castsetStatic<uint32_t>;
				break;
			case INT32:
				castget = castgetStatic<int32_t>;
				castset = castsetStatic<int32_t>;
				break;
			case UINT64:
				castget = castgetStatic<uint64_t>;
				castset = castsetStatic<uint64_t>;
				break;
			case INT64:
				castget = castgetStatic<int64_t>;
				castset = castsetStatic<int64_t>;
				break;
			case FLOAT32:
				castget = castgetStatic<float>;
				castset = castsetStatic<float>;
				break;
			case FLOAT64:
				castget = castgetStatic<double>;
				castset = castsetStatic<double>;
				break;
			case FLOAT128:
				castget = castgetStatic<long double>;
				castset = castsetStatic<long double>;
				break;
			case COMPLEX64:
				castget = castgetStatic<cfloat_t>;
				castset = castsetStatic<cfloat_t>;
				break;
			case COMPLEX128:
				castget = castgetStatic<cdouble_t>;
				castset = castsetStatic<cdouble_t>;
				break;
			case COMPLEX256:
				castget = castgetStatic<cquad_t>;
				castset = castsetStatic<cquad_t>;
				break;
			case RGB24:
				castget = castgetStatic<rgb_t>;
				castset = castsetStatic<rgb_t>;
				break;
			case RGBA32:
				castget = castgetStatic<rgba_t>;
				castset = castsetStatic<rgba_t>;
				break;
			default:
			case UNKNOWN_TYPE:
				castget = castgetStatic<uint8_t>;
				castset = castsetStatic<uint8_t>;
				throw std::invalid_argument("Unknown type to NDAccess");
				break;
		}
	};
	
	/**
	 * @brief Gets value linear position in array, then casts to T
	 *
	 * @return value
	 */
	T operator[](int64_t index)
	{
		return castget(this->parent->__getAddr(index)); 
	};
	
	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @return value
	 */
	T get(const std::vector<int64_t>& index)
	{
		return castget(this->parent->__getAddr(index)); 
	};
	
	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @return value
	 */
	T operator[](const std::vector<int64_t>& index)
	{
		return castget(this->parent->__getAddr(index)); 
	};
	
	/**
	 * @brief Casts to the appropriate type then sets array at given index.
	 *
	 * @return current value
	 */
	void set(T v, const std::vector<int64_t>& index)
	{
		return castset(this->parent->__getAddr(index), v); 
	};
	
	/**
	 * @brief Casts to the appropriate type then sets array at given index.
	 *
	 * @return current value
	 */
	void set(T v, int64_t index)
	{
		return castset(this->parent->__getAddr(index), v); 
	};
	
	int64_t tlen() { return this->parent->tlen(); };
	
protected:
	

	/**
	 * @brief This is a wrapper function that will be called to safely cast 
	 * from the underlying type.
	 *
	 * @tparam U Underlying type of pixel, figured out in the constructor
	 * @param ptr Pointer to memory where the pixel is.
	 *
	 * @return Correctly cast value
	 */
	template <typename U>
	static T castgetStatic(void* ptr)
	{
		return (T)(*(static_cast<U*>(ptr)));
	};
	
	/**
	 * @brief This is a wrapper function that will be called to safely cast 
	 * to the underlying type.
	 *
	 * @tparam U Underlying type of pixel, figured out in the constructor
	 * @param ptr Pointer to memory where the pixel is.
	 * @param val new value to write
	 *
	 */
	template <typename U>
	static void castsetStatic(void* ptr, const T& val)
	{
		(*(static_cast<U*>(ptr))) = (U)val;
	};

	/**
	 * @brief Where to get the dat a from. Also the shared_ptr prevents dealloc
	 */
	std::shared_ptr<NDArray> parent;

	/**
	 * @brief Function pointer to the correct function for casting from the 
	 * underlying type
	 *
	 * @param ptr location in memory where the pixel is stored
	 */
	T (*castget)(void* ptr);

	/**
	 * @brief Function pointer to the correct function for casting to the 
	 * underlying type. This should be set during construction.
	 *
	 *
	 * @param ptr pointer to memory 
	 * @param v value to cast and write
	 */
	void (*castset)(void* ptr, const T& v);

};

/**
 * @brief The purpose of this class is to view an image as a 3D 
 * image rather than a ND image. Therefore all dimensions above the third will
 * be ignored and index 0 will be used. 
 *
 * @tparam T Type of value to cast and return
 */
template<typename T>
class Pixel3DView : public NDAccess<T>
{
public:
	Pixel3DView(std::shared_ptr<NDArray> in) : NDAccess<T>(in) 
	{ };

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @return value
	 */
	T operator()(int64_t x=0, int64_t y=0, int64_t z=0)
	{
		return this->castget(this->parent->__getAddr(x,y,z,0)); 
	};
	
	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @return value
	 */
	T get(int64_t x=0, int64_t y=0, int64_t z=0)
	{
		return this->castget(this->parent->__getAddr(x,y,z,0)); 
	};
	
	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @return value
	 */
	void set(T v, int64_t x=0, int64_t y=0, int64_t z=0)
	{
		this->castset(this->parent->__getAddr(x,y,z,0), v); 
	};
	
protected:
	
	// Remove functions that aren't relevent from NDAccess
	T operator[](int64_t i) { (void)(i); return T(); };
	T get(const std::vector<int64_t>& i) {  (void)(i); return T(); };
	T operator[](const std::vector<int64_t>& i) { (void)(i); return T(); };
};


/**
 * @brief The purpose of this class is to view an image as a 3D+vector dimension
 * image rather than a 4+D image. Therefore all dimensions above the third are 
 * cast as a vector. If there is demand I may create a matrixx verion as well
 *
 * @tparam T Type of value to cast and return
 */
template<typename T>
class Vector3DView : public NDAccess<T>
{
public:
	Vector3DView(std::shared_ptr<NDArray> in) : NDAccess<T>(in) 
	{ };

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @return value
	 */
	T operator()(int64_t x=0, int64_t y=0, int64_t z=0, int64_t t=0)
	{
		return this->castget(this->parent->__getAddr(x,y,z,t)); 
	};
	
	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @return value
	 */
	T get(int64_t x=0, int64_t y=0, int64_t z=0, int64_t t=0)
	{
		return this->castget(this->parent->__getAddr(x,y,z,t)); 
	};
	
	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @return value
	 */
	void set(T v, int64_t x=0, int64_t y=0, int64_t z=0, int64_t t=0)
	{
		this->castset(this->parent->__getAddr(x,y,z,t), v); 
	};
	
protected:
	
	// Remove functions that aren't relevent from NDAccess
	T operator[](int64_t i) { (void)(i); return T(); };
	T get(const std::vector<int64_t>& i) {  (void)(i); return T(); };
	T operator[](const std::vector<int64_t>& i) { (void)(i); return T(); };
};

/* Linear Kernel Sampling */
double linKern(double x)
{
	return fabs(1-fmin(1,fabs(x)));
}

/**
 * @brief The purpose of this class is to view an image as a continuous 
 * 3D+vector dimension image rather than a 4+D image. Therefore all dimensions 
 * above the third are cast as a vector and interpolation is only performed
 * between 3D points, with the 4th dimension assumed to be non-spatial. The
 * would be applicable if the upper dimensions are of a different type 
 * than the first 3. 
 *
 * @tparam T Type of value to cast and return
 */
template<typename T>
class LinInterp3DView : public Vector3DView<T>
{
public:
	LinInterp3DView(std::shared_ptr<NDArray> in, 
				BoundaryConditionT bound = ZEROFLUX) 
				: Vector3DView<T>(in), m_boundmethod(bound)
	{ };

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @return value
	 */
	T operator()(double x=0, double y=0, double z=0, int64_t t=0)
	{
		return get(x,y,z,t);
	};
	
	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @return value
	 */
	T get(double x=0, double y=0, double z=0, int64_t t=0)
	{
		// figure out size of dimensions in parent
		size_t dim[4];
		dim[0] = this->parent->dim(0);
		dim[1] = this->parent->ndim() > 1 ? this->parent->dim(1) : 1;
		dim[2] = this->parent->ndim() > 2 ? this->parent->dim(2) : 1;
		dim[3] = this->parent->tlen();
		
		// deal with t being outside bounds
		if(t < 0 || t >= dim[3]) {
			if(m_boundmethod == ZEROFLUX) {
				// clamp
				t = clamp<int64_t>(0, dim[3]-1, t);
			} else if(m_boundmethod == WRAP) {
				// wrap
				t = wrap<int64_t>(0, dim[3]-1, t);
			} else {
				return 0;
			}
		}

		// initialize variables
		double cindex[3] = {x,y,z};
		int64_t index[3];
		const int KPOINTS = 2;
		const int DIM = 3;

		// 1D version of the weights and indices
		double karray[DIM][KPOINTS];
		int64_t indarray[DIM][KPOINTS];
		
		for(int dd = 0; dd < DIM; dd++) {
			indarray[dd][0] = floor(cindex[dd]);
			indarray[dd][1] = indarray[dd][0]+1; //make sure they aren't the same
			karray[dd][0] = linKern(indarray[dd][0]-cindex[dd]);
			karray[dd][1] = linKern(indarray[dd][1]-cindex[dd]);
		}

		bool iioutside = false;
//		outside = false;

		// compute weighted pixval by iterating over neighbors, which are 
		// combinations of KPOINTS
		T pixval = 0;
		double weight = 0;
		div_t result;
		for(int ii = 0 ; ii < pow(KPOINTS, DIM); ii++) {
			weight = 1;

			//set index
			result.quot = ii;
			iioutside = false;
			for(int dd = 0; dd < DIM; dd++) {
				result = std::div(result.quot, KPOINTS);
				weight *= karray[dd][result.rem];
				index[dd] = indarray[dd][result.rem];
				iioutside = iioutside || index[dd] < 0 || index[dd] >= dim[dd];
			}

			// might prevent optimization
			//			if(weight == 0)
			//				continue;

			// if the current point maps outside, then we need to deal with it
//			outside = (weight != 0 && iioutside) || outside;
			if(iioutside) {
				if(m_boundmethod == ZEROFLUX) {
					// clamp
					for(size_t dd=0; dd<DIM; dd++)
						index[dd] = clamp<int64_t>(0, dim[dd]-1, index[dd]);
				} else if(m_boundmethod == WRAP) {
					// wrap
					for(size_t dd=0; dd<DIM; dd++) 
						wrap<int64_t>(0, dim[dd]-1, index[dd]);
				} else {
					// set wieght to zero, then just clamp
					weight = 0;
					for(size_t dd=0; dd<DIM; dd++)
						index[dd] = clamp<int64_t>(0, dim[dd]-1, index[dd]);
				}
			} 

			T v = this->castget(this->parent->__getAddr(index[0], index[1],index[2],t));
			pixval += weight*v; 
		}

		return pixval;
	}
	
	BoundaryConditionT m_boundmethod;
protected:
	
	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @return value
	 */
	// Remove functions that aren't relevent from Base
	void set(T v, int64_t x=0, int64_t y=0, int64_t z=0, int64_t t=0) 
	{
		(void)(v); (void)(x); (void)(y); (void)(z); (void)(t);
	};
	T operator[](int64_t i) { (void)(i); return T(); };
	T get(const std::vector<int64_t>& i) { (void)(i); return T(); };
	T operator[](const std::vector<int64_t>& i) { (void)(i); return T(); };
};

/**
 * @brief The purpose of this class is to view an image as a continuous 
 * 3D+vector dimension image rather than a 4+D image. Therefore all dimensions 
 * above the third are cast as a vector and interpolation is only performed
 * between 3D points, with the 4th dimension assumed to be non-spatial. The
 * would be applicable if the upper dimensions are of a different type 
 * than the first 3. 
 *
 * @tparam T Type of value to cast and return
 */
template<typename T>
class NNInterp3DView : public Vector3DView<T>
{
public:
	NNInterp3DView(std::shared_ptr<NDArray> in, 
				BoundaryConditionT bound = ZEROFLUX) 
				: Vector3DView<T>(in), m_boundmethod(bound)
	{ };

	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @return value
	 */
	T operator()(double x=0, double y=0, double z=0, int64_t t=0)
	{
		return get(x,y,z,t); 
	};
	
	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @return value
	 */
	T get(double x=0, double y=0, double z=0, int64_t t=0)
	{
		// interpolate
		int64_t i = round(x);
		int64_t j = round(y);
		int64_t k = round(z);
		size_t xdim = this->parent->dim(0);
		size_t ydim = this->parent->ndim() > 1 ? this->parent->dim(1) : 1;
		size_t zdim = this->parent->ndim() > 2 ? this->parent->dim(2) : 1;
		size_t tdim = this->parent->tlen(); 

		bool xout = (i < 0 || i >= xdim);
		bool yout = (j < 0 || j >= ydim);
		bool zout = (k < 0 || k >= zdim);
		bool tout = (t < 0 || t >= tdim);
		
		if(xout || yout || zout || tout) {
//			outside = true;
			switch(m_boundmethod) {
				case ZEROFLUX:
					i = clamp<int64_t>(0, xdim-1, i);
					j = clamp<int64_t>(0, ydim-1, j);
					k = clamp<int64_t>(0, zdim-1, k);
					t = clamp<int64_t>(0, tdim-1, t);
					break;
				case WRAP:
					i = wrap<int64_t>(0, xdim-1, i);
					j = wrap<int64_t>(0, ydim-1, j);
					k = wrap<int64_t>(0, zdim-1, k);
					t = wrap<int64_t>(0, tdim-1, t);
					break;
				case CONSTZERO:
				default:
					return 0;
					break;
			}
		}

		return this->castget(this->parent->__getAddr(i,j,k,t)); 
	};
	
	BoundaryConditionT m_boundmethod;;
private:
	


	/**
	 * @brief Gets value at array index and then casts to T
	 *
	 * @return value
	 */
	// Remove functions that aren't relevent from Base
	void set(T v, int64_t x=0, int64_t y=0, int64_t z=0, int64_t t=0) 
	{
		(void)(v); (void)(x); (void)(y); (void)(z); (void)(t);
	};
	T operator[](int64_t i) { (void)(i); return T(); };
	T get(const std::vector<int64_t>& i) { (void)(i); return T(); };
	T operator[](const std::vector<int64_t>& i) { (void)(i); return T(); };
};

} // namespace npl

#endif //ACCESSORS_H
