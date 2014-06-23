#ifndef NDIMAGE_H
#define NDIMAGE_H

#include "ndarray.h"

#include <string>
#include <iostream>

class NDImage;

// simply reads an image in its native type
NDImage* readNDImage(std::string filename);
void writeNDImage(NDImage* img, std::string filename);

// enforces given pixel type
template <typename T>
NDImage* readNDImage(std::string filename);


/**
 * @brief NDImage can basically be used like an NDArray, with the addition
 * of orientation related additions.
 */
class NDImage : public virtual NDArray
{
};


/**
 * @brief NDImageStore is a version of NDArray that has an orientation matrix
 *
 * @tparam D 	Number of dimensions
 * @tparam T	Pixel type
 */
template <int D, typename T>
class NDImageStore :  public NDArrayStore<D,T>, public NDImage
{

public:
	/**
	 * @brief Create an image with default orientation, of the specified size
	 *
	 * @param dim	number of image dimensions
	 * @param size	vector of size dim, with the image size
	 * @param orient orientation
	 */

	NDImageStore(std::initializer_list<size_t> a_args);
	NDImageStore(size_t size[D]);
	

	/*************************************************************************
	 * Coordinate Transform Functions
	 ************************************************************************/
	int indToPt(double index[D], double ras[D]);
	int ptToInd(double ras[D], double index[D]);
	int indToPt(std::initializer_list<double> index[D], 
			std::initializer_list<double> ras[D]);
	int ptToInd(std::initializer_list<double> ras[D], 
			std::initializer_list<double> index[D]);
	

	void orientDefault();
	void updateAffine();
	void printSelf();

	// used to transform index to RAS (Right Handed Coordinate System)
	double _m_dir[D*D];
	double _m_space[D];
	double _m_origin[D];

	// chache of the affine index -> RAS (right handed coordiante system)
	double _m_affine[(D+1)*(D+1)];

//	NDImage(size_t dim, size_t* size);
//
//	/* Functions which operate pixelwise on the data */
//	template <F>
//	void apply(void* data);
//	void apply(void(*cb)(double), void* data);

};


/**
 * @brief Constructor with initializer list
 *
 * @param a_args dimensions of input, the length of this initializer list
 * may not be fully used if a_args is longer than D. If it is shorter
 * then D then additional dimensions are left as size 1.
 */
template <int D,typename T>
NDImageStore<D,T>::NDImageStore(std::initializer_list<size_t> a_args) :
	NDImageStore<D,T>(a_args)
{
	orientDefault();
}

/**
 * @brief Constructor with array to set size
 *
 * @param dim dimensions of input 
 */
template <int D,typename T>
NDImageStore<D,T>::NDImageStore(size_t dim[D]) : NDArrayStore<D,T>(dim)  
{
	orientDefault();
}


/**
 * @brief Default orientation (dir=ident, space=1 and origin=0)
 */
template <int D,typename T>
void NDImageStore<D,T>::orientDefault()
{
	
	for(size_t ii=0; ii<D; ii++) {
		_m_space[ii] = 1;
		_m_origin[ii] = 0;
		for(size_t jj=0; jj<D; jj++) {
			_m_dir[ii*D+jj] = (ii==jj);
		}
	}

	updateAffine();
}

/**
 * @brief Updates index->RAS affine transform cache
 */
template <int D,typename T>
void NDImageStore<D,T>::updateAffine()
{
	// first DxD section
	for(size_t ii=0; ii<D; ii++) {
		for(size_t jj=0; jj<D; jj++) {
			_m_affine[ii*(D+1)+jj] = _m_dir[ii*D+jj]*_m_space[jj];
		}
	}
		
	// bottom row
	for(size_t jj=0; jj<D; jj++) 
		_m_affine[D*(D+1)+jj] = 0;
	
	// last column
	for(size_t ii=0; ii<D; ii++) 
		_m_affine[ii*(D+1)+D] = 0;

	// bottom right
	_m_affine[D*(D+1)+D] = 1;
}

/**
 * @brief Just dump image information
 */
template <int D,typename T>
void NDImageStore<D,T>::printSelf()
{
	std::cerr << D << "D Image\n[";
	for(size_t ii=0; ii<D; ii++)
		std::cerr << dim(ii) << ", ";
	std::cerr << "]\n";

	std::cerr << "Orientation:\nOrigin: [";
	for(size_t ii=0; ii<D; ii++) 
		std::cerr << _m_origin[ii] << ", ";
	std::cerr << "\nSpacing: [";
	for(size_t ii=0; ii<D; ii++) 
		std::cerr << _m_space[ii] << ", ";
	std::cerr << "\nDirection:\n";
	for(size_t ii=0; ii<D; ii++) {
		std::cerr << "[";
		for(size_t jj=0; jj<D; jj++) 
			std::cerr << _m_dir[ii*D+jj] << ", ";
		std::cerr << "]\n";
	}
	std::cerr << "\nAffine:\n";
	for(size_t ii=0; ii<D+1; ii++) {
		std::cerr << "[";
		for(size_t jj=0; jj<D+1; jj++) 
			std::cerr << _m_affine[ii*(D+1)+jj] << ", ";
		std::cerr << "]\n";
	}
}

#endif 
