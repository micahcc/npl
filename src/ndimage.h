#ifndef NDIMAGE_H
#define NDIMAGE_H

#include "ndarray.h"

#include <string>
#include <iostream>

class NDImage;

// simply reads an image in its native type
NDImage* readNDImage(std::string filename);
void writeNDImage(NDImage* img, std::string filename);

NDImage* createNDImage(size_t dim

/**
 * @brief NDImage can basically be used like an NDArray, with the addition
 * of orientation related additions.
 */
class NDImage : public virtual NDArray
{
};


/**
 * @brief NDImageStore is a version of NDArray that has an orientation matrix.
 * Right now it also has additional data that is unique to nifti. Eventually
 * this class will be forked into a subclass, and this will only have the 
 * orientation.
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

private:
	// used to transform index to RAS (Right Handed Coordinate System)
	double m_dir[D*D];
	double m_space[D];
	double m_origin[D];
	std::string m_units[D];

	// chache of the affine index -> RAS (right handed coordiante system)
	double m_affine[(D+1)*(D+1)];
	
	/*
	 * medical image specific stuff, eventually these should be moved to a 
	 * medical image subclass
	 */

	// each slice is given its relative time, with 0 as the first
	vector<double> m_slice_timing;
	
	// < 0 indicate unset variables
	int m_freqdim;
	int m_phasedim;
	int m_slicedim;

	/*
	 * nifti specific stuff, eventually these should be moved to a nifti 
	 * image subclass
	 */

	// raw values for slice data, < 0 indicate unset
	double m_slice_duration;
	int m_slice_start;
	int m_slice_end;

	// SEQ, RSEQ, ALT, RALT, ALT_P1, RALT_P1
	// SEQ (sequential): 	slice_start .. slice_end
	// RSEQ (reverse seq): 	slice_end .. slice_start
	// ALT (alternated): 	slice_start, slice_start+2, .. slice_end|slice_end-1,
	// 						slice_start+1 .. slice_end|slice_end-1
	// RALT (reverse alt): 	slice_end, slice_end-2, .. slice_start|slice_start+1,
	// 						slice_end-1 .. slice_start|slice_start+1
	// ALT_P1 (siemens alt):slice_start+1, slice_start+3, .. slice_end|slice_end-1,
	// 						slice_start .. slice_end|slice_end-1
	// RALT (reverse alt): 	slice_end-1, slice_end-3, .. slice_start|slice_start+1,
	// 						slice_end-2 .. slice_start|slice_start+1
	std::string m_slice_order;

	// if quaternians are the original direction base, then this stores the 
	// raw quaternian values, to prevent roundoff errors
	bool use_quaterns;;
	double quaterns[4];

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
	NDArrayStore<D,T>(dim), m_freqdim(-1), m_phasedim(-1), m_slicedim(-1),
	m_slice_duration(-1), m_slice_start(-1), m_slice_end(-1), use_quaterns(0)
{
	orientDefault();
}

/**
 * @brief Constructor with array to set size
 *
 * @param dim dimensions of input 
 */
template <int D,typename T>
NDImageStore<D,T>::NDImageStore(size_t dim[D]) : 
	NDArrayStore<D,T>(dim), m_freqdim(-1), m_phasedim(-1), m_slicedim(-1),
	m_slice_duration(-1), m_slice_start(-1), m_slice_end(-1), use_quaterns(0)
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
		m_space[ii] = 1;
		m_origin[ii] = 0;
		for(size_t jj=0; jj<D; jj++) {
			m_dir[ii*D+jj] = (ii==jj);
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
			m_affine[ii*(D+1)+jj] = m_dir[ii*D+jj]*m_space[jj];
		}
	}
		
	// bottom row
	for(size_t jj=0; jj<D; jj++) 
		m_affine[D*(D+1)+jj] = 0;
	
	// last column
	for(size_t ii=0; ii<D; ii++) 
		m_affine[ii*(D+1)+D] = 0;

	// bottom right
	m_affine[D*(D+1)+D] = 1;
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
		std::cerr << m_origin[ii] << ", ";
	std::cerr << "\nSpacing: [";
	for(size_t ii=0; ii<D; ii++) 
		std::cerr << m_space[ii] << ", ";
	std::cerr << "\nDirection:\n";
	for(size_t ii=0; ii<D; ii++) {
		std::cerr << "[";
		for(size_t jj=0; jj<D; jj++) 
			std::cerr << m_dir[ii*D+jj] << ", ";
		std::cerr << "]\n";
	}
	std::cerr << "\nAffine:\n";
	for(size_t ii=0; ii<D+1; ii++) {
		std::cerr << "[";
		for(size_t jj=0; jj<D+1; jj++) 
			std::cerr << m_affine[ii*(D+1)+jj] << ", ";
		std::cerr << "]\n";
	}
}

template <typename T, typename DIMT>
NDImage* createNDImage(size_t ndim, DIMT* dims)
{
	const size_t MAXDIM = 7;
	size_t truedim[MAXDIM]
	switch(ndim) {
		case 1:
			for(size_t ii=0; ii<ndim; ii++)
				truedim[ii] = (size_t)dims[ii];
			return NDImageStore<1, T>(truedim);
		case 2:
			for(size_t ii=0; ii<ndim; ii++)
				truedim[ii] = (size_t)dims[ii];
			return NDImageStore<2, T>(truedim);
		case 3:
			for(size_t ii=0; ii<ndim; ii++)
				truedim[ii] = (size_t)dims[ii];
			return NDImageStore<3, T>(truedim);
		case 4:
			for(size_t ii=0; ii<ndim; ii++)
				truedim[ii] = (size_t)dims[ii];
			return NDImageStore<4, T>(truedim);
		case 5:
			for(size_t ii=0; ii<ndim; ii++)
				truedim[ii] = (size_t)dims[ii];
			return NDImageStore<5, T>(truedim);
		case 6:
			for(size_t ii=0; ii<ndim; ii++)
				truedim[ii] = (size_t)dims[ii];
			return NDImageStore<6, T>(truedim);
		case 7:
			for(size_t ii=0; ii<ndim; ii++)
				truedim[ii] = (size_t)dims[ii];
			return NDImageStore<7, T>(truedim);
		default:
			std::cerr << "Unsupported dimension: " << ndim << std::endl;
			return NULL;

	}
	return NULL;
}


#endif 
