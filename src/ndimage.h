#ifndef NDIMAGE_H
#define NDIMAGE_H

#include "ndarray.h"
#include "npltypes.h"

#include <string>
#include <iostream>
#include <cassert>

namespace npl {

class NDImage;

/**
 * @brief NDImage can basically be used like an NDArray, with the addition
 * of orientation related additions.
 */
class NDImage : public virtual NDArray
{
public:

	enum PixelT { UINT8=2, INT16=4, INT32=8, FLOAT32=16, COMPLEX64=32,
		FLOAT64=64, RGB24=128, INT8=256, UINT16=512, UINT32=768, INT64=1024,
		NIFTI_TYPE_UINT64=1280, FLOAT128=1536, COMPLEX128=1792,
		COMPLEX256=2048, RGBA32=2304 };

	virtual double& space(size_t d) = 0;
	virtual double& origin(size_t d) = 0;
	virtual double& direction(size_t d1, size_t d2) = 0;
	virtual double& affine(size_t d1, size_t d2) = 0;
	virtual void updateAffine() = 0;
	

//	NDImage() : m_freqdim(-1), m_phasedim(-1), m_slicedim(-1), 
//				m_slice_duration(-1), m_slice_start(-1), m_slice_end(-1), 
//				use_quaterns(0) 
//	{
//
//	};

	/*
	 * medical image specific stuff, eventually these should be moved to a 
	 * medical image subclass
	 */

	// each slice is given its relative time, with 0 as the first
	std::vector<double> m_slice_timing;
	
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
 * @brief NDImageStore is a version of NDArray that has an orientation matrix.
 * Right now it also has additional data that is unique to nifti. Eventually
 * this class will be forked into a subclass, and this will only have the 
 * orientation.
 *
 * @tparam D 	Number of dimensions
 * @tparam T	Pixel type
 */
template <int D, typename T>
class NDImageStore :  public virtual NDArrayStore<D,T>, public virtual NDImage
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
	NDImageStore(const std::vector<size_t>& a_args);
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

	PixelT type() const;
	double& space(size_t d);
	double& origin(size_t d);
	double& direction(size_t d1, size_t d2);
	double& affine(size_t d1, size_t d2);
	const double& space(size_t d) const;
	const double& origin(size_t d) const;
	const double& direction(size_t d1, size_t d2) const;
	const double& affine(size_t d1, size_t d2) const;

	std::string getUnits(size_t d1, double d2);

private:
	// used to transform index to RAS (Right Handed Coordinate System)
	double m_dir[D*D];
	double m_space[D];
	double m_origin[D];
	std::string m_units[D];
	
	// chache of the affine index -> RAS (right handed coordiante system)
	double m_affine[(D+1)*(D+1)];

	
	PixelT pixeltype;

};

// simply reads an image in its native type
NDImage* readNDImage(std::string filename, bool verbose);
NDImage* createNDImage(size_t ndim, size_t* dims, NDImage::PixelT);
int writeNDImage(NDImage* img, std::string fn, bool nifti2 = true);


} // npl
#endif 
