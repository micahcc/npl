#include "ndarray.h"

#include <string>

class NDImage;

// simply reads an image in its native type
NDImage* readNDImage(std::string filename);

// enforces given pixel type
template <typename T>
NDImage* readNDImage(std::string filename);

NDImage* readNiftiImage(std::string filename);


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
	NDImageStore(size_t size[D]) : NDArrayStore<D,T>(size), _m_orient({D,D}) {};

//	NDImage(size_t dim, size_t* size);
//
//	/* Functions which operate pixelwise on the data */
//	template <F>
//	void apply(void* data);
//	void apply(void(*cb)(double), void* data);

	Matrix _m_orient;

};

