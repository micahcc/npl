#include "ndarray.h"

#include <string>

class NDImage;

// simply reads an image in its native type
NDImage* readNDImage(std::string filename);

// enforces given pixel type
template <typename T>
NDImage* readNDImage(std::string filename);

NDImage* readNiftiImage(std::string filename);

class NDImage : public virtual NDArray
{
};

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
	NDImageStore(size_t size[D]) : NDArrayStore<D,T>(size), m_orient({D,D}) {} ;
//	NDImage(size_t dim, size_t* size);
//
//	/* Functions which operate pixelwise on the data */
//	template <F>
//	void apply(void* data);
//	void apply(void(*cb)(double), void* data);

private:

	Matrix m_orient;

};

