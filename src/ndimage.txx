
/**
 * @brief Constructor with initializer list
 *
 * @param a_args dimensions of input, the length of this initializer list
 * may not be fully used if a_args is longer than D. If it is shorter
 * then D then additional dimensions are left as size 1.
 */
template <int D,typename T>
NDImageStore<D,T>::NDImageStore(std::initializer_list<size_t> a_args) :
	NDArrayStore<D,T>(a_args)
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
	NDArrayStore<D,T>(dim)
{
	orientDefault();
}

/**
 * @brief Constructor with array to set size
 *
 * @param dim dimensions of input 
 */
template <int D,typename T>
NDImageStore<D,T>::NDImageStore(const std::vector<size_t>& dim) : 
	NDArrayStore<D,T>(dim)
{
	orientDefault();
}

/**
 * @brief Returns the enumerated value for the image pixel type
 *
 * @return 
 */
template <int D,typename T>
NDImage::PixelT NDImageStore<D,T>::type() const 
{
	return pixeltype;
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
	size_t truedim[MAXDIM];
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

template <int D, typename T>
double& NDImageStore<D,T>::space(size_t d) 
{ 
	assert(d<D); 
	return m_space[d]; 
};

template <int D, typename T>
double& NDImageStore<D,T>::origin(size_t d) 
{
	assert(d<D); 
	return m_origin[d]; 
};

template <int D, typename T>
double& NDImageStore<D,T>::direction(size_t d1, size_t d2) 
{
	assert(d1 < D && d2 < D);
	return m_dir[D*d1+d2]; 
};

template <int D, typename T>
double& NDImageStore<D,T>::affine(size_t d1, size_t d2) 
{ 
	assert(d1 < D+1 && d2 < D+1);
	return m_dir[(D+1)*d1+d2]; 
};

template <int D, typename T>
const double& NDImageStore<D,T>::space(size_t d) const 
{ 
	assert(d < D);
	return m_space[d]; 
};

template <int D, typename T>
const double& NDImageStore<D,T>::origin(size_t d) const 
{
	assert(d < D);
	return m_origin[d]; 
};

template <int D, typename T>
const double& NDImageStore<D,T>::direction(size_t d1, size_t d2) const 
{ 
	assert(d1 < D && d2 < D);
	return m_dir[D*d1+d2]; 
};

template <int D, typename T>
const double& NDImageStore<D,T>::affine(size_t d1, size_t d2) const 
{
	assert(d1 < D+1 && d2 < D+1);
	return m_dir[(D+1)*d1+d2]; 
};


