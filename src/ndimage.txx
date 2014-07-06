
namespace npl {

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

template <int D, typename T>
void NDImageStore<D,T>::write(std::string filename, double version) const
{
	std::string mode = "wb";
	const size_t BSIZE = 1024*1024; //1M
	gzFile gz;

	// remove .gz to find the "real" format, 
	std::string fn_nz;
	if(fn.substr(fn.size()-3, 3) == ".gz") {
		fn_nz = fn.substr(0, fn.size()-3);
	} else {
		// if no .gz, then make encoding "transparent" (plain)
		fn_nz = fn;
		mode += 'T';
	}
	
	// go ahead and open
	gz = gzopen(fn.c_str(), mode.c_str());
	gzbuffer(gz, BSIZE);

	if(fn_nz.substr(fn_nz.size()-4, 4) == ".nii") {
		if(version >= 2) {
			if(writeNifti2Image(gz) != 0) {
				std::cerr << "Error writing" << std:: endl;
				gzclose(gz);
				return -1;
			}
		} else {
			if(writeNifti1Image(gz) != 0) {
				std::cerr << "Error writing" << std:: endl;
				gzclose(gz);
				return -1;
			}
		}
	} else {
		std::cerr << "Unknown filetype: " << fn_nz.substr(fn_nz.rfind('.')) 
			<< std::endl;
		gzclose(gz);
		return -1;
	}

	gzclose(gz);
	return 0;
}

template <int D, typename T>
void NDImageStore<D,T>::writeNifti1Image(gzFile file) const
{
	int32_t headsize = 348;
	char unused[10+18+4+2+1];
	char dim_info = 0;
	short numdims = (short)ndim();
	short dimsize[7] = {0,0,0,0,0,0,0};

	gzwrite(file, &headsize, sizeof(headsize));
	gzwrite(file, unused, sizeof(unused));

	if(m_freqdim >= 0) dim_info |= (char)(((m_freqdim+1)) & 0x03);
	if(m_phasedim >= 0) dim_info |= (char)(((m_phasedim+1)) & 0x03);
	if(m_slicedim >= 0) dim_info |= (char)(((m_slicedim+1)) & 0x03);
	gzwrite(file, dim_info, sizeof(char));

	// TODO HERE
	for(size_t ii=0; ii<ndim(); ii++) {

	}

	gzwrite(file, numdims, sizeof(numdims));
	gzwrite(file, dimsize, sizeof(numdims));

	header.ndim = out->ndim();
	for(size_t dd=0; dd<out->ndim(); dd++) {
		header.dim[dd] = out->dim(dd);
		header.pixdim[dd] = out->space(dd);
	}

	std::cerr << "Error NiftiWriter not yet implemented" << std::endl;
	throw (-1);
	
	if(out->quatern) {
		
	}
	double a = 0.5*sqrt(1+R11+R22+R33);
    header.quatern_b = 0.25*(R32-R23)/a;
	header.quatern_c = 0.25*(R13-R31)/a;
	header.quatern_d = 0.25*(R21-R12)/a

	// read the pixels
	// note x is the fastest in nifti, for us it is the slowest
	switch(header.datatype) {
		case DT_INT32:
			break;
		case DT_FLOAT:
			break;
	}
	return 0;
}

} //npl
